"""
Reduced-order mechanical models for structures made of repeating cells, built on the
discretisation of omc.mech_bloch (same element matrices, grids and DOF numbering), with
sensitivities to every cell's fraction map.

- `BlochROM`: Bloch mode synthesis (BMS; Krattiger & Hussein, PRE 90, 063306 (2014); local
  interface reduction and residual enhancement as in their generalised BMS, JCP 357 (2018))
  for the bands of a unit cell or supercell. The cell is cut open along the node lines i = 0
  and j = 0 (copies per side: segments "L", "B" and four corner copies "C"); the rest is the
  interior I. Craig–Bampton on the open cell: fixed-interface modes (the lowest n_fixed of
  K_II, M_II, or those nearest σ₀ when sigma0_hz > 0) and constraint modes −A_II⁻¹A_IB,
  A = K − σ₀M. Boundary reduction: each corner keeps its DOFs, lifted by "hat" ramps along the
  adjacent segments (so rigid motions stay exact); each segment gets vectors that vanish at
  the corners, from its two-sided interface problem (two copies of the cell glued along it):
  residual vectors, the glued flexibility applied to the loads the retained columns put on the
  segment, compressed by SVD (n_residual), optionally plus that problem's lowest modes
  (n_iface). Interior residual vectors (A_II⁻¹ minus the kept modes, applied to the boundary
  columns' inertia loads; n_interior_residual) cut the fixed-mode truncation. Both copies of a
  segment share one basis, so u_R = e^{iθ1} u_L holds exactly in the reduced coordinates:
  K_r(k) = Σ_d e^{i d·θ} K_d over nine k-independent dense blocks, one small dense Hermitian
  eigenproblem per k. `reproject` re-evaluates a new fraction map on the same basis.

- `ChainROM`: component mode synthesis of a finite chain of cells along a1 (a cavity of N
  columns with its shield rows, closed along a2 at k·a2 = 0, periodic or free-ended along a1),
  each cell with its own fraction raster and optional absorber η (complex mass ρ(1 − iη) as in
  omc.cavity, so K − λ(M − iA) stays complex symmetric and Q = Re f / 2 Im f). Each distinct
  raster is reduced on its own by a frequency-shifted Craig–Bampton: fixed-interface modes
  nearest the target σ, interface lifts −A_II⁻¹A_IΓΦ_Γ exact at σ, and their first moment
  A_II⁻¹M_II·lifts (first order in λ − σ). All interfaces share one basis Φ_Γ (the Bloch-wave
  traces at σ of a few reference cells: block Krylov space of the glued two-cell interface
  flexibility, ranked by the interface QEP's |ln μ|), so a cell's reduced matrices depend on
  that cell alone: `set_cell` re-reduces one raster, or with frozen=True projects it on the
  basis of its previous self or of neighbours (no factorisation). Absorbers only add VᵀAV per
  instance. The chain is assembled block-banded and solved by sparse shift-invert.

Sensitivities. Both models evaluate the full-model formula with the reconstructed mode
x = V q, dλ/df_e = u_eᵀ(K_e − λ M_e^c)u_e / xᵀM^c x (unconjugated, complex symmetric; the
Hermitian form for Bloch modes). With the basis held fixed (BlochROM.reproject, frozen chain
cells) this is the exact gradient of the reduced eigenvalue. When the basis is rebuilt with the
design, the basis-derivative term is 2(dV q)ᵀ r with r = (K − λM)x the full-model residual of
the reconstructed mode, so the projected gradient is the full-model gradient to the reduced
model's eigenvector accuracy.

Reconstruction: `BlochROM.reconstruct` gives a vector on the BlochSolver DOFs (for
omc.mech_adjoint, omc.coupling.mech_mode); `ChainROM.full_vector` on the DOFs of the whole
chain's BlochSolver (that of omc.cavity.mech_cavity on the same rasters), with `ChainROM.grid()`
and `mass_operator()` standing in for that solver in coupling.mech_mode (g0, m_eff) without
assembling it.

Units as omc.mech_bloch: µm, Hz; eigenvalues λ = (2π f / OMEGA_UNIT)².
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .linalg import factorize, factorize_indefinite
from .mech_adjoint import BandSensitivity, find_clusters
from .mech_bloch import OMEGA_UNIT, BlochSolver, ElementPart

SIGMA0_HZ = -0.5e9  # shift of the static reductions (as BlochSolver.solve): K − σ₀M is positive definite


def lam_of_hz(f_hz: float) -> float:
    return float(np.sign(f_hz) * (2 * np.pi * f_hz / OMEGA_UNIT) ** 2)


def hz_of_lam(lam):
    """Frequency in Hz of eigenvalues λ (complex λ gives complex frequencies, Im > 0 decaying)."""
    lam = np.asarray(lam)
    if np.iscomplexobj(lam):
        return np.sqrt(lam.astype(complex)) * OMEGA_UNIT / (2 * np.pi)
    return OMEGA_UNIT * np.sqrt(np.clip(lam, 0, None)) / (2 * np.pi)


# --- the open cell ----------------------------------------------------------------------------


def _dofs_per_node(solver: BlochSolver) -> int:
    if solver.model == "slab":
        raise ValueError("mech_rom supports the 2D-mesh models (plane_stress/strain, slab_p, stack), not 'slab'")
    nd = solver.parts[0].Ke.shape[0]
    if solver.model == "stack":
        order = nd // 12 - 1
        return 3 * (len(solver.parts) * order + 1)
    return nd // 4


def part_fractions(fraction=None, layers=None, model: str = "plane_stress") -> list[np.ndarray]:
    """The fraction map of each part of a BlochSolver built with these arguments."""
    return [np.asarray(L.fraction, float) for L in layers] if model == "stack" else [np.asarray(fraction, float)]


@dataclass
class OpenCell:
    """
    A BlochSolver's mesh cut open along the axes in open_axes: each periodic DOF p on node
    line i = 0 (axis 1) and/or j = 0 (axis 2) gets a copy per side, copy (c1, c2) ∈ {0, 1}²
    (c = 1: the copy reached by wrapping, i.e. the right / top side). Closed axes keep the
    periodic identification with the fixed phase e^{iθ}. Open DOF o is (p[o], copy[o]).

    eopen[part]: (ne, nd) open DOF of each element DOF; ephase[part]: the closed-axis phase of
    each element DOF (None when all are 1). weights[part]: each kept element's fraction.
    node_i, node_j, comp: grid node and basis index of every periodic DOF.
    """

    solver: BlochSolver
    weights: list[np.ndarray]
    eopen: list[np.ndarray]
    ephase: list[np.ndarray] | None
    p: np.ndarray
    copy: np.ndarray
    node_i: np.ndarray
    node_j: np.ndarray
    comp: np.ndarray
    dpn: int

    @property
    def n_open(self) -> int:
        return len(self.p)

    def assemble(self, which: str = "K", eta: list[np.ndarray] | None = None) -> sp.csr_matrix:
        """Σ_e w_e P_eᴴ E_e P_e over all parts, E = Ke ("K") or Me ("M"); eta (per part, per element) scales Me by η_e ("A")."""
        rows, cols, vals = [], [], []
        for q, part in enumerate(self.solver.parts):
            E = part.Ke if which == "K" else part.Me
            w = self.weights[q] if which != "A" else self.weights[q] * eta[q]
            sel = np.flatnonzero(w != 0)
            eo = self.eopen[q][sel]
            nd = eo.shape[1]
            v = w[sel, None, None] * E[None]
            if self.ephase is not None:
                ph = self.ephase[q][sel]
                v = np.conj(ph)[:, :, None] * v * ph[:, None, :]
            rows.append(np.broadcast_to(eo[:, :, None], (len(sel), nd, nd)).ravel())
            cols.append(np.broadcast_to(eo[:, None, :], (len(sel), nd, nd)).ravel())
            vals.append(v.ravel())
        n = self.n_open
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))

    def element_vectors(self, part: int, x: np.ndarray) -> np.ndarray:
        """(ne, nd, nv) open-cell vectors x (n_open, nv) gathered onto part's element DOFs (with closed-axis phases)."""
        U = x[self.eopen[part]]
        if self.ephase is not None:
            U = U * self.ephase[part][..., None]
        return U


def open_cell(solver: BlochSolver, fractions: list[np.ndarray], open_axes=(True, True), theta=(0.0, 0.0)) -> OpenCell:
    dpn = _dofs_per_node(solver)
    n1, n2 = solver.shape
    keys, phases = [], []
    closed_phase = not all(open_axes) and np.any(np.asarray(theta)[[not a for a in open_axes]] != 0)
    for part in solver.parts:
        if np.any(part.edof < 0):
            raise ValueError("mirror-fixed DOFs are not supported")
        c1 = part.W1 if open_axes[0] else 0 * part.W1
        c2 = part.W2 if open_axes[1] else 0 * part.W2
        keys.append((part.edof.astype(np.int64) * 2 + c1) * 2 + c2)
        if closed_phase:
            ph = np.ones(part.edof.shape, complex)
            if not open_axes[0]:
                ph = ph * np.exp(1j * theta[0] * part.W1)
            if not open_axes[1]:
                ph = ph * np.exp(1j * theta[1] * part.W2)
            phases.append(ph)
    allk, inv = np.unique(np.concatenate([k.ravel() for k in keys]), return_inverse=True)
    eopen, start = [], 0
    for k in keys:
        eopen.append(inv[start : start + k.size].reshape(k.shape))
        start += k.size
    p = allk // 4
    copy = np.stack([(allk // 2) % 2, allk % 2], axis=1)
    node = solver.dofs // dpn
    weights = [f.ravel()[part.cell] for f, part in zip(fractions, solver.parts)]
    return OpenCell(solver, weights, eopen, phases if closed_phase else None, p, copy, node // n2, node % n2, solver.dofs % dpn, dpn)


def _selection(n_rows: int, rows: np.ndarray) -> sp.csr_matrix:
    """n_rows × len(rows) 0/1 matrix with column c selecting row rows[c] (rows[c] < 0: empty column)."""
    ok = rows >= 0
    return sp.csr_matrix((np.ones(ok.sum()), (rows[ok], np.flatnonzero(ok))), shape=(n_rows, len(rows)))


def _blocked_product(A: sp.spmatrix, G: np.ndarray, H: np.ndarray | None = None, block: int = 256) -> np.ndarray:
    """Hᵀ A G (H defaults to G), by column blocks of G to bound the memory."""
    H = G if H is None else H
    out = np.empty((H.shape[1], G.shape[1]), dtype=np.result_type(A.dtype, G.dtype, H.dtype))
    for s in range(0, G.shape[1], block):
        out[:, s : s + block] = H.T @ (A @ G[:, s : s + block])
    return out


def _m_orthonormal(B: np.ndarray, M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """An M-orthonormal basis of span(B), dropping directions below tol (relative)."""
    g = B.T @ (M @ B)
    e, V = np.linalg.eigh((g + g.T) / 2)
    keep = e > tol * e.max()
    return B @ (V[:, keep] / np.sqrt(e[keep]))


def _eigs_nearest(K, M, sigma: float, n: int, solve=None, tol: float = 1e-10):
    """n eigenpairs of the real symmetric pencil (K, M) nearest sigma, by shift-invert; ascending."""
    if solve is None:
        solve = factorize(K - sigma * M, hpd=sigma < 0) if sigma < 0 else factorize_indefinite(K - sigma * M)
    op = spla.LinearOperator(K.shape, matvec=solve, dtype=float)
    n = min(n, K.shape[0] - 2)
    lam, X = spla.eigsh(K, k=n, M=M, sigma=sigma, which="LM", OPinv=op, tol=tol)
    o = np.argsort(lam)
    return lam[o], X[:, o]


# --- Bloch mode synthesis ---------------------------------------------------------------------


class BlochROM:
    """
    Bloch mode synthesis of one (super)cell; the arguments before n_fixed are BlochSolver's.

    n_fixed: fixed-interface modes (the lowest; nearest σ₀ when sigma0_hz > 0). n_residual:
    residual vectors per segment, an int or (n_L, n_B) for the i = 0 (along a2) and j = 0
    (along a1) lines. n_iface: extra glued-problem modes per segment (default none).
    n_interior_residual: interior residual vectors. f_ref_hz: frequency weighting the inertia
    loads. sigma0_hz: shift of the condensation; the default (slightly negative) suits the
    lowest bands, a frequency inside the band of interest (e.g. a line-defect window, solved
    with solve(..., window_hz=…)) makes the condensation exact there. Defaults (60, 120, 60) give
    ≤ 5e-4 relative on the lowest 6–7 bands of the 4 GHz cells at n = 64–96, mesh-independently.

    Attributes: n_reduced (size of the k-problems), f_fixed_max (Hz), n_residual (per segment),
    n_interior_residual, timings (s).
    """

    def __init__(
        self,
        lattice,
        fraction: np.ndarray | None = None,
        material=None,
        n_fixed: int = 60,
        n_iface: int | tuple[int, int] = 0,
        n_residual: int | tuple[int, int] = 120,
        n_interior_residual: int = 60,
        f_ref_hz: float = 5e9,
        residual_tol: float = 1e-6,
        sigma0_hz: float = SIGMA0_HZ,
        **solver_kwargs,
    ):
        t0 = time.perf_counter()
        self.solver = BlochSolver(lattice, fraction, material, **solver_kwargs)
        self.lattice = lattice
        fr = part_fractions(fraction, solver_kwargs.get("layers"), solver_kwargs.get("model", "plane_stress"))
        self.fractions = fr
        oc = self.cell = open_cell(self.solver, fr, (True, True))
        K0, M0 = oc.assemble("K"), oc.assemble("M")
        self.timings = {"assemble": time.perf_counter() - t0}

        n1, n2 = self.solver.shape
        i, j = oc.node_i[oc.p], oc.node_j[oc.p]
        c1, c2 = oc.copy.T
        corner, left, bottom = (i == 0) & (j == 0), (i == 0) & (j != 0), (j == 0) & (i != 0)
        interior = ~(corner | left | bottom)
        I = np.flatnonzero(interior)
        n_open = oc.n_open
        # Segments: periodic DOFs of the segment, and for each copy w the open DOF (or -1).
        segs = {}
        for name, mask, copies in (
            ("L", left, [(0, 0), (1, 0)]),
            ("B", bottom, [(0, 0), (0, 1)]),
            ("C", corner, [(0, 0), (1, 0), (0, 1), (1, 1)]),
        ):
            pp = np.unique(oc.p[mask])
            pos = {}
            for w in copies:
                sel = np.flatnonzero(mask & (c1 == w[0]) & (c2 == w[1]))
                rows = np.full(len(pp), -1)
                rows[np.searchsorted(pp, oc.p[sel])] = sel
                pos[w] = rows
            segs[name] = (pp, pos)

        KII, MII = K0[I][:, I].tocsc(), M0[I][:, I].tocsc()
        sig0 = lam_of_hz(sigma0_hz)
        t1 = time.perf_counter()
        shifted = sig0 > 0  # condensation exact inside the band of interest: indefinite LU
        solve = factorize_indefinite(KII - sig0 * MII) if shifted else factorize(KII - sig0 * MII, hpd=True)
        self.sigma0 = sig0
        lam_f, Phi = _eigs_nearest(KII, MII, sig0, n_fixed, solve)
        self.timings["fixed_modes"] = time.perf_counter() - t1
        self.f_fixed_max = float(hz_of_lam(lam_f[-1]))

        # Boundary basis. Corners: kept, each with a "hat" lift (a linear ramp to zero along the
        # adjacent segments, same basis function), so the corner columns sum to rigid motions.
        # Segments: vectors vanishing at the corners, from the two-sided interface problem (two
        # copies of the cell glued along the segment, all other boundaries clamped): optionally
        # its n_iface lowest modes, and residual vectors, its flexibility (at σ₀) applied to the
        # loads the retained columns put on the segment (residual-mode enhancement), compressed
        # by SVD. Both copies of a segment share one basis, so the Bloch condition stays exact.
        t1 = time.perf_counter()
        A0 = (K0 - sig0 * M0).tocsr()
        AI = A0[I]
        AII = AI[:, I].tocsc()
        pair = lambda v: (v, v) if np.isscalar(v) else tuple(v)  # noqa: E731
        n_if, n_res = dict(zip("LB", pair(n_iface))), dict(zip("LB", pair(n_residual)))

        def lift_cols(rows, T):
            """Open-cell columns with values T on the copy `rows` (-1: absent) and their interior static response."""
            Y = np.zeros((n_open, T.shape[1]))
            ok = rows >= 0
            Y[rows[ok]] = T[ok]
            Y[I] = -solve(AI @ Y)
            return Y

        pc, cpos = segs["C"]
        corner_cols = {w: lift_cols(rows, np.eye(len(pc))) for w, rows in cpos.items()} if len(pc) else {}
        rel = {"L": [(0, 0), (0, 1)], "B": [(0, 0), (1, 0)]}
        for name in ("L", "B"):
            pp, pos = segs[name]
            if not len(pp) or not len(pc):
                continue
            along = (oc.node_j[pp] / n2) if name == "L" else (oc.node_i[pp] / n1)
            for r in rel[name]:
                ramp = along if (r[1] if name == "L" else r[0]) else 1 - along
                Psi = ramp[:, None] * (oc.comp[pp][:, None] == oc.comp[pc][None, :])
                for w, rows in pos.items():
                    wc = (w[0] + r[0], w[1] + r[1])
                    if wc in corner_cols:
                        corner_cols[wc] += lift_cols(rows, Psi)

        # Loads of the retained columns (unit M-norm; interior modes weighted by their
        # participation ~ λ_ref/λ_j), as stiffness and λ_ref-weighted inertia: P = [A g, λ_ref M g],
        # and W = A_II⁻¹ P_I, so that the load on a segment copy is P_s − A_sI W.
        lam_ref = lam_of_hz(f_ref_hz)
        GI = np.zeros((n_open, Phi.shape[1]))
        GI[I] = Phi
        Gr = np.concatenate([GI] + list(corner_cols.values()), axis=1)
        wgt = 1 / np.sqrt(np.maximum(np.einsum("ij,ij->j", Gr, M0 @ Gr), 1e-300))
        wgt[: Phi.shape[1]] *= np.minimum(1.0, np.abs(lam_ref - sig0) / np.abs(lam_f - sig0))
        Gr *= wgt
        P = np.concatenate([A0 @ Gr, lam_ref * (M0 @ Gr)], axis=1)
        del Gr
        W = solve(P[I])

        self.f_iface_max, self.n_residual = {}, {}
        groups = []  # (q-block name, copy w, G block (n_open × m))
        for name in ("L", "B"):
            pp, pos = segs[name]
            if not len(pp):
                continue
            (w0, r0), (w1, r1) = pos.items()
            liv = np.flatnonzero((r0 >= 0) | (r1 >= 0))
            S0, S1 = _selection(n_open, r0[liv]), _selection(n_open, r1[liv])
            Dm = (S0.T @ M0 @ S0 + S1.T @ M0 @ S1).tocsc()
            # Glued system [interior A, interior B, shared segment]: A's right copy is B's left copy,
            # at σ₀ (indefinite for a shifted ROM: the dynamic flexibility matters there).
            gsig = sig0
            D = (S0.T @ A0 @ S0 + S1.T @ A0 @ S1).tocsc()
            C1, C0 = (AI @ S1).tocsc(), (AI @ S0).tocsc()
            Gm = sp.bmat([[AII, None, C1], [None, AII, C0], [C1.T, C0.T, D]], format="csc")
            gsolve = factorize_indefinite(Gm) if shifted else factorize(Gm, hpd=True)
            nl = len(liv)
            T = []
            if n_if[name]:
                MI_, M1, M0_ = MII, (M0[I] @ S1).tocsc(), (M0[I] @ S0).tocsc()
                Mg = sp.bmat([[MI_, None, M1], [None, MI_, M0_], [M1.T, M0_.T, Dm]], format="csc")
                lam_g, Xg = _eigs_nearest(Gm + gsig * Mg, Mg, gsig, n_if[name], gsolve)
                T.append(Xg[-nl:])
                self.f_iface_max[name] = float(hz_of_lam(lam_g[-1]))
            if n_res[name]:
                F = []
                for rows in (r0, r1):
                    rr = rows[liv]
                    ok = rr >= 0
                    Fw = np.zeros((nl, P.shape[1]))
                    Fw[ok] = P[rr[ok]] - A0[rr[ok]][:, I] @ W
                    F.append(Fw)
                F = np.concatenate(F, axis=1)
                rhs = np.zeros((Gm.shape[0], F.shape[1]))
                rhs[-nl:] = F
                R = gsolve(rhs)[-nl:]
                Lc = np.linalg.cholesky(Dm.toarray())
                U, sv, _ = np.linalg.svd(Lc.T @ R, full_matrices=False)
                r = int(min(n_res[name], np.sum(sv > residual_tol * sv[0])))
                T.append(np.linalg.solve(Lc.T, U[:, :r]))
            del gsolve, Gm
            Ts = _m_orthonormal(np.concatenate(T, axis=1), Dm)
            self.n_residual[name] = Ts.shape[1]
            Tfull = np.zeros((len(pp), Ts.shape[1]))
            Tfull[liv] = Ts
            for w, rows in pos.items():
                groups.append((name, w, lift_cols(rows, Tfull)))
        del P, W
        for w, Y in corner_cols.items():
            groups.append(("C", w, Y))
        self.timings["constraint_modes"] = time.perf_counter() - t1

        # Interior residual enhancement: static response of the truncated fixed-interface modes,
        # (A_II⁻¹ − Φ(Λ − σ₀)⁻¹Φᵀ) M_I· g, to the inertia loads of the boundary columns g,
        # compressed in the M_II norm and appended to Φ (interior-only, k-independent).
        self.n_interior_residual = 0
        if n_interior_residual:
            t1 = time.perf_counter()
            Gb = np.concatenate([g[2] for g in groups], axis=1)
            Gb = Gb / np.sqrt(np.maximum(np.einsum("ij,ij->j", Gb, M0 @ Gb), 1e-300))
            L = (M0 @ Gb)[I]
            del Gb
            R = solve(L) - Phi @ ((Phi.T @ L) / (lam_f - sig0)[:, None])
            g = R.T @ (MII @ R)
            e, V = np.linalg.eigh((g + g.T) / 2)
            e, V = e[::-1], V[:, ::-1]
            r = int(min(n_interior_residual, np.sum(e > residual_tol**2 * e[0])))
            R = R @ V[:, :r]
            R -= Phi @ (Phi.T @ (MII @ R))
            R = _m_orthonormal(R, MII)
            Phi = np.concatenate([Phi, R], axis=1)
            self.n_interior_residual = R.shape[1]
            self.timings["interior_residual"] = time.perf_counter() - t1

        # Basis G = [Φ_I | groups], k-independent reduced matrices, then the nine phase classes.
        t1 = time.perf_counter()
        GI = np.zeros((n_open, Phi.shape[1]))
        GI[I] = Phi
        blocks = [("I", (0, 0), GI)] + groups
        self.G = np.concatenate([b[2] for b in blocks], axis=1)
        KG, MG = _blocked_product(K0, self.G), _blocked_product(M0, self.G)
        KG, MG = (KG + KG.T) / 2, (MG + MG.T) / 2
        names = []
        for name, *_ in blocks:
            if name not in names:
                names.append(name)
        width = {name: next(b[2].shape[1] for b in blocks if b[0] == name) for name in names}
        qstart = dict(zip(names, np.cumsum([0] + [width[n] for n in names])[:-1]))
        self.n_reduced = int(sum(width.values()))
        gcol, gstart = [], 0
        for name, w, B in blocks:
            gcol.append((name, w, gstart, B.shape[1]))
            gstart += B.shape[1]
        self._blocks = gcol
        self._qstart, self._width = qstart, width
        self._set_classes(KG, MG)
        self.timings["project"] = time.perf_counter() - t1
        self.timings["total"] = time.perf_counter() - t0

    def _set_classes(self, KG: np.ndarray, MG: np.ndarray) -> None:
        """The nine k-independent blocks K_d, M_d of K_r(k) = Σ_d e^{i d·θ} K_d from the open-cell reduced matrices."""
        classes = {}
        n = self.n_reduced
        for na, wa, sa, ma in self._blocks:
            for nb, wb, sb, mb in self._blocks:
                d = (wb[0] - wa[0], wb[1] - wa[1])
                if d not in classes:
                    classes[d] = (np.zeros((n, n)), np.zeros((n, n)))
                Kd, Md = classes[d]
                qa, qb = self._qstart[na], self._qstart[nb]
                Kd[qa : qa + ma, qb : qb + mb] += KG[sa : sa + ma, sb : sb + mb]
                Md[qa : qa + ma, qb : qb + mb] += MG[sa : sa + ma, sb : sb + mb]
        self.classes = classes

    def reproject(self, fraction: np.ndarray | None = None, layers=None) -> "BlochROM":
        """
        A copy on a new fraction map (or new `layers` fractions for a stack) with this basis held
        fixed: Galerkin on the new K, M, no reduction (about one assembly and two products). Its
        eigenvalues are smooth in the fraction and `sensitivities` is their exact gradient (on
        the elements kept by the original raster; others stay dropped). For line searches and
        for consistent finite differences; re-reduce (a new BlochROM) when the design has moved.
        """
        import copy

        new = copy.copy(self)
        fr = part_fractions(fraction, layers, self.solver.model)
        new.fractions = fr
        new.cell = copy.copy(self.cell)
        new.cell.weights = [f.ravel()[part.cell] for f, part in zip(fr, self.solver.parts)]
        K0, M0 = new.cell.assemble("K"), new.cell.assemble("M")
        KG, MG = _blocked_product(K0, self.G), _blocked_product(M0, self.G)
        new._set_classes((KG + KG.T) / 2, (MG + MG.T) / 2)
        return new

    # -- per k ------------------------------------------------------------------------------

    def theta(self, k) -> np.ndarray:
        return self.lattice.vectors @ np.asarray(k, dtype=float)

    def matrices(self, k) -> tuple[np.ndarray, np.ndarray]:
        """Reduced Hermitian K_r(k), M_r(k)."""
        th = self.theta(k)
        n = self.n_reduced
        K, M = np.zeros((n, n), complex), np.zeros((n, n), complex)
        for d, (Kd, Md) in self.classes.items():
            ph = np.exp(1j * (d[0] * th[0] + d[1] * th[1]))
            K += ph * Kd
            M += ph * Md
        return K, M

    def solve(self, k, n_bands: int = 10, vectors: bool = False, window_hz: tuple[float, float] | None = None):
        """
        The n_bands lowest frequencies (Hz) at k, or with window_hz = (lo, hi) all those in that
        window (for a ROM shifted into the band of interest, sigma0_hz > 0); with vectors also
        the reduced eigenvectors q (n_reduced × n, M_r-orthonormal).
        """
        K, M = self.matrices(k)
        if window_hz is None:
            lam, q = sla.eigh(K, M, subset_by_index=(0, n_bands - 1), driver="gvx", overwrite_a=True, overwrite_b=True)
        else:
            lam, q = sla.eigh(K, M, subset_by_value=tuple(lam_of_hz(f) for f in window_hz), driver="gvx", overwrite_a=True, overwrite_b=True)
        f = hz_of_lam(lam)
        return (f, q) if vectors else f

    def bands(self, ks, n_bands: int = 10, window_hz: tuple[float, float] | None = None):
        """Bands over ks: an array (nk, n_bands), or with window_hz a list of arrays (the count may vary with k)."""
        out = [self.solve(k, n_bands, window_hz=window_hz) for k in ks]
        return out if window_hz is not None else np.array(out)

    def open_vector(self, k, q: np.ndarray) -> np.ndarray:
        """Open-cell DOF values G T(k) q of reduced vectors q (n_reduced × nv)."""
        th = self.theta(k)
        q = np.atleast_2d(q.T).T
        out = np.zeros((self.G.shape[0], q.shape[1]), complex)
        for name, w, s, m in self._blocks:
            qa = self._qstart[name]
            out += np.exp(1j * (w[0] * th[0] + w[1] * th[1])) * (self.G[:, s : s + m] @ q[qa : qa + m])
        return out

    def reconstruct(self, k, q: np.ndarray) -> np.ndarray:
        """Eigenvectors on the BlochSolver's DOFs (as solver.solve(..., vectors=True) returns them)."""
        xo = self.open_vector(k, q)
        oc = self.cell
        th = self.theta(k)
        x = np.zeros((self.solver.ndof, xo.shape[1]), complex)
        # Any copy of a DOF gives its periodic value after removing the copy's phase.
        ph = np.exp(-1j * (oc.copy[:, 0] * th[0] + oc.copy[:, 1] * th[1]))
        x[oc.p] = xo * ph[:, None]
        return x

    def sensitivities(self, k, n_bands: int = 10, degenerate: str = "mean", rtol: float = 1e-4, keep_vectors: bool = False) -> BandSensitivity:
        """
        As omc.mech_adjoint.sensitivities (BandSensitivity, dfreq (nparts, nb, n1, n2) in Hz
        per unit fraction), from the reconstructed modes: exact to the reduced model's accuracy.
        """
        f, q = self.solve(k, n_bands, vectors=True)
        lam = (2 * np.pi * f / OMEGA_UNIT) ** 2
        xo = self.open_vector(k, q)
        n1, n2 = self.solver.shape
        oc = self.cell
        dlam = np.zeros((len(self.solver.parts), n_bands, n1 * n2))
        for p, part in enumerate(self.solver.parts):
            U = oc.element_vectors(p, xo)
            KU, MU = part.Ke @ U, part.Me @ U
            eK = np.einsum("eav,eav->ev", U.conj(), KU).real
            eM = np.einsum("eav,eav->ev", U.conj(), MU).real
            g = eK - lam[None] * eM
            for b in range(n_bands):
                dlam[p, b] = np.bincount(part.cell, weights=g[:, b], minlength=n1 * n2)
        # q is M_r-orthonormal, so xᴴ M x = 1.
        clusters = find_clusters(lam, rtol)
        scale = np.where(lam > 0, f / (2 * np.where(lam > 0, lam, 1)), 0.0)
        dfreq = dlam * scale[None, :, None]
        if degenerate == "mean":
            for c in clusters:
                if len(c) > 1:
                    dfreq[:, c] = dfreq[:, c].mean(axis=1, keepdims=True)
        X = self.reconstruct(k, q) if keep_vectors else None
        return BandSensitivity(np.asarray(k, float), f, lam, dfreq.reshape(len(self.solver.parts), n_bands, n1, n2), clusters, X)


# --- chains of cells (component mode synthesis) --------------------------------------------------


def _key(*arrays) -> str:
    h = hashlib.sha1()
    for a in arrays:
        if a is None:
            h.update(b"none")
        else:
            a = np.ascontiguousarray(a)
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()


class _LightSolver:
    """The parts of a BlochSolver that the open cell, the sensitivities and coupling.mech_mode read."""

    def __init__(self, template: BlochSolver, parts, dofs):
        self.lattice, self.shape, self.model = template.lattice, template.shape, template.model
        self.thickness, self.parity, self.basis = template.thickness, template.parity, template.basis
        self.n = template.n
        self.parts, self.dofs, self.ndof = parts, dofs, len(dofs)


class CellGrid:
    """
    The lattice-aligned grid shared by all cells of a chain. Element matrices and connectivity
    are taken once from a template BlochSolver; `solver(fraction)` then builds the
    BlochSolver-compatible parts (kept elements, local DOFs, wrap counts) of any raster without
    assembling anything (identical to BlochSolver for the 2D-mesh models).
    """

    def __init__(self, lattice, fraction, material, min_fraction: float = 1e-3, **solver_kwargs):
        self.template = BlochSolver(lattice, fraction, material, min_fraction=min_fraction, **solver_kwargs)
        t = self.template
        self.dpn = _dofs_per_node(t)
        self.min_fraction = min_fraction
        n1, n2 = t.shape
        corners = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
        el = np.stack([g.ravel() for g in np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")], axis=1)
        gn = el[:, None, :] + corners[None]
        self.el, self.w1, self.w2 = el, gn[..., 0] // n1, gn[..., 1] // n2
        self.node = (gn[..., 0] % n1) * n2 + gn[..., 1] % n2
        self.offsets = []
        for part in t.parts:
            nd = part.Ke.shape[0] // 4
            self.offsets.append(t.dofs[part.edof[0][:nd]] % self.dpn)

    def solver(self, fractions: list[np.ndarray]) -> _LightSolver:
        t = self.template
        n2 = t.shape[1]
        raw = []
        for frac, part, off in zip(fractions, t.parts, self.offsets):
            f = frac[self.el[:, 0], self.el[:, 1]]
            keep = f > self.min_fraction
            edof = (self.node[keep][..., None] * self.dpn + off).reshape(keep.sum(), -1)
            raw.append((part, edof, np.repeat(self.w1[keep], len(off), axis=1), np.repeat(self.w2[keep], len(off), axis=1),
                        self.el[keep, 0] * n2 + self.el[keep, 1]))
        used = np.unique(np.concatenate([r[1].ravel() for r in raw]))
        local = np.full(used.max() + 1, -1)
        local[used] = np.arange(len(used))
        parts = [ElementPart(p.Ke, p.Me, cell, local[edof], W1, W2) for p, edof, W1, W2, cell in raw]
        return _LightSolver(t, parts, used)


@dataclass
class ChainCell:
    """One cell raster of a chain: its (light) solver, the open cell (a1 open, a2 closed), lossless K, M."""

    fraction: np.ndarray
    solver: _LightSolver
    cell: OpenCell
    K: sp.csr_matrix
    M: sp.csr_matrix
    interior: np.ndarray  # open DOFs off the interface line
    sel_L: np.ndarray  # interface index (j·dpn + b) → open DOF of the left copy, or -1
    sel_R: np.ndarray


@dataclass
class CellBasis:
    """
    A cell's reduction basis V (n_open × (n_int + 2m)): n_int interior-only columns (zero on both
    interface copies), then m columns with trace Φ_Γ on the left copy and m on the right; the
    lossless reduced K_r, M_r (dense). f_fixed: frequencies of the fixed-interface modes kept.
    """

    V: np.ndarray
    n_int: int
    K: np.ndarray
    M: np.ndarray
    f_fixed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    seconds: float = 0.0
    frozen: bool = False


@dataclass
class CellReduction:
    """A cell instance: its basis and, with an absorber, A_r = Vᵀ A V (dense)."""

    basis: CellBasis
    A: np.ndarray | None

    @property
    def V(self):
        return self.basis.V

    @property
    def n_int(self):
        return self.basis.n_int

    @property
    def K(self):
        return self.basis.K

    @property
    def M(self):
        return self.basis.M


def _block_nearest(K, M, sigma: float, n: int, solve, iters: int = 6, tol: float = 1e-3, seed: int = 0):
    """
    The n eigenpairs of the real symmetric pencil (K, M) nearest σ, by block shift-invert
    subspace iteration (block n + max(10, n/2), Rayleigh–Ritz each step) with multi-RHS solves.
    Stops when the n Ritz pairs have relative residual < tol or after iters steps: the result
    is used as a basis (its span near σ is what matters), not as converged eigenpairs.
    """
    N = K.shape[0]
    n = min(n, N - 2)
    p = min(N - 1, n + max(10, n // 2))
    K, M = K.tocsr(), M.tocsr()
    X = np.random.default_rng(seed).standard_normal((N, p))
    for _ in range(iters):
        X = _m_orthonormal(solve(M @ X), M)
        KX = K @ X
        G = X.T @ KX
        lam, Y = np.linalg.eigh((G + G.T) / 2)
        o = np.argsort(np.abs(lam - sigma))
        lam, Y = lam[o], Y[:, o]
        X, KX = X @ Y, KX @ Y
        R = KX[:, :n] - (M @ X[:, :n]) * lam[:n]
        if np.max(np.linalg.norm(R, axis=0) / np.maximum(np.linalg.norm(KX[:, :n], axis=0), 1e-300)) < tol:
            break
    o = np.argsort(lam[:n])
    return lam[:n][o], X[:, :n][:, o]


def _project(c: ChainCell, V: np.ndarray, n_int: int) -> CellBasis:
    K, M = _blocked_product(c.K, V), _blocked_product(c.M, V)
    return CellBasis(V, n_int, (K + K.T) / 2, (M + M.T) / 2)


def _reduce_raster(c: ChainCell, Phi_G: np.ndarray, sigma: float, n_fixed: int, n_moment: int = 0) -> CellBasis:
    """
    Frequency-shifted Craig–Bampton of one raster: n_fixed fixed-interface modes nearest σ, and
    lifts with traces Φ_Γ on the left / right copy and interior response −A_II⁻¹A_IΓΦ_Γ
    (A = K − σM, exact at σ), less their Φ components (same span, better conditioned).
    n_moment > 0 adds the dominant directions of the lifts' first moment A_II⁻¹M_II Y (their
    λ-derivative at σ, less Φ), so the interior response is right to first order in λ − σ.
    """
    t0 = time.perf_counter()
    I, n_open, m = c.interior, c.cell.n_open, Phi_G.shape[1]
    KII, MII = c.K[I][:, I].tocsc(), c.M[I][:, I].tocsc()
    A = (c.K - sigma * c.M).tocsr()
    solve = factorize_indefinite(A[I][:, I])
    lam, Phi = _block_nearest(KII, MII, sigma, n_fixed, solve)
    Y = np.zeros((n_open, 2 * m))
    for s, rows in enumerate((c.sel_L, c.sel_R)):
        ok = rows >= 0
        Y[rows[ok], s * m : (s + 1) * m] = Phi_G[ok]
    Y[I] = -solve(A[I] @ Y)
    Y[I] -= Phi @ (Phi.T @ (MII @ Y[I]))
    if n_moment:
        R = solve(MII @ Y[I])
        R -= Phi @ (Phi.T @ (MII @ R))
        g = R.T @ (MII @ R)
        e, W = np.linalg.eigh((g + g.T) / 2)
        R = R @ W[:, ::-1][:, : min(n_moment, int(np.sum(e > 1e-12 * e.max())))]
        R -= Phi @ (Phi.T @ (MII @ R))
        Phi = np.concatenate([Phi, _m_orthonormal(R, MII)], axis=1)
        Y[I] -= Phi @ (Phi.T @ (MII @ Y[I]))
        lam = np.concatenate([lam, np.full(Phi.shape[1] - len(lam), np.nan)])
    V = np.zeros((n_open, Phi.shape[1] + Y.shape[1]))
    V[I, : Phi.shape[1]] = Phi
    V[:, Phi.shape[1] :] = Y
    b = _project(c, V, Phi.shape[1])
    b.f_fixed = hz_of_lam(lam)
    b.seconds = time.perf_counter() - t0
    return b


def _reduce_task(args):
    return _reduce_raster(*args)


class ChainROM:
    """
    Component mode synthesis of a chain of cells along a1.

    cell_lattice: lattice of one cell (a1 = (pitch, 0), a2), as in omc.cavity: every cell is on
    the same n_cell × n2 grid. fractions: per-cell rasters (n_cell, n2). etas: per-cell absorber
    η on the same grid (None: lossless). periodic: the last cell joins the first (as
    omc.cavity's supercell), else the chain has free ends. Each cell is closed along a2 at
    k·a2 = 0. target_hz: shift σ of the reductions and of the solve. n_fixed: fixed-interface
    modes per cell (nearest σ). n_iface: size of the interface basis Φ_Γ shared by all interfaces
    (None: the exact interface, for checks). ref_cells: cells whose interfaces define Φ_Γ
    (default: up to n_ref distinct lossless rasters spread along the chain). solver_kwargs: as
    BlochSolver (slab_p, even, t = 0.22, order 2 by default).

    Caching: the lossless matrices and the basis are per raster, so cells with the same raster
    (e.g. a mirror and its absorbing copies) are reduced once; an absorber only adds Vᵀ A V.
    `set_cell` changes one cell, re-reducing it (or, frozen=True, projecting it on the basis of
    a neighbour or of its previous self: no factorisation); `solve` gives complex frequencies;
    `full_vector`/`grid`/`mass_operator` the mode on the whole chain for omc.coupling;
    `sensitivities` ∂λ/∂fraction per cell.
    """

    def __init__(
        self,
        cell_lattice,
        fractions: list[np.ndarray],
        material,
        etas: list[np.ndarray | None] | None = None,
        target_hz: float = 7e9,
        n_fixed: int = 20,
        n_iface: int | None = 100,
        periodic: bool = True,
        ref_cells: list[int] | None = None,
        n_ref: int = 2,
        krylov: tuple[int, int, int] = (20, 3, 40),
        workers: int = 1,
        anchors: list[int] | None = None,
        n_moment: int = 40,
        **solver_kwargs,
    ):
        self.lattice = cell_lattice
        self.material = material
        kw = dict(model="slab_p", thickness=0.22, parity="even", order=2)
        kw.update(solver_kwargs)
        self.solver_kwargs = kw
        self.periodic = periodic
        self.permc = "NATURAL"
        self.sigma = lam_of_hz(target_hz)
        self.target_hz = target_hz
        self.n_fixed = n_fixed
        self.n_moment = n_moment
        self.timings = {}
        t0 = time.perf_counter()
        self.grid_template = CellGrid(cell_lattice, np.asarray(fractions[0], float), material, **kw)
        self.shape = np.asarray(fractions[0]).shape
        self.dpn = self.grid_template.dpn
        self.basis = self.grid_template.template.basis
        self._cells: dict[str, ChainCell] = {}
        self._absorbers: dict[tuple[str, str], sp.csr_matrix] = {}
        self._etas: dict[str, np.ndarray] = {}
        self._bases: dict[str, CellBasis] = {}
        self._reductions: dict[tuple[str, str], CellReduction] = {}
        self.keys: list[tuple[str, str]] = []
        etas = etas or [None] * len(fractions)
        for f, e in zip(fractions, etas):
            self.keys.append(self._add_cell(f, e))
        self.timings["cells"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if ref_cells is None:
            seen, cand = set(), []
            for c, (fk, ek) in enumerate(self.keys):
                if ek == "none" and fk not in seen:
                    seen.add(fk)
                    cand.append(c)
            cand = cand or [0]
            ref_cells = [cand[i] for i in np.unique(np.linspace(0, len(cand) - 1, min(n_ref, len(cand))).round().astype(int))]
        self.ref_cells = ref_cells
        if n_iface is None:
            present = self._present()
            self.Phi_G = np.eye(len(present))[:, present]
            self.f_iface = None
        else:
            self.Phi_G, self.f_iface = self._interface_basis([self._cells[self.keys[c][0]] for c in ref_cells], n_iface, *krylov)
        self.m = self.Phi_G.shape[1]
        self.timings["interface"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        rasters = list(dict.fromkeys(k[0] for k in self.keys))
        todo = rasters if anchors is None else list(dict.fromkeys(self.keys[a][0] for a in anchors))
        if workers > 1 and len(todo) > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(workers) as pool:
                args = [(self._cells[fk], self.Phi_G, self.sigma, n_fixed, self.n_moment) for fk in todo]
                for fk, b in zip(todo, pool.map(_reduce_task, args)):
                    self._bases[fk] = b
        else:
            for fk in todo:
                self._bases[fk] = self._reduce(self._cells[fk])
        if anchors is not None:
            # Other rasters (taper cells): projected on the nearest anchors' bases (frozen, no factorisation).
            apos = np.array(sorted(anchors))
            for c, (fk, _) in enumerate(self.keys):
                if fk in self._bases:
                    continue
                near = apos[np.argsort(np.abs(apos - c))[:2]]
                srcs = [self.keys[a][0] for a in near]
                self._bases[fk] = self.frozen_basis(self._cells[fk], [(self._cells[k], self._bases[k]) for k in dict.fromkeys(srcs)])
        for key in dict.fromkeys(self.keys):
            self._reductions[key] = self._instance(key)
        self.timings["reduce"] = time.perf_counter() - t0

    # -- cells -------------------------------------------------------------------------------

    def _add_cell(self, fraction, eta) -> tuple[str, str]:
        fraction = np.asarray(fraction, float)
        fk = _key(fraction)
        if fk not in self._cells:
            s = self.grid_template.solver([fraction])
            oc = open_cell(s, [fraction], (True, False))
            K, M = oc.assemble("K"), oc.assemble("M")
            on_line = oc.node_i[oc.p] == 0
            idx = oc.node_j[oc.p] * oc.dpn + oc.comp[oc.p]
            sel = {}
            for side in (0, 1):
                o = np.flatnonzero(on_line & (oc.copy[:, 0] == side))
                rows = np.full(self.shape[1] * oc.dpn, -1)
                rows[idx[o]] = o
                sel[side] = rows
            self._cells[fk] = ChainCell(fraction, s, oc, K, M, np.flatnonzero(~on_line), sel[0], sel[1])
        if eta is None or not np.any(eta):
            return fk, "none"
        eta = np.asarray(eta, float)
        ek = _key(eta)
        self._etas[ek] = eta
        if (fk, ek) not in self._absorbers:
            c = self._cells[fk]
            self._absorbers[(fk, ek)] = c.cell.assemble("A", [eta.ravel()[part.cell] for part in c.solver.parts])
        return fk, ek

    def _present(self) -> np.ndarray:
        present = np.zeros(self.shape[1] * self.dpn, bool)
        for c in self._cells.values():
            present |= (c.sel_L >= 0) | (c.sel_R >= 0)
        return present

    def _interface_basis(self, refs: list[ChainCell], m: int, n_start: int = 20, n_krylov: int = 3, block: int = 40, tol: float = 1e-8):
        """
        Φ_Γ, the interface basis shared by all interfaces, from the reference cells at the shift σ.
        For each reference cell, two copies glued along the interface (outer interfaces clamped):
        the responses to n_start random interface loads, then n_krylov block steps of its dynamic
        interface flexibility S_glued(σ)⁻¹ applied to the loads that motion of the outer
        interfaces puts on the shared one (through either cell). The vectors are ranked by the
        Bloch waves at σ: the interface QEP (S_RL/μ + S_LL + S_RR + μ S_LR)φ = 0 (u_n = μⁿφ on
        interface n of a chain of this cell) projected on that space, small |ln μ| (propagating,
        slowly decaying) first; Gram–Schmidt in that order (interleaving the references) to m.
        """
        sig = self.sigma
        nif = len(refs[0].sel_L)
        rng = np.random.default_rng(1)
        traces, Dms = [], []
        for c in refs:
            I, n_open = c.interior, c.cell.n_open
            A = (c.K - sig * c.M).tocsr()
            AI = A[I]
            lu = factorize_indefinite(AI[:, I])
            liv = np.flatnonzero((c.sel_L >= 0) | (c.sel_R >= 0))
            nl = len(liv)
            SL, SR = _selection(n_open, c.sel_L[liv]), _selection(n_open, c.sel_R[liv])
            D = (SR.T @ A @ SR + SL.T @ A @ SL).tocsc()
            CR, CL = (AI @ SR).tocsc(), (AI @ SL).tocsc()
            glu = factorize_indefinite(sp.bmat([[AI[:, I], None, CR], [None, AI[:, I], CL], [CR.T, CL.T, D]], format="csc"))
            Dm = (SR.T @ c.M @ SR + SL.T @ c.M @ SL).tocsc()
            Dms.append((liv, Dm))
            Lc = np.linalg.cholesky(Dm.toarray())
            nG = 2 * len(I) + nl

            def gsolve(F):
                rhs = np.zeros((nG, F.shape[1]))
                rhs[-nl:] = F
                return glu(rhs)[-nl:]

            def far_loads(B):
                out = []
                for far, near in ((SL, SR), (SR, SL)):  # cell A: far = its left copy; cell B: its right copy
                    Y = np.asarray(far @ B)
                    Y[I] = -lu(AI @ Y)
                    out.append(np.asarray(near.T @ (A @ Y)))
                return np.concatenate(out, axis=1)

            def compress(B, prev):
                if prev is not None:
                    B = B - prev @ (prev.T @ (Dm @ B))
                U, sv, _ = np.linalg.svd(Lc.T @ B, full_matrices=False)
                r = int(min(block, np.sum(sv > tol * max(sv[0], 1e-300))))
                return np.linalg.solve(Lc.T, U[:, :r])

            B = compress(gsolve(rng.standard_normal((nl, n_start))), None)
            basis = B
            for _ in range(n_krylov):
                B = compress(gsolve(far_loads(B)), basis)
                if not B.shape[1]:
                    break
                basis = np.concatenate([basis, B], axis=1)
            nk = basis.shape[1]
            Sxy = {}
            for yname, ycopy in (("L", SL), ("R", SR)):
                Y = np.asarray(ycopy @ basis)
                Y[I] = -lu(AI @ Y)
                AY = A @ Y
                for xname, xcopy in (("L", SL), ("R", SR)):
                    Sxy[xname + yname] = basis.T @ np.asarray(xcopy.T @ AY)
            Z, E = np.zeros((nk, nk)), np.eye(nk)
            mu, Yq = sla.eig(np.block([[Z, E], [-Sxy["RL"], -(Sxy["LL"] + Sxy["RR"])]]), np.block([[E, Z], [Z, Sxy["LR"]]]))
            ok = np.isfinite(mu) & (np.abs(mu) > 0)
            mu, Yq = mu[ok], Yq[:nk, ok]
            o = np.argsort(np.abs(np.log(np.abs(mu))))
            waves = basis @ Yq[:, o]
            ranked = np.concatenate([np.stack([w.real, w.imag], axis=1) for w in waves.T] + [basis], axis=1)
            T = np.zeros((nif, ranked.shape[1]))
            T[liv] = ranked
            traces.append((T, np.abs(np.log(np.abs(mu[o])))))
        Dm = np.zeros((nif, nif))
        for liv, d in Dms:
            Dm[np.ix_(liv, liv)] += d.toarray() / len(refs)
        live = np.flatnonzero(np.diag(Dm) > 0)
        L = np.linalg.cholesky(Dm[np.ix_(live, live)])
        width = max(t.shape[1] for t, _ in traces)
        Q = np.zeros((len(live), 0))
        for i in range(width):
            for t, _ in traces:
                if i >= t.shape[1] or Q.shape[1] >= m:
                    continue
                x = L.T @ t[live, i]
                nx = np.linalg.norm(x)
                for _ in range(2):
                    x = x - Q @ (Q.T @ x)
                if np.linalg.norm(x) > 1e-6 * nx:
                    Q = np.concatenate([Q, (x / np.linalg.norm(x))[:, None]], axis=1)
        Phi = np.zeros((nif, Q.shape[1]))
        Phi[live] = np.linalg.solve(L.T, Q)
        # DOFs present on some cell's interface but not on the reference ones: keep them exactly.
        extra = np.flatnonzero(self._present() & ~np.isin(np.arange(nif), live))
        if len(extra):
            E = np.zeros((nif, len(extra)))
            E[extra, np.arange(len(extra))] = 1.0
            Phi = np.concatenate([Phi, E], axis=1)
        return Phi, np.concatenate([k for _, k in traces])

    def _reduce(self, c: ChainCell) -> CellBasis:
        return _reduce_raster(c, self.Phi_G, self.sigma, self.n_fixed, self.n_moment)

    def _project(self, c: ChainCell, V: np.ndarray, n_int: int) -> CellBasis:
        return _project(c, V, n_int)

    def _instance(self, key: tuple[str, str]) -> CellReduction:
        b = self._bases[key[0]]
        if key[1] == "none":
            return CellReduction(b, None)
        A = _blocked_product(self._absorbers[key], b.V)
        return CellReduction(b, (A + A.T) / 2)

    # -- frozen bases (no factorisation) -----------------------------------------------------

    def _dof_keys(self, cc: ChainCell) -> np.ndarray:
        oc = cc.cell
        return ((oc.node_i[oc.p] * self.shape[1] + oc.node_j[oc.p]) * self.dpn + oc.comp[oc.p]) * 2 + oc.copy[:, 0]

    def _carry(self, src: ChainCell, cols: np.ndarray, dst: ChainCell) -> np.ndarray:
        """Columns on src's open DOFs carried to dst's open DOFs by DOF identity (zero where absent)."""
        ks, kd = self._dof_keys(src), self._dof_keys(dst)
        order = np.argsort(ks)
        pos = np.clip(np.searchsorted(ks[order], kd), 0, len(ks) - 1)
        hit = ks[order][pos] == kd
        out = np.zeros((dst.cell.n_open, cols.shape[1]), cols.dtype)
        out[hit] = cols[order[pos[hit]]]
        return out

    def frozen_basis(self, c: ChainCell, sources: list[tuple[ChainCell, CellBasis]], tol: float = 1e-8) -> CellBasis:
        """
        Basis for raster c from other rasters' bases, without a factorisation or eigensolve: the
        interior-only columns of all sources carried over by DOF identity and re-orthonormalised
        in c's mass (dependent directions dropped); the interface columns keep their traces Φ_Γ
        exactly, with the interior parts of the first source (the lift differences of the others
        join the interior block). Galerkin on c's own K, M, so the eigenvalue gradient with this
        basis held fixed is exactly the projected full-model sensitivity.
        """
        t0 = time.perf_counter()
        m = self.m
        n_open, I = c.cell.n_open, c.interior
        interior, lift0 = [], None
        for src, b in sources:
            V = self._carry(src, b.V, c)
            interior.append(V[:, : b.n_int])
            if lift0 is None:
                lift0 = V[:, b.n_int :]
            else:
                interior.append(V[:, b.n_int :] - lift0)
        B = np.concatenate(interior, axis=1)
        on = np.ones(n_open, bool)
        on[I] = False
        B[on] = 0
        Y = lift0.copy()
        Y[on] = 0
        for s, rows in enumerate((c.sel_L, c.sel_R)):
            ok = rows >= 0
            Y[rows[ok], s * m : (s + 1) * m] = self.Phi_G[ok]
        B = _m_orthonormal(B, c.M, tol)
        b = self._project(c, np.concatenate([B, Y], axis=1), B.shape[1])
        b.seconds, b.frozen = time.perf_counter() - t0, True
        return b

    def set_cell(self, index: int, fraction: np.ndarray, eta="keep", frozen: bool = False, sources: list[int] | None = None) -> float:
        """
        Replace cell `index` (eta="keep" keeps its absorber map). Only that raster is reduced
        (frozen=False), or projected on the bases of the cells `sources` (default: the cell's
        previous raster) with frozen=True. Returns the seconds spent.
        """
        old = self.keys[index]
        if isinstance(eta, str):
            eta = self._etas.get(old[1]) if old[1] != "none" else None
        key = self._add_cell(fraction, eta)
        self.keys[index] = key
        t0 = time.perf_counter()
        if key[0] not in self._bases:
            c = self._cells[key[0]]
            if frozen:
                src = [old[0]] if sources is None else [self.keys[s][0] for s in sources]
                self._bases[key[0]] = self.frozen_basis(c, [(self._cells[k], self._bases[k]) for k in src])
            else:
                self._bases[key[0]] = self._reduce(c)
        if key not in self._reductions:
            self._reductions[key] = self._instance(key)
        return time.perf_counter() - t0

    def reduction(self, index: int) -> CellReduction:
        return self._reductions[self.keys[index]]

    def chain_cell(self, index: int) -> ChainCell:
        return self._cells[self.keys[index][0]]

    def eta(self, index: int) -> np.ndarray | None:
        return self._etas.get(self.keys[index][1])

    # -- assembly and solve -----------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        return len(self.keys)

    def _layout(self):
        """
        Global offsets, interleaved cell by cell so the reduced matrices are block banded:
        [Γ_0, η_0, Γ_1, η_1, …] (+ Γ_N for an open chain). Returns (n_lines, m, (interface
        offsets, interior offsets), size).
        """
        N, m = self.n_cells, self.m
        n_lines = N if self.periodic else N + 1
        ioff, offs, o = [], [], 0
        for c in range(N):
            ioff.append(o)
            o += m
            offs.append(o)
            o += self.reduction(c).n_int
        if not self.periodic:
            ioff.append(o)
            o += m
        return n_lines, m, (ioff, offs), o

    def _cell_dofs(self, c: int, n_lines: int, m: int, offs) -> np.ndarray:
        ioff, coff = offs
        left = ioff[c] + np.arange(m)
        right = ioff[(c + 1) % n_lines] + np.arange(m)
        return np.concatenate([coff[c] + np.arange(self.reduction(c).n_int), left, right])

    def assemble(self, phase1: float = 0.0):
        """Global reduced K, M^c = M − iA (sparse, block banded). phase1: Bloch phase across the periodic seam."""
        n_lines, m, offs, n = self._layout()
        rows, cols, kv, mv = [], [], [], []
        lossy = any(self.reduction(c).A is not None for c in range(self.n_cells))
        for c in range(self.n_cells):
            r = self.reduction(c)
            d = self._cell_dofs(c, n_lines, m, offs)
            Kc = r.K
            Mc = r.M if r.A is None else r.M - 1j * r.A
            if self.periodic and c == self.n_cells - 1 and phase1:
                ph = np.ones(len(d), complex)
                ph[r.n_int + m :] = np.exp(1j * phase1)
                Kc = np.conj(ph)[:, None] * Kc * ph[None]
                Mc = np.conj(ph)[:, None] * Mc * ph[None]
            rows.append(np.repeat(d, len(d)))
            cols.append(np.tile(d, len(d)))
            kv.append(np.ravel(Kc))
            mv.append(np.ravel(Mc))
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        K = sp.csr_matrix((np.concatenate(kv), (rows, cols)), shape=(n, n))
        M = sp.csr_matrix((np.concatenate(mv), (rows, cols)), shape=(n, n))
        if not lossy and not phase1:
            K, M = K.real, M.real
        return K, M

    def solve(self, n_modes: int = 20, target_hz: float | None = None, tol: float = 1e-10, phase1: float = 0.0):
        """
        n_modes modes nearest target_hz (default: the reduction shift): complex frequencies (Hz,
        Im > 0 decaying), sorted by real part, and reduced eigenvectors Q (n_reduced × n_modes),
        normalised to qᵀ M^c q = 1 (unconjugated; for lossless real modes, the M-norm).
        Q_m = Re f / (2 Im f).
        """
        t0 = time.perf_counter()
        K, M = self.assemble(phase1)
        sig = self.sigma if target_hz is None else lam_of_hz(target_hz)
        Asig = (K - sig * M).tocsc()
        if np.iscomplexobj(Asig.data) or np.iscomplexobj(M.data):
            lu = spla.splu(Asig.astype(complex), permc_spec=self.permc)
            op = spla.LinearOperator(K.shape, matvec=lambda v: lu.solve(np.asarray(M @ v, complex)), dtype=complex)
            mu, Q = spla.eigs(op, k=n_modes, which="LM", tol=tol)
            lam = sig + 1 / mu
        else:
            lu = spla.splu(Asig, permc_spec=self.permc)
            op = spla.LinearOperator(K.shape, matvec=lu.solve, dtype=float)
            lam, Q = spla.eigsh(K, k=n_modes, M=M, sigma=sig, which="LM", OPinv=op, tol=tol)
            lam = lam.astype(complex)
        w = hz_of_lam(lam.astype(complex))
        o = np.argsort(w.real)
        w, Q, lam = w[o], Q[:, o], lam[o]
        Q = Q / np.sqrt(np.einsum("iv,iv->v", Q, M @ Q))
        self.timings["solve"] = time.perf_counter() - t0
        self.n_reduced = K.shape[0]
        self.last = (w, Q, lam)
        return w, Q

    # -- reconstruction ---------------------------------------------------------------------

    def cell_vectors(self, c: int, Q: np.ndarray, phase1: float = 0.0) -> np.ndarray:
        """Open-cell DOF values (n_open_c × nv) of cell c for reduced vectors Q."""
        n_lines, m, offs, _ = self._layout()
        d = self._cell_dofs(c, n_lines, m, offs)
        r = self.reduction(c)
        q = Q[d]
        if self.periodic and c == self.n_cells - 1 and phase1:
            q = q.astype(complex)
            q[r.n_int + m :] *= np.exp(1j * phase1)
        return r.V @ q

    def grid(self) -> "ChainGrid":
        """A light stand-in for the whole chain's BlochSolver (lattice, shape, dofs, basis) for coupling.mech_mode."""
        from .geometry import Lattice2D

        n_cell, n2 = self.shape
        N = self.n_cells
        A = self.lattice.vectors
        dofs = np.unique(np.concatenate([self._global_open_dofs(c) for c in range(N)]))
        frac = np.concatenate([self.chain_cell(c).fraction for c in range(N)], axis=0)
        shape = (N * n_cell + (0 if self.periodic else 1), n2)
        kw = self.solver_kwargs
        return ChainGrid(Lattice2D(np.array([N * A[0], A[1]])), shape, dofs, self.basis, kw["model"], kw.get("thickness"), kw.get("parity"), frac)

    def _global_open_dofs(self, c: int) -> np.ndarray:
        """Whole-chain grid DOF of each open DOF of cell c (the right copy sits on the next cell's line)."""
        n_cell, n2 = self.shape
        oc = self.chain_cell(c).cell
        gi = c * n_cell + oc.node_i[oc.p] + oc.copy[:, 0] * n_cell
        if self.periodic:
            gi = gi % (self.n_cells * n_cell)
        return (gi * n2 + oc.node_j[oc.p]) * self.dpn + oc.comp[oc.p]

    def full_vector(self, Q: np.ndarray, phase1: float = 0.0, grid: "ChainGrid | None" = None) -> np.ndarray:
        """Modes on the whole chain's DOFs (ChainGrid.dofs = omc.cavity.mech_cavity(...).solver.dofs on the same rasters)."""
        g = grid or self.grid()
        Q = np.atleast_2d(Q.T).T
        x = np.zeros((len(g.dofs), Q.shape[1]), complex)
        for c in range(self.n_cells):
            pos = np.searchsorted(g.dofs, self._global_open_dofs(c))
            x[pos] = self.cell_vectors(c, Q, phase1)  # interface DOFs: both sides carry the same Φ_Γ ξ
        return x

    def mass_operator(self, grid: "ChainGrid | None" = None) -> "ChainMass":
        """Lossless M of the whole chain as an operator on full_vector's DOFs (coupling.mech_mode(M=...))."""
        return ChainMass(self, grid)

    def modal_mass(self, Q: np.ndarray) -> np.ndarray:
        """∫ρ|u|² (solver units) of reduced vectors, cell by cell: qᴴ M_r q summed (lossless mass)."""
        Q = np.atleast_2d(Q.T).T
        out = np.zeros(Q.shape[1])
        n_lines, m, offs, _ = self._layout()
        for c in range(self.n_cells):
            q = Q[self._cell_dofs(c, n_lines, m, offs)]
            out += np.real(np.einsum("iv,iv->v", q.conj(), self.reduction(c).M @ q))
        return out

    # -- sensitivities ------------------------------------------------------------------------

    def sensitivities(self, Q: np.ndarray, lam: np.ndarray) -> list[np.ndarray]:
        """
        ∂λ/∂fraction of each mode (columns of Q, normalised qᵀM^c q = 1, eigenvalues lam) per
        cell: a list over cells of complex arrays (nv, n_cell, n2), the full-model formula
        u_eᵀ(K_e − λ(1 − iη_e)M_e)u_e with the reconstructed mode (the absorber scales with the
        fraction, as omc.cavity.absorber_matrix). Exact for the reduced model with the bases held
        fixed; with re-reduction, exact to the reduced model's accuracy.
        """
        Q = np.atleast_2d(Q.T).T
        lam = np.atleast_1d(lam)
        n_cell, n2 = self.shape
        out = []
        for c in range(self.n_cells):
            cc = self.chain_cell(c)
            xo = self.cell_vectors(c, Q)
            eta = self.eta(c)
            g = np.zeros((Q.shape[1], n_cell * n2), complex)
            for p, part in enumerate(cc.solver.parts):
                U = cc.cell.element_vectors(p, xo)
                eK = np.einsum("eav,eav->ev", U, part.Ke @ U)
                eM = np.einsum("eav,eav->ev", U, part.Me @ U)
                fac = 1 - 1j * (eta.ravel()[part.cell] if eta is not None else 0.0)
                ge = eK - lam[None] * np.asarray(fac)[..., None] * eM if eta is not None else eK - lam[None] * eM
                for v in range(Q.shape[1]):
                    g[v] += np.bincount(part.cell, weights=ge[:, v].real, minlength=n_cell * n2)
                    g[v] += 1j * np.bincount(part.cell, weights=ge[:, v].imag, minlength=n_cell * n2)
            out.append(g.reshape(Q.shape[1], n_cell, n2))
        return out


@dataclass
class ChainGrid:
    """What coupling.mech_mode reads from a BlochSolver, for the whole chain."""

    lattice: object
    shape: tuple[int, int]
    dofs: np.ndarray
    basis: list
    model: str
    thickness: float
    parity: str
    fraction: np.ndarray

    @property
    def ndof(self) -> int:
        return len(self.dofs)


class ChainMass:
    """x ↦ M x on ChainGrid DOFs (lossless mass), cell by cell."""

    def __init__(self, rom: ChainROM, grid: ChainGrid | None = None):
        self.rom = rom
        g = grid or rom.grid()
        self.maps = [(np.searchsorted(g.dofs, rom._global_open_dofs(c)), rom.chain_cell(c).M) for c in range(rom.n_cells)]
        self.shape = (len(g.dofs), len(g.dofs))

    def __matmul__(self, x):
        out = np.zeros_like(x, dtype=np.result_type(x.dtype, float))
        for pos, M in self.maps:
            np.add.at(out, pos, M @ x[pos])
        return out


def chain_absorber(n_cell: int, n2: int, lattice_vectors: np.ndarray, columns: tuple[int, int], pitch: float,
                   absorb: list[int], eta_max: float, power: float = 2.0) -> np.ndarray:
    """
    η on the whole chain grid (N n_cell, n2), as omc.cavity.column_absorber (t ramps 0 → 1 over
    the listed columns away from column 0, η = eta_max t^power), for every element.
    """
    c0, c1 = columns
    N = c1 - c0 + 1
    n1 = N * n_cell
    x_c = (c0 + c1) / 2 * pitch
    V = lattice_vectors
    i, j = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    s1, s2 = (i + 0.5) / n1 - 0.5, (j + 0.5) / n2 - 0.5
    xs = s1 * V[0, 0] + s2 * V[1, 0]
    eta = np.zeros((n1, n2))
    for side in (-1, 1):
        cols = sorted([c for c in absorb if np.sign(c) == side], key=abs)
        if not cols:
            continue
        x_in = cols[0] * pitch - x_c - side * pitch / 2
        x_out = cols[-1] * pitch - x_c + side * pitch / 2
        t = (xs - x_in) / (x_out - x_in)
        msk = (t >= 0) & (t <= 1)
        eta[msk] = np.maximum(eta[msk], eta_max * t[msk] ** power)
    return eta
