"""
Piezoelectric finite elements for layered, patterned membranes (e.g. X-cut LN on Si).

Unknowns are the displacement u (3 components) and the electric potential φ, coupled in the
quasi-static approximation through the stress-charge relations T = C^E S - e^T E,
D = e S + ε^S E, E = -∇φ. Stationarity of the electric enthalpy gives the symmetric,
indefinite system

    [ K - ω² M   K_uφ ] [u]   [f ]
    [ K_uφ^T    -K_φφ ] [φ] = [-q],

K_uφ = ∫ B^T e^T ∇N_φ, K_φφ = ∫ ∇N_φ^T ε ∇N_φ (positive definite), q the surface charge.

Mesh: as the "stack" model of `mech_bloch`. In-plane, an n1 × n2 grid of identical bilinear
(Q1) parallelogram elements spanning the cell edge vectors A1, A2 (rows of `cell`), from
`origin`. Through the stack, layers listed bottom to top, each carrying the hierarchical
`lobatto` functions of its own order; the linear end functions are shared with the
neighbouring layer, so u and φ are continuous across interfaces. A layer may be patterned:
each element carries a solid fraction f, its mechanical and piezoelectric matrices are scaled
by f, and its dielectric matrix is f ε_solid + (1 - f) ε_background. Layers without material
(air, vacuum) carry φ only; layers without permittivity (metal) carry u only. Elements with
nothing in them are dropped, and so are the DOFs only they touch.

Electrodes are massless, infinitely thin equipotentials on a layer interface: the φ DOFs of
the interface function on the nodes of the masked elements are merged into one terminal DOF
(a metal layer may be added separately for mass loading). In the reduced system the terminal
row is Σ_electrode (K_uφ^T u - K_φφ φ) = -Q, so a terminal is either driven (V given, Q is
the reaction) or floating (Q = 0 and V unknown). Electrodes of kind "ground" are always at 0.
Admittance Y = iωQ/V for the e^{iωt} convention (as COMSOL). k_eff² = (f_oc² - f_sc²)/f_oc²
from the eigenfrequencies with the terminals shorted and open.

Boundaries in-plane, per direction: "free", "clamped" (u = 0 on both edges; φ free) or
"periodic" (Bloch: u(x + A_d) = exp(i k·A_d) u(x), also for φ). Faces normal to z are free
and charge-free unless covered by an electrode. Reduction to the independent DOFs is one
sparse map P (images of periodic nodes carry their Bloch phase, electrode nodes map to their
terminal, clamped DOFs are dropped), and the reduced system is P^H A P.

Absorbers: a layer's `mass_scale` multiplies the density per element by a complex factor
(COMSOL-style damping layer; 1 - i α(x) is lossy for e^{iωt}); `stretch` gives each element
column / row a complex coordinate stretch s, i.e. element edge A_d/n_d → s A_d/n_d, a
frequency-independent PML (s = 1 - iβ ramp: |R| ~ 1e-5 for SH0 in validation/piezo_absorber.py;
TODO: tune for S0/A0 and for patterned ports).

Solves: harmonic responses, admittance and shift-invert eigenmodes factorise the reduced
indefinite system with `factorize_indefinite` (SuperLU on a METIS ordering). When the system
is real (k = 0 or a real Bloch phase, no absorber) it is factorised in real arithmetic and
the eigenmodes come from Lanczos with M-inner product; otherwise Arnoldi on
(K_eff - σM)^{-1} M, complex symmetric. The eigenproblem is posed on u only: each operator
application solves the full (u, φ) system with zero charge on the free φ DOFs, i.e. on the
piezo-stiffened K_eff = K + K_uφ K_φφ^{-1} K_uφ^T without forming it.

Design variables and sensitivities: each layer's fraction map, and the in-plane crystal angle
θ of the layers given with `angle` (material in the crystal frame, rotated by x_cut(θ); θ in
degrees, period 180°; one shared θ for all such layers, e.g. the sub-layers of a sloped LN
ridge). A(ω²) is linear in the fractions and in the rotated tensors, whose θ-derivatives come
from ∂R/∂θ and the Bond matrix (`materials.rotation_derivatives`). With x = (u, φ) a mode
(φ the static response to u, zero charge on the free potential), the piezo-stiffened Rayleigh
quotient is stationary in φ, so

    dλ = xᴴ dA(λ) x / (uᴴ M u),   dA(λ) = [[dK - λ dM, dC], [dCᵀ, -dE]]

(the mass enters only the uu block; xᵀ instead of xᴴ for a lossy system at real Bloch phase).
Near-degenerate clusters get their mean, as in `mech_adjoint`. For a driven response, any
quantity J with dJ = zᵀ dA y (the admittance, or a linear functional of the field such as a
modal overlap at a port plane) needs one adjoint solve with the same factorisation
(transposed), and z, y are mapped back to the grid with conj(P) and P. Every derivative is
then a sum of element bilinear forms: no matrix is assembled per design variable. Elements
dropped from a layer (fraction ≤ min_fraction) get zero gradient; LN ⊆ Si is up to the
parametrisation. Fraction gradients are per element of each layer, (n_layers, n1, n2);
θ-gradients are per degree.

Discretisation: Q1 in-plane needs about 30 elements per acoustic wavelength for 0.3 %, and at
least order 2 per layer through thin piezo layers (order 1 overstiffens by several %); see
research/piezo_convergence.py. Sloped sidewalls are staircased by sub-layers of stepped
fraction, which converges erratically at the 0.3 % level.

Units inside: µm, stiffness in 1e11 Pa, density in 1e3 kg/m³ (ω = 1e10 √λ rad/s, as in
mech_bloch), e in C/m², permittivity in 1e-11 F/m. Then φ is in units of 1e5 V and charge in
1e-12 C. Inputs and outputs of the public methods are SI (Hz, V, C, S, F), except lengths
(µm) and wavevectors (1/µm).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial import legendre

from .linalg import factorize, factorize_indefinite
from .materials import EPS0, Elastic, Piezoelectric, rotation_derivatives, x_cut, x_cut_derivative
from .mech_bloch import C_UNIT, GAUSS2, OMEGA_UNIT, RHO_UNIT, _q1_shape, lobatto

E_UNIT = 1.0  # C/m²
EPS_UNIT = 1e-11  # F/m, so that EPS_UNIT * PHI_UNIT = E_UNIT * 1 µm
PHI_UNIT = C_UNIT * 1e-6 / E_UNIT  # V: e φ / L balances C u / L with u in µm
Q_UNIT = E_UNIT * 1e-12  # C: D × µm²

_CORNERS = np.array(list(itertools.product((0, 1), repeat=2)))


@dataclass
class PiezoLayer:
    """
    One layer of the stack, listed bottom to top. thickness in µm.

    fraction: (n1, n2) solid fraction per element, or a scalar for a uniform layer.
    material: Elastic or Piezoelectric (already rotated to the lab frame), or None (no u).
    eps_r: relative permittivity of the solid part (scalar or 3 × 3); defaults to the
        material's ε^S for a Piezoelectric. None: no φ in the solid part (e.g. a metal).
    background_eps_r: permittivity of the unfilled part (1 for air); None: no φ there.
    order: highest polynomial degree through the layer.
    mass_scale: complex factor on the density per element (scalar or (n1, n2)).
    angle: in-plane crystal angle θ in degrees: the material is then in the crystal frame and
        used as material.rotated(x_cut(θ)); θ is a design variable (`PiezoSolver.set_angle`).
    """

    thickness: float
    fraction: np.ndarray | float = 1.0
    material: Elastic | Piezoelectric | None = None
    eps_r: float | np.ndarray | None = None
    background_eps_r: float | None = None
    order: int = 2
    mass_scale: np.ndarray | complex | None = None
    angle: float | None = None


@dataclass
class Electrode:
    """
    Equipotential on layer interface `interface` (0 = bottom of layer 0, len(layers) = top of
    the stack) covering the masked elements (bool (n1, n2)). kind: "terminal" or "ground".
    Several electrodes with the same name form one terminal.
    """

    name: str
    mask: np.ndarray
    interface: int
    kind: str = "terminal"


@dataclass
class HarmonicResult:
    freq: float
    u: np.ndarray  # (n1+1, n2+1, n_functions, 3) displacement in µm per node and thickness function
    phi: np.ndarray  # (n1+1, n2+1, n_functions) potential in V
    charge: dict[str, complex] = field(default_factory=dict)  # C per terminal
    voltage: dict[str, complex] = field(default_factory=dict)  # V per terminal


@dataclass
class _Part:
    """
    One layer's elements: DOFs on the full grid (udof, pdof: (n_el, ·)), element class
    (Jacobian) index, fraction f and mass scale ms per element, the solid mask, and element
    matrices per class (None where absent): Ke, Me, Ce, Es (solid ε), Eb (background ε), and
    their θ-derivatives dKe, dCe, dEs for an angled layer.
    """

    udof: np.ndarray
    pdof: np.ndarray
    cls: np.ndarray
    f: np.ndarray
    ms: np.ndarray
    solid: np.ndarray
    diel: np.ndarray
    Ke: np.ndarray | None = None
    Me: np.ndarray | None = None
    Ce: np.ndarray | None = None
    Es: np.ndarray | None = None
    Eb: np.ndarray | None = None
    dKe: np.ndarray | None = None
    dCe: np.ndarray | None = None
    dEs: np.ndarray | None = None


@dataclass
class EigenSensitivity:
    """
    Eigenmodes and their gradients. freqs: (nm,) Hz (complex). dfreq_angle: (nm,) Hz/degree.
    dfreq_frac: (n_layers, nm, n1, n2) Hz per unit fraction. clusters: index arrays of
    near-degenerate groups (their members carry the cluster-mean gradient). u, phi as in
    `eigenmodes`.
    """

    freqs: np.ndarray
    dfreq_angle: np.ndarray
    dfreq_frac: np.ndarray
    clusters: list[np.ndarray]
    u: np.ndarray
    phi: np.ndarray


@dataclass
class CouplingSensitivity:
    """Short/open-circuit pairs (Hz), k_eff² and its gradients (per degree; per unit fraction)."""

    f_sc: np.ndarray
    f_oc: np.ndarray
    k2: np.ndarray
    dk2_angle: np.ndarray
    dk2_frac: np.ndarray
    overlap: np.ndarray
    short: EigenSensitivity
    open: EigenSensitivity
    pair: np.ndarray


@dataclass
class MotionalBranch:
    """
    BVD motional branches of the short-circuit modes at one terminal. freqs (Hz), C_m (F),
    C0 (clamped capacitance, F), g_em (rad/s) to a resonator of impedance Z_r (Ω), and for the
    modes asked for, dg_angle (rad/s per degree) and dg_frac ((n_layers, nm, n1, n2), rad/s per
    unit fraction; NaN for modes not asked for). u: mode shapes as in `eigenmodes`.
    """

    freqs: np.ndarray
    C_m: np.ndarray
    C0: float
    g_em: np.ndarray
    Z_r: float
    dg_angle: np.ndarray
    dg_frac: np.ndarray
    u: np.ndarray

    @property
    def L_m(self) -> np.ndarray:
        return 1 / ((2 * np.pi * self.freqs) ** 2 * self.C_m)

    def gamma_e(self, R0: float = 50.0) -> np.ndarray:
        """Radiation rate (rad/s) of each mode into a load R0 across the terminal: R0 ω² C_m."""
        return R0 * (2 * np.pi * self.freqs) ** 2 * self.C_m


def _bilinear(Z: np.ndarray, mats: np.ndarray, X: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """Σ_ab Z[e, a, v] mats[cls[e], a, b] X[e, b, v] for every element e and vector v."""
    out = np.zeros((Z.shape[0], Z.shape[2]), np.result_type(Z, X, mats))
    for c in np.unique(cls):
        m = cls == c
        out[m] = np.einsum("eav,eav->ev", Z[m], np.einsum("ab,ebv->eav", mats[c], X[m]))
    return out


def _layer_element(J: np.ndarray, thickness: float, order: int, C, rho, e, eps_list):
    """
    Element matrices of one layer over a Q1 element with Jacobian J (2 × 2, complex for a
    stretched PML element). u DOFs: node-major, then thickness function, then component; φ
    DOFs: node-major, then thickness function. Returns Kuu, Muu (None without C), Kuφ (None
    without e) and one Kφφ per permittivity in eps_list. Internal units.
    """
    detJ = np.linalg.det(J)
    if detJ.real < 0:
        detJ = -detJ
    Jinv_T = np.linalg.inv(J).T
    h = thickness / 2
    zq, zw = legendre.leggauss(order + 2)
    V, dV = lobatto(order, zq)
    dV = dV / h
    nn = 4 * (order + 1)
    comps = np.tile(np.arange(3), nn)
    dtype = complex if np.iscomplexobj(J) else float
    Kuu = np.zeros((3 * nn, 3 * nn), dtype) if C is not None else None
    Muu = np.zeros((3 * nn, 3 * nn), dtype) if C is not None else None
    Kup = np.zeros((3 * nn, nn), dtype) if e is not None else None
    Kpp = [np.zeros((nn, nn), dtype) for _ in eps_list]
    same = comps[:, None] == comps[None, :]
    for xi in itertools.product(GAUSS2, repeat=2):
        N, dN = _q1_shape(_CORNERS, np.array(xi))
        gxy = dN @ Jinv_T.T
        for q in range(len(zw)):
            val = np.outer(N, V[:, q]).ravel()
            G = np.empty((nn, 3), dtype)
            G[:, 0] = np.outer(gxy[:, 0], V[:, q]).ravel()
            G[:, 1] = np.outer(gxy[:, 1], V[:, q]).ravel()
            G[:, 2] = np.outer(N, dV[:, q]).ravel()
            w = detJ / 4 * h * zw[q]
            if C is not None or e is not None:
                B = _strain(np.repeat(G, 3, axis=0), comps)
            if C is not None:
                Kuu += w * B.T @ C @ B
                vu = np.repeat(val, 3)
                Muu += w * rho * np.outer(vu, vu) * same
            if e is not None:
                Kup += w * B.T @ e.T @ G.T
            for K, eps in zip(Kpp, eps_list):
                K += w * G @ eps @ G.T
    return Kuu, Muu, Kup, Kpp


def _strain(g: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """Voigt strain (xx, yy, zz, yz, xz, xy; engineering shear) of DOFs along comps with gradients g."""
    B = np.zeros((6, len(comps)), g.dtype)
    gx, gy, gz = g.T
    ux, uy, uz = comps == 0, comps == 1, comps == 2
    B[0] = gx * ux
    B[1] = gy * uy
    B[2] = gz * uz
    B[3] = gz * uy + gy * uz
    B[4] = gz * ux + gx * uz
    B[5] = gy * ux + gx * uy
    return B


def _eps_tensor(eps) -> np.ndarray:
    eps = np.asarray(eps, dtype=float)
    return (eps * np.eye(3) if eps.ndim == 0 else eps) * EPS0 / EPS_UNIT


class PiezoSolver:
    """
    Piezoelectric FEM of a layered, patterned stack on a structured in-plane grid.

    cell: (2, 2) rows A1, A2, the domain edge vectors in µm (a rectangle Lx × Ly is
    diag(Lx, Ly); a hex unit cell is `HexLattice.vectors`). origin: position of the grid
    corner. layers: `PiezoLayer`s bottom to top; their fraction arrays fix (n1, n2), else
    pass shape. bc: per in-plane direction "free", "clamped" or "periodic". electrodes:
    `Electrode`s. stretch: per direction None or complex (n_d,) PML stretch per element
    column / row.
    """

    def __init__(
        self,
        cell: np.ndarray,
        layers: Sequence[PiezoLayer],
        bc: tuple[str, str] = ("free", "free"),
        electrodes: Sequence[Electrode] = (),
        origin: Sequence[float] = (0.0, 0.0),
        shape: tuple[int, int] | None = None,
        stretch: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
        min_fraction: float = 1e-3,
    ):
        self.cell = np.asarray(cell, dtype=float)
        self.origin = np.asarray(origin, dtype=float)
        self.layers = list(layers)
        self.bc = tuple(bc)
        self.electrodes = list(electrodes)
        if shape is None:
            arrays = [np.shape(L.fraction) for L in self.layers if np.ndim(L.fraction) == 2]
            arrays += [E.mask.shape for E in self.electrodes]
            shape = arrays[0]
        n1, n2 = self.shape = tuple(shape)
        self.n_el = n1 * n2
        orders = [L.order for L in self.layers]
        self.offsets = np.concatenate([[0], np.cumsum(orders)])  # interface function index
        self.n_func = F = int(self.offsets[-1]) + 1
        self.z = np.concatenate([[0.0], np.cumsum([L.thickness for L in self.layers])])

        # Element classes: distinct (stretch1, stretch2) pairs, one Jacobian each.
        s1 = np.ones(n1, complex) if stretch[0] is None else np.asarray(stretch[0], complex)
        s2 = np.ones(n2, complex) if stretch[1] is None else np.asarray(stretch[1], complex)
        self.complex_geometry = stretch[0] is not None or stretch[1] is not None
        pairs = np.stack(np.broadcast_arrays(s1[:, None], s2[None, :]), axis=-1).reshape(-1, 2)
        uniq, cls = np.unique(pairs, axis=0, return_inverse=True)
        cls = cls.ravel()
        jac = []
        for a, b in uniq:
            J = np.column_stack([self.cell[0] / n1 * a, self.cell[1] / n2 * b])
            jac.append(J if self.complex_geometry else J.real)

        el = np.stack(np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij"), -1).reshape(-1, 2)
        enode = (el[:, None, 0] + _CORNERS[None, :, 0]) * (n2 + 1) + el[:, None, 1] + _CORNERS[None, :, 1]
        self.n_node = (n1 + 1) * (n2 + 1)

        self.min_fraction = min_fraction
        self._jac, self._cls, self._enode = jac, cls, enode
        self.angle = next((L.angle for L in self.layers if L.angle is not None), None)
        self.parts = [self._make_part(l) for l in range(len(self.layers))]
        self._assemble()

    # --- assembly and design variables --------------------------------------------------

    def _layer_tensors(self, L: PiezoLayer):
        """Internal-unit C, rho, e, ε_solid of a layer (lab frame) and, if angled, dC, de, dε per degree."""
        mat = L.material
        d = (None, None, None)
        if L.angle is not None:
            if not isinstance(mat, (Piezoelectric, Elastic)):
                raise ValueError("an angled layer needs a material in the crystal frame")
            R, dR = x_cut(L.angle), x_cut_derivative(L.angle)
            if isinstance(mat, Piezoelectric):
                dC, de, deps = rotation_derivatives(mat, R, dR)
                d = (dC / C_UNIT, de / E_UNIT, deps * EPS0 / EPS_UNIT if L.eps_r is None else None)
            else:
                pz = Piezoelectric(mat.name, mat.C, mat.rho, np.zeros((3, 6)), np.eye(3))
                d = (rotation_derivatives(pz, R, dR)[0] / C_UNIT, None, None)
            mat = mat.rotated(R)
        C = mat.C / C_UNIT if mat is not None else None
        rho = mat.rho / RHO_UNIT if mat is not None else None
        e = mat.e / E_UNIT if isinstance(mat, Piezoelectric) and np.any(mat.e) else None
        if e is None:
            d = (d[0], None, d[2])
        eps_s = L.eps_r if L.eps_r is not None else (mat.eps_r if isinstance(mat, Piezoelectric) else None)
        return C, rho, e, eps_s, d

    def _make_part(self, l: int) -> _Part:
        L = self.layers[l]
        n1, n2 = self.shape
        F = self.n_func
        f = np.broadcast_to(np.asarray(L.fraction, dtype=float), (n1, n2)).ravel().copy()
        C, rho, e, eps_s, (dC, de, deps) = self._layer_tensors(L)
        if C is None and eps_s is None and L.background_eps_r is None:
            raise ValueError(f"layer {l} has neither material nor permittivity")
        eps_list = ([_eps_tensor(eps_s)] if eps_s is not None else []) + (
            [_eps_tensor(L.background_eps_r)] if L.background_eps_r is not None else []
        )
        mats = [_layer_element(J, L.thickness, L.order, C, rho, e, eps_list) for J in self._jac]
        funcs = self.offsets[l] + np.arange(L.order + 1)
        enode = self._enode
        ms = np.broadcast_to(np.asarray(1.0 if L.mass_scale is None else L.mass_scale), (n1, n2)).ravel()
        solid = f > self.min_fraction
        diel = np.zeros(self.n_el, bool)
        if eps_s is not None:
            diel |= solid
        if L.background_eps_r is not None:
            diel |= f < 1 - self.min_fraction
        part = _Part(
            udof=(enode[:, :, None, None] * 3 * F + 3 * funcs[None, None, :, None] + np.arange(3)).reshape(self.n_el, -1),
            pdof=(enode[:, :, None] * F + funcs[None, None, :]).reshape(self.n_el, -1),
            cls=self._cls,
            f=f,
            ms=ms,
            solid=solid if C is not None else np.zeros(self.n_el, bool),
            diel=diel,
        )
        stack = lambda i, j=None: np.stack([m[i] if j is None else m[i][j] for m in mats])  # noqa: E731
        if C is not None:
            part.Ke, part.Me = stack(0), stack(1)
        if e is not None:
            part.Ce = stack(2)
        if eps_s is not None:
            part.Es = stack(3, 0)
        if L.background_eps_r is not None:
            part.Eb = stack(3, 1 if eps_s is not None else 0)
        if L.angle is not None:
            dm = [
                _layer_element(J, L.thickness, L.order, dC, 0.0, de, [deps] if deps is not None else [])
                for J in self._jac
            ]
            part.dKe = np.stack([m[0] for m in dm])
            part.dCe = np.stack([m[2] for m in dm]) if de is not None else None
            part.dEs = np.stack([m[3][0] for m in dm]) if deps is not None else None
        return part

    def _assemble(self) -> None:
        K_t, M_t, C_t, E_t = [], [], [], []  # (rows, cols, vals) per block
        for P in self.parts:
            s, c = P.solid, P.cls
            if P.Ke is not None:
                K_t.append(_triplets(P.udof[s], P.udof[s], P.Ke[c[s]] * P.f[s, None, None]))
                M_t.append(_triplets(P.udof[s], P.udof[s], P.Me[c[s]] * (P.f * P.ms)[s, None, None]))
                if P.Ce is not None:
                    C_t.append(_triplets(P.udof[s], P.pdof[s], P.Ce[c[s]] * P.f[s, None, None]))
            d = P.diel
            if d.any():
                Ee = 0.0
                if P.Es is not None:
                    Ee = Ee + P.Es[c[d]] * P.f[d, None, None]
                if P.Eb is not None:
                    Ee = Ee + P.Eb[c[d]] * (1 - P.f[d])[:, None, None]
                E_t.append(_triplets(P.pdof[d], P.pdof[d], Ee))

        # Compact numbering of the DOFs that some element touches.
        cat = lambda ts, i: np.concatenate([t[i] for t in ts]) if ts else np.zeros(0, int)  # noqa: E731
        self.udofs = np.unique(cat(K_t, 0))
        self.pdofs = np.unique(np.concatenate([cat(E_t, 0), cat(C_t, 1)]))
        nu, npd = len(self.udofs), len(self.pdofs)
        uloc = lambda d: np.searchsorted(self.udofs, d)  # noqa: E731
        ploc = lambda d: np.searchsorted(self.pdofs, d)  # noqa: E731

        def build(ts, rmap, cmap, shape):
            if not ts:
                return sp.csr_matrix(shape)
            r, c, v = (np.concatenate([t[i] for t in ts]) for i in range(3))
            return sp.coo_matrix((v, (rmap(r), cmap(c))), shape=shape).tocsr()

        self.K = build(K_t, uloc, uloc, (nu, nu))
        self.M = build(M_t, uloc, uloc, (nu, nu))
        self.Kup = build(C_t, uloc, ploc, (nu, npd))
        self.Kpp = build(E_t, ploc, ploc, (npd, npd))
        self.ndof = nu + npd
        self.lossy = self.complex_geometry or any(np.iscomplexobj(P.ms) and np.any(P.ms.imag) for P in self.parts)

        # Electrodes: φ DOFs of the interface function on the nodes of the masked elements.
        F = self.n_func
        self.terminals: list[str] = []
        self.ground: set[str] = set()
        self._elec_pdofs: dict[str, np.ndarray] = {}
        for E in self.electrodes:
            nodes = np.unique(self._enode[np.asarray(E.mask, bool).ravel()])
            d = nodes * F + self.offsets[E.interface]
            d = d[np.isin(d, self.pdofs)]
            if E.name not in self._elec_pdofs:
                self.terminals.append(E.name)
                self._elec_pdofs[E.name] = d
            else:
                self._elec_pdofs[E.name] = np.union1d(self._elec_pdofs[E.name], d)
            if E.kind == "ground":
                self.ground.add(E.name)
            elif E.kind != "terminal":
                raise ValueError(f"electrode kind {E.kind!r}")

    def set_angle(self, angle: float) -> None:
        """Set the crystal angle θ (degrees) of every angled layer and reassemble."""
        for l, L in enumerate(self.layers):
            if L.angle is not None:
                L.angle = float(angle)
                self.parts[l] = self._make_part(l)
        self.angle = float(angle)
        self._assemble()

    def set_fraction(self, layer: int, fraction: np.ndarray) -> None:
        """Replace one layer's fraction map (n1, n2) and reassemble (the DOF set may change)."""
        P = self.parts[layer]
        P.f = np.broadcast_to(np.asarray(fraction, dtype=float), self.shape).ravel().copy()
        L = self.layers[layer]
        L.fraction = P.f.reshape(self.shape)
        P.solid = (P.f > self.min_fraction) if P.Ke is not None else np.zeros(self.n_el, bool)
        P.diel = np.zeros(self.n_el, bool)
        if P.Es is not None:
            P.diel |= P.f > self.min_fraction
        if P.Eb is not None:
            P.diel |= P.f < 1 - self.min_fraction
        self._assemble()

    # --- geometry helpers -------------------------------------------------------------

    def centers(self) -> np.ndarray:
        """Element centres (n1, n2, 2) in µm."""
        n1, n2 = self.shape
        s = (np.arange(n1) + 0.5) / n1
        t = (np.arange(n2) + 0.5) / n2
        return self.origin + s[:, None, None] * self.cell[0] + t[None, :, None] * self.cell[1]

    def interface_function(self, interface: int) -> int:
        return int(self.offsets[interface])

    # --- reduction --------------------------------------------------------------------

    def _wrap(self, nodes: np.ndarray, theta: np.ndarray):
        """Base node and Bloch phase of each grid node."""
        n1, n2 = self.shape
        i, j = nodes // (n2 + 1), nodes % (n2 + 1)
        ph = np.ones(len(nodes), complex)
        if self.bc[0] == "periodic":
            w = i // n1
            i = i % n1
            ph *= np.exp(1j * theta[0] * w)
        if self.bc[1] == "periodic":
            w = j // n2
            j = j % n2
            ph *= np.exp(1j * theta[1] * w)
        return i * (n2 + 1) + j, ph

    def _reduction(self, k):
        """
        P_u (n_u × n_ur) and P_p (n_p × (n_pf + n_terminals)); the last columns of P_p are the
        terminals in self.terminals order. Also whether the potential needs a gauge.
        """
        theta = self.cell @ np.asarray(k, dtype=float)
        n1, n2 = self.shape
        F = self.n_func
        # u
        node, slot = self.udofs // (3 * F), self.udofs % (3 * F)
        base, ph = self._wrap(node, theta)
        i, j = node // (n2 + 1), node % (n2 + 1)
        keep = np.ones(len(node), bool)
        if self.bc[0] == "clamped":
            keep &= (i > 0) & (i < n1)
        if self.bc[1] == "clamped":
            keep &= (j > 0) & (j < n2)
        key = base * 3 * F + slot
        uk, col = np.unique(key[keep], return_inverse=True)
        Pu = sp.csr_matrix((ph[keep], (np.flatnonzero(keep), col)), shape=(len(node), len(uk)))
        # φ
        node, slot = self.pdofs // F, self.pdofs % F
        base, ph = self._wrap(node, theta)
        key = base * F + slot
        term = np.full(len(key), -1)
        for t, name in enumerate(self.terminals):
            bkeys = np.unique(key[np.isin(self.pdofs, self._elec_pdofs[name])])
            hit = np.isin(key, bkeys)
            if np.any(term[hit] >= 0):
                raise ValueError(f"electrode {name!r} overlaps another")
            term[hit] = t
        free = term < 0
        fk, fcol = np.unique(key[free], return_inverse=True)
        col = np.empty(len(key), int)
        col[free] = fcol
        col[~free] = len(fk) + term[~free]
        Pp = sp.csr_matrix((ph, (np.arange(len(key)), col)), shape=(len(key), len(fk) + len(self.terminals)))
        periodic_phase = [theta[d] for d in range(2) if self.bc[d] == "periodic"]
        gauge = all(abs(np.angle(np.exp(1j * t))) < 1e-12 for t in periodic_phase)
        return Pu, Pp, len(fk), gauge

    def reduced(self, omega2: complex, k=(0.0, 0.0)):
        """
        Reduced system Pᴴ A(ω²) P (internal units, ω² in units of OMEGA_UNIT²), mass matrix
        Pᴴ M P on the u block, and the reduction (Pu, Pp, n_pf, gauge).
        """
        Pu, Pp, npf, gauge = self._reduction(k)
        PuH, PpH = Pu.conj().T.tocsr(), Pp.conj().T.tocsr()
        Kr = PuH @ (self.K - omega2 * self.M) @ Pu
        Mr = PuH @ self.M @ Pu
        Cr = PuH @ self.Kup @ Pp
        Cl = PpH @ self.Kup.T @ Pu
        Er = PpH @ self.Kpp @ Pp
        A = sp.bmat([[Kr, Cr], [Cl, -Er]], format="csc")
        return A, Mr, (Pu, Pp, npf, gauge)

    def _partition(self, red, fixed: dict[str, complex]):
        """Unknown and fixed reduced indices; fixed terminals get their (internal) potential."""
        Pu, Pp, npf, gauge = red
        nur = Pu.shape[1]
        nt = len(self.terminals)
        fixed = dict(fixed)
        for g in self.ground:
            fixed[g] = 0.0
        fix_idx = [nur + npf + self.terminals.index(n) for n in fixed]
        fix_val = np.array([fixed[n] / PHI_UNIT for n in fixed], complex)
        if gauge and not fixed and npf > 0:  # floating potential: pin one free φ DOF
            fix_idx.append(nur)
            fix_val = np.append(fix_val, 0.0)
        mask = np.ones(nur + npf + nt, bool)
        mask[fix_idx] = False
        return np.flatnonzero(mask), np.array(fix_idx, int), fix_val

    def _expand(self, y_u, y_p, red):
        Pu, Pp, _, _ = red
        n1, n2 = self.shape
        F = self.n_func
        u = np.zeros((self.n_node * 3 * F,), complex)
        u[self.udofs] = Pu @ y_u
        phi = np.zeros((self.n_node * F,), complex)
        phi[self.pdofs] = Pp @ y_p
        return u.reshape(n1 + 1, n2 + 1, F, 3), phi.reshape(n1 + 1, n2 + 1, F) * PHI_UNIT

    # --- solves -----------------------------------------------------------------------

    def harmonic(
        self,
        freq: float,
        voltages: dict[str, complex],
        floating: Sequence[str] = (),
        k=(0.0, 0.0),
        force: np.ndarray | None = None,
    ) -> HarmonicResult:
        """
        Driven response at freq (Hz): terminals in `voltages` at those voltages (V), those in
        `floating` floating (Q = 0), every other terminal grounded. force: optional nodal
        forces (N) on the displacement DOFs, shaped like HarmonicResult.u.
        """
        return self._harmonic_multi(freq, [voltages], floating, k, force)[0]

    def _harmonic_multi(self, freq, drives, floating, k, force=None, context=False):
        w2 = (2 * np.pi * freq / OMEGA_UNIT) ** 2
        A, _, red = self.reduced(w2, k)
        fixed_names = [n for n in self.terminals if n not in floating and n not in self.ground]
        unk, fix, _ = self._partition(red, {n: 0.0 for n in fixed_names})
        Auu = _real_if_possible(A[unk][:, unk])
        Auf = A[unk][:, fix]
        solve = factorize_indefinite(Auu)
        nur = red[0].shape[1]
        b = np.zeros(A.shape[0], complex)
        if force is not None:  # N -> internal force unit C_UNIT · µm²
            b[:nur] = red[0].conj().T @ (np.asarray(force).ravel()[self.udofs] / (C_UNIT * 1e-12))
        results, ys = [], []
        for drive in drives:
            _, _, fval = self._partition(red, {n: drive.get(n, 0.0) for n in fixed_names})
            y = np.zeros(A.shape[0], complex)
            y[fix] = fval
            y[unk] = solve(b[unk] - Auf @ fval)
            r = A @ y
            npf = red[2]
            charge = {n: -r[nur + npf + t] * Q_UNIT for t, n in enumerate(self.terminals)}
            volt = {n: y[nur + npf + t] * PHI_UNIT for t, n in enumerate(self.terminals)}
            u, phi = self._expand(y[:nur], y[nur:], red)
            results.append(HarmonicResult(freq, u, phi, charge, volt))
            ys.append(y)
        if context:
            return results, dict(A=A, red=red, unk=unk, fix=fix, solve=solve, ys=ys, omega2=w2)
        return results

    def admittance(self, freqs: Sequence[float], ports: Sequence[str] | None = None, floating=(), k=(0.0, 0.0)):
        """
        Short-circuit admittance matrix Y (len(freqs), n_ports, n_ports) in S: Y_ij = iω Q_i
        with V_j = 1 V and every other non-floating terminal grounded.
        """
        ports = [p for p in self.terminals if p not in self.ground] if ports is None else list(ports)
        Y = np.zeros((len(freqs), len(ports), len(ports)), complex)
        for a, f in enumerate(freqs):
            res = self._harmonic_multi(f, [{p: 1.0} for p in ports], floating, k)
            for j, r in enumerate(res):
                for i, p in enumerate(ports):
                    Y[a, i, j] = 1j * 2 * np.pi * f * r.charge[p]
        return Y

    def capacitance(self, ports: Sequence[str] | None = None, k=(0.0, 0.0)) -> np.ndarray:
        """Clamped (u = 0) electrostatic short-circuit capacitance matrix in F."""
        ports = [p for p in self.terminals if p not in self.ground] if ports is None else list(ports)
        _, Pp, npf, gauge = self._reduction(k)
        E = (Pp.conj().T @ self.Kpp @ Pp).tocsr()
        red = (sp.csr_matrix((0, 0)), Pp, npf, gauge)
        names = [n for n in self.terminals if n not in self.ground]
        unk, fix, _ = self._partition(red, {n: 0.0 for n in names})
        solve = factorize(E[unk][:, unk], hpd=not self.complex_geometry)
        Cm = np.zeros((len(ports), len(ports)), complex)
        for j, p in enumerate(ports):
            _, _, fval = self._partition(red, {n: (1.0 if n == p else 0.0) for n in names})
            y = np.zeros(E.shape[0], complex)
            y[fix] = fval
            y[unk] = solve(-(E[unk][:, fix] @ fval))
            r = E @ y  # = -(reduced A y) with u = 0, i.e. the terminal charges
            for i, q in enumerate(ports):
                Cm[i, j] = r[npf + self.terminals.index(q)] * Q_UNIT
        return Cm.real if not self.complex_geometry else Cm

    def eigenmodes(
        self,
        n_modes: int = 10,
        shift_hz: complex = -0.5e9,
        k=(0.0, 0.0),
        electrical: str = "short",
        vectors: bool = False,
        tol: float = 1e-10,
        _reduced: bool = False,
    ):
        """
        n_modes eigenfrequencies (complex, Hz; Im > 0 decays as e^{iωt}) nearest shift_hz, by
        shift-invert on the displacement with the potential eliminated in each solve.
        electrical: "short" (every terminal at 0 V) or "open" (terminals floating, Q = 0;
        "ground" electrodes stay at 0). A real negative shift selects the lowest modes.
        Returns freqs, or (freqs, u, phi) with u (n_modes, n1+1, n2+1, F, 3) and phi in V
        per unit of the (arbitrarily normalised) mode.
        """
        if np.isrealobj(shift_hz) and shift_hz < 0:
            sigma = -((2 * np.pi * shift_hz / OMEGA_UNIT) ** 2)
        else:
            sigma = (2 * np.pi * shift_hz / OMEGA_UNIT) ** 2
        A, Mr, red = self.reduced(sigma, k)
        Pu, Pp, npf, gauge = red
        if electrical == "short":
            fixed = {n: 0.0 for n in self.terminals}
        elif electrical == "open":
            fixed = {}
        else:
            raise ValueError(electrical)
        unk, fix, _ = self._partition(red, fixed)
        nur = Pu.shape[1]
        Auu = _real_if_possible(A[unk][:, unk])
        Mr = _real_if_possible(Mr.tocsr())
        solve = factorize_indefinite(Auu)
        n_unk = len(unk)
        real = np.isrealobj(Auu) and np.isrealobj(Mr) and np.isrealobj(sigma)
        dtype = float if real else complex

        def op(v):
            rhs = np.zeros(n_unk, dtype)
            rhs[:nur] = Mr @ v
            return solve(rhs)[:nur]

        OP = spla.LinearOperator((nur, nur), matvec=op, dtype=dtype)
        v0 = np.random.default_rng(0).uniform(0.5, 1.5, nur)  # deterministic start: reproducible mode sets
        if real:  # (K_eff - σM)^{-1} M is M-symmetric: Lanczos in real arithmetic

            def opinv(v):  # (K_eff - σM)^{-1}; ARPACK applies M itself
                rhs = np.zeros(n_unk)
                rhs[:nur] = v
                return solve(rhs)[:nur]

            dummy = spla.LinearOperator((nur, nur), matvec=lambda v: v, dtype=float)
            OPinv = spla.LinearOperator((nur, nur), matvec=opinv, dtype=float)
            lam, V = spla.eigsh(dummy, k=n_modes, M=Mr, sigma=sigma, which="LM", OPinv=OPinv, tol=tol, v0=v0.real)
            nu = 1 / (lam - sigma)
        else:
            nu, V = spla.eigs(OP, k=n_modes, which="LM", tol=tol, v0=v0.astype(dtype))
            lam = sigma + 1 / nu
        freqs = OMEGA_UNIT * np.sqrt(lam.astype(complex)) / (2 * np.pi)
        order = np.argsort(freqs.real)
        freqs = freqs[order]
        if not vectors:
            return freqs
        us, phis, ys = [], [], []
        for m in order:
            rhs = np.zeros(n_unk, dtype)
            rhs[:nur] = Mr @ V[:, m]
            x = solve(rhs) / nu[m]
            y = np.zeros(A.shape[0], complex)
            y[unk] = x
            y[:nur] = V[:, m]
            u, phi = self._expand(y[:nur], y[nur:], red)
            us.append(u)
            phis.append(phi)
            ys.append(y)
        if _reduced:  # reduced vectors and partition, for motional-branch sensitivities
            return freqs, np.array(us), np.array(phis), dict(ys=np.array(ys), red=red, unk=unk, Mr=Mr)
        return freqs, np.array(us), np.array(phis)

    def coupling(self, n_modes: int = 10, shift_hz: float = -0.5e9, k=(0.0, 0.0)):
        """
        Short- and open-circuit eigenfrequencies (Hz, real parts) and k_eff² per mode, the
        open-circuit modes paired with the short-circuit ones by displacement overlap.
        """
        fs, us, _ = self.eigenmodes(n_modes, shift_hz, k, "short", vectors=True)
        fo, uo, _ = self.eigenmodes(n_modes, shift_hz, k, "open", vectors=True)
        a = us.reshape(n_modes, -1)
        b = uo.reshape(n_modes, -1)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        overlap = np.abs(a.conj() @ b.T)
        pair = overlap.argmax(axis=1)
        fs, fo = fs.real, fo.real[pair]
        k2 = np.divide(fo**2 - fs**2, fo**2, out=np.zeros_like(fo), where=fo > 0)
        return fs, fo, k2, overlap.max(axis=1)


    # --- sensitivities ------------------------------------------------------------------

    def _full(self, y_u, y_p, red, conj: bool = False):
        """Reduced (u, φ) vectors (columns) to flat full-grid vectors in internal units; with conj, mapped by conj(P)."""
        Pu, Pp, _, _ = red
        if conj:
            Pu, Pp = Pu.conj(), Pp.conj()
        F = self.n_func
        y_u, y_p = np.atleast_2d(y_u.T).T, np.atleast_2d(y_p.T).T
        u = np.zeros((self.n_node * 3 * F, y_u.shape[1]), complex)
        u[self.udofs] = Pu @ y_u
        phi = np.zeros((self.n_node * F, y_p.shape[1]), complex)
        phi[self.pdofs] = Pp @ y_p
        return u, phi

    def _forms(self, zu, zp, xu, xp, omega2, frac: bool = True):
        """
        Element bilinear forms z_eᵀ (∂A_e) x_e for every vector pair (columns): the derivative
        with respect to each element's fraction, (n_layers, nv, n1, n2), and with respect to θ
        (per degree), (nv,). ∂A_e/∂f = [[Ke - ω² ms Me, Ce], [Ceᵀ, -(Es - Eb)]] on solid
        elements; ∂A/∂θ = Σ_e f_e [[dKe, dCe], [dCeᵀ, -dEs]].
        """
        nv = xu.shape[1]
        n1, n2 = self.shape
        g_frac = np.zeros((len(self.parts), nv, n1, n2), complex)
        g_ang = np.zeros(nv, complex)
        for l, P in enumerate(self.parts):
            s = P.solid
            if not s.any() and not (P.diel.any() and P.Es is not None):
                continue
            Zu, Xu = zu[P.udof[s]], xu[P.udof[s]]
            Zp, Xp = zp[P.pdof[s]], xp[P.pdof[s]]
            c = P.cls[s]
            g = np.zeros((s.sum(), nv), complex)
            ga = np.zeros((s.sum(), nv), complex)
            if P.Ke is not None:
                g += _bilinear(Zu, P.Ke, Xu, c) - omega2 * P.ms[s, None] * _bilinear(Zu, P.Me, Xu, c)
                if P.dKe is not None:
                    ga += _bilinear(Zu, P.dKe, Xu, c)
            if P.Ce is not None:
                g += _bilinear(Zu, P.Ce, Xp, c) + _bilinear(Zp, P.Ce.transpose(0, 2, 1), Xu, c)
                if P.dCe is not None:
                    ga += _bilinear(Zu, P.dCe, Xp, c) + _bilinear(Zp, P.dCe.transpose(0, 2, 1), Xu, c)
            if P.Es is not None:
                dE = P.Es - (P.Eb if P.Eb is not None else 0.0)
                g -= _bilinear(Zp, dE, Xp, c)
                if P.dEs is not None:
                    ga -= _bilinear(Zp, P.dEs, Xp, c)
            if frac:
                gf = np.zeros((self.n_el, nv), complex)
                gf[s] = g
                g_frac[l] = gf.T.reshape(nv, n1, n2)
            g_ang += (P.f[s, None] * ga).sum(axis=0)
        return g_frac, g_ang

    def _transpose_ok(self, k) -> str:
        """'H' for a Hermitian system, 'T' for a complex-symmetric one (lossy, real Bloch phases)."""
        if not self.lossy:
            return "H"
        theta = self.cell @ np.asarray(k, dtype=float)
        if all(abs(np.exp(1j * theta[d]) - 1) < 1e-12 for d in range(2) if self.bc[d] == "periodic"):
            return "T"
        raise NotImplementedError("eigen-sensitivities of a lossy Bloch problem need left eigenvectors")

    def eigen_sensitivities(
        self,
        n_modes: int = 10,
        shift_hz: complex = -0.5e9,
        k=(0.0, 0.0),
        electrical: str = "short",
        rtol: float = 1e-4,
        degenerate: str = "mean",
    ) -> EigenSensitivity:
        """
        Eigenmodes as in `eigenmodes` and the gradients of their frequencies with respect to θ
        and every layer's fraction map. Modes whose frequencies differ by less than rtol form a
        cluster; with degenerate="mean" its members carry the gradient of the cluster mean.
        """
        kind = self._transpose_ok(k)
        freqs, U, PHI = self.eigenmodes(n_modes, shift_hz, k, electrical, vectors=True)
        nm = len(freqs)
        xu = U.reshape(nm, -1).T
        xp = PHI.reshape(nm, -1).T / PHI_UNIT
        zu, zp = (xu.conj(), xp.conj()) if kind == "H" else (xu, xp)
        lam = (2 * np.pi * freqs / OMEGA_UNIT) ** 2
        norm = np.einsum("iv,iv->v", zu[self.udofs], self.M @ xu[self.udofs])
        gf = np.zeros((len(self.parts), nm, *self.shape), complex)
        ga = np.zeros(nm, complex)
        for m in range(nm):  # each mode at its own ω² = λ
            a, b = self._forms(zu[:, m : m + 1], zp[:, m : m + 1], xu[:, m : m + 1], xp[:, m : m + 1], lam[m])
            gf[:, m], ga[m] = a[:, 0], b[0]
        scale = np.where(np.abs(lam) > 0, freqs / (2 * np.where(np.abs(lam) > 0, lam, 1)), 0.0) / norm
        dfa = ga * scale
        dff = gf * scale[None, :, None, None]
        if kind == "H":
            dfa, dff = dfa.real, dff.real
        clusters = _clusters(freqs.real, rtol)
        if degenerate == "mean":
            for c in clusters:
                if len(c) > 1:
                    dfa[c] = dfa[c].mean()
                    dff[:, c] = dff[:, c].mean(axis=1, keepdims=True)
        return EigenSensitivity(freqs, dfa, dff, clusters, U, PHI)

    def coupling_sensitivities(self, n_modes: int = 10, shift_hz: float = -0.5e9, k=(0.0, 0.0), rtol: float = 1e-4):
        """
        k_eff² = 1 - λ_sc/λ_oc per short-circuit mode (open-circuit partner by displacement
        overlap, as `coupling`) and its gradients: dk² = (λ_sc dλ_oc - λ_oc dλ_sc) / λ_oc².
        """
        sc = self.eigen_sensitivities(n_modes, shift_hz, k, "short", rtol)
        oc = self.eigen_sensitivities(n_modes, shift_hz, k, "open", rtol)
        a = sc.u.reshape(n_modes, -1)
        b = oc.u.reshape(n_modes, -1)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        overlap = np.abs(a.conj() @ b.T)
        pair = overlap.argmax(axis=1)
        fs, fo = sc.freqs.real, oc.freqs.real[pair]
        ok = fo > 0
        k2 = np.where(ok, 1 - fs**2 / np.where(ok, fo, 1) ** 2, 0.0)
        # k² = 1 - (fs/fo)²: dk² = 2 fs² / fo² (dfo/fo - dfs/fs)
        r = np.where(ok & (fs > 0), fs**2 / np.where(ok, fo, 1) ** 2, 0.0)
        inv_s = np.where(fs > 0, 1 / np.where(fs > 0, fs, 1), 0.0)
        inv_o = np.where(ok, 1 / np.where(ok, fo, 1), 0.0)
        dk2a = 2 * r * (oc.dfreq_angle.real[pair] * inv_o - sc.dfreq_angle.real * inv_s)
        dk2f = 2 * r[None, :, None, None] * (
            oc.dfreq_frac.real[:, pair] * inv_o[None, :, None, None] - sc.dfreq_frac.real * inv_s[None, :, None, None]
        )
        return CouplingSensitivity(fs, fo, k2, dk2a, dk2f, overlap.max(axis=1), sc, oc, pair)

    def _driven_gradient(self, ctx, y, z_red):
        """dJ = z_redᵀ (dA y) for the driven state y: gradients with respect to fractions and θ."""
        red = ctx["red"]
        nur = red[0].shape[1]
        xu, xp = self._full(y[:nur], y[nur:], red)
        zu, zp = self._full(z_red[:nur], z_red[nur:], red, conj=True)
        gf, ga = self._forms(zu, zp, xu, xp, ctx["omega2"])
        return gf[:, 0], ga[0]

    def admittance_gradient(self, freq: float, port: str, floating: Sequence[str] = (), k=(0.0, 0.0)):
        """
        Y = iωQ/V of `port` (1 V on it, other non-floating terminals grounded) and its
        complex gradients dY/dθ (S/degree) and dY/dfraction (S, (n_layers, n1, n2)). One
        adjoint solve with the transposed factorisation.
        """
        res, ctx = self._harmonic_multi(freq, [{port: 1.0}], floating, k, context=True)
        A, red, unk, solve = ctx["A"], ctx["red"], ctx["unk"], ctx["solve"]
        t = red[0].shape[1] + red[2] + self.terminals.index(port)
        # Q = -(A y)_t: dQ = -(dA y)_t + A_t,unk A_uu⁻¹ (dA y)_unk = zᵀ dA y
        z = np.zeros(A.shape[0], complex)
        z[t] = -1.0
        z[unk] = solve(np.asarray(A.tocsr()[t][:, unk].todense()).ravel(), "T")
        gf, ga = self._driven_gradient(ctx, ctx["ys"][0], z)
        iw = 1j * 2 * np.pi * freq * Q_UNIT
        return 1j * 2 * np.pi * freq * res[0].charge[port], iw * ga, iw * gf

    def field_gradient(
        self,
        freq: float,
        voltages: dict[str, complex],
        weights: np.ndarray,
        floating: Sequence[str] = (),
        k=(0.0, 0.0),
        force: np.ndarray | None = None,
    ):
        """
        J = Σ weights · u (u in µm on the full grid, weights shaped like HarmonicResult.u,
        e.g. the conjugate of a port mode on a port plane for a modal amplitude) for the
        driven response, and dJ/dθ (per degree), dJ/dfraction. A real objective such as the
        power |J|² follows as 2 Re(conj(J) dJ).
        """
        res, ctx = self._harmonic_multi(freq, [voltages], floating, k, force, context=True)
        A, red, unk, solve = ctx["A"], ctx["red"], ctx["unk"], ctx["solve"]
        nur = red[0].shape[1]
        g = np.asarray(weights).ravel()[self.udofs]
        c = np.zeros(A.shape[0], complex)
        c[:nur] = red[0].T @ g  # J = gᵀ Pu y_u
        y = ctx["ys"][0]
        J = c @ y
        z = np.zeros(A.shape[0], complex)
        z[unk] = -solve(c[unk], "T")  # dJ = -c_unkᵀ A_uu⁻¹ (dA y)_unk
        gf, ga = self._driven_gradient(ctx, y, z)
        return J, ga, gf


    def _mass_forms(self, zu, xu):
        """Element forms z_eᵀ (∂M/∂f_e) x_e = ms z_eᵀ Me x_e per layer, (n_layers, nv, n1, n2); M does not depend on θ."""
        nv = xu.shape[1]
        out = np.zeros((len(self.parts), nv, *self.shape), complex)
        for l, P in enumerate(self.parts):
            s = P.solid
            if P.Me is None or not s.any():
                continue
            g = np.zeros((self.n_el, nv), complex)
            g[s] = P.ms[s, None] * _bilinear(zu[P.udof[s]], P.Me, xu[P.udof[s]], P.cls[s])
            out[l] = g.T.reshape(nv, *self.shape)
        return out

    def motional(
        self,
        port: str,
        n_modes: int = 10,
        shift_hz: float = -0.5e9,
        k=(0.0, 0.0),
        Z_r: float = 700.0,
        gradient: Sequence[int] = (),
    ) -> MotionalBranch:
        """
        Butterworth-Van Dyke branches of the short-circuit modes as seen from terminal `port`
        (every other terminal grounded), and the electromechanical coupling to a microwave
        resonator of impedance Z_r.

        Eliminating u and φ from the driven system with the modes (K_eff - λM) u_m = 0,
        u_mᴴ M u_m = 1 gives exactly Y(ω)/(iω) = C₀ + Σ_m α_m²/(ω_m² - ω²), where C₀ is the
        clamped capacitance and α_m = -Q_m the charge the mode induces on the shorted
        terminal (its row of the reduced system applied to the mode). Hence the motional
        branch C_m = α_m²/ω_m², L_m = 1/α_m², and with a resonator of C_r = 1/(ω Z_r),
        g_em = (ω_m/2)√(C_m/C_r) = (ω_m/2)√(C_m ω_m Z_r), so g_em² = γ_e ω_m Z_r/(4R₀) with
        γ_e = R₀ω_m²C_m (as `system.electromechanical_g`). C_m and g_em are per unit cell for a
        Bloch-periodic structure (∝ number of periods N for a finite one, g_em ∝ √N).

        gradient: indices (into the returned modes) whose dg_em/dθ, dg_em/dfraction are wanted
        (real or Hermitian systems). With J = |α|², d ln g = dλ/(4λ) + dJ/(2J), and
        dα = ℓᵀ dA(λ) y - ½ α yᴴ dM y, ℓ = e_t - conj(w), where w solves the bordered system
        [[T(λ), By], [yᴴB, 0]] [w; μ] = [c̄; 0] (T the shorted system at λ, c its terminal row):
        the eigenvector derivative without forming it, one extra factorisation per mode.
        """
        kind = self._transpose_ok(k)
        if kind != "H":
            raise NotImplementedError("motional-branch sensitivities need a lossless system")
        freqs, U, PHI, ctx = self.eigenmodes(n_modes, shift_hz, k, "short", vectors=True, _reduced=True)
        red, unk, Mr, ys = ctx["red"], ctx["unk"], ctx["Mr"], ctx["ys"]
        Pu, Pp, npf, _ = red
        nur = Pu.shape[1]
        t = nur + npf + self.terminals.index(port)
        lam = (2 * np.pi * freqs.real / OMEGA_UNIT) ** 2
        A, _, _ = self.reduced(0.0, k)  # the terminal row has no mass term
        c = np.asarray(A.tocsr()[t].todense()).ravel()
        norm = np.sqrt(np.einsum("iv,iv->v", ys[:, :nur].conj().T, Mr @ ys[:, :nur].T).real)
        ys = ys / norm[:, None]
        alpha = ys @ c
        rigid = freqs.real < 1e6  # zero-frequency (rigid-body) modes carry no motional branch
        C_m = np.where(rigid, 0.0, np.abs(alpha) ** 2 / np.where(rigid, 1.0, lam) * Q_UNIT / PHI_UNIT)
        omega = 2 * np.pi * freqs.real
        g = 0.5 * omega * np.sqrt(C_m * omega * Z_r)
        C0 = float(np.real(self.capacitance([port], k)[0, 0]))
        nl = len(self.parts)
        dga = np.full(len(freqs), np.nan)
        dgf = np.full((nl, len(freqs), *self.shape), np.nan)
        for m in gradient:
            y = ys[m]
            xu, xp = self._full(y[:nur], y[nur:], red)
            # dλ = yᴴ dA(λ) y (normalised mode)
            gf_l, ga_l = self._forms(xu.conj(), xp.conj(), xu, xp, lam[m])
            # bordered solve for w
            Am, _, _ = self.reduced(lam[m], k)
            T = Am.tocsr()[unk][:, unk]
            By = np.zeros(len(unk), complex)
            By[:nur] = Mr @ y[:nur]
            Tb = sp.bmat([[T, sp.csr_matrix(By[:, None])], [sp.csr_matrix(By.conj()[None, :]), None]], format="csc")
            rhs = np.append(c[unk].conj(), 0.0)
            w = factorize_indefinite(Tb)(rhs)[:-1]
            ell = np.zeros(A.shape[0], complex)
            ell[unk] = -w.conj()
            ell[t] += 1.0
            zu, zp = self._full(ell[:nur], ell[nur:], red, conj=True)
            gf_a, ga_a = self._forms(zu, zp, xu, xp, lam[m])
            mf = self._mass_forms(xu.conj(), xu)
            dal_f = gf_a[:, 0] - 0.5 * alpha[m] * mf[:, 0]
            dal_a = ga_a[0]
            J = abs(alpha[m]) ** 2
            dJ_f = 2 * np.real(np.conj(alpha[m]) * dal_f)
            dJ_a = 2 * np.real(np.conj(alpha[m]) * dal_a)
            dga[m] = g[m] * (ga_l[0].real / (4 * lam[m]) + dJ_a / (2 * J))
            dgf[:, m] = g[m] * (gf_l[:, 0].real / (4 * lam[m]) + dJ_f / (2 * J))
        return MotionalBranch(freqs.real, C_m, C0, g, Z_r, dga, dgf, U)


def _clusters(f: np.ndarray, rtol: float) -> list[np.ndarray]:
    """Groups of sorted frequencies closer than rtol (relative)."""
    groups, start = [], 0
    for i in range(1, len(f) + 1):
        if i == len(f) or f[i] - f[i - 1] > rtol * max(abs(f[i]), 1e-12):
            groups.append(np.arange(start, i))
            start = i
    return groups


def _real_if_possible(A: sp.spmatrix) -> sp.spmatrix:
    """A as a real matrix when its imaginary part vanishes (k = 0, no absorber): real SuperLU is several times faster."""
    if np.iscomplexobj(A.data) and not np.any(A.data.imag):
        A = A.copy()
        A.data = A.data.real
    return A


def _triplets(rdof: np.ndarray, cdof: np.ndarray, vals: np.ndarray):
    """COO triplets of element matrices vals (n_el, nr, nc) on DOFs rdof (n_el, nr), cdof (n_el, nc)."""
    nr, nc = rdof.shape[1], cdof.shape[1]
    r = np.broadcast_to(rdof[:, :, None], (len(rdof), nr, nc)).ravel()
    c = np.broadcast_to(cdof[:, None, :], (len(cdof), nr, nc)).ravel()
    return r, c, vals.ravel()
