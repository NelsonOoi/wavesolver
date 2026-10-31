"""
Finite cavities as chains of cells (mirror → taper → defect → …) in one long supercell:
mechanical cavity modes by shift-invert, per-column envelopes, and a 1D tight-binding envelope
model built from per-cell Bloch bands.

A cavity is a sequence of cells along x, each given either as a solid-fraction raster on the
per-cell lattice-aligned grid (n_cell × n2, any free-form design) or as hole polygons centred
on its own cell (`chain_domain`). `gpt07_cavity` builds the drawn GPT07 dagger row of
`gpt_omc.build` (mirror taper, defect, waveguide taper, …) on top of the same structure.

Domain. Columns c0 … c1 of the dagger row (column n centred at x = n·a) with the shield rows
above and below, as one supercell a1 = (N a, 0), a2 = (a/2, L_y), N = c1 − c0 + 1 and L_y as in
geometry.line_defect_supercell, centred on the middle column, so the in-plane grid of n_cell
elements per column is the per-cell grid of `line_defect_supercell` repeated (same element
shape, nodes aligned): per-cell Bloch modes and cavity fields can be compared node by node.
The supercell closes the row on itself (column c1 next to c0) and the shield on its mirror
image across y = ±L_y/2, as the line-defect supercell does. Holes are the drawn `Layout` holes
(Crystal.drawn(), no SEM correction, no fillets).

Mechanics. `slab_p`, even z-parity, at k = 0 of the supercell: K, M real symmetric, so the
modes near a target frequency come from shift-invert with a real indefinite LU
(linalg.factorize_indefinite). An absorber adds a complex mass −iη(x) M_e over chosen
columns (ρ → ρ(1 − iη)): with e^{iωt}, Im ω > 0 is decay, Q = Re ω / 2 Im ω. The absorber is
a numerical device standing in for whatever the row continues into; its Q is only meaningful
if it is converged in η and length.

Units as omc.mech_bloch: µm, Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from shapely import affinity
from shapely.geometry import Polygon

from .geometry import Lattice2D, gpt_omc, solid_fraction
from .linalg import factorize_indefinite
from .mech_adjoint import element_forms, element_vectors
from .mech_bloch import OMEGA_UNIT, BlochSolver

DAGGER_REGIONS = ("input", "mirror", "defect", "waveguide", "exit")


@dataclass
class CavityDomain:
    """
    A run of cells (columns c0…c1, column n centred at lab x = n·pitch) as a supercell centred on
    x_c (lab x = local x + x_c). The pattern is either hole polygons in local coordinates
    (`holes`) or per-cell fraction rasters (`cell_fractions`, each (n_cell, n2), concatenated
    along a1). row_offset/row_pitch only set `half_width`, the |y| window of the defect rows.
    """

    lattice: Lattice2D
    holes: list[Polygon] | None
    columns: tuple[int, int]
    x_c: float
    pitch: float
    row_offset: float
    row_pitch: float
    shield_rows: int
    regions: dict[int, str]
    cell_fractions: list[np.ndarray] | None = None

    def fraction(self, n1: int, n2: int, supersample: int = 4) -> np.ndarray:
        """Solid fraction (n1, n2) of the whole domain: rasterised holes, or the concatenated cell rasters."""
        if self.cell_fractions is not None:
            f = np.concatenate(self.cell_fractions, axis=0)
            if f.shape != (n1, n2):
                raise ValueError(f"cell rasters give {f.shape}, the solver grid is {(n1, n2)}")
            return f
        return solid_fraction(self.lattice, self.holes, (n1, n2), supersample=supersample)

    @property
    def n_columns(self) -> int:
        return self.columns[1] - self.columns[0] + 1

    def column_x(self, n: int) -> float:
        """Local x of the centre of column n."""
        return n * self.pitch - self.x_c

    @property
    def half_width(self) -> float:
        """|y| bound of the dagger rows (as in the line-defect benchmarks)."""
        return self.row_offset + self.row_pitch / 2


def gpt07_cavity(
    crystal: gpt_omc.Crystal | None = None, columns: tuple[int, int] = (-8, 12), shield_rows: int = 4, skew: bool = True
) -> CavityDomain:
    """
    The drawn GPT07 dagger row from column c0 to c1 (0 = defect; negative: mirror then input
    side, positive: waveguide then exit side) with shield_rows blade rows on each side.
    skew=False makes the supercell rectangular, a2 = (0, L_y): the outermost blade rows then
    meet square-stacked across y = ±L_y/2 (a seam shield_rows rows from the cavity), but the
    lattice-aligned grid becomes symmetric under y → −y, so σ_y is exact (`mech_cavity(sym_y=True)`).
    """
    layout = gpt_omc.build(crystal or gpt_omc.Crystal())
    c = layout.crystal
    a, rp = c.pitch, c.row_pitch
    c0, c1 = columns
    x_c = (c0 + c1) / 2 * a
    x_lo, x_hi = c0 * a - a / 2, c1 * a + a / 2  # half-open: each hole counted once across the seam
    holes, regions = [], {}
    for h in layout.holes:
        x, y = h.center
        if h.region in DAGGER_REGIONS:
            if c0 <= h.column <= c1:
                holes.append(affinity.translate(h.shape, -x_c, 0.0))
                regions[h.column] = h.region
        elif h.region == "shield" and h.row <= shield_rows and x_lo <= x < x_hi:
            holes.append(affinity.translate(h.shape, -x_c, 0.0))
    height = 2 * (c.row_offset + shield_rows * rp) + rp
    lattice = Lattice2D(np.array([[(c1 - c0 + 1) * a, 0.0], [a / 2 if skew else 0.0, height]]))
    return CavityDomain(lattice, holes, columns, x_c, a, c.row_offset, rp, shield_rows, regions)


def chain_domain(
    cells: list,
    pitch: float,
    height: float,
    first_column: int = 0,
    skew: bool = False,
    half_width: float | None = None,
    regions: dict[int, str] | None = None,
) -> CavityDomain:
    """
    A cavity from a sequence of cells along x (first_column, first_column + 1, …): each cell is a
    fraction raster (n_cell, n2) on its own lattice-aligned grid (same shape for all cells), or a
    list of hole polygons in that cell's centred coordinates (holes may cross into neighbours).
    The supercell is a1 = (N pitch, 0), a2 = (pitch/2 if skew else 0, height). half_width: |y|
    window of the defect region used for localisation tags (default height/6).
    """
    N = len(cells)
    c0, c1 = first_column, first_column + N - 1
    x_c = (c0 + c1) / 2 * pitch
    lattice = Lattice2D(np.array([[N * pitch, 0.0], [pitch / 2 if skew else 0.0, height]]))
    hw = height / 6 if half_width is None else half_width
    if all(isinstance(c, np.ndarray) for c in cells):
        return CavityDomain(lattice, None, (c0, c1), x_c, pitch, hw, 0.0, 0, regions or {}, [np.asarray(c, float) for c in cells])
    holes = [affinity.translate(h, (c0 + i) * pitch - x_c, 0.0) for i, cell in enumerate(cells) for h in cell]
    return CavityDomain(lattice, holes, (c0, c1), x_c, pitch, hw, 0.0, 0, regions or {})


def taper_cell(n: int, crystal: gpt_omc.Crystal | None = None) -> gpt_omc.DaggerShape:
    """The dagger shape of column n of the drawn row (mirror taper for n < 0, waveguide taper for n > 0)."""
    c = (crystal or gpt_omc.Crystal()).drawn()
    if n == 0:
        return c.defect
    if n < 0:
        return c.defect.toward(c.mirror, c.mirror_taper.weight(-n))
    return c.defect.toward(c.waveguide, c.waveguide_taper.weight(n))


# --- mechanics -------------------------------------------------------------------------------


@dataclass
class MechCavity:
    """A slab_p solver on a CavityDomain, its fraction, and optional absorber profile η per element."""

    domain: CavityDomain
    solver: BlochSolver
    fraction: np.ndarray
    n_cell: int
    P: sp.csr_matrix | None = None  # σ_y-even basis (ndof × n_reduced), when sym_y

    def element_xy(self, part: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Local (x, y) of the centre of each kept element of a part."""
        n1, n2 = self.solver.shape
        cell = self.solver.parts[part].cell
        i, j = cell // n2, cell % n2
        V = self.domain.lattice.vectors
        s1, s2 = (i + 0.5) / n1 - 0.5, (j + 0.5) / n2 - 0.5
        return s1 * V[0, 0] + s2 * V[1, 0], s1 * V[0, 1] + s2 * V[1, 1]

    def element_column(self) -> np.ndarray:
        """Dagger-row column of each kept element (by the element's in-plane grid index along a1)."""
        n2 = self.solver.shape[1]
        i = self.solver.parts[0].cell // n2
        return self.domain.columns[0] + i // self.n_cell


def mech_cavity(
    domain: CavityDomain, material, n_cell: int = 15, n2: int | None = None, order: int = 4, supersample: int = 4, sym_y: bool = False,
    min_fraction: float = 1e-3, loads: list | None = None,
) -> MechCavity:
    """
    slab_p (even) solver on the domain, n_cell elements per column along x and n2 along a2
    (default ≈ a/n_cell, rounded to even; both taken from the rasters of a raster-built domain). sym_y (rectangular domain only): keep the σ_y-even
    subspace, u(x, −y) = (u_x, −u_y, u_z)(x, y), as an exact reduction P of the DOFs.
    min_fraction: elements kept by the solver (≈ 1e-9 for shape gradients, see omc.coupling_adjoint).
    loads: BlochSolver membrane loads [(material, thickness µm, fraction (n1, n2))] on the same grid
    (e.g. an LN/Al transducer on part of the chain); fixed, not design variables.
    """
    if domain.cell_fractions is not None:
        n_cell, n2 = domain.cell_fractions[0].shape
    n1 = n_cell * domain.n_columns
    if n2 is None:
        n2 = 2 * int(round(domain.lattice.vectors[1, 1] / (domain.pitch / n_cell) / 2))
    frac = domain.fraction(n1, n2, supersample)
    solver = BlochSolver(domain.lattice, frac, material, model="slab_p", thickness=0.22, parity="even", order=order, min_fraction=min_fraction,
                         loads=loads)
    cav = MechCavity(domain, solver, frac, n_cell)
    if sym_y:
        if domain.lattice.vectors[1, 0] != 0 or n2 % 2:
            raise ValueError("sym_y needs a rectangular domain (skew=False) and an even n2")
        cav.P = _sym_y_basis(solver)
    return cav


def _sym_y_basis(solver: BlochSolver) -> sp.csr_matrix:
    """Orthonormal basis of the σ_y-even DOF subspace: node (i, j) ↔ (i, −j mod n2), u_y flips sign."""
    n1, n2 = solver.shape
    dpn = len(solver.basis)
    g = solver.dofs
    node, b = g // dpn, g % dpn
    i, j = node // n2, node % n2
    partner = (i * n2 + (-j) % n2) * dpn + b
    sign = np.where(np.array([c for c, _ in solver.basis])[b] == 1, -1.0, 1.0)
    pos = np.full(g.max() + 1, -1)
    pos[g] = np.arange(len(g))
    q = pos[partner]
    if np.any(q < 0):
        raise ValueError("DOFs are not mirror symmetric")
    me = np.arange(len(g))
    self_pair = q == me
    keep_self = self_pair & (sign > 0)
    first = (~self_pair) & (me < q)
    rows, cols, vals = [], [], []
    ns = int(keep_self.sum())
    rows.append(me[keep_self]); cols.append(np.arange(ns)); vals.append(np.ones(ns))
    nf = int(first.sum())
    c = ns + np.arange(nf)
    rows += [me[first], q[first]]; cols += [c, c]; vals += [np.full(nf, 1 / np.sqrt(2)), sign[first] / np.sqrt(2)]
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(g), ns + nf))


def absorber_matrix(cav: MechCavity, eta: np.ndarray) -> sp.csr_matrix:
    """Σ_e η_e f_e M_e on the global DOFs at k = 0 (η per kept element; elements with η = 0 skipped)."""
    part = cav.solver.parts[0]
    sel = np.flatnonzero(eta > 0)
    f = cav.fraction.ravel()[part.cell[sel]]
    edof = part.edof[sel]
    nd = edof.shape[1]
    rows = np.repeat(edof, nd, axis=1).ravel()
    cols = np.tile(edof, (1, nd)).ravel()
    vals = ((eta[sel] * f)[:, None, None] * part.Me[None]).ravel()
    ok = (rows >= 0) & (cols >= 0)
    n = cav.solver.ndof
    return sp.csr_matrix((vals[ok], (rows[ok], cols[ok])), shape=(n, n))


def column_absorber(cav: MechCavity, columns: list[int], eta_max: float, power: float = 2.0) -> np.ndarray:
    """η per element: eta_max·t^power over the listed columns, t ramping 0 → 1 away from the cavity centre (column 0)."""
    xs, _ = cav.element_xy()
    eta = np.zeros(len(xs))
    for side in (-1, 1):
        cols = sorted([c for c in columns if np.sign(c) == side], key=abs)
        if not cols:
            continue
        x_in = cav.domain.column_x(cols[0]) - side * cav.domain.pitch / 2
        x_out = cav.domain.column_x(cols[-1]) + side * cav.domain.pitch / 2
        t = (xs - x_in) / (x_out - x_in)
        m = (t >= 0) & (t <= 1)
        eta[m] = np.maximum(eta[m], eta_max * t[m] ** power)
    return eta


def cavity_modes(cav: MechCavity, target_hz: float, n_modes: int = 12, eta: np.ndarray | None = None, tol: float = 1e-9):
    """
    n_modes modes nearest target_hz: complex frequencies (Hz, Im > 0 decaying when an absorber
    is given) sorted by real part, and eigenvectors (ndof, n_modes).
    """
    K, M = cav.solver.matrices(np.zeros(2))
    K, M = K.real.tocsc(), M.real.tocsc()
    P = cav.P
    if P is not None:
        K, M = (P.T @ K @ P).tocsc(), (P.T @ M @ P).tocsc()
    sigma = (2 * np.pi * target_hz / OMEGA_UNIT) ** 2
    if eta is not None and np.any(eta > 0) and P is not None:
        A = (P.T @ absorber_matrix(cav, eta) @ P).tocsc()
    elif eta is not None and np.any(eta > 0):
        A = absorber_matrix(cav, eta).tocsc()
    if eta is None or not np.any(eta > 0):
        solve = factorize_indefinite(K - sigma * M)
        op = spla.LinearOperator(K.shape, matvec=solve, dtype=float)
        lam, X = spla.eigsh(K, k=n_modes, M=M, sigma=sigma, which="LM", OPinv=op, tol=tol)
    else:
        Mc = (M - 1j * A).tocsc()
        solve = factorize_indefinite(K - sigma * Mc)
        op = spla.LinearOperator(K.shape, matvec=lambda v: solve(Mc @ v), dtype=complex)
        lam, X = spla.eigs(op, k=n_modes, which="LM", tol=tol)
        lam = sigma + 1 / lam
    w = np.sqrt(lam.astype(complex)) * OMEGA_UNIT / (2 * np.pi)
    order = np.argsort(w.real)
    X = X[:, order]
    return w[order], (P @ X if P is not None else X)


def column_mass(cav: MechCavity, X: np.ndarray, y_window: float | None = None) -> np.ndarray:
    """∫ρ|u|² per dagger-row column (columns × modes), optionally only elements with |y| < y_window."""
    part = cav.solver.parts[0]
    _, eM = element_forms(part, element_vectors(part, np.zeros(2), X))
    w = cav.fraction.ravel()[part.cell][:, None] * eM
    if y_window is not None:
        _, ys = cav.element_xy()
        w = w * (np.abs(ys) < y_window)[:, None]
    col = cav.element_column() - cav.domain.columns[0]
    return np.array([np.bincount(col, weights=w[:, b], minlength=cav.domain.n_columns) for b in range(X.shape[1])]).T


def participation(column_weight: np.ndarray) -> float:
    """N_eff = Σ w_n / max w_n of a per-column weight (energy or ∫ρ|u|²)."""
    return float(column_weight.sum() / column_weight.max())


# --- envelope (local band edge) model ----------------------------------------------------------


def envelope_modes(edges: np.ndarray, curvature: np.ndarray, edge: str, n_modes: int = 3):
    """
    Tight-binding (full-band k·p) envelope model on the column lattice. Column n carries one
    band f_n(k) = e_n + 2 t_n cos(k a) fitted to its local band edge f_n and curvature
    D_n = |d²f/d(ka)²| there: t_n = D_n / 2, and e_n = f_n − 2 t_n for a band maximum at Γ
    (edge="gamma": mechanics) or e_n = f_n + 2 t_n for a band minimum at X (edge="x": optics).
    H = diag(e_n) + t_{n+½} (n ↔ n+1), t_{n+½} the mean of the neighbours; ψ = 0 past the ends.
    A cell whose band [e − 2t, e + 2t] excludes the frequency is a mirror for it (band-gap
    confinement, which a parabolic model misses when the mirror band edge moves the wrong way).
    Returns all eigenvalues and eigenvectors (columns × modes) sorted by the weight of ψ² in the
    three central columns (most cavity-like first), n_modes of them. An X-point mode alternates
    sign between columns; its envelope is |ψ|.
    """
    edges = np.asarray(edges, float)
    t = np.abs(np.asarray(curvature, float)) / 2
    e = edges - 2 * t if edge == "gamma" else edges + 2 * t
    th = (t[1:] + t[:-1]) / 2
    H = np.diag(e) + np.diag(th, 1) + np.diag(th, -1)
    w, V = np.linalg.eigh(H)
    return w, V


def cavity_like(V: np.ndarray, centre: int, n_modes: int = 3) -> np.ndarray:
    """Indices of the eigenvectors with the most weight in columns centre−1 … centre+1."""
    wgt = (V[max(centre - 1, 0) : centre + 2] ** 2).sum(axis=0)
    return np.argsort(-wgt)[:n_modes]


def _legendre_weights(mode) -> np.ndarray:
    return np.array([2 / (2 * m + 1) for _, m in mode.basis])


def mirror_y_parity(cav: MechCavity, x: np.ndarray) -> float:
    """Re⟨u, σ_y u⟩/⟨u, u⟩ over the nodes, σ_y u = (u_x, −u_y, u_z)(x, −y) (lumped nodal mass)."""
    from .coupling import coefficients_at, mech_mode

    mode = mech_mode(cav.solver, np.zeros(2), 1.0, x, real=False, M=sp.eye(cav.solver.ndof))
    n1, n2 = cav.solver.shape
    V = cav.domain.lattice.vectors
    s1 = (np.arange(n1) / n1 - 0.5)[:, None] * np.ones((1, n2))
    s2 = (np.arange(n2) / n2 - 0.5)[None, :] * np.ones((n1, 1))
    img = coefficients_at(mode, s1 + 2 * s2 * V[1, 0] / V[0, 0], -s2, renormalise=False)
    c = np.moveaxis(mode.coef, -1, 0)
    f = cav.fraction
    wn = sum(np.roll(np.roll(f, p, 0), q, 1) for p in (0, 1) for q in (0, 1)) / 4
    sign = np.array([-1.0 if comp == 1 else 1.0 for comp, _ in mode.basis])[:, None, None]
    lw = _legendre_weights(mode)[:, None, None]
    return float(np.real((wn * lw * sign * np.conj(c) * img).sum()) / (wn * lw * np.abs(c) ** 2).sum())


def bloch_amplitudes(cav: MechCavity, x: np.ndarray, bloch_coef: np.ndarray) -> np.ndarray:
    """
    Signed envelope of a cavity mode on the per-cell Bloch mode: A_n = ⟨b, u|_n⟩ / ⟨b, b⟩ per
    column n, with b the nodal coefficients (n_cell, n2, nb) of a Γ-point Bloch mode on the same
    local grid (coupling.mech_mode(...).coef of a per-cell solver), both normalised to max|u| = 1.
    """
    from .coupling import mech_mode

    mode = mech_mode(cav.solver, np.zeros(2), 1.0, x, M=sp.eye(cav.solver.ndof))
    lw = _legendre_weights(mode)
    nc = cav.n_cell
    out = []
    for n in range(cav.domain.n_columns):
        c = mode.coef[n * nc : (n + 1) * nc]
        out.append(np.real(np.sum(lw * np.conj(bloch_coef) * c)) / np.real(np.sum(lw * np.abs(bloch_coef) ** 2)))
    return np.array(out)
