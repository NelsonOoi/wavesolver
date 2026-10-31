"""
Optomechanical vacuum coupling g0 between an optical Bloch mode (wavesolver.optics_pwe, TE-like) and a
mechanical Bloch mode (wavesolver.mech_bloch, "slab_p") on the same patterned slab.

A mechanical mode of amplitude x displaces the material by x·û(r), with û normalised to
max|û| = 1. To first order the optical frequency moves by δω = (∂ω/∂x) x, and

    g0 = (∂ω/∂x) x_zpf,   x_zpf = √(ħ / (2 m_eff Ω_m)),   m_eff = ∫ρ|û|² dV / max|û|².

The pair (m_eff, x_zpf) depends on where |û| is normalised, but g0 does not, since
x_zpf ∝ 1/√m_eff ∝ max|û| and ∂ω/∂x ∝ 1/max|û| (this is COMSOL's solid.disp convention, as in
the user's nanobeam GA models). The sign of g0 follows the (arbitrary) sign of û; only the
relative sign of the two terms below is physical.

Two terms, both first-order perturbations of the PWE eigenvalue λ = (ω/c)², δω/ω = δλ/2λ:

- Moving boundary. The material is advected, F(r) → F(r − u), so the voxel fill changes by
  δF = −u·∇F, and δλ_MB = Σ_voxels (∂λ/∂F) δF with ∂λ/∂F from
  `PWESolver.voxel_sensitivity` (Kottke tensor and normal term included). The interfaces are
  sub-voxel, so δF must stay in the partly filled voxels: −u·∇F is taken on a supersampled
  in-plane raster and block-averaged (`advected_fill`), and the slab faces move exactly. The
  same formula on the optical voxels (`advected_fill_coarse`) is 20-40 % off.
- Photoelastic. Inside Si the impermeability changes by Δ(ε⁻¹) = p : S (Voigt p_IJ S_J,
  engineering shear strain, `materials.SI_P` rotated to the lab frame), i.e.
  δε = −ε (p:S) ε. With E = ε⁻¹ e, e the periodic part of ∇×H (`PWESolver.curl_h`),
  δλ_PE / λ = −Σ E*·δε E / Σ E*·ε E = Σ F ε_Si² E*·(p:S)E / Σ e*·ε⁻¹ e, weighted by the Si
  fill F of each voxel.

The mechanical displacement of "slab_p" is Q1 in-plane on the lattice-aligned grid and a sum
of Legendre polynomials P_m(2z/t) through the thickness; it is interpolated onto points of
the optical cell (in-plane: sub-cells of the lattice-aligned grid of geometry.solid_fraction;
z: the middle of the covered part of slab_cell's voxels), so the two grids need not
coincide (both lattice-aligned on the same cell; for the strain the mechanical grid must
refine the optical one by an integer). Strain uses the in-plane Q1 gradients and the
z-derivative of the Legendre series at the element centroids, weighted by the element
fraction (the Si strain of a partly filled element), averaged onto each voxel
(`strain_on_voxels`).

Symmetry. The lattice-aligned (skewed) grids keep the C2 rotation of the GPT cell exactly but
not its mirrors σ_x, σ_y, so a symmetric optical mode couples weakly to (σ_y, σ_x) = (−, −)
mechanical modes; that leakage is the discretisation error estimate of g0.

Phase matching. Both fields are Bloch modes of one (super)cell. A band-edge optical mode at
k_o (e.g. X) has |E|² with the lattice period, so only mechanical modes with k_m ≡ 0 (Γ)
give a non-zero per-cell overlap; `g0` requires that. The numbers are per (super)cell: both
modes are normalised over one cell. For a cavity of N cells with slowly varying envelopes
ψ_o (optical, on |E|²) and ψ_m (mechanical), g0 ≈ g0_cell · Σ|ψ_o|²ψ_m / (Σ|ψ_o|² · max|ψ_m|)
/ √(Σψ_m² / max ψ_m²): for uniform envelopes over N cells, g0_cell / √N (the mass grows
as N, the fractional frequency shift per unit amplitude does not change).

Differentiability (stage 2). Everything is explicit in the in-plane fraction f through the
cell (F = f g, ∇F by central differences, ε⁻¹ by Kottke), the optical eigenpair (∂λ/∂F, E),
the mechanical eigenpair (û, Ω_m, m_eff) and the interpolation (linear in the mode). The
adjoint will need ∂(∂λ/∂F)/∂f (a second-order optical term) and the mechanical eigenvector
derivative; neither is implemented.

Units: µm, displacement amplitude per µm, frequencies returned in rad/s (and Hz as g/2π).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre

from .materials import SI_P
from .mech_bloch import BlochSolver
from .optics_pwe import C_UM_PER_PS, PWESolver, SlabCell

HBAR = 1.054571817e-34  # J s
MASS_UNIT = 1e-15  # kg: density 1e3 kg/m³ times µm³ (wavesolver.mech_bloch units)


# --- mechanical mode as a field ------------------------------------------------------------


@dataclass
class MechMode:
    """
    One "slab_p" Bloch mode as nodal coefficients, normalised to max|û| = 1.

    coef: (n1, n2, nb) coefficients of the thickness basis at node (i, j), which sits at
    centred lattice coordinates (i/n1 − ½, j/n2 − ½); zero at nodes with no DOF. has_dof:
    (n1, n2). basis: (component, Legendre degree) of each coefficient. mass: m_eff in kg.
    realness: ‖Im û‖/‖û‖ after the global phase was chosen to make û as real as possible.
    fraction: the solver's element fractions (n1, n2), if given, for `strain_on_voxels`.
    """

    lattice: object
    k: np.ndarray
    freq: float
    coef: np.ndarray
    has_dof: np.ndarray
    basis: list[tuple[int, int]]
    thickness: float
    mass: float
    realness: float
    fraction: np.ndarray | None = None

    @property
    def omega(self) -> float:
        return 2 * np.pi * self.freq

    @property
    def x_zpf(self) -> float:
        """Zero-point amplitude √(ħ / 2 m_eff Ω) in m."""
        return np.sqrt(HBAR / (2 * self.mass * self.omega))

    def _legendre(self, z: np.ndarray, derivative: bool = False) -> np.ndarray:
        """P_m(2z/t) (or its z-derivative) of each basis function at z (clipped to the slab)."""
        s = np.clip(2 * np.asarray(z, dtype=float) / self.thickness, -1, 1)
        out = []
        for _, m in self.basis:
            c = np.eye(m + 1)[m]
            out.append(legendre.legval(s, legendre.legder(c)) * 2 / self.thickness if derivative else legendre.legval(s, c))
        return np.array(out)


def mech_mode(
    solver: BlochSolver,
    k: np.ndarray,
    freq: float,
    x: np.ndarray,
    real: bool = True,
    nz_max: int = 9,
    M=None,
    fraction: np.ndarray | None = None,
) -> MechMode:
    """
    Wrap eigenvector x (solver.ndof) of a "slab_p" solver at k, frequency freq (Hz).

    The global phase is chosen so that Σ x² is real and positive (û real for a
    non-degenerate standing mode at Γ or X); with real=True the imaginary remainder is dropped,
    giving the physical standing-wave displacement. max|û| is taken over the nodes and nz_max
    points through the thickness (faces included); m_eff = xᴴ M x / max|û|² (M: M(k), assembled
    if not given). fraction: the element fractions the solver was built from (needed for the
    photoelastic term, see `strain_on_voxels`).
    """
    if solver.model != "slab_p":
        raise ValueError("mech_mode needs a slab_p solver")
    x = np.asarray(x, dtype=complex)
    x = x * np.exp(-0.5j * np.angle(np.sum(x * x)))
    realness = np.linalg.norm(x.imag) / np.linalg.norm(x)
    if real:
        x = x.real.astype(complex)
    n1, n2 = solver.shape
    dpn = len(solver.basis)
    full = np.zeros(n1 * n2 * dpn, dtype=complex)
    full[solver.dofs] = x
    has = np.zeros(n1 * n2 * dpn, dtype=bool)
    has[solver.dofs] = True
    coef = full.reshape(n1, n2, dpn)
    has_dof = has.reshape(n1, n2, dpn).any(axis=2)
    mode = MechMode(
        solver.lattice, np.asarray(k, float), float(freq), coef, has_dof, solver.basis, solver.thickness, 1.0, realness, fraction
    )
    z = np.linspace(-solver.thickness / 2, solver.thickness / 2, nz_max)
    u = _combine(mode, coef.reshape(-1, dpn).T, mode._legendre(z))  # (3, nodes, nz)
    umax = np.sqrt((np.abs(u) ** 2).sum(axis=0)).max()
    if M is None:
        _, M = solver.matrices(k)
    mode.coef = coef / umax
    mode.mass = float(np.real(np.vdot(x, M @ x))) / umax**2 * MASS_UNIT
    return mode


def _combine(mode: MechMode, c: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Components u_comp = Σ_b [comp_b = comp] c_b P_b. c: (nb, *pts); P: (nb, *z) → (3, *pts, *z)."""
    comps = np.array([cb for cb, _ in mode.basis])
    return np.array([np.tensordot(c[comps == comp], P[comps == comp], axes=(0, 0)) for comp in range(3)])


def _corners(mode: MechMode, s1: np.ndarray, s2: np.ndarray):
    """Bilinear corners of points at centred lattice coordinates: node indices, weights, Bloch phases, local coords."""
    n1, n2 = mode.coef.shape[:2]
    t1 = (np.asarray(s1, float) + 0.5) * n1
    t2 = (np.asarray(s2, float) + 0.5) * n2
    i0, j0 = np.floor(t1).astype(int), np.floor(t2).astype(int)
    a, b = t1 - i0, t2 - j0
    theta = mode.lattice.vectors @ mode.k
    out = []
    for d1 in (0, 1):
        for d2 in (0, 1):
            i, j = i0 + d1, j0 + d2
            w = (a if d1 else 1 - a) * (b if d2 else 1 - b)
            phase = np.exp(1j * (np.floor_divide(i, n1) * theta[0] + np.floor_divide(j, n2) * theta[1]))
            out.append((i % n1, j % n2, d1, d2, w, phase))
    return out, a, b


def coefficients_at(mode: MechMode, s1: np.ndarray, s2: np.ndarray, renormalise: bool = True) -> np.ndarray:
    """
    Thickness-basis coefficients (nb, *shape) at in-plane points (centred lattice coordinates),
    by Q1 interpolation with the Bloch phases of the wrapped nodes. With renormalise, corners
    without DOFs (inside holes) are left out and the weights rescaled, so points just outside
    the solid get the displacement of the nearby boundary rather than a fraction of it.
    """
    corners, _, _ = _corners(mode, s1, s2)
    c = 0
    wsum = 0
    for i, j, _, _, w, phase in corners:
        m = mode.has_dof[i, j] if renormalise else 1.0
        c = c + (w * m * phase)[..., None] * mode.coef[i, j]
        wsum = wsum + w * m
    if renormalise:
        c = c / np.where(wsum > 0, wsum, 1)[..., None]
    return np.moveaxis(c, -1, 0)


def displacement(mode: MechMode, s1: np.ndarray, s2: np.ndarray, z: np.ndarray, renormalise: bool = True) -> np.ndarray:
    """û (3, *inplane_shape, *z_shape) at in-plane points (centred lattice coordinates) and heights z (µm)."""
    return _combine(mode, coefficients_at(mode, s1, s2, renormalise), mode._legendre(z))


def strain(mode: MechMode, s1: np.ndarray, s2: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Engineering strain (6, *inplane_shape, *z_shape) of û in Voigt order (xx, yy, zz, yz, xz,
    xy), from the Q1 gradients of the element containing each point (plain interpolation:
    every node of a kept element carries DOFs) and the z-derivative of the Legendre series.
    """
    n1, n2 = mode.coef.shape[:2]
    corners, a, b = _corners(mode, s1, s2)
    c = d1c = d2c = 0
    for i, j, d1, d2, w, phase in corners:
        v = phase[..., None] * mode.coef[i, j]
        c = c + w[..., None] * v
        d1c = d1c + ((b if d2 else 1 - b) * (1 if d1 else -1) * n1)[..., None] * v  # ∂/∂s1
        d2c = d2c + ((a if d1 else 1 - a) * (1 if d2 else -1) * n2)[..., None] * v  # ∂/∂s2
    rb = mode.lattice.reciprocal / (2 * np.pi)  # ∇ = Σ_i b_i ∂/∂s_i
    gx = rb[0, 0] * d1c + rb[1, 0] * d2c
    gy = rb[0, 1] * d1c + rb[1, 1] * d2c
    P, dP = mode._legendre(z), mode._legendre(z, derivative=True)
    mv = lambda q: np.moveaxis(q, -1, 0)
    u_z = _combine(mode, mv(c), dP)  # ∂u/∂z
    u_x = _combine(mode, mv(gx), P)
    u_y = _combine(mode, mv(gy), P)
    return np.array([u_x[0], u_y[1], u_z[2], u_z[1] + u_y[2], u_z[0] + u_x[2], u_y[0] + u_x[1]])


# --- optical side --------------------------------------------------------------------------


def voxel_points(cell: SlabCell) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Centred lattice coordinates s1 (N1, 1), s2 (1, N2) of the voxel columns (the fraction
    cells of geometry.solid_fraction) and the voxel heights z (N3,) of slab_cell.
    """
    n1, n2, nz = cell.eps.shape
    s1 = ((np.arange(n1) + 0.5) / n1 - 0.5)[:, None]
    s2 = ((np.arange(n2) + 0.5) / n2 - 0.5)[None, :]
    dz = cell.height / nz
    z = np.arange(nz) * dz
    return s1, s2, np.where(z > cell.height / 2, z - cell.height, z)


@dataclass
class OpticalMode:
    """One band of the last PWESolver solve: λ = (ω/c)², e = ∇×H (periodic part), E = ε⁻¹e, ∂λ/∂F."""

    solver: PWESolver
    band: int
    freq_thz: float
    lam: float
    e: np.ndarray
    E: np.ndarray
    dlam_dF: np.ndarray

    @property
    def omega(self) -> float:
        return 2 * np.pi * self.freq_thz * 1e12


def optical_mode(solver: PWESolver, band: int) -> OpticalMode:
    f = float(solver._last_freqs[band])
    e = solver.curl_h(band)
    w = solver.cell.eps_inv
    E = np.array(
        [
            w[0] * e[0] + w[5] * e[1] + w[4] * e[2],
            w[5] * e[0] + w[1] * e[1] + w[3] * e[2],
            w[4] * e[0] + w[3] * e[1] + w[2] * e[2],
        ]
    )
    lam = (2 * np.pi * f / C_UM_PER_PS) ** 2
    return OpticalMode(solver, band, f, lam, e, E, solver.voxel_sensitivity(band))


@dataclass
class _CellOnly:
    """Stand-in for a PWESolver when the optical field comes from elsewhere (only .cell is used)."""

    cell: SlabCell


def field_voxel_sensitivity(cell: SlabCell, e: np.ndarray) -> np.ndarray:
    """
    (∂λ/∂F)/λ (N1, N2, N3) for a field e ∝ D = ε E (3, N1, N2, N3) on the cell's voxels, of any
    normalisation: `PWESolver.voxel_sensitivity` with N·(…) replaced by (…)/Σ e*·ε⁻¹ e, so a PWE
    eigenvector gives exactly voxel_sensitivity / λ. For fields from another solver (e.g. meep
    DFT fields interpolated onto the voxels) this is the first-order shift of the Rayleigh
    quotient λ = Σ e*·ε⁻¹ e / Σ |h|²-type normalisation, which is stationary for an eigenmode.
    """
    n1, n2, nz = cell.eps.shape
    d_eps = cell.eps_slab - cell.eps_background
    d_inv = 1 / cell.eps_slab - 1 / cell.eps_background
    a = -d_eps / cell.eps**2
    b = d_inv - a
    aniso = 1 / cell.eps_background + cell.fill * d_inv - 1 / cell.eps
    dz = cell.height / nz
    rb = cell.lattice.reciprocal / (2 * np.pi)
    w = cell.eps_inv
    den = (
        w[0] * np.abs(e[0]) ** 2 + w[1] * np.abs(e[1]) ** 2 + w[2] * np.abs(e[2]) ** 2
        + 2 * w[3] * np.real(np.conj(e[1]) * e[2]) + 2 * w[4] * np.real(np.conj(e[0]) * e[2]) + 2 * w[5] * np.real(np.conj(e[0]) * e[1])
    ).sum()
    ne = (cell.normals * e).sum(axis=0)
    out = a * (np.abs(e) ** 2).sum(axis=0) + b * np.abs(ne) ** 2
    r = np.real(np.conj(e) * ne)
    r -= cell.normals * (cell.normals * r).sum(axis=0)
    s = 2 * aniso / cell.grad_norm * r
    ca = lambda v, axis, scale: -(np.roll(v, -1, axis) - np.roll(v, 1, axis)) * scale / 2
    out += ca(s[0] * rb[0, 0] + s[1] * rb[0, 1], 0, n1)
    out += ca(s[0] * rb[1, 0] + s[1] * rb[1, 1], 1, n2)
    out += ca(s[2], 2, 1 / dz)
    return out / den


def optical_mode_from_field(cell: SlabCell, e: np.ndarray, freq_thz: float, E: np.ndarray | None = None) -> OpticalMode:
    """
    An OpticalMode (for `moving_boundary_shift`, `photoelastic_shift`, `g0`) from a field e ∝ D
    on the voxels of `cell` (e.g. a meep cavity mode), with λ normalised to 1 and ∂λ/∂F from
    `field_voxel_sensitivity`. E defaults to ε⁻¹ e with the cell's Kottke tensor.
    """
    e = np.asarray(e, dtype=complex)
    if E is None:
        w = cell.eps_inv
        E = np.array(
            [
                w[0] * e[0] + w[5] * e[1] + w[4] * e[2],
                w[5] * e[0] + w[1] * e[1] + w[3] * e[2],
                w[4] * e[0] + w[3] * e[1] + w[2] * e[2],
            ]
        )
    return OpticalMode(_CellOnly(cell), -1, float(freq_thz), 1.0, e, E, field_voxel_sensitivity(cell, e))


def fill_gradient(cell: SlabCell) -> np.ndarray:
    """∇F (3, N1, N2, N3), Cartesian, 1/µm: slab_cell's central differences (= n·√(|∇F|² + η²))."""
    return cell.normals * cell.grad_norm


def advected_fill_coarse(cell: SlabCell, u: np.ndarray) -> np.ndarray:
    """
    δF = −u·∇F on the optical voxels themselves (u: (3, N1, N2, N3), µm). Reference only: the
    central differences spread the change of a partly filled voxel into its full and empty
    neighbours, which is 20-40 % off on a 24-voxel blade cell (validation/g0_validate.py).
    """
    return -(np.real(u) * fill_gradient(cell)).sum(axis=0)


def block_mean(a: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """Mean of (n1·m1, n2·m2, ...) over m1 × m2 blocks → (n1, n2, ...)."""
    m1, m2 = a.shape[0] // n1, a.shape[1] // n2
    return a.reshape(n1, m1, n2, m2, *a.shape[2:]).mean(axis=(1, 3))


def advected_fill(cell: SlabCell, fine_fraction: np.ndarray, u_fn) -> np.ndarray:
    """
    First-order change δF (N1, N2, N3) of the optical voxel fill when the slab is displaced by
    u_fn(s1, s2, z) → (3, *s.shape, len(z)) (centred lattice coordinates, µm; e.g.
    `displacement` of a MechMode).

    In-plane: δf = −u·∇f on a raster m1 × m2 times finer than the optical grid (fine_fraction,
    (N1·m1, N2·m2) from geometry.solid_fraction; the cell's fraction should be its block mean),
    with ∇f by central differences there, then averaged back onto the optical voxels; u_∥ is
    taken at the middle of each voxel's covered height. This converges as 1/m to the exact
    boundary integral ∮(u·n) (0.6 % at m = 8, 0.2 % at m = 16 on the blade-cell dilation).
    Faces: the slab faces at ±t/2 move by u_z there, so the voxel holding each face gains
    ⟨f u_z(±t/2)⟩ / dz (sign: outward), exactly as rebuilding slab_cell with the moved face.
    Edges where sidewalls meet faces are second order. Linear in u; explicit in the fill.
    """
    n1, n2, nz = cell.eps.shape
    M1, M2 = fine_fraction.shape
    dz = cell.height / nz
    g = cell.z_cover
    t = g.sum() * dz
    _, _, z = voxel_points(cell)
    rb = cell.lattice.reciprocal / (2 * np.pi)
    d1 = (np.roll(fine_fraction, -1, 0) - np.roll(fine_fraction, 1, 0)) * M1 / 2
    d2 = (np.roll(fine_fraction, -1, 1) - np.roll(fine_fraction, 1, 1)) * M2 / 2
    gx, gy = d1 * rb[0, 0] + d2 * rb[1, 0], d1 * rb[0, 1] + d2 * rb[1, 1]
    s1 = ((np.arange(M1) + 0.5) / M1 - 0.5)[:, None] * np.ones((1, M2))
    s2 = ((np.arange(M2) + 0.5) / M2 - 0.5)[None, :] * np.ones((M1, 1))
    dF = np.zeros((n1, n2, nz))
    sel = np.flatnonzero(g > 0)
    zc = (np.maximum(z[sel] - dz / 2, -t / 2) + np.minimum(z[sel] + dz / 2, t / 2)) / 2
    u = np.real(u_fn(s1, s2, zc))  # (3, M1, M2, nsel)
    dF[:, :, sel] = block_mean(-(u[0] * gx[..., None] + u[1] * gy[..., None]), n1, n2) * g[sel]
    for sign in (1, -1):
        face = sign * t / 2
        j = np.flatnonzero((z - dz / 2 < face) & (face <= z + dz / 2))
        uz = np.real(u_fn(s1, s2, np.array([face])))[2, ..., 0]
        dF[:, :, j] += sign * block_mean(fine_fraction * uz, n1, n2)[..., None] / dz
    return dF


def moving_boundary_shift(opt: OpticalMode, dF: np.ndarray) -> float:
    """δω/ω = Σ (∂λ/∂F) δF / 2λ for a voxel fill change δF (N1, N2, N3), e.g. from `advected_fill`."""
    return float((opt.dlam_dF * dF).sum() / (2 * opt.lam))


def photoelastic_shift(opt: OpticalMode, S: np.ndarray, p: np.ndarray = SI_P) -> float:
    """
    δω/ω for an engineering strain field S (6, N1, N2, N3) in the Si (lab-frame p, Voigt 6 × 6):
    Δ(ε⁻¹) = p S inside Si, weighted by the Si fill F of each voxel.
    """
    cell = opt.solver.cell
    d = np.tensordot(p, np.real(S), axes=(1, 0))  # Δη Voigt (6, ...), tensor components as TENSOR
    E = opt.E
    Ec = np.conj(E)
    q = (
        d[0] * np.abs(E[0]) ** 2
        + d[1] * np.abs(E[1]) ** 2
        + d[2] * np.abs(E[2]) ** 2
        + 2 * d[3] * np.real(Ec[1] * E[2])
        + 2 * d[4] * np.real(Ec[0] * E[2])
        + 2 * d[5] * np.real(Ec[0] * E[1])
    )
    num = (cell.fill * cell.eps_slab**2 * q).sum()
    w = cell.eps_inv
    e = opt.e
    den = (
        w[0] * np.abs(e[0]) ** 2
        + w[1] * np.abs(e[1]) ** 2
        + w[2] * np.abs(e[2]) ** 2
        + 2 * w[3] * np.real(np.conj(e[1]) * e[2])
        + 2 * w[4] * np.real(np.conj(e[0]) * e[2])
        + 2 * w[5] * np.real(np.conj(e[0]) * e[1])
    ).sum()
    return float(num / den / 2)


@dataclass
class Coupling:
    """g0 per (super)cell, rad/s, split into moving-boundary and photoelastic parts; dω/dx in rad/s per m."""

    g_mb: float
    g_pe: float
    dw_dx_mb: float
    dw_dx_pe: float
    m_eff: float
    x_zpf: float

    @property
    def g0(self) -> float:
        return self.g_mb + self.g_pe


def strain_on_voxels(mode: MechMode, cell: SlabCell, m: int | None = None, to_mode=None) -> np.ndarray:
    """
    to_mode: optional map (s1, s2) → (s1, s2) from the optical cell's centred lattice
    coordinates to the mode's, for an optical window that is a block of the mechanical grid
    (then m must be given).
    Solid-phase engineering strain (6, N1, N2, N3) of û on the optical voxels, at the middle of
    each voxel's covered height (zero outside the slab), averaged over m × m in-plane
    sub-points per voxel. By default m = n_mech / n_opt (the mechanical grid must then refine
    the optical one), so the sub-points are the element centroids, where Q1 gradients are
    superconvergent; any sub-point set symmetric within the elements gives the same mean.

    A partly filled element (fraction f, stiffness and mass both scaled by f) is a soft
    homogenised material whose strain ε_h is not that of the Si sliver it stands for: near-empty
    elements flap. The Si strain is estimated from the homogenised stress, σ_h / C = f ε_h, so
    the strain is weighted by the element fraction (mode.fraction, required). Without this the
    boundary layer dominates the photoelastic integral (sign flip on the GPT breathing mode, and
    5-10× more leakage into symmetry-forbidden modes); f ε_h agrees with dropping the elements
    with f < 0.9 to 1 % (research/g0_line_defect.py notes).
    """
    if mode.fraction is None:
        raise ValueError("strain_on_voxels needs mech_mode(..., fraction=...)")
    if m is None:
        if to_mode is not None:
            raise ValueError("strain_on_voxels: give m with to_mode")
        m = mode.coef.shape[0] // cell.eps.shape[0]
        if m * cell.eps.shape[0] != mode.coef.shape[0] or m * cell.eps.shape[1] != mode.coef.shape[1]:
            raise ValueError("default sub-points need the mechanical grid to be an integer refinement of the optical grid")
    n1, n2, nz = cell.eps.shape
    dz = cell.height / nz
    g = cell.z_cover
    t = g.sum() * dz
    _, _, z = voxel_points(cell)
    s1 = ((np.arange(n1 * m) + 0.5) / (n1 * m) - 0.5)[:, None] * np.ones((1, n2 * m))
    s2 = ((np.arange(n2 * m) + 0.5) / (n2 * m) - 0.5)[None, :] * np.ones((n1 * m, 1))
    if to_mode is not None:
        s1, s2 = to_mode(s1, s2)
    sel = np.flatnonzero(g > 0)
    zc = (np.maximum(z[sel] - dz / 2, -t / 2) + np.minimum(z[sel] + dz / 2, t / 2)) / 2
    S = np.zeros((6, n1, n2, nz))
    mn1, mn2 = mode.fraction.shape
    f_el = mode.fraction[np.floor((s1 + 0.5) * mn1).astype(int) % mn1, np.floor((s2 + 0.5) * mn2).astype(int) % mn2]
    S_solid = np.real(strain(mode, s1, s2, zc)) * f_el[None, ..., None]
    S[..., sel] = np.moveaxis(block_mean(np.moveaxis(S_solid, 0, -1), n1, n2), -1, 0)
    return S


def g0(opt: OpticalMode, mode: MechMode, fine_fraction: np.ndarray, p: np.ndarray = SI_P, strain_supersample: int | None = None) -> Coupling:
    """
    Per-cell vacuum coupling between an optical mode and a mechanical mode at Γ on the same
    cell. fine_fraction: the supersampled raster for `advected_fill`; p: lab-frame Si
    photoelastic tensor (Voigt); strain_supersample: sub-points per voxel side for the strain
    (default: the mechanical element centroids).
    """
    theta = mode.lattice.vectors @ mode.k
    if np.abs(np.angle(np.exp(1j * theta))).max() > 1e-9:
        raise ValueError("phase matching: the mechanical mode must be at Γ (k_m ≡ 0) for a per-cell overlap")
    cell = opt.solver.cell
    dF = advected_fill(cell, fine_fraction, lambda a, b, z: displacement(mode, a, b, z))
    rel_mb = moving_boundary_shift(opt, dF)
    rel_pe = photoelastic_shift(opt, strain_on_voxels(mode, cell, strain_supersample), p)
    dw_mb, dw_pe = opt.omega * rel_mb * 1e6, opt.omega * rel_pe * 1e6  # per µm → per m
    xz = mode.x_zpf
    return Coupling(dw_mb * xz, dw_pe * xz, dw_mb, dw_pe, mode.mass, xz)


__all__ = [
    "Coupling",
    "advected_fill",
    "advected_fill_coarse",
    "block_mean",
    "MechMode",
    "OpticalMode",
    "coefficients_at",
    "displacement",
    "field_voxel_sensitivity",
    "fill_gradient",
    "g0",
    "mech_mode",
    "strain_on_voxels",
    "moving_boundary_shift",
    "optical_mode",
    "optical_mode_from_field",
    "photoelastic_shift",
    "strain",
    "voxel_points",
]
