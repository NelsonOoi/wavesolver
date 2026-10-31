"""
Device-level figure of merit of the piezo-optomechanical transducer and its combined adjoint.

One mechanical eigenmode of one structure (Si cavity, phononic channel, LN transducer; a
`PiezoSolver`, short circuit, Γ point) carries both couplings: g0 to the optical mode (PWE,
Si only, in an optical window of the same grid) and g_em to the microwave resonator (through
the charge the mode induces on the terminal). The figure of merit is the three-mode model of
omc.system (`three_mode_response`): conversion efficiency at the resonance (fom="eta") or its
integral over the signal frequency, ∫η dν/2π = η_peak × equivalent bandwidth (fom="eta_bw").

    η(ν) = κ_ex κ_r,ex |T_ac(ν)|²,  T = M(ν)⁻¹,  M = [[κ/2 − iν, ig, 0], [ig, (γ − Γ₊)/2 − iν, ig_em], [0, ig_em, κ_r/2 − iν]]

with g = g0√n_c, red-detuned pump (Δ = −ω_m), Γ₊ the Stokes rate, κ_i = ω_o/Q_i (+ κ_rad if
asked), κ_ex = κ_i (critical, from the measured Q_i), γ = γ_i + γ_leak. T_ac = (ig)(ig_em)/det M
in closed form. At ν = 0 and Γ₊ → 0 this is system.three_mode_efficiency_resonant,
η_o η_r 4C_oC_e/(1 + C_o + C_e)². The model is written in JAX, so ∂η/∂(g0, g_em, ω_m, ω_o,
γ_leak, κ_rad) are exact; `eta_partials` is checked against system.three_mode_response.

Adapter (task 1). `piezo_mech_mode` turns the Si-layer displacement of a PiezoSolver mode in
the optical window into a coupling.MechMode: the Si layer's Lobatto functions through the
thickness are re-expanded exactly in Legendre polynomials P_d(2z/t) (degree ≤ the layer
order, all three components), on the window's nodes. coupling.g0 and the shared
coupling_adjoint.optomech_functional then work unchanged. The map from the reduced
eigenvector to the MechMode coefficients is a sparse matrix S, so ∂g0/∂y = Sᵀ ∂g0/∂coef. The
PWE window is periodic in y while the mechanics continues past it, so coupling's
interpolation wraps in the window's outermost element row; `check_optical_window` reports the
optical energy there and in LN-covered columns (both must be negligible: the PWE models Si
only).

Combined adjoint (task 2). With ŷ the mode (any scale; all quantities are scale invariant),
λ its eigenvalue (piezo-stiffened Rayleigh quotient), T(λ) the shorted system and
B = diag(M, 0):

    dη = Σ_q η_q dq,  q ∈ {g0, g_em, ω_m, ω_o, γ_leak, κ_rad}
       = [explicit design terms] + c_λ dλ − wᵀ dA(λ) ŷ,
    [[T(λ), Bŷ], [ŷᵀB, 0]] [w; μ] = [η_g0 ∂g0/∂ŷ + η_gem ∂g_em/∂ŷ; 0],

one bordered solve (mech_adjoint.eigenvector_adjoint_solver) for the weighted right-hand side,
dλ = ŷᵀdA(λ)ŷ / ŷᵀBŷ, c_λ = −η_g0 g0/4λ + η_gem g_em/4λ + η_ωm ω_m/2λ. Explicit terms:
g0's (moving-boundary raster, strain fraction weighting, the optical fill through ε⁻¹ and the
optical eigenvector: one MINRES, weighted by η_g0), g_em's terminal row (dα = e_tᵀ dA ŷ), and
the mass in both normalisations. All are element bilinear forms (PiezoSolver._forms), so
the gradient covers every layer's fraction map and the LN angle θ at once.

Constraints (tasks 3-4): f_m and λ_o (values and gradients); mechanical leakage
γ_leak = 2 Im ω of the same mode in a copy of the structure with a PML (complex symmetric:
the left eigenvector is the transpose, dλ = ŷᵀdA ŷ/ŷᵀBŷ, PiezoSolver.eigen_sensitivities);
`leakage_split` separates the leakage into absorber regions (exact first-order split of
Im λ over the lossy elements, e.g. parasitic edges vs an intended mechanical port; values
only, the gradient is that of the total γ_leak); Q_o through
a `QoModel` (callable returning Q and dQ/df_o): `LightConeQ` is a PWE light-cone proxy with
an adjoint gradient, `ExternalQ` the stub for omc.cavity's finite-cavity method (agent D).
Fab rules (Si ≥ 80 nm, LN ≥ 200 nm) are constraints of the parametrisation
(omc.parametrize: bridge widths, Zhou length scale) and are not repeated here.

Units: rates in rad/s, frequencies in Hz, lengths µm, fractions dimensionless.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .coupling import MASS_UNIT, HBAR, MechMode, block_mean
from .coupling_adjoint import block_mean_adjoint, optomech_functional
from .materials import SI_P
from .mech_adjoint import eigenvector_adjoint_solver
from .mech_bloch import OMEGA_UNIT
from .optics_pwe import C_UM_PER_PS, PWESolver, slab_cell
from .piezo import PHI_UNIT, Q_UNIT, PiezoSolver

jax.config.update("jax_enable_x64", True)
TWO_PI = 2 * np.pi
QUANTITIES = ("g0", "g_em", "omega_m", "omega_o", "gamma_leak", "kappa_rad")


# --- system figure of merit (JAX) ----------------------------------------------------------


@dataclass(frozen=True)
class SystemParams:
    """
    Nominal transducer parameters: n_c intracavity photons; measured
    optical Q_i at critical coupling (κ_ex = κ_i = ω_o/Q_i); γ_i/2π = 30 kHz; a single-mode,
    overcoupled microwave resonator of κ_r/2π = 1 MHz (κ_r,ex 0.9, κ_r,i 0.1 MHz), Z_r = 700 Ω.
    leak_in_gamma: γ = γ_i + γ_leak in η; rad_in_kappa: κ_i = ω_o/Q_i + κ_rad in η (off: the
    measured Q_i already holds all optical loss; Q_rad is then a constraint only).
    fom: "eta" (resonant efficiency) or "eta_bw" (∫η dν/2π in Hz = η_peak × equivalent
    bandwidth; n_nu points of a sinh-mapped grid out to ±nu_span × the widest linewidth).
    """

    n_c: float = 10.0
    Q_i: float = 2.4e5
    gamma_i: float = TWO_PI * 30e3
    kappa_r_ex: float = TWO_PI * 0.9e6
    kappa_r_i: float = TWO_PI * 0.1e6
    Z_r: float = 700.0
    leak_in_gamma: bool = True
    rad_in_kappa: bool = False
    fom: str = "eta"
    n_nu: int = 4001
    nu_span: float = 12.0


def _det(nu, g, gem, kappa, gamma_eff, kappa_r):
    a = kappa / 2 - 1j * nu
    d = gamma_eff / 2 - 1j * nu
    f = kappa_r / 2 - 1j * nu
    return a * d * f + a * gem**2 + f * g**2  # det of M with b = ig, e = ig_em


def _fom(v, P: SystemParams):
    g0, gem, wm, wo, gl, krad = v[0], v[1], v[2], v[3], v[4], v[5]
    kappa_meas = wo / P.Q_i
    kappa_i = kappa_meas + (krad if P.rad_in_kappa else 0.0)
    kappa_ex = kappa_meas
    kappa = kappa_i + kappa_ex
    gamma = P.gamma_i + (gl if P.leak_in_gamma else 0.0)
    g2 = g0**2 * P.n_c
    gp = g2 * kappa / (kappa**2 / 4 + 4 * wm**2)  # Stokes rate at Δ = −ω_m
    kr = P.kappa_r_ex + P.kappa_r_i
    num = kappa_ex * P.kappa_r_ex * g2 * gem**2
    if P.fom == "eta":
        return num / jnp.abs(_det(0.0, jnp.sqrt(g2), gem, kappa, gamma - gp, kr)) ** 2
    # ν = w_small sinh(s): resolves the narrowest line (resonator or dressed mechanics) and spans
    # nu_span × the widest; both widths are smooth in the inputs, so the quadrature is too.
    w_big = gamma + 4 * g2 / kappa + kr + 2 * gem
    w_small = 1 / (1 / kr + 1 / (gamma + 4 * g2 / kappa))
    smax = jnp.arcsinh(P.nu_span * w_big / w_small)
    s = jnp.linspace(-1.0, 1.0, P.n_nu) * smax
    nu = w_small * jnp.sinh(s)
    eta = num / jnp.abs(_det(nu, jnp.sqrt(g2), gem, kappa, gamma - gp, kr)) ** 2
    return jnp.trapezoid(eta * w_small * jnp.cosh(s), s) / TWO_PI


_fom_vg = {}


def eta_partials(values: dict, P: SystemParams) -> tuple[float, dict]:
    """FOM and its partials with respect to QUANTITIES (missing ones default to 0)."""
    key = P
    if key not in _fom_vg:
        _fom_vg[key] = jax.jit(jax.value_and_grad(lambda v: _fom(v, P)))
    v = jnp.array([float(values.get(q, 0.0)) for q in QUANTITIES])
    val, g = _fom_vg[key](v)
    return float(val), dict(zip(QUANTITIES, np.asarray(g)))


def system_check(values: dict, P: SystemParams) -> dict:
    """η from omc.system (three_mode_response at ν = 0, and the closed form) for the same values."""
    from .system import Resonator, Transducer, three_mode_efficiency_resonant, three_mode_response

    wo = values["omega_o"]
    lam = TWO_PI * 299792458.0 / wo
    gi = P.gamma_i + (values.get("gamma_leak", 0.0) if P.leak_in_gamma else 0.0)
    p = Transducer(Q_i=P.Q_i, gamma_i=gi, g0=values["g0"], f_m=values["omega_m"] / TWO_PI, wavelength=lam)
    r = Resonator(kappa_ex=P.kappa_r_ex, kappa_i=P.kappa_r_i, g_em=values["g_em"])
    eta0, _ = three_mode_response(p, r, P.n_c, np.array([0.0]))
    return dict(eta_response=float(eta0[0]), eta_closed=float(three_mode_efficiency_resonant(p, r, P.n_c)))


# --- PiezoSolver mode → coupling.MechMode ----------------------------------------------------


def _lobatto_to_legendre(order: int) -> np.ndarray:
    """T (order+1, order+1): Lobatto function q = Σ_d T[d, q] P_d(s) (omc.mech_bloch.lobatto order)."""
    T = np.zeros((order + 1, order + 1))
    T[0, 0], T[1, 0] = 0.5, -0.5
    T[0, order], T[1, order] = 0.5, 0.5
    for m in range(2, order + 1):
        c = 1 / np.sqrt(2 * (2 * m - 1))
        T[m, m - 1] += c
        T[m - 2, m - 1] -= c
    return T


@dataclass
class OpticalWindow:
    """
    The optical (PWE) cell as a block of the PiezoSolver grid: all n1 columns (the PWE is
    periodic along a1 with the same period) and element rows j0 … j0 + n2w − 1, which must be
    centred on the PWE cell's centre. Optics N1 × N2 × nz (n1, n2w integer multiples),
    supersample `sub` fine-raster cells per mechanical element side (moving-boundary raster).
    """

    j0: int
    n2w: int
    n_opt: tuple[int, int]
    nz: int
    height: float
    eps_si: float
    k_opt: np.ndarray
    sub: int = 4
    thickness: float | None = None  # Si thickness; default: the solver's layer 0

    def lattice(self, sol: PiezoSolver):
        from .geometry import Lattice2D

        dy = sol.cell[1] / sol.shape[1]
        return Lattice2D(np.array([sol.cell[0], dy * self.n2w]))

    def fine_window(self, fine: np.ndarray) -> np.ndarray:
        return fine[:, self.j0 * self.sub : (self.j0 + self.n2w) * self.sub]

    def optical_fraction(self, fine: np.ndarray) -> np.ndarray:
        return block_mean(self.fine_window(fine), *self.n_opt)


class PiezoSiMap:
    """Sparse map S: reduced u of a PiezoSolver → Legendre MechMode coefficients of the Si layer in the window."""

    def __init__(self, sol: PiezoSolver, red, win: OpticalWindow):
        n1, n2 = sol.shape
        p = sol.layers[0].order
        F = sol.n_func
        self.order = p
        self.basis = [(c, d) for c in range(3) for d in range(p + 1)]
        nb = len(self.basis)
        T = _lobatto_to_legendre(p)
        i, jw = np.meshgrid(np.arange(n1), np.arange(win.n2w), indexing="ij")
        node = (i * (n2 + 1) + win.j0 + jw).ravel()
        nw = len(node)
        rows, cols, vals = [], [], []
        for c in range(3):
            for d in range(p + 1):
                b = c * (p + 1) + d
                for q in range(p + 1):
                    if T[d, q] != 0:
                        rows.append(np.arange(nw) * nb + b)
                        cols.append(node * 3 * F + 3 * q + c)
                        vals.append(np.full(nw, T[d, q]))
        G = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nw * nb, sol.n_node * 3 * F))
        Pu = red[0]
        self.S = (G[:, sol.udofs] @ Pu).tocsr()
        if np.iscomplexobj(self.S.data) and not np.any(self.S.data.imag):
            self.S.data = self.S.data.real
        self.shape = (n1, win.n2w, nb)
        P0 = sol.parts[0]
        nodes = np.unique(sol._enode[P0.solid])
        base, _ = sol._wrap(nodes, np.zeros(2))
        has = np.zeros(sol.n_node, bool)
        has[base] = True
        has[nodes] = True
        self.has_dof = has[node].reshape(n1, win.n2w)
        self.thickness = sol.layers[0].thickness if win.thickness is None else win.thickness

    def coef(self, y_u: np.ndarray) -> np.ndarray:
        return (self.S @ y_u).reshape(self.shape)


def piezo_mech_mode(smap: PiezoSiMap, y_u: np.ndarray, freq: float, mass: float, lattice, fraction: np.ndarray) -> tuple[MechMode, float]:
    """
    MechMode of the Si layer in the window for the reduced displacement y_u, normalised to
    max|û| = 1 over the window nodes and 9 heights (coupling.mech_mode's convention, applied to
    the window). mass: y_uᴴ M y_u of the whole structure (internal units) for the same y_u.
    Returns the mode and the normalisation max|u|.
    """
    coef = smap.coef(np.real(y_u))
    mode = MechMode(lattice, np.zeros(2), float(freq), coef.astype(complex), smap.has_dof, smap.basis, smap.thickness, 1.0, 0.0, fraction)
    z = np.linspace(-smap.thickness / 2, smap.thickness / 2, 9)
    P = mode._legendre(z)
    comps = np.array([c for c, _ in smap.basis])
    u = np.array([np.tensordot(coef[..., comps == c], P[comps == c], axes=(-1, 0)) for c in range(3)])
    umax = float(np.sqrt((u**2).sum(axis=0)).max())
    mode.coef = mode.coef / umax
    mode.mass = mass / umax**2 * MASS_UNIT
    return mode, umax


def check_optical_window(opt_solver: PWESolver, band: int, ln_columns: np.ndarray | None = None, rows: int = 1) -> dict:
    """
    Optical |H|² fraction in the `rows` outermost voxel rows of the window (where the window's
    periodic wrap differs from the continuing mechanics) and, if ln_columns (N1, N2) bool marks
    in-plane positions with LN above the Si, in those columns. Both should be ≲ 1e-3.
    """
    h = (np.abs(opt_solver.h_field(band)) ** 2).sum(axis=(0, 3))
    out = dict(edge=float((h[:, :rows].sum() + h[:, -rows:].sum()) / h.sum()))
    if ln_columns is not None:
        out["ln"] = float(h[np.asarray(ln_columns, bool)].sum() / h.sum())
    return out


# --- optical Q interface -------------------------------------------------------------------


class QoModel(Protocol):
    """Radiation-limited optical Q of band `band` of a solved PWESolver, and dQ/d(optical fill f_o)."""

    def __call__(self, opt_solver: PWESolver, band: int) -> tuple[float, np.ndarray]: ...


class ExternalQ:
    """
    Stub for the finite-cavity Q_o of omc.cavity (agent D: light-cone Fourier proxy of a cavity
    supercell field, or meep FDTD). Pass a callable fn(opt_solver, band) -> (Q, dQ/df_o).
    """

    def __init__(self, fn: Callable | None = None):
        self.fn = fn

    def __call__(self, opt_solver: PWESolver, band: int) -> tuple[float, np.ndarray]:
        if self.fn is None:
            raise NotImplementedError("finite-cavity Q_o (omc.cavity) is not connected yet")
        return self.fn(opt_solver, band)


def _lightcone_q(xr, xi, f, kvec, absq, u, v, cellA, height, z0, g, dg, rb, eta, es, eb, mask):
    """Q proxy = k0² L_z / (2 Σ_{|k∥|<k0} k_z |h₂D(G∥; z0)|²), x normalised; k0 = √λ (Rayleigh)."""
    from .coupling_adjoint import _apply, _curl, _eps_tensor

    s = 1 / jnp.sqrt(jnp.sum(xr**2 + xi**2))
    hr, hi = xr * s, xi * s
    e = _curl(hr, hi, u, v, absq)
    _, _, w, _, _, _ = _eps_tensor(f, g, dg, rb, eta, es, eb)
    N = e[0].size
    lam = N * jnp.sum(jnp.real(jnp.conj(e) * _apply(w, e)))
    k0 = jnp.sqrt(lam)
    h = (hr + 1j * hi).reshape((2,) + absq.shape)
    H = h[0] * u + h[1] * v  # Cartesian plane-wave amplitudes (3, N1, N2, N3)
    n3 = absq.shape[2]
    Gz = 2 * jnp.pi * jnp.fft.fftfreq(n3, d=height / n3)
    h2 = (H * jnp.exp(1j * Gz * z0)[None, None, None, :]).sum(axis=3)  # (3, N1, N2)
    kpar2 = kvec[0] ** 2 + kvec[1] ** 2  # (N1, N2) |k + G∥|²
    kz = jnp.where(mask > 0, jnp.sqrt(jnp.where(mask > 0, k0**2 - kpar2, 1.0)), 0.0)  # no sqrt(0) in the gradient
    P = jnp.sum(mask * kz * (jnp.abs(h2) ** 2).sum(axis=0))
    return k0**2 * height / (2 * P)


class LightConeQ:
    """
    Light-cone proxy of Q_rad for a PWE (super)cell mode: the plane-wave components of H at a
    plane z0 above the slab with |k + G∥| < k0 radiate with S_z = ½Z₀|h|² k_z/k0 on each side,
    and the stored energy is ½μ₀∫|H|². Meaningful for a finite cavity in a large supercell
    (Γ point); a guided Bloch mode below the light line has no light-cone components (Q = ∞,
    zero gradient). The gradient goes through the optical adjoint (PWESolver.adjoint_solve)
    and the ε⁻¹ map, as g0's optical part. z0: plane height above the slab centre (µm).
    """

    def __init__(self, z0: float = 0.3, tol: float = 1e-10):
        self.z0, self.tol = z0, tol
        self._vg = jax.jit(jax.value_and_grad(_lightcone_q, argnums=(0, 1, 2)))

    def __call__(self, opt_solver: PWESolver, band: int) -> tuple[float, np.ndarray]:
        from .coupling_adjoint import _curl, _optical_inputs, _theta_bilinear_grad

        s = opt_solver
        oi = _optical_inputs(s)
        q = s.G + np.array([s._k[0], s._k[1], 0.0])[:, None, None, None]
        kvec = q[:2, :, :, 0]
        lam = (2 * np.pi * s._last_freqs[band] / C_UM_PER_PS) ** 2
        mask = (kvec[0] ** 2 + kvec[1] ** 2 < lam).astype(float)
        if not mask.any():
            return np.inf, np.zeros(oi["f"].shape)
        x = s.last_vectors[:, band]
        ostate = (jnp.asarray(s.u), jnp.asarray(s.v), jnp.asarray(s.absq))
        cellA = abs(np.linalg.det(s.cell.lattice.vectors))
        Q, (gr, gi, gf) = self._vg(
            jnp.asarray(x.real), jnp.asarray(x.imag), oi["f"], jnp.asarray(kvec), ostate[2], ostate[0], ostate[1], cellA,
            s.cell.height, self.z0, oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"], jnp.asarray(mask),
        )
        r = np.asarray(gr) + 1j * np.asarray(gi)
        w, _ = s.adjoint_solve(band, r, tol=self.tol)
        ew = _curl(jnp.asarray(w.real), jnp.asarray(w.imag), *ostate)
        ex = _curl(jnp.asarray(x.real), jnp.asarray(x.imag), *ostate)
        gt = _theta_bilinear_grad(oi["f"], ew, ex, oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"])
        return float(Q), np.asarray(gf) - np.asarray(gt)


# --- leakage -------------------------------------------------------------------------------


def leakage_split(sol: PiezoSolver, u_full: np.ndarray, phi_full: np.ndarray, freq: complex, regions: dict[str, np.ndarray]) -> dict:
    """
    Split of a lossy mode's decay rate γ = 2 Im ω over absorber regions (bool element masks
    (n1, n2)). From Σ_e ŷᴴA_e(λ)ŷ = 0, with ŷᴴA_eŷ and ŷᴴM_eŷ real on every lossless element,
    Im λ = Σ_lossy Im ŷᴴA_e(Re λ)ŷ / Re ŷᴴMŷ exactly, element by element, so each region's
    share is γ_region = Ω Im λ_region / √(Re λ) (to first order in Im λ / Re λ): e.g. parasitic
    edge absorbers vs the absorber terminating an intended mechanical port. u_full, phi_full:
    one mode as from eigenmodes (φ in V). Assumes absorber elements without a dielectric
    background term (background_eps_r None there). Returns γ (rad/s) per region,
    their sum and 2 Im ω of the eigenvalue, as a check.
    """
    xu = np.asarray(u_full).reshape(-1, 1)
    xp = (np.asarray(phi_full) / PHI_UNIT).reshape(-1, 1)
    lam = (2 * np.pi * freq / OMEGA_UNIT) ** 2
    gf, _ = sol._forms(xu.conj(), xp.conj(), xu, xp, lam.real)
    contrib = np.zeros(sol.shape)
    for l, P in enumerate(sol.parts):
        contrib += np.imag(gf[l, 0] * P.f.reshape(sol.shape))
    norm = np.real(xu[sol.udofs, 0].conj() @ (sol.M @ xu[sol.udofs, 0]))
    # γ = 2 Im ω, ω = Ω√λ: Im ω ≈ Ω Im λ / (2√Re λ)
    w_of = lambda im_lam: OMEGA_UNIT * im_lam / np.sqrt(lam.real)  # noqa: E731
    out = {name: float(w_of(contrib[np.asarray(m, bool)].sum() / norm)) for name, m in regions.items()}
    out["sum"] = float(sum(out.values()))
    out["eigen"] = float(2 * np.imag(2 * np.pi * freq))
    return out


# --- combined evaluation -------------------------------------------------------------------


@dataclass
class Design:
    """Si fine raster (n1·sub, n2·sub) over the whole PiezoSolver grid, fractions of layers 1… (n1, n2), LN angle (deg)."""

    si_fine: np.ndarray
    layers: list[np.ndarray]
    angle: float | None = None


@dataclass
class Evaluation:
    """
    fom (η or ∫η dν/2π) and quantities (g0, g_em in rad/s, f_m, f_o in Hz, γ_leak rad/s),
    partials ∂fom/∂q, gradients on the design (dict: si_fine, layers (list), angle) of the FOM
    and of each constraint value in `constraints`, timings and diagnostics in info.
    """

    fom: float
    q: dict
    partials: dict
    grad: dict
    constraints: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    mode_u: np.ndarray | None = None
    opt_x: np.ndarray | None = None


class DeviceModel:
    """
    The test-device pipeline: a lossless PiezoSolver (η, g0, g_em, f_m), optionally a PML copy
    (γ_leak), and the PWE optics in the window. Design changes go through `set_design`.
    port: terminal of the transducer; P: SystemParams; qo: optional QoModel (constraint).
    """

    def __init__(self, sol: PiezoSolver, win: OpticalWindow, port: str, P: SystemParams = SystemParams(),
                 sol_pml: PiezoSolver | None = None, qo: QoModel | None = None, p_si: np.ndarray = SI_P,
                 targets: dict | None = None, leak_regions: dict | None = None):
        self.sol, self.win, self.port, self.P = sol, win, port, P
        self.sol_pml, self.qo, self.p_si = sol_pml, qo, p_si
        self.targets = dict(f_m=4e9, wavelength=1.55, leak_frac=0.1, q_ratio=10.0) | (targets or {})
        self.leak_regions = leak_regions
        self.opt_ref = None

    # design → solvers
    def set_design(self, d: Design) -> None:
        n1, n2 = self.sol.shape
        fsi = block_mean(d.si_fine, n1, n2)
        for s in [self.sol] + ([self.sol_pml] if self.sol_pml is not None else []):
            s.set_fraction(0, fsi)
            for l, fr in enumerate(d.layers, start=1):
                s.set_fraction(l, fr)
            if d.angle is not None:
                s.set_angle(d.angle)
        self.design = d

    def optics(self, n_bands: int = 6, tol: float = 1e-8, maxiter: int = 600) -> PWESolver:
        w = self.win
        fo = w.optical_fraction(self.design.si_fine)
        t = self.sol.layers[0].thickness if w.thickness is None else w.thickness
        s = PWESolver(slab_cell(w.lattice(self.sol), fo, t, w.height, w.nz, w.eps_si), parity="even")
        if self.opt_ref is not None:
            s._prev = self.opt_ref._prev
        s.solve(w.k_opt, n_bands, tol=tol, maxiter=maxiter)
        return s

    def mechanics(self, n_modes: int, shift_hz: float):
        return self.sol.eigenmodes(n_modes, shift_hz, (0.0, 0.0), "short", vectors=True, _reduced=True)

    def evaluate(self, mode_ref: np.ndarray | float, band_ref: int | np.ndarray, n_modes: int = 12, shift_hz: float = 3.0e9,
                 n_bands: int = 6, opt_tol: float = 1e-8, gradient: bool = True, weights: dict | None = None) -> Evaluation:
        """
        mode_ref: the mechanical mode to follow, as a reference displacement field
        (n1+1, n2+1, F, 3) (largest |overlap|) or a frequency in Hz (nearest); band_ref: the
        optical band index, or a reference eigenvector (largest overlap). weights: override the
        FOM partials (e.g. {"g0": 1} for dg0 alone), for checks.
        """
        info: dict = {}
        t0 = time.perf_counter()
        so = self.optics(n_bands, opt_tol)
        if np.ndim(band_ref) == 0:
            band = int(band_ref)
        else:
            band = int(np.argmax(np.abs(so.last_vectors[:, :n_bands].conj().T @ band_ref)))
        info["t_optics"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        freqs, U, PHI, ctx = self.mechanics(n_modes, shift_hz)
        info["t_mech_eigen"] = time.perf_counter() - t1
        if np.ndim(mode_ref) == 0:
            m = int(np.argmin(np.abs(freqs.real - mode_ref)))
        else:
            ref = np.asarray(mode_ref).ravel()
            ov = [abs(np.vdot(U[j].ravel(), ref)) / np.linalg.norm(U[j]) for j in range(len(freqs))]
            m = int(np.argmax(ov))
        others = np.delete(freqs.real, m)
        if np.abs(others - freqs[m].real).min() < 1e-4 * freqs[m].real:
            warnings.warn("mechanical mode is (near-)degenerate")
        res = self._quantities(so, band, freqs[m].real, ctx, m, gradient, weights, info)
        res.mode_u = U[m]
        res.opt_x = so.last_vectors[:, band].copy()
        info["t_total"] = time.perf_counter() - t0
        info["band"], info["mode"] = band, m
        return res

    # --- the core ---------------------------------------------------------------------------

    def _quantities(self, so, band, freq, ctx, m, gradient, weights, info) -> Evaluation:
        sol, win, P = self.sol, self.win, self.P
        red, unk, Mr = ctx["red"], ctx["unk"], ctx["Mr"]
        Pu, Pp, npf, _ = red
        nur = Pu.shape[1]
        y = np.real(ctx["ys"][m]).copy()  # Γ, lossless: real
        n1, n2 = sol.shape
        lat = win.lattice(sol)
        fine = self.design.si_fine
        fsi = block_mean(fine, n1, n2)
        fsi_w = fsi[:, win.j0 : win.j0 + win.n2w]
        smap = PiezoSiMap(sol, red, win)
        # scale ŷ so that max|u| = 1 in the window (g0 and g_em do not depend on it)
        mode, umax = piezo_mech_mode(smap, y[:nur], freq, float(y[:nur] @ (Mr @ y[:nur])), lat, fsi_w)
        y = y / umax
        yu = y[:nur]
        Myu = Mr @ yu
        yMy = float(yu @ Myu)
        A0, _, _ = sol.reduced(0.0)
        A0 = A0.tocsr().real
        lam = float(y @ (A0 @ y)) / yMy  # piezo-stiffened Rayleigh quotient
        omega_m = OMEGA_UNIT * np.sqrt(lam)
        x_zpf = np.sqrt(HBAR / (2 * MASS_UNIT * yMy * omega_m))
        t = nur + npf + sol.terminals.index(self.port)
        c = np.real(np.asarray(A0[t].todense()).ravel())
        alpha = float(c @ y)
        C_m = alpha**2 / (lam * yMy) * Q_UNIT / PHI_UNIT
        g_em = 0.5 * omega_m * np.sqrt(C_m * omega_m * P.Z_r)
        # optics + g0 core
        t2 = time.perf_counter()
        core = optomech_functional(so, band, mode, win.fine_window(fine), fsi_w, self.p_si, adjoint=gradient)
        g0 = core["L"] * x_zpf
        omega_o = 2 * np.pi * so._last_freqs[band] * 1e12
        info["t_g0_core"] = time.perf_counter() - t2
        info.update({f"g0_{k}": v for k, v in core["info"].items()})
        q = dict(g0=g0, g_em=g_em, omega_m=omega_m, omega_o=omega_o, gamma_leak=0.0, kappa_rad=0.0, f_m=omega_m / TWO_PI,
                 f_o=omega_o / TWO_PI, alpha=alpha, C_m=C_m, x_zpf=x_zpf, m_eff=MASS_UNIT * yMy)
        info["window"] = check_optical_window(so, band)
        # leakage (PML copy)
        leak = None
        if self.sol_pml is not None:
            t3 = time.perf_counter()
            leak = self._leakage(freq, m, ctx, gradient)
            q["gamma_leak"] = leak["gamma"]
            info["t_leak"] = time.perf_counter() - t3
            info["leak_match"] = leak["match"]
            if "split" in leak:
                info["leak_split"] = leak["split"]
        qo_val = None
        if self.qo is not None:
            t4 = time.perf_counter()
            qo_val = self.qo(so, band)
            info["t_qo"] = time.perf_counter() - t4
            if P.rad_in_kappa and np.isfinite(qo_val[0]):
                q["kappa_rad"] = omega_o / qo_val[0]
        fom, partials = eta_partials(q, P)
        if weights is not None:
            partials = {k: float(weights.get(k, 0.0)) for k in QUANTITIES}
        ev = Evaluation(fom, q, partials, {}, info=info)
        tg = self.targets
        cons = dict(f_m=(q["f_m"] - tg["f_m"]) / tg["f_m"], wavelength=(C_UM_PER_PS / (q["f_o"] / 1e12) - tg["wavelength"]) / tg["wavelength"])
        if leak is not None:
            cons["leak"] = q["gamma_leak"] / (tg["leak_frac"] * P.gamma_i) - 1.0
        if qo_val is not None:
            cons["q_o"] = np.log(qo_val[0] / (tg["q_ratio"] * P.Q_i)) if np.isfinite(qo_val[0]) else np.inf
        ev.constraints = cons
        if not gradient:
            return ev

        # ---------------- gradients ----------------
        t5 = time.perf_counter()
        eg0, egem, ewm, ewo = (partials[k] for k in ("g0", "g_em", "omega_m", "omega_o"))
        # right-hand side on the unknowns (u first, then free φ)
        dg0_dy = np.zeros(len(unk))
        dg0_dy[:nur] = x_zpf * (smap.S.T @ core["gcoef"].reshape(-1)) - g0 * Myu / yMy
        By = np.zeros(len(unk))
        By[:nur] = Myu
        yk = y[unk]
        dgem_dy = g_em * (c[unk] / alpha - By / yMy)
        r = eg0 * dg0_dy + egem * dgem_dy
        Al, _, _ = sol.reduced(lam)
        T = Al.tocsr()[unk][:, unk]
        Bm = sp.block_diag([Mr, sp.csr_matrix((len(unk) - nur, len(unk) - nur))], format="csr")
        info["mech_rhs_overlap"] = abs(r @ yk) / (np.linalg.norm(r) * np.linalg.norm(yk) + 1e-300)
        solve = eigenvector_adjoint_solver(T, Bm, 0.0, yk)
        w = np.zeros(len(y))
        w[unk] = np.real(solve(r))
        info["mech_adjoint_residual"] = float(np.linalg.norm(T @ w[unk] - (r - By * (yk @ r) / yMy)) / np.linalg.norm(r))
        et = np.zeros(len(y))
        et[t] = 1.0
        Z = np.stack([w, y, et], axis=1)
        zu, zp = sol._full(Z[:nur], Z[nur:], red, conj=True)
        xu, xp = sol._full(np.repeat(y[:nur, None], 3, 1), np.repeat(y[nur:, None], 3, 1), red)
        gfr, gang = sol._forms(zu, zp, xu, xp, lam)
        mf = sol._mass_forms(xu[:, :1], xu[:, :1])[:, 0].real  # ŷᵀ dM ŷ per layer
        gfr, gang = gfr.real, gang.real
        dlam_f = gfr[:, 1] / yMy  # dλ per layer fraction
        dlam_a = gang[1] / yMy
        c_lam = -eg0 * g0 / (4 * lam) + egem * g_em / (4 * lam) + ewm * omega_m / (2 * lam)
        c_alpha = egem * g_em / alpha
        c_mass = -(eg0 * g0 + egem * g_em) / (2 * yMy)
        d_layers = -gfr[:, 0] + c_lam * dlam_f + c_alpha * gfr[:, 2] + c_mass * mf
        d_angle = -gang[0] + c_lam * dlam_a + c_alpha * gang[2]
        # g0 explicit: strain weighting (Si fraction in the window), moving-boundary raster, optics
        d_layers[0][:, win.j0 : win.j0 + win.n2w] += eg0 * x_zpf * core["gfm"]
        d_fine = np.zeros_like(fine)
        sl = slice(win.j0 * win.sub, (win.j0 + win.n2w) * win.sub)
        fw_shape = (fine.shape[0], win.n2w * win.sub)
        d_fine[:, sl] += eg0 * x_zpf * core["gfine"] + eg0 * x_zpf * block_mean_adjoint(core["d_opt"], fw_shape)
        # ω_o through κ_i = ω_o/Q_i: dω_o/df_o from the optical band sensitivity
        dfo = None
        if ewo != 0 or True:
            dfo = so.sensitivities([band])[0] * 2 * np.pi * 1e12  # rad/s per unit f_o
            d_fine[:, sl] += ewo * block_mean_adjoint(dfo, fw_shape)
        # f_m constraint gradient (per layer, angle)
        dfm_layers = omega_m / (2 * lam) * dlam_f / TWO_PI
        dfm_angle = omega_m / (2 * lam) * dlam_a / TWO_PI
        # leakage into η and its constraint
        if leak is not None:
            el = partials["gamma_leak"]
            d_layers = d_layers + el * leak["d_layers"]
            d_angle = d_angle + el * leak["d_angle"]
        if qo_val is not None and P.rad_in_kappa and np.isfinite(qo_val[0]):
            # κ_rad = ω_o/Q: dκ = dω_o/Q − ω_o dQ/Q²
            ek = partials["kappa_rad"]
            d_fine[:, sl] += ek * block_mean_adjoint(dfo / qo_val[0] - omega_o * qo_val[1] / qo_val[0] ** 2, fw_shape)
        info["t_gradient"] = time.perf_counter() - t5

        def to_design(dl, dfine=None, dang=0.0):
            dsf = block_mean_adjoint(dl[0], fine.shape) + (0 if dfine is None else dfine)
            return dict(si_fine=dsf, layers=[dl[l] for l in range(1, len(dl))], angle=float(dang))

        ev.grad["fom"] = to_design(d_layers, d_fine, d_angle)
        gcons = dict(f_m=to_design(dfm_layers / self.targets["f_m"], None, dfm_angle / self.targets["f_m"]))
        # λ_o = c/f_o: dλ/λ = −df/f
        dwl = np.zeros_like(fine)
        dwl[:, sl] = -block_mean_adjoint(dfo, fw_shape) / omega_o
        gcons["wavelength"] = to_design(np.zeros_like(d_layers), dwl * (C_UM_PER_PS / (q["f_o"] / 1e12)) / self.targets["wavelength"], 0.0)
        if leak is not None:
            s_ = 1 / (self.targets["leak_frac"] * P.gamma_i)
            gcons["leak"] = to_design(s_ * leak["d_layers"], None, s_ * leak["d_angle"])
        if qo_val is not None and np.isfinite(qo_val[0]):
            dq = np.zeros_like(fine)
            dq[:, sl] = block_mean_adjoint(qo_val[1] / qo_val[0], fw_shape)
            gcons["q_o"] = to_design(np.zeros_like(d_layers), dq, 0.0)
        ev.grad["constraints"] = gcons
        ev.grad["quantities"] = dict(d_layers_parts=dict(dlam=dlam_f), dfo=dfo)
        return ev

    def _leakage(self, freq, m, ctx, gradient):
        """γ_leak = 2 Im ω of the PML copy's mode matched to the lossless one (outside the PML), and its gradient."""
        sp_ = self.sol_pml
        es = sp_.eigen_sensitivities(6, complex(freq), (0.0, 0.0), "short") if gradient else None
        if es is None:
            fr, Uq, PHq = sp_.eigenmodes(6, freq + 0j, (0.0, 0.0), "short", vectors=True)
        else:
            fr, Uq, PHq = es.freqs, es.u, es.phi
        u0 = self.sol._expand(np.real(ctx["ys"][m])[: ctx["red"][0].shape[1]], np.real(ctx["ys"][m])[ctx["red"][0].shape[1]:], ctx["red"])[0]
        mask = np.ones(u0.shape[:2], bool)
        if self.leak_regions is not None:
            for reg in self.leak_regions.values():
                rm = np.zeros(u0.shape[:2], bool)
                rm[:-1, :-1] = reg
                rm[1:, :-1] |= reg
                rm[:-1, 1:] |= reg
                rm[1:, 1:] |= reg
                mask &= ~rm
        a = (u0 * mask[:, :, None, None]).ravel()
        ov = [abs(np.vdot((Uq[j] * mask[:, :, None, None]).ravel(), a)) / (np.linalg.norm(Uq[j] * mask[:, :, None, None]) * np.linalg.norm(a)) for j in range(len(fr))]
        j = int(np.argmax(ov))
        out = dict(gamma=float(2 * np.imag(2 * np.pi * fr[j])), match=(float(ov[j]), complex(fr[j])))
        if self.leak_regions is not None:
            out["split"] = leakage_split(sp_, Uq[j], PHq[j], fr[j], self.leak_regions)
        if es is not None:
            out["d_layers"] = 2 * TWO_PI * np.imag(es.dfreq_frac[:, j])
            out["d_angle"] = 2 * TWO_PI * float(np.imag(es.dfreq_angle[j]))
        return out


def flatten(grad: dict) -> np.ndarray:
    """Design gradient dict → one vector (si_fine, layers, angle), the order of `Design` flattening."""
    return np.concatenate([grad["si_fine"].ravel()] + [g.ravel() for g in grad["layers"]] + [np.atleast_1d(grad["angle"])])


__all__ = [
    "Design", "DeviceModel", "Evaluation", "ExternalQ", "LightConeQ", "OpticalWindow", "PiezoSiMap", "QoModel",
    "SystemParams", "check_optical_window", "eta_partials", "flatten", "leakage_split", "piezo_mech_mode", "system_check",
]
