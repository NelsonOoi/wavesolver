"""
Adjoint gradient of the optomechanical vacuum coupling g0 (wavesolver.coupling) with respect to the
solid-fraction maps that build the geometry.

g0 as wavesolver.coupling computes it depends on three in-plane rasters, treated here as independent
inputs (they are tied together only by how the caller builds them):

- f_o (N1, N2): the optical fill the PWESolver's cell was built from (slab_cell: Kottke ε⁻¹
  with regularised normals, the eigenpair λ, x, and the photoelastic fill weight);
- f_m (n1, n2): the mechanical element fractions (BlochSolver K, M; the eigenpair Ω_m, û;
  and the fraction weighting of the strain in `strain_on_voxels`);
- f_fine (N1·m1, N2·m2): the supersampled raster that transports the moving boundary
  (`advected_fill`: −u·∇f by central differences, plus the faces' f·u_z).

`g0_gradient` returns ∂g0/∂f_o, ∂g0/∂f_m and ∂g0/∂f_fine at a solved pair of modes.
`fine_total_gradient` chains them onto f_fine when, as coupling.py intends, f_o and f_m are
block means of f_fine (f_o = block_mean(f_fine, N1, N2), f_m = block_mean(f_fine, n1, n2)); a
shape parameter θ then follows from the raster's Jacobian (wavesolver.parametrize).

Method. Write g0 = L(x, û; f) · x_zpf with L = ω_o (δω/ω per unit amplitude) the optical
frequency pull per metre of mechanical amplitude, linear in û, and x_zpf = √(ħ / 2 m Ω_m),
m = ûᴴMû (û in coupling.py's normalisation). Then:

1. Normalisation and max|û|. g0 is homogeneous of degree 0 in û: L ∝ |û| and x_zpf ∝ 1/|û|.
   So the max|û| = 1 normalisation (COMSOL's convention, m_eff = ∫ρ|û|²/max|û|²) cancels
   exactly and g0 is smooth even though max is not; no surrogate is needed for g0. The
   gradient of m_eff itself (reported as d_meff, and x_zpf) is taken at a fixed reference
   point: the node and height where |û| peaks at the current design. That is exact as long
   as the argmax does not jump (piecewise smooth; FD-checked away from switches).
2. Optics. L depends on the unit eigenvector x through e = ∇×H (ε⁻¹ e ∝ E), on λ (taken as the
   Rayleigh quotient of x, so its f_o-dependence is Hellmann–Feynman), and explicitly on f_o
   through ε⁻¹ (Kottke tensor and normals), the voxel sensitivity ∂λ/∂F (a second-order
   optical term) and the photoelastic fill weight. The whole optical functional is written
   in JAX (a re-implementation of slab_cell's ε⁻¹, voxel_sensitivity and photoelastic_shift,
   checked against the numpy versions at every call), so ∂L/∂x and ∂L/∂f_o|explicit come by
   reverse mode. L is invariant to the phase and scale of x, so rᴴx = 0 for r = ∂L/∂x, and
   the adjoint is (Θ − λ) w = r, w ⊥ x (PWESolver.adjoint_solve: MINRES with the deflated
   MPB preconditioner); the eigenvector term is −Re(wᴴ ∂Θ/∂f_o x) = −∂/∂f_o [N Re Σ e_w* ε⁻¹ e_x],
   again by JAX through the same ε⁻¹ map.
3. Mechanics (Γ only, real K, M). The fields that L sees, δF (moving boundary, linear in û,
   explicit in f_fine) and the Si strain S (linear in û, explicit in f_m through the
   fraction weighting), are re-implemented in JAX at Γ and pulled back by one VJP, giving
   ∂L/∂û, ∂L/∂f_fine and ∂L/∂f_m|strain. With a = ∂g0/∂û (⊥ û, as g0 is scale invariant) the
   adjoint z solves (K − λM) z = a, ûᴴMz = 0 (mech_adjoint.eigenvector_adjoint_solver:
   bordered system, SuperLU), and per element
       ∂g0/∂f_e = x_zpf ∂L/∂f_e|strain − z_eᵀ(K_e − λM_e)û_e
                  − g0/(4λ) ∂λ/∂f_e − (g0/2) û_eᵀM_eû_e / ûᵀMû,
   the last two from x_zpf ∝ (m Ω_m)^(-1/2) with Ω_m ∝ √λ and m explicit in M.
   Elements dropped by the solver (f ≤ min_fraction) get zero gradient, as in mech_adjoint.

Assumptions. Both modes are non-degenerate (warned otherwise: optics via the relative gap to
the other computed bands, mechanics via `mech_freqs` and the realness of the Γ mode); the
mechanical mode is at Γ (phase matching, as coupling.g0). Accuracy follows the forward
eigensolver tolerances: the gradient is exact for the computed eigenpairs only as far as
they are converged (LOBPCG tol ≲ 1e-6 for FD-level agreement).

Caveats found in the checks (validation/g0_gradient_check.py):
- Element threshold. BlochSolver drops elements with f ≤ min_fraction (default 1e-3) and the
  gradient of a dropped element is zero, so shape gradients through a smoothed raster miss
  the contribution of boundary elements below the threshold (17 % of dg0/dw on the blade
  cell) and FD sees jumps when an element crosses it. Use min_fraction ≈ 1e-9 for shape
  gradients; fraction-level gradients are exact for either setting.
- m_eff with tiny elements. With min_fraction ≈ 0 the max|û| node can be one attached only to
  an almost-empty element (a flapping node), whose amplitude is very sensitive to that
  element's fraction: d_meff is then only locally valid. g0 is not affected (point 1).
Validation (blade cell, cubic Si; coarse GPT line defect): f_m, f_fine and edge f_o
directions 1e-13 to 1e-5; smooth f_o 1e-4 (FD truncation ∝ h², from the regularised normals);
block-mean chain 2e-4 (smooth) and 4e-7 (edge); tripod r, w 3e-5 and 5e-5.

Cost: one optical MINRES solve (typically a few hundred operator applications, comparable to
the forward LOBPCG), one SuperLU factorisation of the bordered mechanical system (shared by
the g0 and m_eff right-hand sides), and a few JAX passes over the voxel fields.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .coupling import HBAR, MASS_UNIT, _corners, mech_mode, voxel_points
from .materials import SI_P
from .mech_adjoint import eigenvector_adjoint_solver
from .mech_bloch import OMEGA_UNIT, BlochSolver
from .optics_pwe import C_UM_PER_PS, TENSOR, PWESolver

jax.config.update("jax_enable_x64", True)


# --- optics in JAX -------------------------------------------------------------------------


def _abs2(z):
    return jnp.real(z) ** 2 + jnp.imag(z) ** 2


def _eps_tensor(f, g, dg, rb, eta, es, eb):
    """slab_cell's fill F, ⟨ε⟩, Kottke ε⁻¹ (6, ...), normals, |∇F|_η and c = ⟨ε⁻¹⟩ − ⟨ε⟩⁻¹ from f (N1, N2)."""
    n1, n2 = f.shape
    dfs1 = (jnp.roll(f, -1, 0) - jnp.roll(f, 1, 0)) * n1 / 2
    dfs2 = (jnp.roll(f, -1, 1) - jnp.roll(f, 1, 1)) * n2 / 2
    gfx = dfs1 * rb[0, 0] + dfs2 * rb[1, 0]
    gfy = dfs1 * rb[0, 1] + dfs2 * rb[1, 1]
    F = f[:, :, None] * g
    grad = jnp.stack([gfx[..., None] * g, gfy[..., None] * g, f[:, :, None] * dg])
    norm = jnp.sqrt((grad**2).sum(axis=0) + eta**2)
    nrm = grad / norm
    eps = eb + F * (es - eb)
    aniso = 1 / eb + F * (1 / es - 1 / eb) - 1 / eps
    w = jnp.stack([aniso * nrm[i] * nrm[j] + (1 / eps if i == j else 0.0) for i, j in TENSOR])
    return F, eps, w, nrm, norm, aniso


def _apply(w, e):
    return jnp.stack(
        [
            w[0] * e[0] + w[5] * e[1] + w[4] * e[2],
            w[5] * e[0] + w[1] * e[1] + w[3] * e[2],
            w[4] * e[0] + w[3] * e[1] + w[2] * e[2],
        ]
    )


def _curl(xr, xi, u, v, absq):
    h = (xr + 1j * xi).reshape((2,) + absq.shape)
    return jnp.fft.ifftn(1j * absq * (h[0] * v - h[1] * u), axes=(1, 2, 3))


def _central_adjoint(a, axis, scale):
    return -(jnp.roll(a, -1, axis) - jnp.roll(a, 1, axis)) * scale / 2


def _optics_functional(xr, xi, f, dF, S, p, g, dg, rb, dz, eta, es, eb, u, v, absq):
    """
    L = ω_o (δω/ω)_MB+PE · 1e6: the optical frequency pull (rad/s per metre of mechanical
    amplitude) for a voxel fill change dF and an engineering strain S, as
    coupling.moving_boundary_shift + photoelastic_shift, with x normalised here.
    """
    s = 1 / jnp.sqrt(jnp.sum(xr**2 + xi**2))
    e = _curl(xr * s, xi * s, u, v, absq)
    F, eps, w, nrm, norm, aniso = _eps_tensor(f, g, dg, rb, eta, es, eb)
    n1, n2, nz = F.shape
    N = n1 * n2 * nz
    D = _apply(w, e)
    den = jnp.sum(jnp.real(jnp.conj(e) * D))
    lam = N * den
    # PWESolver.voxel_sensitivity
    a = -(es - eb) / eps**2
    b = (1 / es - 1 / eb) - a
    ne = (nrm * e).sum(axis=0)
    dlam = N * (a * _abs2(e).sum(axis=0) + b * _abs2(ne))
    r = jnp.real(jnp.conj(e) * ne)
    r = r - nrm * (nrm * r).sum(axis=0)
    sv = 2 * N * aniso / norm * r
    dlam = dlam + _central_adjoint(sv[0] * rb[0, 0] + sv[1] * rb[0, 1], 0, n1)
    dlam = dlam + _central_adjoint(sv[0] * rb[1, 0] + sv[1] * rb[1, 1], 1, n2)
    dlam = dlam + _central_adjoint(sv[2], 2, 1 / dz)
    rel_mb = (dlam * dF).sum() / (2 * lam)
    # photoelastic_shift: E = ε⁻¹ e = D
    d = jnp.tensordot(p, S, axes=(1, 0))
    q = (
        d[0] * _abs2(D[0])
        + d[1] * _abs2(D[1])
        + d[2] * _abs2(D[2])
        + 2 * d[3] * jnp.real(jnp.conj(D[1]) * D[2])
        + 2 * d[4] * jnp.real(jnp.conj(D[0]) * D[2])
        + 2 * d[5] * jnp.real(jnp.conj(D[0]) * D[1])
    )
    rel_pe = (F * es**2 * q).sum() / den / 2
    omega = C_UM_PER_PS * jnp.sqrt(lam) * 1e12
    return omega * 1e6 * (rel_mb + rel_pe), (rel_mb, rel_pe, lam)


_optics_vg = jax.jit(jax.value_and_grad(_optics_functional, argnums=(0, 1, 2, 3, 4), has_aux=True))


def _theta_bilinear(f, ew, ex, g, dg, rb, eta, es, eb):
    """Re(wᴴ Θ(f) x) = N Re Σ e_w* ε⁻¹(f) e_x for fixed fields."""
    _, _, w, _, _, _ = _eps_tensor(f, g, dg, rb, eta, es, eb)
    return ew[0].size * jnp.sum(jnp.real(jnp.conj(ew) * _apply(w, ex)))


_theta_bilinear_grad = jax.jit(jax.grad(_theta_bilinear))
_eps_tensor_jit = jax.jit(_eps_tensor)


# --- mechanical fields in JAX (Γ point, real) ----------------------------------------------


@partial(jax.jit, static_argnames=("n1", "n2", "nz", "ms"))
def _mech_fields(coef, fine, fm, c, n1, n2, nz, ms):
    """
    (δF, S) of coupling.advected_fill and coupling.strain_on_voxels for nodal coefficients
    coef (nodes, nb) at Γ; c holds the precomputed interpolation tables (`_MechTables`).
    """
    M1, M2 = fine.shape
    rb, rbf = c["rb"], c["rbf"]
    # moving boundary
    cf = (c["wt_f"][..., None] * coef[c["idx_f"]]).sum(axis=0)  # (P, nb)
    u = jnp.einsum("pb,cbz->cpz", cf, c["P"]).reshape(3, M1, M2, -1)
    d1 = (jnp.roll(fine, -1, 0) - jnp.roll(fine, 1, 0)) * M1 / 2
    d2 = (jnp.roll(fine, -1, 1) - jnp.roll(fine, 1, 1)) * M2 / 2
    gx, gy = d1 * rbf[0, 0] + d2 * rbf[1, 0], d1 * rbf[0, 1] + d2 * rbf[1, 1]
    inpl = -(u[0] * gx[..., None] + u[1] * gy[..., None])
    inpl = inpl.reshape(n1, M1 // n1, n2, M2 // n2, -1).mean(axis=(1, 3)) * c["gsel"]
    dF = jnp.zeros((n1, n2, nz)).at[:, :, c["sel"]].set(inpl)
    for key, sign in (("top", 1.0), ("bot", -1.0)):
        uz = (cf @ c["Pz_" + key]).reshape(M1, M2)
        val = (fine * uz).reshape(n1, M1 // n1, n2, M2 // n2).mean(axis=(1, 3))
        dF = dF.at[:, :, c["j_" + key]].add(sign * val[..., None] / c["dz"])
    # strain at the sub-points, fraction weighted
    cs = (c["w_s"][..., None] * coef[c["idx_s"]]).sum(axis=0)
    d1c = (c["w1_s"][..., None] * coef[c["idx_s"]]).sum(axis=0)
    d2c = (c["w2_s"][..., None] * coef[c["idx_s"]]).sum(axis=0)
    gxc, gyc = rb[0, 0] * d1c + rb[1, 0] * d2c, rb[0, 1] * d1c + rb[1, 1] * d2c
    u_z = jnp.einsum("pb,cbz->cpz", cs, c["dP"])
    u_x = jnp.einsum("pb,cbz->cpz", gxc, c["P"])
    u_y = jnp.einsum("pb,cbz->cpz", gyc, c["P"])
    S6 = jnp.stack([u_x[0], u_y[1], u_z[2], u_z[1] + u_y[2], u_z[0] + u_x[2], u_y[0] + u_x[1]])
    S6 = S6 * fm.reshape(-1)[c["el"]][None, :, None]
    S6 = S6.reshape(6, n1, ms, n2, ms, -1).mean(axis=(2, 4))
    S = jnp.zeros((6, n1, n2, nz)).at[..., c["sel"]].set(S6)
    return dF, S


@partial(jax.jit, static_argnames=("n1", "n2", "nz", "ms"))
def _mech_fields_vjp(coef, fine, fm, c, cot_dF, cot_S, n1, n2, nz, ms):
    _, pull = jax.vjp(lambda a, b, d: _mech_fields(a, b, d, c, n1, n2, nz, ms), coef, fine, fm)
    return pull((cot_dF, cot_S))


def _mech_tables(mode, cell, fine_shape, m_s: int | None, to_mode=None) -> dict:
    """
    Interpolation tables for `_mech_fields`: corner nodes and weights, Legendre values.
    to_mode: map from the optical cell's centred lattice coordinates to the mode's (an optical
    window on a larger mechanical domain, cavity_optics.GridWindow.to_mode; m_s then required).
    """
    n1, n2, nz = cell.eps.shape
    mn1, mn2 = mode.coef.shape[:2]
    if m_s is None and to_mode is not None:
        raise ValueError("give strain_supersample with to_mode")
    if m_s is None:
        m_s = mn1 // n1
        if m_s * n1 != mn1 or m_s * n2 != mn2:
            raise ValueError("default strain sub-points need the mechanical grid to be an integer refinement of the optical grid")
    M1, M2 = fine_shape
    dz = cell.height / nz
    g = cell.z_cover
    t = g.sum() * dz
    _, _, z = voxel_points(cell)
    sel = np.flatnonzero(g > 0)
    zc = (np.maximum(z[sel] - dz / 2, -t / 2) + np.minimum(z[sel] + dz / 2, t / 2)) / 2
    comps = np.array([cb for cb, _ in mode.basis])
    onehot = (comps[None, :] == np.arange(3)[:, None]).astype(float)  # (3, nb)
    P = mode._legendre(zc)  # (nb, nsel)
    dP = mode._legendre(zc, derivative=True)
    out = dict(rb=mode.lattice.reciprocal / (2 * np.pi), rbf=cell.lattice.reciprocal / (2 * np.pi), P=onehot[:, :, None] * P[None],
               dP=onehot[:, :, None] * dP[None])
    out.update(gsel=g[sel], sel=sel, dz=dz)
    for key, sign in (("top", 1), ("bot", -1)):
        face = sign * t / 2
        out["j_" + key] = np.flatnonzero((z - dz / 2 < face) & (face <= z + dz / 2))
        out["Pz_" + key] = mode._legendre(np.array([face]))[:, 0] * (comps == 2)
    # fine points, renormalised Q1 weights (coefficients_at, renormalise=True); Γ: phases 1
    s1 = ((np.arange(M1) + 0.5) / M1 - 0.5)[:, None] * np.ones((1, M2))
    s2 = ((np.arange(M2) + 0.5) / M2 - 0.5)[None, :] * np.ones((M1, 1))
    if to_mode is not None:
        s1, s2 = to_mode(s1, s2)
    corners, _, _ = _corners(mode, s1, s2)
    idx = np.array([(i * mn2 + j).ravel() for i, j, *_ in corners])
    w = np.array([(wc * mode.has_dof[i, j]).ravel() for i, j, _, _, wc, _ in corners])
    wsum = w.sum(axis=0)
    out.update(idx_f=idx, wt_f=w / np.where(wsum > 0, wsum, 1))
    # strain sub-points (coupling.strain: plain Q1 and its gradients)
    s1 = ((np.arange(n1 * m_s) + 0.5) / (n1 * m_s) - 0.5)[:, None] * np.ones((1, n2 * m_s))
    s2 = ((np.arange(n2 * m_s) + 0.5) / (n2 * m_s) - 0.5)[None, :] * np.ones((n1 * m_s, 1))
    if to_mode is not None:
        s1, s2 = to_mode(s1, s2)
    corners, a, b = _corners(mode, s1, s2)
    out["idx_s"] = np.array([(i * mn2 + j).ravel() for i, j, *_ in corners])
    out["w_s"] = np.array([wc.ravel() for *_, wc, _ in corners])
    out["w1_s"] = np.array([((b if d2 else 1 - b) * (1 if d1 else -1) * mn1).ravel() for _, _, d1, d2, _, _ in corners])
    out["w2_s"] = np.array([((a if d1 else 1 - a) * (1 if d2 else -1) * mn2).ravel() for _, _, d1, d2, _, _ in corners])
    ei = np.floor((s1 + 0.5) * mn1).astype(int) % mn1
    ej = np.floor((s2 + 0.5) * mn2).astype(int) % mn2
    out["el"] = (ei * mn2 + ej).ravel()
    return {k: jnp.asarray(v) for k, v in out.items()}, m_s


# --- result --------------------------------------------------------------------------------


@dataclass
class G0Gradient:
    """
    g0 (rad/s, per (super)cell) and its gradients (rad/s per unit fraction) with respect to
    the optical fill f_o (N1, N2), the mechanical element fractions f_m (n1, n2) and the
    moving-boundary raster f_fine (M1, M2), as independent inputs. g_mb, g_pe: the two terms
    of g0. m_eff (kg) and d_meff (kg per unit fraction, at the fixed max|û| reference point),
    mech_freq (Hz) and d_mech_freq. The sign of g0 follows the sign of the mechanical
    eigenvector, and the gradients are those of that signed g0. info: timings, solver stats.
    """

    g0: float
    g_mb: float
    g_pe: float
    m_eff: float
    x_zpf: float
    opt_freq_thz: float
    mech_freq: float
    d_opt: np.ndarray
    d_mech: np.ndarray
    d_fine: np.ndarray
    d_meff: np.ndarray | None = None
    d_mech_freq: np.ndarray | None = None
    info: dict = field(default_factory=dict)


def _optical_inputs(solver: PWESolver) -> dict:
    """Arrays describing the solver's cell for the JAX optics, with f_o and η recovered from it."""
    c = solver.cell
    n1, n2, nz = solver.shape
    g = c.z_cover
    dz = c.height / nz
    jmax = int(np.argmax(g))
    f = c.fill[:, :, jmax] / g[jmax]
    eta = float(np.sqrt(np.median(c.grad_norm**2 * np.clip(1 - (c.normals**2).sum(axis=0), 0, None))))
    out = dict(
        f=jnp.asarray(f),
        g=jnp.asarray(g),
        dg=jnp.asarray((np.roll(g, -1) - np.roll(g, 1)) / (2 * dz)),
        rb=jnp.asarray(c.lattice.reciprocal / (2 * np.pi)),
        dz=dz,
        eta=eta,
        es=float(c.eps_slab),
        eb=float(c.eps_background),
    )
    w = np.asarray(_eps_tensor_jit(out["f"], out["g"], out["dg"], out["rb"], eta, out["es"], out["eb"])[2])
    err = np.abs(w - c.eps_inv).max()
    if err > 1e-10:
        warnings.warn(f"JAX rebuild of the cell's ε⁻¹ differs by {err:.1e}: was the cell built by slab_cell?")
    out["eps_err"] = err
    return out


def optomech_functional(
    opt_solver: PWESolver,
    band: int,
    mode,
    fine_fraction: np.ndarray,
    mech_fraction: np.ndarray,
    p: np.ndarray = SI_P,
    strain_supersample: int | None = None,
    adjoint: bool = True,
    minres_tol: float = 1e-10,
    minres_maxiter: int = 3000,
    to_mode=None,
    minres_jax: bool = False,
) -> dict:
    """
    The part of g0 shared by every mechanical model: L = ω_o (δω/ω)_MB+PE · 1e6 (rad/s per
    metre of mechanical amplitude) for the Γ-point MechMode `mode` (real coefficients on its
    node grid, any normalisation: L is linear in them), and its gradients
      gcoef (nodes, nb): ∂L/∂(mode coefficients), the right-hand side of a mechanical adjoint;
      gfine: ∂L/∂f_fine (moving-boundary raster, explicit);
      gfm: ∂L/∂(mode.fraction) (strain fraction weighting, explicit);
      d_opt: dL/df_o including the optical eigenvector (one MINRES; only with adjoint).
    g0 = L x_zpf. Used by `g0_gradient` (BlochSolver) and wavesolver.codesign (PiezoSolver modes).
    to_mode: the optical cell is a window of the mode's domain (see `_mech_tables`; gfm is then
    on the whole mechanical grid, gfine on the window's raster). minres_jax: the optical adjoint
    by PWESolver.adjoint_solve(jax=True).
    """
    info: dict = {}
    t0 = time.perf_counter()
    cell = opt_solver.cell
    n1, n2, nz = cell.eps.shape
    nn1, nn2, dpn = mode.coef.shape
    coef = np.real(mode.coef).reshape(nn1 * nn2, dpn)
    tab, ms = _mech_tables(mode, cell, fine_fraction.shape, strain_supersample, to_mode)
    coef_j, fine_j, fm_j = jnp.asarray(coef), jnp.asarray(fine_fraction, dtype=float), jnp.asarray(mech_fraction, dtype=float)
    dF, S = _mech_fields(coef_j, fine_j, fm_j, tab, n1, n2, nz, ms)
    oi = _optical_inputs(opt_solver)
    x = opt_solver.last_vectors[:, band]
    ostate = (jnp.asarray(opt_solver.u), jnp.asarray(opt_solver.v), jnp.asarray(opt_solver.absq))
    args = (oi["g"], oi["dg"], oi["rb"], oi["dz"], oi["eta"], oi["es"], oi["eb"]) + ostate
    (L, (rel_mb, rel_pe, lam_o)), (gxr, gxi, gf, gdF, gS) = _optics_vg(
        jnp.asarray(x.real), jnp.asarray(x.imag), oi["f"], dF, S, jnp.asarray(p, dtype=float), *args
    )
    out = dict(L=float(L), rel_mb=float(rel_mb), rel_pe=float(rel_pe), lam_o=float(lam_o), info=info)
    info["t_fields"] = time.perf_counter() - t0
    info["eps_rebuild_err"] = oi["eps_err"]
    if adjoint:
        t1 = time.perf_counter()
        r = np.asarray(gxr) + 1j * np.asarray(gxi)
        info["opt_rhs_overlap"] = abs(np.vdot(x, r)) / np.linalg.norm(r)
        w, oinfo = opt_solver.adjoint_solve(band, r, tol=minres_tol, maxiter=minres_maxiter, jax=minres_jax)
        info.update({f"opt_{k}": v for k, v in oinfo.items()})
        ew = _curl(jnp.asarray(w.real), jnp.asarray(w.imag), *ostate)
        ex = _curl(jnp.asarray(x.real), jnp.asarray(x.imag), *ostate)
        gtheta = _theta_bilinear_grad(oi["f"], ew, ex, oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"])
        out["d_opt"] = np.asarray(gf) - np.asarray(gtheta)
        info["t_opt_adjoint"] = time.perf_counter() - t1
    gcoef, gfine, gfm = _mech_fields_vjp(coef_j, fine_j, fm_j, tab, gdF, gS, n1, n2, nz, ms)
    out.update(gcoef=np.asarray(gcoef), gfine=np.asarray(gfine), gfm=np.asarray(gfm))
    return out


def g0_gradient(
    opt_solver: PWESolver,
    band: int,
    mech_solver: BlochSolver,
    mech_freq: float,
    mech_vector: np.ndarray,
    fine_fraction: np.ndarray,
    mech_fraction: np.ndarray,
    p: np.ndarray = SI_P,
    mech_freqs: np.ndarray | None = None,
    strain_supersample: int | None = None,
    minres_tol: float = 1e-10,
    minres_maxiter: int = 3000,
    meff: bool = True,
) -> G0Gradient:
    """
    g0 between band `band` of opt_solver's last solve and the Γ-point "slab_p" mode
    (mech_freq Hz, eigenvector mech_vector of mech_solver.solve(0, ..., vectors=True)), and
    its adjoint gradients (see the module docstring). fine_fraction: coupling.advected_fill's
    raster; mech_fraction: the fractions mech_solver was built from; p: lab-frame Si
    photoelastic tensor (Voigt; materials.SI_P is cubic Si with [100] ∥ x). mech_freqs: all
    computed mechanical frequencies, to warn on degeneracy. meff: also d m_eff / d f_m.
    """
    info: dict = {}
    t0 = time.perf_counter()
    k0 = np.zeros(2)
    if mech_solver.model != "slab_p" or len(mech_solver.parts) != 1:
        raise ValueError("g0_gradient needs a single-part slab_p BlochSolver")
    if mech_freqs is not None:
        others = np.asarray(mech_freqs)[np.abs(np.asarray(mech_freqs) - mech_freq) > 0]
        if len(others) and np.abs(others - mech_freq).min() < 1e-4 * mech_freq:
            warnings.warn(f"mechanical mode at {mech_freq / 1e9:.4f} GHz is (near-)degenerate; its g0 gradient is not defined")
    # Mechanical mode: real at Γ, in coupling.py's normalisation.
    y = np.asarray(mech_vector, dtype=complex)
    y = y * np.exp(-0.5j * np.angle(np.sum(y * y)))
    realness = np.linalg.norm(y.imag) / np.linalg.norm(y)
    if realness > 1e-6:
        warnings.warn(f"Γ mode is not real (‖Im‖/‖u‖ = {realness:.1e}): degenerate partner mixed in?")
    Kc, Mc = mech_solver.matrices(k0)
    K, M = Kc.real.tocsr(), Mc.real.tocsr()
    mode = mech_mode(mech_solver, k0, mech_freq, y.real.astype(complex), M=Mc, fraction=mech_fraction)
    nn1, nn2, dpn = mode.coef.shape
    coef = np.real(mode.coef).reshape(nn1 * nn2, dpn)
    yh = coef.reshape(-1)[mech_solver.dofs]  # û on the solver's DOFs
    My = M @ yh
    yMy = float(yh @ My)
    lam_m = float(yh @ (K @ yh)) / yMy  # Rayleigh quotient
    omega_m = OMEGA_UNIT * np.sqrt(lam_m)
    x_zpf = np.sqrt(HBAR / (2 * yMy * MASS_UNIT * omega_m))

    core = optomech_functional(
        opt_solver, band, mode, fine_fraction, mech_fraction, p, strain_supersample, minres_tol=minres_tol, minres_maxiter=minres_maxiter
    )
    info.update(core["info"])
    L, rel_mb, rel_pe, lam_o = core["L"], core["rel_mb"], core["rel_pe"], core["lam_o"]
    g0 = L * x_zpf
    d_opt = x_zpf * core["d_opt"]
    gcoef, gfine, gfm = core["gcoef"], core["gfine"], core["gfm"]
    t2 = time.perf_counter()
    d_fine = x_zpf * np.asarray(gfine)
    a = x_zpf * np.asarray(gcoef).reshape(-1)[mech_solver.dofs] - g0 * My / yMy
    info["mech_rhs_overlap"] = abs(a @ yh) / (np.linalg.norm(a) * np.linalg.norm(yh))
    solve = eigenvector_adjoint_solver(K, M, lam_m, yh)
    info["t_mech_factor"] = time.perf_counter() - t2
    rhs = [a]
    if meff:
        # m_eff = ûᵀMû / |û(r*)|² at the fixed peak point r* (node, height) of mech_mode.
        zz = np.linspace(-mech_solver.thickness / 2, mech_solver.thickness / 2, 9)
        Pz = mode._legendre(zz)  # (nb, 9)
        comps = np.array([cb for cb, _ in mode.basis])
        onehot = (comps[None, :] == np.arange(3)[:, None]).astype(float)
        un = np.einsum("nb,cb,bz->cnz", coef, onehot, Pz)
        u2 = (un**2).sum(axis=0)
        node, zi = np.unravel_index(np.argmax(u2), u2.shape)
        ustar = un[:, node, zi]
        grad_u2 = np.zeros_like(coef)
        grad_u2[node] = 2 * (ustar @ onehot) * Pz[:, zi]
        u2s = float(ustar @ ustar)
        m_eff = MASS_UNIT * yMy / u2s
        a_m = MASS_UNIT * (2 * My / u2s - yMy / u2s**2 * grad_u2.reshape(-1)[mech_solver.dofs])
        rhs.append(a_m)
        info["meff_ref_point"] = (int(node), int(zi), u2s)
    else:
        m_eff = MASS_UNIT * yMy
    zs = [solve(rr) for rr in rhs]
    res = np.linalg.norm((K @ zs[0] - lam_m * (M @ zs[0])) - (a - My * (yh @ a) / yMy)) / np.linalg.norm(a)
    info["mech_residual"] = res

    part = mech_solver.parts[0]
    ok = part.edof >= 0
    gather = lambda vec: np.where(ok, vec[np.maximum(part.edof, 0)], 0.0)  # (ne, nd)
    Ye = gather(yh)
    kY, mY = Ye @ part.Ke.T, Ye @ part.Me.T
    yKy_e, yMy_e = (Ye * kY).sum(axis=1), (Ye * mY).sum(axis=1)
    rY = kY - lam_m * mY
    nmech = nn1 * nn2
    cells = lambda v: np.bincount(part.cell, weights=v, minlength=nmech).reshape(nn1, nn2)
    dlam_m = cells(yKy_e - lam_m * yMy_e) / yMy
    d_mech = (
        x_zpf * np.asarray(gfm)
        - cells((gather(zs[0]) * rY).sum(axis=1))
        - g0 / (4 * lam_m) * dlam_m
        - g0 / 2 * cells(yMy_e) / yMy
    )
    d_meff = None
    if meff:
        d_meff = MASS_UNIT * cells(yMy_e) / u2s - cells((gather(zs[1]) * rY).sum(axis=1))
    d_mech_freq = mech_freq / (2 * lam_m) * dlam_m
    info["t_mech_adjoint"] = time.perf_counter() - t2
    info["t_total"] = time.perf_counter() - t0
    info["realness"] = realness
    return G0Gradient(
        g0=g0,
        g_mb=omega_rel(lam_o, rel_mb) * x_zpf,
        g_pe=omega_rel(lam_o, rel_pe) * x_zpf,
        m_eff=m_eff,
        x_zpf=float(np.sqrt(HBAR / (2 * m_eff * omega_m))),
        opt_freq_thz=C_UM_PER_PS * np.sqrt(lam_o) / (2 * np.pi),
        mech_freq=mech_freq,
        d_opt=d_opt,
        d_mech=d_mech,
        d_fine=d_fine,
        d_meff=d_meff,
        d_mech_freq=d_mech_freq,
        info=info,
    )


def omega_rel(lam: float, rel: float) -> float:
    """dω/dx in rad/s per m from a relative shift per µm: ω_o · rel · 1e6."""
    return C_UM_PER_PS * np.sqrt(lam) * 1e12 * rel * 1e6


def block_mean_adjoint(G: np.ndarray, fine_shape: tuple[int, int]) -> np.ndarray:
    """Adjoint of coupling.block_mean: G (n1, n2) → (M1, M2), spread uniformly over each block."""
    n1, n2 = G.shape
    m1, m2 = fine_shape[0] // n1, fine_shape[1] // n2
    if m1 * n1 != fine_shape[0] or m2 * n2 != fine_shape[1]:
        raise ValueError(f"{G.shape} does not divide the raster {fine_shape}")
    return np.repeat(np.repeat(G, m1, axis=0), m2, axis=1) / (m1 * m2)


def fine_total_gradient(res: G0Gradient, d_mech: np.ndarray | None = None) -> np.ndarray:
    """
    dg0/df_fine when f_o = block_mean(f_fine, N1, N2) and f_m = block_mean(f_fine, n1, n2)
    (coupling.py's convention for the moving-boundary raster): explicit raster term plus the
    block-mean adjoints of the optical and mechanical gradients. d_mech overrides
    res.d_mech (e.g. d_meff, to chain m_eff the same way, with res.d_opt and d_fine set to 0).
    """
    shape = res.d_fine.shape
    dm = res.d_mech if d_mech is None else d_mech
    return res.d_fine + block_mean_adjoint(res.d_opt, shape) + block_mean_adjoint(dm, shape)


__all__ = ["G0Gradient", "block_mean_adjoint", "fine_total_gradient", "g0_gradient", "optomech_functional"]
