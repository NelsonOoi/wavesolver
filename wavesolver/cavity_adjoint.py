"""
Adjoint gradient of a finite cavity's g0 (and of f_o, f_m) with respect to the moving-boundary
raster of the whole mechanical domain: the optics in a window of the mechanical grid
(wavesolver.cavity_optics), the mechanics a σ_y-reduced Γ-point cavity mode (wavesolver.cavity).

The method is wavesolver.coupling_adjoint.g0_gradient's, with three changes:
- the optical functional and the mechanical fields it sees are evaluated on the window
  (`optomech_functional(..., to_mode=win.to_mode)`), so ∂L/∂f_o and ∂L/∂f_fine live on the
  window and ∂L/∂f_m on the whole mechanical grid;
- the optical adjoint (Θ − λ) w = r runs in JAX at a loose tolerance: the cavity mode sits
  in a dense spectrum of delocalised supercell modes, but r is localised with the mode and the
  eigenvector term −Re(wᴴ ∂Θ/∂f x) only sees w where x is, so the components the loose solve
  leaves out (near-λ, delocalised) hardly contribute (checked against FD in
  research/linedefect_cavity_opt.py `check`);
- the mechanical adjoint (K − λM) z = a, ûᵀMz = 0 is solved in the σ_y-even subspace (P
  orthonormal, z = P z_r), where the mode is simple; the element terms use the full z, û.

Rasters (coupling.py's convention): f_o = block_mean(window of f_fine), f_m = block_mean(f_fine)
over the mechanical elements, so everything chains onto f_fine (`CavityGradient.d_*`).
Units: g0 rad/s, f_o THz, f_m Hz; gradients per unit fraction of a fine pixel.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as np

from .cavity import MechCavity
from .cavity_optics import CavityOptics
from .coupling import HBAR, MASS_UNIT, MechMode, mech_mode
from .coupling_adjoint import _apply, _curl, _eps_tensor, _optical_inputs, _theta_bilinear_grad, block_mean_adjoint, omega_rel, optomech_functional
from .linalg import factorize_indefinite
from .mech_adjoint import eigenvector_adjoint_solver
from .mech_bloch import OMEGA_UNIT
from .optics_pwe import C_UM_PER_PS, PWESolver


@dataclass
class CavityGradient:
    """g0 (rad/s; sign of the mechanical eigenvector) and d g0, d f_o, d f_m on the whole fine raster."""

    g0: float
    g_mb: float
    g_pe: float
    f_o: float  # THz
    f_m: float  # Hz
    m_eff: float  # kg (max|û| = 1)
    d_g0: np.ndarray
    d_fo: np.ndarray
    d_fm: np.ndarray
    d_g0_mech: np.ndarray | None = None  # the part of d_g0 through the mechanical element fractions (the rest is the window raster's)
    info: dict = field(default_factory=dict)


def cavity_g0_gradient(co: CavityOptics, s: PWESolver, cav: MechCavity, mech_freq: float, x_mech: np.ndarray, fine_shape: tuple[int, int],
                       band: int = 0, adjoint_tol: float = 1e-3, adjoint_maxiter: int = 600, extra=None) -> CavityGradient:
    """
    co: set_design done on the same design and co.mode the cavity mode (mech_freq Hz, eigenvector
    x_mech on cav.solver's DOFs, real at Γ, closed cavity: no absorber); s: the solved window
    PWESolver (co.solve(...).solver); fine_shape: the whole domain's fine raster.
    extra: callable g0 → (a, r_e, gf_e) (see optomech_functional's rhs_extra): d_g0 is then the
    gradient of a·g0 + E (E another functional of the optical mode) by the same optical adjoint.
    """
    info: dict = {}
    t0 = time.perf_counter()
    solver, P = cav.solver, cav.P
    k0 = np.zeros(2)
    y = np.asarray(x_mech, dtype=complex)
    y = y * np.exp(-0.5j * np.angle(np.sum(y * y)))
    info["realness"] = float(np.linalg.norm(y.imag) / np.linalg.norm(y))
    if info["realness"] > 1e-6:
        warnings.warn(f"cavity mode is not real (‖Im‖/‖u‖ = {info['realness']:.1e})")
    Kc, Mc = solver.matrices(k0)
    K, M = Kc.real.tocsr(), Mc.real.tocsr()
    mode = mech_mode(solver, k0, mech_freq, y.real.astype(complex), M=Mc, fraction=cav.fraction)
    nn1, nn2, dpn = mode.coef.shape
    coef = np.real(mode.coef).reshape(nn1 * nn2, dpn)
    yh = coef.reshape(-1)[solver.dofs]
    My = M @ yh
    yMy = float(yh @ My)
    lam_m = float(yh @ (K @ yh)) / yMy
    omega_m = OMEGA_UNIT * np.sqrt(lam_m)
    x_zpf = np.sqrt(HBAR / (2 * yMy * MASS_UNIT * omega_m))
    info["t_setup"] = time.perf_counter() - t0

    a_g = [1.0]

    def rhs_extra(L):  # the combined right-hand side: L·x_zpf = g0 carries the weight a(g0), E is unscaled
        a_g[0], r_e, gf_e = extra(L * x_zpf)
        return a_g[0] * x_zpf, r_e, gf_e

    core = optomech_functional(s, band, mode, co.fine, cav.fraction, co.p, strain_supersample=co.ratio, minres_tol=adjoint_tol,
                               minres_maxiter=adjoint_maxiter, to_mode=co.win.to_mode, minres_jax=True, rhs_extra=rhs_extra if extra else None)
    info.update(core["info"])
    L, rel_mb, rel_pe, lam_o = core["L"], core["rel_mb"], core["rel_pe"], core["lam_o"]
    g0 = L * x_zpf

    t1 = time.perf_counter()
    a = x_zpf * core["gcoef"].reshape(-1)[solver.dofs] - g0 * My / yMy
    if P is not None:
        solve = eigenvector_adjoint_solver((P.T @ K @ P).tocsr(), (P.T @ M @ P).tocsr(), lam_m, P.T @ yh)
        z = P @ solve(P.T @ a)
        info["mech_odd_rhs"] = float(np.linalg.norm(a - P @ (P.T @ a)) / np.linalg.norm(a))
    else:
        z = eigenvector_adjoint_solver(K, M, lam_m, yh)(a)
    info["t_mech_adjoint"] = time.perf_counter() - t1

    part = solver.parts[0]
    ok = part.edof >= 0
    gather = lambda vec: np.where(ok, vec[np.maximum(part.edof, 0)], 0.0)  # noqa: E731
    Ye = gather(yh)
    kY, mY = Ye @ part.Ke.T, Ye @ part.Me.T
    yKy_e, yMy_e = (Ye * kY).sum(axis=1), (Ye * mY).sum(axis=1)
    rY = kY - lam_m * mY
    cells = lambda v: np.bincount(part.cell, weights=v, minlength=nn1 * nn2).reshape(nn1, nn2)  # noqa: E731
    dlam_m = cells(yKy_e - lam_m * yMy_e) / yMy
    d_mech = x_zpf * core["gfm"] - cells((gather(z) * rY).sum(axis=1)) - g0 / (4 * lam_m) * dlam_m - g0 / 2 * cells(yMy_e) / yMy

    # onto the whole fine raster
    w = co.win
    m1, m2 = fine_shape[0] // w.n1m, fine_shape[1] // w.n2m
    sl = (slice(w.i0 * m1, (w.i0 + w.n1w) * m1), slice(w.j0 * m2, (w.j0 + w.n2w) * m2))
    d_g0_mech = block_mean_adjoint(d_mech, fine_shape)
    if extra:  # d_opt is already the combined a·g0 + E optical part
        d_g0 = a_g[0] * d_g0_mech
        d_g0[sl] += a_g[0] * x_zpf * core["gfine"] + block_mean_adjoint(core["d_opt"], co.fine.shape)
    else:
        d_g0 = d_g0_mech.copy()
        d_g0[sl] += x_zpf * core["gfine"] + block_mean_adjoint(x_zpf * core["d_opt"], co.fine.shape)
    d_fo = np.zeros(fine_shape)
    d_fo[sl] = block_mean_adjoint(s.sensitivities([band])[0], co.fine.shape)
    d_fm = block_mean_adjoint(mech_freq / (2 * lam_m) * dlam_m, fine_shape)
    info["t_total"] = time.perf_counter() - t0
    return CavityGradient(
        g0=g0,
        g_mb=omega_rel(lam_o, rel_mb) * x_zpf,
        g_pe=omega_rel(lam_o, rel_pe) * x_zpf,
        f_o=C_UM_PER_PS * np.sqrt(lam_o) / (2 * np.pi),
        f_m=mech_freq,
        m_eff=mode.mass,
        d_g0=d_g0,
        d_fo=d_fo,
        d_fm=d_fm,
        d_g0_mech=d_g0_mech,
        info=info,
    )


def light_cone_mask(s: PWESolver, f_thz: float) -> np.ndarray:
    """Flat mask (as last_vectors' rows) of the plane waves with in-plane |q| < ω/c (k = 0: q = G)."""
    q = np.hypot(s.G[0], s.G[1])
    inside = q < 2 * np.pi * f_thz / C_UM_PER_PS
    return np.concatenate([inside.ravel(), inside.ravel()])


def light_cone_gradient(co: CavityOptics, s: PWESolver, fine_shape: tuple[int, int], band: int = 0, adjoint_tol: float = 1e-3,
                        adjoint_maxiter: int = 600) -> tuple[float, np.ndarray, dict]:
    """
    LC = xᴴPx / xᴴx, the share of the cavity mode's plane-wave power inside the light cone (P: in-plane
    |q| < ω_o/c, held at the current ω_o), a proxy for the radiation loss 1/Q_rad, and dLC/df_fine
    on the whole fine raster: one optical adjoint (Θ − λ) w = 2(P − LC)x, then −Re(wᴴ ∂Θ/∂f x)
    through the window's fill (the cone edge moving with ω_o is left out: second order for a
    mode centred at X, far from the cone).
    """
    x = s.last_vectors[:, band]
    x = x / np.linalg.norm(x)
    P = light_cone_mask(s, float(s._last_freqs[band]))
    lc = float(np.sum(np.abs(x[P]) ** 2))
    r = 2 * (np.where(P, x, 0) - lc * x)
    w, info = s.adjoint_solve(band, r, tol=adjoint_tol, maxiter=adjoint_maxiter, jax=True)
    oi = _optical_inputs(s)
    st = (np.asarray(s.u), np.asarray(s.v), np.asarray(s.absq))
    ew = _curl(w.real, w.imag, *st)
    ex = _curl(x.real, x.imag, *st)
    d_opt = -np.asarray(_theta_bilinear_grad(oi["f"], ew, ex, oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"]))
    win = co.win
    m1, m2 = fine_shape[0] // win.n1m, fine_shape[1] // win.n2m
    out = np.zeros(fine_shape)
    out[win.i0 * m1 : (win.i0 + win.n1w) * m1, win.j0 * m2 : (win.j0 + win.n2w) * m2] = block_mean_adjoint(d_opt, co.fine.shape)
    return lc, out, info


def _log_volume(xr, xi, f, g, dg, rb, eta, es, eb, u, v, absq):
    """log of the participation volume Σu)² / Σu² (voxels) of the electric energy density u = Re e*·ε⁻¹e (x normalised here)."""
    sc = 1 / jnp.sqrt(jnp.sum(xr**2 + xi**2))
    e = _curl(xr * sc, xi * sc, u, v, absq)
    _, _, w, _, _, _ = _eps_tensor(f, g, dg, rb, eta, es, eb)
    dens = jnp.real(jnp.conj(e) * _apply(w, e)).sum(axis=0)
    return 2 * jnp.log(dens.sum()) - jnp.log((dens**2).sum())


_log_volume_vg = jax.jit(jax.value_and_grad(_log_volume, argnums=(0, 1, 2)))


def mode_volume(s: PWESolver, band: int = 0) -> float:
    """`mode_volume_gradient`'s V in (λ/n_Si)³, without the gradient."""
    x = s.last_vectors[:, band]
    oi = _optical_inputs(s)
    st = (jnp.asarray(s.u), jnp.asarray(s.v), jnp.asarray(s.absq))
    lv, _ = _log_volume_vg(jnp.asarray(x.real), jnp.asarray(x.imag), oi["f"], oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"], *st)
    n1, n2, nz = s.shape
    dV = abs(np.linalg.det(s.cell.lattice.vectors)) * s.cell.height / (n1 * n2 * nz)
    return float(np.exp(lv)) * dV / (C_UM_PER_PS / float(s._last_freqs[band]) / np.sqrt(s.cell.eps_slab)) ** 3


def mode_volume_gradient(co: CavityOptics, s: PWESolver, fine_shape: tuple[int, int], band: int = 0, adjoint_tol: float = 1e-3,
                         adjoint_maxiter: int = 600) -> tuple[float, np.ndarray, dict]:
    """
    Optical mode volume V = (Σu)² / Σu² · dV in (λ/n_Si)³, u = D·E per voxel (the participation-ratio
    volume: smooth, unlike ∫u / max u, and within a small factor of it for a single-lobed cavity
    envelope), and d log V / df_fine on the whole fine raster: the explicit ε⁻¹ dependence by JAX,
    the eigenvector part by one optical adjoint, and the (λ/n)³ normalisation's 3 d log f_o.
    """
    x = s.last_vectors[:, band]
    oi = _optical_inputs(s)
    st = (jnp.asarray(s.u), jnp.asarray(s.v), jnp.asarray(s.absq))
    args = (oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"]) + st
    lv, (gr, gi, gf) = _log_volume_vg(jnp.asarray(x.real), jnp.asarray(x.imag), oi["f"], *args)
    r = np.asarray(gr) + 1j * np.asarray(gi)
    w, info = s.adjoint_solve(band, r, tol=adjoint_tol, maxiter=adjoint_maxiter, jax=True)
    ew = _curl(jnp.asarray(w.real), jnp.asarray(w.imag), *st)
    ex = _curl(jnp.asarray(x.real), jnp.asarray(x.imag), *st)
    gtheta = _theta_bilinear_grad(oi["f"], ew, ex, oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"])
    f_thz = float(s._last_freqs[band])
    # V in (λ/n)³ carries +3 log f_o (Hellmann–Feynman, as cavity_g0_gradient's d_fo)
    d_opt = np.asarray(gf) - np.asarray(gtheta) + 3 / f_thz * s.sensitivities([band])[0]
    n1, n2, nz = s.shape
    dV = abs(np.linalg.det(s.cell.lattice.vectors)) * s.cell.height / (n1 * n2 * nz)
    lam_n = C_UM_PER_PS / float(s._last_freqs[band]) / np.sqrt(s.cell.eps_slab)
    V = float(np.exp(lv)) * dV / lam_n**3
    win = co.win
    m1, m2 = fine_shape[0] // win.n1m, fine_shape[1] // win.n2m
    out = np.zeros(fine_shape)
    out[win.i0 * m1 : (win.i0 + win.n1w) * m1, win.j0 * m2 : (win.j0 + win.n2w) * m2] = block_mean_adjoint(d_opt, co.fine.shape)
    return V, out, info


def mech_freq_gradient(cav: MechCavity, freq: float, x_mech: np.ndarray, fine_shape: tuple[int, int]) -> np.ndarray:
    """df/df_fine (Hz per unit fraction) of a mechanical cavity mode (Hellmann–Feynman, as cavity_g0_gradient's d_fm)."""
    solver = cav.solver
    K, M = (A.real.tocsr() for A in solver.matrices(np.zeros(2)))
    y = np.real(np.asarray(x_mech))
    yMy = float(y @ (M @ y))
    lam = float(y @ (K @ y)) / yMy
    part = solver.parts[0]
    ok = part.edof >= 0
    Ye = np.where(ok, y[np.maximum(part.edof, 0)], 0.0)
    e = ((Ye @ part.Ke.T) * Ye).sum(axis=1) - lam * ((Ye @ part.Me.T) * Ye).sum(axis=1)
    n1, n2 = solver.shape
    dlam = np.bincount(part.cell, weights=e, minlength=n1 * n2).reshape(n1, n2) / yMy
    return block_mean_adjoint(freq / (2 * lam) * dlam, fine_shape)


def mech_share(cav: MechCavity, x_mech: np.ndarray, mask: np.ndarray) -> float:
    """`mech_share_gradient`'s P without the gradient."""
    solver = cav.solver
    _, M = solver.matrices(np.zeros(2))
    y = np.real(np.asarray(x_mech))
    num = 0.0
    for part, frac in zip(solver.parts, solver.part_fractions):
        ok = part.edof >= 0
        Ye = np.where(ok, y[np.maximum(part.edof, 0)], 0.0)
        e = ((Ye @ part.Me.T) * Ye).sum(axis=1) * frac.ravel()[part.cell]
        num += e[mask.ravel()[part.cell]].sum()
    return float(num / (y @ (M.real @ y)))


def mech_share_gradient(cav: MechCavity, freq: float, x_mech: np.ndarray, mask: np.ndarray, fine_shape: tuple[int, int]) -> tuple[float, np.ndarray, dict]:
    """
    P = yᵀM_T y / yᵀM y, the kinetic-energy share of a closed-cavity mode in the elements flagged by
    mask (n1, n2) of the mechanical grid, over all of the solver's parts (the slab and its fixed
    membrane loads, e.g. the LN/Al of a transducer section), and dP/df_fine through the slab's
    (part 0) fractions: the explicit mass term (yₑᵀMₑyₑ([e ∈ T] − P)/yᵀMy) plus the eigenvector
    term by one reduced mechanical adjoint (K − λM) z = ∂P/∂y = 2(M_T − P M)y / yᵀMy (⊥ y, as P is
    scale invariant), −zₑᵀ(Kₑ − λMₑ)yₑ.
    """
    t0 = time.perf_counter()
    solver, Pr = cav.solver, cav.P
    K, M = (A.real.tocsr() for A in solver.matrices(np.zeros(2)))
    y = np.real(np.asarray(x_mech))
    My = M @ y
    den = float(y @ My)
    lam = float(y @ (K @ y)) / den
    MTy = np.zeros_like(y)
    for part, frac in zip(solver.parts, solver.part_fractions):
        ok = part.edof >= 0
        sel = mask.ravel()[part.cell]
        Ye = np.where(ok, y[np.maximum(part.edof, 0)], 0.0)
        contrib = (Ye @ part.Me.T) * (frac.ravel()[part.cell] * sel)[:, None]
        np.add.at(MTy, np.maximum(part.edof, 0)[ok], contrib[ok])
    P = float(y @ MTy) / den
    a = 2 * (MTy - P * My) / den
    if Pr is not None:
        solve = eigenvector_adjoint_solver((Pr.T @ K @ Pr).tocsr(), (Pr.T @ M @ Pr).tocsr(), lam, Pr.T @ y)
        z = Pr @ solve(Pr.T @ a)
    else:
        z = eigenvector_adjoint_solver(K, M, lam, y)(a)
    part = solver.parts[0]
    ok = part.edof >= 0
    gather = lambda v: np.where(ok, v[np.maximum(part.edof, 0)], 0.0)  # noqa: E731
    Ye = gather(y)
    kY, mY = Ye @ part.Ke.T, Ye @ part.Me.T
    yMy_e = (Ye * mY).sum(axis=1)
    sel = mask.ravel()[part.cell].astype(float)
    e = yMy_e * (sel - P) / den - (gather(z) * (kY - lam * mY)).sum(axis=1)
    n1, n2 = solver.shape
    d = np.bincount(part.cell, weights=e, minlength=n1 * n2).reshape(n1, n2)
    return P, block_mean_adjoint(d, fine_shape), dict(t=time.perf_counter() - t0)


def _footprint(xr, xi, f, g, dg, rb, eta, es, eb, u, v, absq, mask):
    """Share of the electric energy density u = Re e*·ε⁻¹e (as _log_volume) in the voxels flagged by mask (n1, n2, nz)."""
    e = _curl(xr, xi, u, v, absq)
    _, _, w, _, _, _ = _eps_tensor(f, g, dg, rb, eta, es, eb)
    dens = jnp.real(jnp.conj(e) * _apply(w, e)).sum(axis=0)
    return jnp.sum(dens * mask) / jnp.sum(dens)


_footprint_vg = jax.jit(jax.value_and_grad(_footprint, argnums=(0, 1, 2)))


def footprint_fraction(s: PWESolver, mask: np.ndarray, band: int = 0) -> float:
    """`footprint_gradient`'s F without the gradient."""
    x = s.last_vectors[:, band]
    oi = _optical_inputs(s)
    st = (jnp.asarray(s.u), jnp.asarray(s.v), jnp.asarray(s.absq))
    F, _ = _footprint_vg(jnp.asarray(x.real), jnp.asarray(x.imag), oi["f"], oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"], *st,
                         jnp.asarray(mask, float))
    return float(F)


def footprint_gradient(co: CavityOptics, s: PWESolver, mask: np.ndarray, fine_shape: tuple[int, int], band: int = 0, adjoint_tol: float = 1e-3,
                       adjoint_maxiter: int = 600) -> tuple[float, np.ndarray, dict]:
    """
    F = Σ_mask u / Σ u, the share of the optical mode's electric energy density in a region of the
    window's PWE grid (mask (n1, n2, nz), e.g. the column above an LN/Al transducer footprint: a
    proxy for the absorption/scattering loss there, 1/Q_LN ∝ F), and dF/df_fine on the whole fine
    raster: the explicit ε⁻¹ dependence by JAX plus the eigenvector part by one optical adjoint.
    F is scale invariant in x, so no normalisation or frequency term.
    """
    x = s.last_vectors[:, band]
    oi = _optical_inputs(s)
    st = (jnp.asarray(s.u), jnp.asarray(s.v), jnp.asarray(s.absq))
    args = (oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"]) + st + (jnp.asarray(mask, float),)
    F, (gr, gi, gf) = _footprint_vg(jnp.asarray(x.real), jnp.asarray(x.imag), oi["f"], *args)
    r = np.asarray(gr) + 1j * np.asarray(gi)
    w, info = s.adjoint_solve(band, r, tol=adjoint_tol, maxiter=adjoint_maxiter, jax=True)
    ew = _curl(jnp.asarray(w.real), jnp.asarray(w.imag), *st)
    ex = _curl(jnp.asarray(x.real), jnp.asarray(x.imag), *st)
    gtheta = _theta_bilinear_grad(oi["f"], ew, ex, oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"])
    d_opt = np.asarray(gf) - np.asarray(gtheta)
    win = co.win
    m1, m2 = fine_shape[0] // win.n1m, fine_shape[1] // win.n2m
    out = np.zeros(fine_shape)
    out[win.i0 * m1 : (win.i0 + win.n1w) * m1, win.j0 * m2 : (win.j0 + win.n2w) * m2] = block_mean_adjoint(d_opt, co.fine.shape)
    return float(F), out, info


def optical_loss_terms(s: PWESolver, mask: np.ndarray, c_lc: float, band: int = 0) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    `optical_loss_gradient`'s LC and F_w with the loss c_lc·LC + F_w's eigenvector gradient r (an
    adjoint_solve right-hand side, at the unit eigenvector) and explicit gradient in the optical fill:
    (LC, F_w, r, gf). Also the `extra` terms of cavity_g0_gradient.
    """
    x = s.last_vectors[:, band]
    x = x / np.linalg.norm(x)
    P = light_cone_mask(s, float(s._last_freqs[band]))
    lc = float(np.sum(np.abs(x[P]) ** 2))
    oi = _optical_inputs(s)
    st = (jnp.asarray(s.u), jnp.asarray(s.v), jnp.asarray(s.absq))
    args = (oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"]) + st + (jnp.asarray(mask, float),)
    F, (gr, gi, gf) = _footprint_vg(jnp.asarray(x.real), jnp.asarray(x.imag), oi["f"], *args)
    r = c_lc * 2 * (np.where(P, x, 0) - lc * x) + np.asarray(gr) + 1j * np.asarray(gi)
    return lc, float(F), r, np.asarray(gf)


def optical_loss_gradient(co: CavityOptics, s: PWESolver, mask: np.ndarray, c_lc: float, fine_shape: tuple[int, int], band: int = 0,
                          adjoint_tol: float = 1e-3, adjoint_maxiter: int = 600) -> tuple[float, float, np.ndarray, dict]:
    """
    A loss rate 1/Q = c_lc·LC + F_w (F_w: `footprint_fraction` with a weighted mask, e.g. Σ_c m_c / K_c
    over loss regions c with Q_c ≈ K_c / F_c) and d(1/Q)/df_fine by ONE optical adjoint: the
    eigenvector right-hand sides of `light_cone_gradient` and `footprint_gradient` add, so their
    weighted sum costs one MINRES solve instead of one each. Returns (LC, F_w, gradient, info).
    """
    lc, F, r, gf = optical_loss_terms(s, mask, c_lc, band)
    w, info = s.adjoint_solve(band, r, tol=adjoint_tol, maxiter=adjoint_maxiter, jax=True)
    oi = _optical_inputs(s)
    st = (jnp.asarray(s.u), jnp.asarray(s.v), jnp.asarray(s.absq))
    x = s.last_vectors[:, band] / np.linalg.norm(s.last_vectors[:, band])
    ew = _curl(jnp.asarray(w.real), jnp.asarray(w.imag), *st)
    ex = _curl(jnp.asarray(x.real), jnp.asarray(x.imag), *st)
    d_opt = gf - np.asarray(_theta_bilinear_grad(oi["f"], ew, ex, oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"]))
    win = co.win
    m1, m2 = fine_shape[0] // win.n1m, fine_shape[1] // win.n2m
    out = np.zeros(fine_shape)
    out[win.i0 * m1 : (win.i0 + win.n1w) * m1, win.j0 * m2 : (win.j0 + win.n2w) * m2] = block_mean_adjoint(d_opt, co.fine.shape)
    return lc, float(F), out, info


# --- radiation Q from the polarisation current ------------------------------------------------
#
# In the H form, ∇×∇×H − k0²H = ∇×J with J = (1 − ε⁻¹) ∇×H the polarisation current, confined to the
# slab. So a box plane wave's amplitude is H(G) = iG×J(G) / (G² − k0²): LC measures J off shell, times a
# denominator set by the box height. The radiated power needs only the amplitudes on the light cone,
# where the box plays no part. For a window of area A, in units ε0 = μ0 = c = 1 (ω = k0),
#     P = Σ_q k0 / (8 k_z A) T(q),   U = ∫ e*·ε⁻¹e d³r / (2 k0²),
# so Q = k0 U / P = 4 A Σ(e*·ε⁻¹e) / (k0² dV Σ_q T(q) / k_z) with Ĵ the voxel sum. The in-plane transform is
# oversampled (pad; (4, 4) is within 1% of (8, 8) on a TF13 cavity), and 1/k_z is averaged over each q cell
# (it is integrable at the cone edge), so the sum varies smoothly as the edge crosses the grid.
#
# T comes by reciprocity from the radiation modes of a uniform slab of ε̄ (Fermi's golden rule, as guided-mode
# expansion): a = ∫ J'·F, with J' = (ε − ε̄)E the current beyond the slab's own response and F the slab's field
# for a unit plane wave incident from the opposite direction. For the even pair of incident waves (z-even mode),
#     TE: a_s = C_s Σ_z cos(κz) ŝ·Ĵ',                        C_s = 2 k_z / (k_z cos(κt/2) − iκ sin(κt/2)),
#     TM: a_p = C_p Σ_z [cos(κz) q̂·Ĵ' + i(q/κ) sin(κz) Ĵ'_z],  C_p = 2 (k_z/k0) / (cos(κt/2) − i(k_z ε̄/κ) sin(κt/2)),
# with κ² = ε̄k0² − q² and T = (|a_s|² + |a_p|²) / 2 (validation/far_field_q_check.py: against the 1D wave equation).
# A membrane reflects grazing light (C → 0 with k_z), so this differs from J radiating into free space (ε̄ = 1),
# which gives most of its power to the cone edge and reads a Q 5–7× lower on TF13 cavities. Against COMSOL's total Q
# with fine outlines (research/cavity_comsol_optics.py --simplify 0.002, or less: the default 5 nm facets act as sidewall
# roughness and limit Q near 1e5) this form is within 1.3× at Q ≈ 7e4, and 4× high at 1e6, a 1/Q gap of ~3e-6 at both.
#
# Solve the mode with PWESolver(no_shell_thz=f_o): the plane waves in the cone within half a k_z spacing of
# the shell, ||q + G| − k0| < π/H, out of the basis. Their denominator above is small and set by the box
# height, so with them the mode hybridises with box modes: J picks up their polarisation, this Q does not
# converge with the solver's tolerance and collapses when f_o crosses one, and Jacobi-Davidson and MINRES stall.


def _cone_weights(k0: float, qx: np.ndarray, qy: np.ndarray, dqx: float, dqy: float, n_y: int = 8) -> np.ndarray:
    """Mean over each q cell of θ(k0 − |q|) / √(k0² − |q|²): exact in q_x, midpoint in q_y."""
    t = (np.arange(n_y) + 0.5) / n_y - 0.5
    out = np.zeros(len(qx))
    for ty in t:
        kap2 = k0**2 - (qy + ty * dqy) ** 2
        inside = kap2 > 0
        kap = np.sqrt(np.where(inside, kap2, 1.0))
        hi = np.clip((qx + dqx / 2) / kap, -1, 1)
        lo = np.clip((qx - dqx / 2) / kap, -1, 1)
        out += np.where(inside, np.arcsin(hi) - np.arcsin(lo), 0.0)
    return out / (n_y * dqx)


def _far_tables(s: PWESolver, k0: float, pad: tuple[int, int]) -> dict:
    """The oversampled in-plane q cells touching the cone (margin of a cell for the edge's move with k0)."""
    n1, n2, nz = s.shape
    L = s.cell.lattice.vectors
    if abs(L[0, 1]) > 1e-12 or abs(L[1, 0]) > 1e-12:
        raise ValueError("far-field Q needs a rectangular window")
    L1, L2 = L[0, 0], L[1, 1]
    N1 = n1 * pad[0]
    dqx, dqy = 2 * np.pi / (L1 * pad[0]), 2 * np.pi / (L2 * pad[1])
    qx_all = np.fft.fftfreq(N1, 1 / N1) * dqx
    m = int(np.ceil((k0 + dqy) / dqy))
    qy_all = np.arange(-m, m + 1) * dqy
    near_x, near_y = np.maximum(np.abs(qx_all) - dqx, 0), np.maximum(np.abs(qy_all) - dqy, 0)
    ix, iy = np.nonzero(near_x[:, None] ** 2 + near_y[None, :] ** 2 < (k0 + 2 * dqx) ** 2)
    dz = s.cell.height / nz
    z = np.arange(nz) * dz
    z = np.where(z > s.cell.height / 2, z - s.cell.height, z)
    sel = np.flatnonzero(s.cell.z_cover > 0)
    y = np.arange(n2) * L2 / n2
    return dict(ix=ix, iy=iy, qx=qx_all[ix], qy=qy_all[iy], dqx=dqx, dqy=dqy, sel=sel, z=z[sel], N1=N1, Ey=np.exp(-1j * np.outer(y, qy_all)),
                dV=abs(np.linalg.det(L)) * s.cell.height / (n1 * n2 * nz), A=L1 * L2, pad=pad, t=float(s.cell.z_cover.sum() * dz), cover=s.cell.z_cover[sel])


def _far_terms(Jq, qx, qy, z, k0, eps_bar=1.0, t=0.0):
    """T(q) of the block comment for Jq (3, nq, ns) on the z planes z, in a slab of ε̄ and thickness t (k_z → 0 beyond the cone's edge; ε̄ = 1: free space)."""
    q2 = qx**2 + qy**2
    eps = 1e-6 * k0
    kz = jnp.sqrt(jnp.maximum(k0**2 - q2, eps**2))
    kap = jnp.sqrt(jnp.maximum(eps_bar * k0**2 - q2, eps**2))
    q = jnp.sqrt(jnp.maximum(q2, eps**2))
    c, s = jnp.cos(kap * t / 2), jnp.sin(kap * t / 2)
    Cs = 2 * kz / (kz * c - 1j * kap * s)
    Cp = 2 * (kz / k0) / (c - 1j * (kz * eps_bar / kap) * s)
    cz, sz = jnp.cos(kap[:, None] * z), jnp.sin(kap[:, None] * z)
    a_s = Cs * (cz * (qx[:, None] * Jq[1] - qy[:, None] * Jq[0])).sum(axis=-1) / q
    a_p = Cp * ((cz * (qx[:, None] * Jq[0] + qy[:, None] * Jq[1])).sum(axis=-1) / q + 1j * (q / kap) * (sz * Jq[2]).sum(axis=-1))
    return (jnp.abs(a_s) ** 2 + jnp.abs(a_p) ** 2) / 2


def _far_current(e, w, sel, N1, Ey, ix, iy, eb):
    """Ĵ'(q, z) (3, nq, ns): (1 − ε̄ ε⁻¹)e on the slab's z planes (eb: ε̄ on them), transformed along y (explicit q_y) and x (padded FFT)."""
    J = e[:, :, :, sel] - eb * _apply(w, e)[:, :, :, sel]
    Jy = jnp.einsum("cijs,jm->cims", J, Ey)
    return jnp.fft.fft(Jy, n=N1, axis=1)[:, ix, iy]


def _far_lnq(xr, xi, f, g, dg, rb, eta, es, eb, u, v, absq, sel, N1, Ey, ix, iy, qx, qy, z, W, k0, c, eps_z, eps_bar, t):
    """ln Q (c: the prefactor 4 A / (k0² dV) · pad1 pad2; eps_z: ε̄ on the z planes, as `_far_tables`' cover) for fixed k0; the mode's scale cancels."""
    e = _curl(xr, xi, u, v, absq)
    _, _, w, _, _, _ = _eps_tensor(f, g, dg, rb, eta, es, eb)
    den = jnp.sum(jnp.real(jnp.conj(e) * _apply(w, e)))
    Jq = _far_current(e, w, sel, N1, Ey, ix, iy, eps_z)
    return jnp.log(c) + jnp.log(den) - jnp.log(jnp.sum(W * _far_terms(Jq, qx, qy, z, k0, eps_bar, t)))


_far_lnq_vg = jax.jit(jax.value_and_grad(_far_lnq, argnums=(0, 1, 2)), static_argnums=(13,))


def mean_slab_eps(s: PWESolver) -> float:
    """The window's mean ε over the slab: a default ε̄ for the far-field Q. A design run should hold one value, so that Q is a smooth function of the design."""
    c = s.cell
    k = int(np.argmax(c.z_cover))
    return c.eps_background + (c.eps_slab - c.eps_background) * float(np.mean(c.fill[:, :, k])) / float(c.z_cover[k])


def _far_setup(s: PWESolver, x, band: int, pad: tuple[int, int], eps_bar: float | None):
    f_thz = float(s._last_freqs[band])
    k0 = 2 * np.pi * f_thz / C_UM_PER_PS
    tab = _far_tables(s, k0, pad)
    tab["eps_bar"] = mean_slab_eps(s) if eps_bar is None else float(eps_bar)
    tab["eps_z"] = 1 + (tab["eps_bar"] - 1) * tab["cover"]
    oi = _optical_inputs(s)
    st = (jnp.asarray(s.u), jnp.asarray(s.v), jnp.asarray(s.absq))
    e = _curl(jnp.asarray(x.real), jnp.asarray(x.imag), *st)
    _, _, w, _, _, _ = _eps_tensor(oi["f"], oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"])
    den = float(jnp.sum(jnp.real(jnp.conj(e) * _apply(w, e))))
    Jq = _far_current(e, w, jnp.asarray(tab["sel"]), tab["N1"], jnp.asarray(tab["Ey"]), jnp.asarray(tab["ix"]), jnp.asarray(tab["iy"]), jnp.asarray(tab["eps_z"]))

    def lnq_at(kk: float) -> float:
        """ln Q at wavenumber kk for the fixed current: the cone edge, the slab's modes and the prefactor move."""
        W = _cone_weights(kk, tab["qx"], tab["qy"], tab["dqx"], tab["dqy"])
        T = _far_terms(Jq, jnp.asarray(tab["qx"]), jnp.asarray(tab["qy"]), jnp.asarray(tab["z"]), kk, tab["eps_bar"], tab["t"])
        c = 4 * tab["A"] / (kk**2 * tab["dV"]) * pad[0] * pad[1]
        return float(np.log(c) + np.log(den) - np.log(float(jnp.sum(jnp.asarray(W) * T))))

    return k0, tab, oi, st, lnq_at


def far_field_q(s: PWESolver, band: int = 0, pad: tuple[int, int] = (4, 4), x: np.ndarray | None = None, eps_bar: float | None = None) -> dict:
    """
    Radiation Q of a window mode (the block comment above), z-even. Q and its k0 sensitivity dlnQ/dk0
    at the fixed current (per µm⁻¹, the cone's move with the frequency). x: another field than the
    band's. eps_bar: ε̄ of the slab's radiation modes (default `mean_slab_eps`; 1: free space).
    """
    x = s.last_vectors[:, band] if x is None else x
    k0, tab, oi, st, lnq_at = _far_setup(s, x, band, pad, eps_bar)
    h = tab["dqx"] / 2
    lnq = lnq_at(k0)
    return dict(Q=float(np.exp(lnq)), lnq=lnq, k0=k0, dlnq_dk0=(lnq_at(k0 + h) - lnq_at(k0 - h)) / (2 * h))


def far_field_q_gradient(co: CavityOptics, s: PWESolver, fine_shape: tuple[int, int], band: int = 0, pad: tuple[int, int] = (4, 4), adjoint_tol: float = 1e-3,
                         adjoint_maxiter: int = 600, eps_bar: float | None = None) -> tuple[float, np.ndarray, dict]:
    """`far_field_q`'s Q and d ln Q / df_fine on the whole fine raster, by one optical adjoint; `far_field_q_terms` gives the terms to merge it with g0's."""
    q, r, gf, dfo_term = far_field_q_terms(s, band, pad, eps_bar)
    w, info = s.adjoint_solve(band, r, tol=adjoint_tol, maxiter=adjoint_maxiter, jax=True)
    oi = _optical_inputs(s)
    st = (jnp.asarray(s.u), jnp.asarray(s.v), jnp.asarray(s.absq))
    x = s.last_vectors[:, band]
    ew = _curl(jnp.asarray(w.real), jnp.asarray(w.imag), *st)
    ex = _curl(jnp.asarray(x.real), jnp.asarray(x.imag), *st)
    d_opt = np.asarray(gf) - np.asarray(_theta_bilinear_grad(oi["f"], ew, ex, oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"])) + dfo_term
    win = co.win
    m1, m2 = fine_shape[0] // win.n1m, fine_shape[1] // win.n2m
    out = np.zeros(fine_shape)
    out[win.i0 * m1 : (win.i0 + win.n1w) * m1, win.j0 * m2 : (win.j0 + win.n2w) * m2] = block_mean_adjoint(d_opt, co.fine.shape)
    return q, out, info


def far_field_q_terms(s: PWESolver, band: int = 0, pad: tuple[int, int] = (4, 4), eps_bar: float | None = None) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    ln Q's gradient pieces at the unit eigenvector: (Q, r, gf, dfo_term) with r the eigenvector right-hand side of
    an optical adjoint, gf the explicit gradient in the optical fill and dfo_term = (∂ln Q/∂k0)(dk0/df_o) the cone's
    move with the frequency (Hellmann–Feynman), all on the window's optical grid.
    """
    x = s.last_vectors[:, band]
    x = x / np.linalg.norm(x)
    k0, tab, oi, st, lnq_at = _far_setup(s, x, band, pad, eps_bar)
    h = tab["dqx"] / 2
    dlnq = (lnq_at(k0 + h) - lnq_at(k0 - h)) / (2 * h)
    W = _cone_weights(k0, tab["qx"], tab["qy"], tab["dqx"], tab["dqy"])
    c = 4 * tab["A"] / (k0**2 * tab["dV"]) * pad[0] * pad[1]
    lnq, (gr, gi, gf) = _far_lnq_vg(jnp.asarray(x.real), jnp.asarray(x.imag), oi["f"], oi["g"], oi["dg"], oi["rb"], oi["eta"], oi["es"], oi["eb"], *st,
                                    jnp.asarray(tab["sel"]), tab["N1"], jnp.asarray(tab["Ey"]), jnp.asarray(tab["ix"]), jnp.asarray(tab["iy"]),
                                    jnp.asarray(tab["qx"]), jnp.asarray(tab["qy"]), jnp.asarray(tab["z"]), jnp.asarray(W), k0, c, jnp.asarray(tab["eps_z"]), tab["eps_bar"], tab["t"])
    r = np.asarray(gr) + 1j * np.asarray(gi)
    dfo_term = dlnq * (2 * np.pi / C_UM_PER_PS) * s.sensitivities([band])[0]
    return float(np.exp(lnq)), r, np.asarray(gf), dfo_term


# --- driven mechanical response --------------------------------------------------------------
#
# g0 of one mechanical eigenmode is undefined at a crossing, and a design step can hand its coupling to a
# neighbour. So drive the mechanics with the optical force instead: F = ∂L/∂û (L, linear in û: optomech_functional),
#     (K − zM) u = F,   z = (2π (f + iγ) / Ω_unit)² = σ + iΓ,
# and with M-normalised modes y_k of K y = λ M y, u = Σ y_k (y_kᵀF) / (λ_k − z), so
#     G² = C Γ Im(Fᵀu) = Σ_k g_k² (Ω_k / Ω) Γ² / ((λ_k − σ)² + Γ²),   C = ħ / (2 m_unit Ω),
# the g_k² of all modes under a unit-peak Lorentzian of half width γ (Hz) at f. A single mode at f gives G = |g0|.
# G² has no mode to track: a crossing moves g² between two modes and leaves their sum in place, and narrowing γ
# puts the coupling into one mode. Its gradient is one mechanical solve and one optical adjoint, the same as g0's:
#     dG² = C Γ [2 dFᵀ Im(u) − Im uᵀ(dK − z dM) u],
# dFᵀv at fixed v = Im(u) being optomech_functional's explicit and optical gradients with v in place of a mode.


@dataclass
class DrivenResponse:
    """G² (rad/s)² at (f, γ), and with gradients: d_J on the fine raster of J = a(G²) + E (see `cavity_driven_response`), d_fo, d_fm."""

    G2: float
    f: float
    gamma: float
    F: np.ndarray  # the optical force on the solver's DOFs
    L_v: float | None = None
    d_J: np.ndarray | None = None
    d_J_mech: np.ndarray | None = None  # the part of d_J through the mechanical element fractions (the rest is the window raster's)
    d_fo: np.ndarray | None = None
    w: np.ndarray | None = None  # the optical adjoint: first order in the eigenvector's residual, J(exact mode) = J − Re(wᴴ(Θx − θx))
    info: dict = field(default_factory=dict)


def _mode_with(mode: MechMode, solver, v: np.ndarray) -> MechMode:
    """`mode` with its coefficients replaced by v on the solver's DOFs (optomech_functional is linear in them)."""
    n1, n2, dpn = mode.coef.shape
    full = np.zeros(n1 * n2 * dpn)
    full[solver.dofs] = v
    return replace(mode, coef=full.reshape(n1, n2, dpn))


def optical_force(co: CavityOptics, s: PWESolver, cav: MechCavity, band: int = 0) -> np.ndarray:
    """F = ∂L/∂û on the mechanical solver's DOFs: L[v] = Fᵀv for any v. Depends on the optical mode and the rasters, not on co.mode."""
    core = optomech_functional(s, band, co.mode, co.fine, cav.fraction, co.p, strain_supersample=co.ratio, adjoint=False, to_mode=co.win.to_mode)
    return np.asarray(core["gcoef"]).reshape(-1)[cav.solver.dofs]


def mode_couplings(F: np.ndarray, cav: MechCavity, freqs: np.ndarray, X: np.ndarray) -> np.ndarray:
    """|g_k| (rad/s) of mechanical modes (freqs Hz, eigenvectors X on the solver's DOFs) from the force, without any further solve."""
    _, M = cav.solver.matrices(np.zeros(2))
    X = np.real(X)
    yMy = np.einsum("ib,ib->b", X, M.real @ X)
    return np.abs(F @ X) * np.sqrt(HBAR / (2 * yMy * MASS_UNIT * 2 * np.pi * np.asarray(freqs)))


def lorentz_g2(g: np.ndarray, freqs: np.ndarray, f: float, gamma: float) -> float:
    """G² from a table of modes (their g_k, rad/s, and frequencies, Hz): the sum in the block comment, in frequency, near f."""
    d = (np.asarray(freqs) ** 2 - (f**2 - gamma**2)) / (2 * f * gamma)
    return float(np.sum(np.asarray(g) ** 2 * (np.asarray(freqs) / f) / (1 + d**2)))


def cavity_driven_response(co: CavityOptics, s: PWESolver, cav: MechCavity, f: float, gamma: float, fine_shape: tuple[int, int], band: int = 0, F: np.ndarray | None = None,
                           adjoint: bool = True, adjoint_tol: float = 1e-3, adjoint_maxiter: int = 600, extra=None, w0: np.ndarray | None = None) -> DrivenResponse:
    """
    G² at f ± γ (Hz) on a closed cavity. co: set_design done, co.mode any mode of the same solver (used for its grid); s: the solved window PWESolver. adjoint: also d_J,
    with J = G² or, given extra: callable G² → (a, r_e, gf_e, dfo_e), J = a(G²) + E with a the local slope dJ/dG², E a functional of the optical mode (eigenvector right-hand
    side r_e, explicit optical-fill gradient gf_e, a term dfo_e added to the optical-grid gradient), all through one optical adjoint.
    w0: that adjoint's start, the w of a previous design.
    """
    t0 = time.perf_counter()
    solver, P = cav.solver, cav.P
    Kc, Mc = solver.matrices(np.zeros(2))
    K, M = Kc.real.tocsr(), Mc.real.tocsr()
    F = optical_force(co, s, cav, band) if F is None else F
    info = dict(t_force=time.perf_counter() - t0)
    t1 = time.perf_counter()
    z = (2 * np.pi * (f + 1j * gamma) / OMEGA_UNIT) ** 2
    Gam = float(z.imag)
    if P is not None:
        solve = factorize_indefinite((P.T @ K @ P - z * (P.T @ M @ P)).tocsc())
        u = P @ solve((P.T @ F).astype(complex))
        info["odd_force"] = float(np.linalg.norm(F - P @ (P.T @ F)) / np.linalg.norm(F))
    else:
        solve = factorize_indefinite((K - z * M).tocsc())
        u = solve(F.astype(complex))
    C = HBAR / (2 * MASS_UNIT * 2 * np.pi * f)
    v = u.imag
    G2 = C * Gam * float(F @ v)
    info["t_response"] = time.perf_counter() - t1
    if not adjoint:
        return DrivenResponse(G2, f, gamma, F, info=info)

    a, r_e, gf_e, dfo_e = extra(G2) if extra is not None else (1.0, 0.0, 0.0, 0.0)
    mode_v = _mode_with(co.mode, solver, v)
    t2 = time.perf_counter()
    core = optomech_functional(s, band, mode_v, co.fine, cav.fraction, co.p, strain_supersample=co.ratio, minres_tol=adjoint_tol, minres_maxiter=adjoint_maxiter,
                               to_mode=co.win.to_mode, minres_jax=True, rhs_extra=lambda _L: (a * 2 * C * Gam, r_e, gf_e), minres_x0=w0)
    info.update(core["info"], t_core=time.perf_counter() - t2)
    part = solver.parts[0]
    ok = part.edof >= 0
    Ue = np.where(ok, u[np.maximum(part.edof, 0)], 0.0)
    n1, n2 = solver.shape
    form = np.imag((Ue * (Ue @ part.Ke.T - z * (Ue @ part.Me.T))).sum(axis=1))
    d_mech = a * C * Gam * (2 * core["gfm"] - np.bincount(part.cell, weights=form, minlength=n1 * n2).reshape(n1, n2))
    w = co.win
    m1, m2 = fine_shape[0] // w.n1m, fine_shape[1] // w.n2m
    sl = (slice(w.i0 * m1, (w.i0 + w.n1w) * m1), slice(w.j0 * m2, (w.j0 + w.n2w) * m2))
    d_J_mech = block_mean_adjoint(d_mech, fine_shape)
    d_J = d_J_mech.copy()
    d_J[sl] += a * 2 * C * Gam * core["gfine"] + block_mean_adjoint(core["d_opt"] + dfo_e, co.fine.shape)
    d_fo = np.zeros(fine_shape)
    d_fo[sl] = block_mean_adjoint(s.sensitivities([band])[0], co.fine.shape)
    info["t_total"] = time.perf_counter() - t0
    return DrivenResponse(G2, f, gamma, F, L_v=core["L"], d_J=d_J, d_J_mech=d_J_mech, d_fo=d_fo, w=core["w"], info=info)


__all__ = ["CavityGradient", "DrivenResponse", "cavity_driven_response", "cavity_g0_gradient", "far_field_q", "far_field_q_gradient", "far_field_q_terms", "footprint_fraction",
           "footprint_gradient", "light_cone_gradient", "lorentz_g2", "mean_slab_eps", "mech_freq_gradient", "mech_share", "mech_share_gradient", "mode_couplings", "mode_volume", "mode_volume_gradient",
           "optical_force", "optical_loss_gradient", "optical_loss_terms"]
