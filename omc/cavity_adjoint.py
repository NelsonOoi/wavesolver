"""
Adjoint gradient of a finite cavity's g0 (and of f_o, f_m) with respect to the moving-boundary
raster of the whole mechanical domain: the optics in a window of the mechanical grid
(omc.cavity_optics), the mechanics a σ_y-reduced Γ-point cavity mode (omc.cavity).

The method is omc.coupling_adjoint.g0_gradient's, with three changes:
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
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .cavity import MechCavity
from .cavity_optics import CavityOptics
from .coupling import HBAR, MASS_UNIT, mech_mode
from .coupling_adjoint import _apply, _curl, _eps_tensor, _optical_inputs, _theta_bilinear_grad, block_mean_adjoint, omega_rel, optomech_functional
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
                       band: int = 0, adjoint_tol: float = 1e-3, adjoint_maxiter: int = 600) -> CavityGradient:
    """
    co: set_design done on the same design and co.mode the cavity mode (mech_freq Hz, eigenvector
    x_mech on cav.solver's DOFs, real at Γ, closed cavity: no absorber); s: the solved window
    PWESolver (co.solve(...).solver); fine_shape: the whole domain's fine raster.
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

    core = optomech_functional(s, band, mode, co.fine, cav.fraction, co.p, strain_supersample=co.ratio, minres_tol=adjoint_tol,
                               minres_maxiter=adjoint_maxiter, to_mode=co.win.to_mode, minres_jax=True)
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


__all__ = ["CavityGradient", "cavity_g0_gradient", "footprint_fraction", "footprint_gradient", "light_cone_gradient", "mech_freq_gradient", "mech_share",
           "mech_share_gradient", "mode_volume", "mode_volume_gradient"]
