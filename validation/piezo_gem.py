"""
Electromechanical coupling g_em of an LN IDT to a microwave resonator, from the piezo FEM.

`PiezoSolver.motional` gives each short-circuit mode's BVD motional capacitance
C_m = α_m²/ω_m² (α_m the charge the M-normalised mode induces on the shorted terminal),
the clamped C0, and g_em = (ω_m/2)√(C_m ω_m Z_r) with Z_r = 700 Ω, plus its adjoint gradient.

Checks:
1. BVD: Y(ω) from driven solves near the strongest mode, fitted with
   Y/(iω) = a + b(ω² - ω_m²) + C_fit ω_m²/(ω_m² - ω²) (ω_m from the eigenmode); C_fit vs C_m.
   And the full modal sum C0 + Σ_m C_m ω_m²/(ω_m² - ω²) over the computed modes vs Y/(iω)
   (the difference is the smooth contribution of the modes not computed).
2. FD check of dg_em/dθ and dg_em/dfraction (Si and both LN sub-layers) on a coarse cell.
3. g_em at 4–5 GHz for the periodic IDT period of validation/piezo_angle.py (250 nm X-cut LN
   on 220 nm cubic Si, 2a = 520 nm, aperture = 600 nm beam) at the design default x_cut(45)
   and at the SH0 optimum, per period and for N periods (C_m ∝ N, g_em ∝ √N for a uniform
   IDT mode), with γ_e into 50 Ω.

    pixi run python validation/piezo_gem.py
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

from piezo_angle import MESH, idt_cell  # noqa: E402

OUT = ROOT / "validation/out"
Z_R = 700.0
BAND = (3.0e9, 7.0e9)


def strongest(mb, band=BAND):
    sel = np.flatnonzero((mb.freqs > band[0]) & (mb.freqs < band[1]))
    return sel[np.argmax(mb.C_m[sel])]


def bvd_check(sol, mb, m):
    fm = mb.freqs[m]
    df = np.array([-0.04, -0.03, -0.02, -0.012, -0.006, 0.006, 0.012, 0.02, 0.03, 0.04])
    # stay clear of neighbouring modes
    others = np.delete(mb.freqs, m)
    df = df[[np.min(np.abs(fm * (1 + d) - others)) > 0.004 * fm for d in df]]
    freqs = fm * (1 + df)
    t0 = time.perf_counter()
    Y = sol.admittance(freqs, ["hot"])[:, 0, 0]
    dt = time.perf_counter() - t0
    w, wm = 2 * np.pi * freqs, 2 * np.pi * fm
    y = (Y / (1j * w)).real
    A = np.column_stack([np.ones_like(w), (w**2 - wm**2) / wm**2, wm**2 / (wm**2 - w**2)])
    coef, *_ = np.linalg.lstsq(A / 1e-18, y / 1e-18, rcond=None)  # aF scale
    fit_res = np.abs(A @ coef - y).max() / np.abs(y).max()
    wn = 2 * np.pi * np.maximum(mb.freqs, 1.0)
    modal = mb.C0 + np.sum(mb.C_m[None, :] * wn[None, :] ** 2 / (wn[None, :] ** 2 - w[:, None] ** 2), axis=1)
    print(f"  BVD fit over {len(freqs)} driven solves ({dt:.1f} s): C_fit = {coef[2] * 1e18:.5f} aF vs modal C_m = {mb.C_m[m] * 1e18:.5f} aF "
          f"(rel. diff {abs(coef[2] / mb.C_m[m] - 1):.1e}; fit residual {fit_res:.1e})")
    print(f"  modal sum ({len(mb.freqs)} modes) vs Y/(iω): background a = {coef[0] * 1e18:.3f} aF vs C0 + other modes; "
          f"max |Y/(iω) - modal sum| = {np.abs(modal - y).max() * 1e18:.3f} aF (C0 = {mb.C0 * 1e18:.2f} aF), "
          f"spread over the band {np.ptp(modal - y) * 1e18:.4f} aF (smooth truncation)")
    return abs(coef[2] / mb.C_m[m] - 1)


def fd_check():
    print("FD check of dg_em (coarse cell, θ = 80°):")
    sol = idt_cell(80.0, 4, 8, 2, 2, 2, 1)
    mb0 = sol.motional("hot", 10, 3.0e9)
    m = strongest(mb0, (3e9, 9e9))
    mb = sol.motional("hot", 10, 3.0e9, gradient=[m])
    f0 = mb.freqs[m]

    def g_of():
        r = sol.motional("hot", 10, 3.0e9)
        return r.g_em[np.argmin(np.abs(r.freqs - f0))]

    worst = 0.0
    h = 1e-3
    sol.set_angle(80.0 + h)
    gp = g_of()
    sol.set_angle(80.0 - h)
    gm = g_of()
    sol.set_angle(80.0)
    fd = (gp - gm) / (2 * h)
    err = abs(mb.dg_angle[m] / fd - 1)
    worst = max(worst, err)
    print(f"  mode {f0 / 1e9:.4f} GHz, g_em/2π = {mb.g_em[m] / 2 / np.pi / 1e3:.2f} kHz per period: dg/dθ analytic {mb.dg_angle[m]:.6e}, FD {fd:.6e} rad/s/° (rel {err:.1e})")
    rng = np.random.default_rng(3)
    for l in (0, 1, 2):
        d = rng.uniform(-1, 1, sol.shape) * sol.parts[l].solid.reshape(sol.shape)
        base = sol.parts[l].f.reshape(sol.shape).copy()
        hf = 1e-4
        sol.set_fraction(l, base + hf * d)
        gp = g_of()
        sol.set_fraction(l, base - hf * d)
        gm = g_of()
        sol.set_fraction(l, base)
        fd = (gp - gm) / (2 * hf)
        an = np.sum(mb.dg_frac[l, m] * d)
        err = abs(an / fd - 1)
        worst = max(worst, err)
        print(f"  layer {l}: dg/dfrac·δ analytic {an:.6e}, FD {fd:.6e} (rel {err:.1e})")
    return worst


def main() -> None:
    OUT.mkdir(exist_ok=True)
    worst_fd = fd_check()
    rows = []
    for theta, label in ((45.0, "x_cut(45) design default"), (93.95, "SH0 k_eff² optimum, validation/piezo_angle.py")):
        sol = idt_cell(theta, *MESH)
        t0 = time.perf_counter()
        mb = sol.motional("hot", 24, 2.5e9)
        m = strongest(mb)
        mb = sol.motional("hot", 24, 2.5e9, gradient=[m])
        dt = time.perf_counter() - t0
        g = mb.g_em[m] / (2 * np.pi)
        ge = mb.gamma_e()[m] / (2 * np.pi)
        print(f"θ = {theta}° ({label}), {sol.ndof} dof, motional + gradient {dt:.1f} s:")
        print(f"  strongest mode {mb.freqs[m] / 1e9:.4f} GHz: C_m = {mb.C_m[m] * 1e18:.3f} aF, C0 = {mb.C0 * 1e18:.1f} aF, "
              f"C_m/C0 = {mb.C_m[m] / mb.C0:.4f}, L_m = {mb.L_m[m]:.3e} H")
        print(f"  per period: g_em/2π = {g / 1e3:.1f} kHz (Z_r = {Z_R:.0f} Ω), γ_e/2π into 50 Ω = {ge:.3g} Hz; "
              f"dg/dθ = {mb.dg_angle[m] / 2 / np.pi / 1e3:+.3f} kHz/°")
        for N in (10, 30, 100):
            print(f"  N = {N:3d} periods ({N * 0.52:.1f} µm): g_em/2π ≈ {g * np.sqrt(N) / 1e6:.3f} MHz, C_m ≈ {N * mb.C_m[m] * 1e15:.3f} fF, C0 ≈ {N * mb.C0 * 1e15:.2f} fF")
        if theta == 45.0:
            bvd_err = bvd_check(sol, mb, m)
        rows.append((theta, mb.freqs[m], mb.C_m[m], mb.C0, mb.g_em[m], mb.dg_angle[m]))
    # θ maximising g_em of the followed mode: secant on dg/dθ = 0 from the k² optimum
    sol = idt_cell(93.95, *MESH)
    th, fm = 93.95, rows[-1][1]
    g_prev = th_prev = None
    for it in range(10):
        mb = sol.motional("hot", 24, 2.5e9)
        m = int(np.argmin(np.abs(mb.freqs - fm)))
        mb = sol.motional("hot", 24, 2.5e9, gradient=[m])
        fm, gm, dg = mb.freqs[m], mb.g_em[m], mb.dg_angle[m]
        print(f"  g_em(θ) opt it {it}: θ = {th:.4f}°  g_em/2π = {gm / 2 / np.pi / 1e6:.4f} MHz per period  dg/dθ = {dg / 2 / np.pi / 1e3:+.3e} kHz/°  f = {fm / 1e9:.4f} GHz", flush=True)
        step = 2.0 * np.sign(dg) if th_prev is None else -dg * (th - th_prev) / (dg - g_prev)
        step = float(np.clip(step, -5, 5))
        if abs(step) < 1e-3:
            break
        th_prev, g_prev = th, dg
        th += step
        sol.set_angle(th)
    print(f"max g_em per period at θ = {th:.2f}°: g_em/2π = {gm / 2 / np.pi / 1e6:.3f} MHz at {fm / 1e9:.4f} GHz")
    print(f"worst FD rel. error of dg_em: {worst_fd:.1e}; BVD C_m rel. diff: {bvd_err:.1e}")
    np.savez(OUT / "piezo_gem.npz", rows=np.array(rows), Z_r=Z_R)


if __name__ == "__main__":
    main()
