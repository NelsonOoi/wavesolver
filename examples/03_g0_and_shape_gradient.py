"""
Optomechanical coupling g0 on one shared geometry, and its adjoint gradient pushed back to shape
parameters.

The tripod is a differentiable ParametricCell (arm length r and width w free). One fine raster
f_fine of it sets the moving-boundary term; its block means are the optical fill (N × N) and the
mechanical fractions (2N × 2N), so both solvers see the same pattern. The TE-like band at M and
the mechanical Γ mode with the largest |g0| are paired. coupling_adjoint.g0_gradient gives dg0 with
respect to all three rasters from one optical and one mechanical adjoint solve;
fine_total_gradient chains them onto f_fine, and fraction_vjp onto (r, w). The (r, w) gradient is
checked against a central difference that re-solves both eigenproblems.

    pixi run python examples/03_g0_and_shape_gradient.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omc.coupling import block_mean, g0, mech_mode, optical_mode  # noqa: E402
from omc.coupling_adjoint import fine_total_gradient, g0_gradient  # noqa: E402
from omc.geometry import HexLattice  # noqa: E402
from omc.materials import SI, SI_P  # noqa: E402
from omc.mech_bloch import BlochSolver  # noqa: E402
from omc.optics_pwe import PWESolver, slab_cell  # noqa: E402
from omc.parametrize import ParametricCell, TripodHole  # noqa: E402

A, T, N_SI = 0.448, 0.22, 3.48
N, NZ, M = 24, 64, 8  # optical grid, z samples, moving-boundary supersampling (fine grid N·M)
K_OPT = np.pi / A * np.array([1.0, 1 / np.sqrt(3)])  # M point


def solve(fine, band=None, mode=None):
    """g0 for a fine raster; picks the best-coupled pair unless (band, mode) are given."""
    fo, fm = block_mean(fine, N, N), block_mean(fine, 2 * N, 2 * N)
    so = PWESolver(slab_cell(HexLattice(A), fo, T, 1.5, NZ, N_SI**2), parity="even")
    so.solve(K_OPT, 4, tol=1e-10, maxiter=800)
    ms = BlochSolver(HexLattice(A), fm, SI, model="slab_p", thickness=T, parity="even", min_fraction=1e-9)
    fmech, X = ms.solve(np.zeros(2), 20, vectors=True)
    pairs = [(band, mode)] if band is not None else [(b, j) for b in range(4) for j in range(3, 20)]
    best = max(
        ((abs(g0(optical_mode(so, b), mech_mode(ms, np.zeros(2), fmech[j], X[:, j], fraction=fm), fine).g0), b, j) for b, j in pairs)
    )
    _, b, j = best
    y = (X[:, j] * np.exp(-0.5j * np.angle(np.sum(X[:, j] ** 2)))).real  # real Γ mode, fixed sign convention
    return so, ms, fmech, y, fm, b, j


def main() -> None:
    cell = ParametricCell(HexLattice(A), TripodHole(), free=("r", "w"))
    theta = cell.theta0()
    fine = cell.fraction(theta, N * M, supersample=2)

    t0 = time.perf_counter()
    so, ms, fmech, y, fm, b, j = solve(fine)
    t_fwd = time.perf_counter() - t0
    g0_gradient(so, b, ms, fmech[j], y.astype(complex), fine, fm, p=SI_P, mech_freqs=fmech)  # JAX compilation
    t0 = time.perf_counter()
    r = g0_gradient(so, b, ms, fmech[j], y.astype(complex), fine, fm, p=SI_P, mech_freqs=fmech)
    t_adj = time.perf_counter() - t0
    print(f"optical band {b} at M ({r.opt_freq_thz:.2f} THz) × mechanical Γ mode {j} ({r.mech_freq / 1e9:.3f} GHz)")
    print(f"g0/2π = {r.g0 / 2 / np.pi / 1e3:+.1f} kHz per cell (moving boundary {r.g_mb / 2 / np.pi / 1e3:+.1f}, "
          f"photoelastic {r.g_pe / 2 / np.pi / 1e3:+.1f}), m_eff {r.m_eff:.2e} kg")
    print(f"timing: forward (both solves, pair search) {t_fwd:.1f} s, g0 + full gradient {t_adj:.1f} s")

    # Chain to the shape parameters: dg0/dθ = (∂f_fine/∂θ)ᵀ dg0/df_fine.
    dg_dtheta = cell.fraction_vjp(theta, fine_total_gradient(r), N * M, supersample=2)
    sign = np.sign(r.g0)
    for i, name in enumerate(cell.free):
        h = 1.5e-5
        g = []
        for s in (1, -1):
            th = theta.copy()
            th[i] += s * h
            so_, ms_, fmech_, y_, fm_, _, _ = solve(cell.fraction(th, N * M, supersample=2), b, j)
            g.append(abs(g0(optical_mode(so_, b), mech_mode(ms_, np.zeros(2), fmech_[j], y_.astype(complex), fraction=fm_),
                            cell.fraction(th, N * M, supersample=2), p=SI_P).g0))
        fd = sign * (g[0] - g[1]) / (2 * h)
        print(f"dg0/d{name}: adjoint {dg_dtheta[i]:.6e}, FD {fd:.6e} rad/s/µm, rel diff {abs(dg_dtheta[i] - fd) / abs(fd):.1e}")


if __name__ == "__main__":
    main()
