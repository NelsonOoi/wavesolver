"""
TE-like bands of an unpatterned 220 nm Si slab against the analytic TE0 slab mode.

With no holes the lowest even band at in-plane k (below the K point) is the TE0 guided mode
with propagation constant β = |k|, which solves κ tan(κ t/2) = γ with κ² = n² k0² - β² and
γ² = β² - k0². This checks the z-discretisation, the interface averaging and the parity
projection.

    pixi run python validation/optics_slab_analytic.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wavesolver.geometry import HexLattice  # noqa: E402
from wavesolver.optics_pwe import C_UM_PER_PS, PWESolver, slab_cell  # noqa: E402

N_SI = 3.48
T = 0.22


def te0_thz(beta: float) -> float:
    def f(k0):
        kappa = np.sqrt(N_SI**2 * k0**2 - beta**2)
        gamma = np.sqrt(beta**2 - k0**2)
        return kappa * np.tan(kappa * T / 2) - gamma

    lo = beta / N_SI * (1 + 1e-12)
    hi = min(beta, np.sqrt((np.pi / T) ** 2 + beta**2) / N_SI) * (1 - 1e-9)
    return brentq(f, lo, hi) * C_UM_PER_PS / (2 * np.pi)


def main() -> None:
    lattice = HexLattice(0.448)
    betas = np.array([4.0, 6.0, 8.0, 9.0])
    ks = np.column_stack([betas, np.zeros_like(betas)])
    exact = np.array([te0_thz(b) for b in betas])
    print("β (1/µm):", betas, " exact TE0 (THz):", np.round(exact, 3))
    for height, nz in ((2.0, 64), (2.0, 128), (3.0, 192)):
        cell = slab_cell(lattice, np.ones((8, 8)), T, height, nz, N_SI**2)
        solver = PWESolver(cell, parity="even")
        t0 = time.perf_counter()
        f = solver.bands(ks, n_bands=1)[:, 0]
        dt = time.perf_counter() - t0
        rel = (f - exact) / exact
        print(f"Lz={height} nz={nz:3d} (dz={1000 * height / nz:4.1f} nm): err {np.round(100 * rel, 3)} %  ({dt:.1f} s)")


if __name__ == "__main__":
    main()
