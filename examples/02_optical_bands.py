"""
TE-like optical bands of a patterned Si membrane by plane-wave expansion, and their gradient.

The same tripod cell as 01 (a = 448 nm, 220 nm Si, n = 3.48) in a 1.5 µm tall supercell. Bands
on a short Γ-M-K-Γ path, the TE-like gap below the light line, then ∂f/∂fill of band 0 at M from
PWESolver.sensitivities, checked against a central difference along a random edge direction.

    pixi run python examples/02_optical_bands.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omc.geometry import HexLattice, blade_hole, solid_fraction  # noqa: E402
from omc.optics_pwe import PWESolver, light_line_thz, slab_cell  # noqa: E402

A, T, N_SI = 0.448, 0.22, 3.48
N, NZ, HEIGHT = 24, 64, 1.5  # in-plane grid, z samples, supercell height (µm)


def main() -> None:
    lattice = HexLattice(A)
    fill = solid_fraction(lattice, [blade_hole(0.172, 0.093, 0.025)], N)
    ks, corners = lattice.gamma_m_k_path(5)

    solver = PWESolver(slab_cell(lattice, fill, T, HEIGHT, NZ, N_SI**2), parity="even")
    t0 = time.perf_counter()
    f = solver.bands(ks, n_bands=4)  # (n_k, n_bands), THz
    print(f"{len(ks)} k-points, {(time.perf_counter() - t0) / len(ks):.1f} s per k-point (first includes JAX compilation)")

    light = light_line_thz(ks)
    below = f < light[:, None]
    top = np.max(np.where(below[:, 0], f[:, 0], -np.inf))
    bottom = np.min(np.where(below[:, 1], f[:, 1], np.inf))
    print(f"TE-like gap between bands 0 and 1, below the light line: {top:.1f}-{bottom:.1f} THz")

    # Gradient of band 0 at M with respect to the in-plane fill.
    k = ks[corners[1]]
    s = PWESolver(slab_cell(lattice, fill, T, HEIGHT, NZ, N_SI**2), parity="even")
    f0 = s.solve(k, 4, tol=1e-8, maxiter=400)[0]
    grad = s.sensitivities([0])[0]  # (N, N) THz per unit fill

    rng = np.random.default_rng(0)
    d = rng.standard_normal(fill.shape) * (fill > 0.01) * (fill < 0.99)
    h = 1e-4

    def f_band(fr):
        return PWESolver(slab_cell(lattice, fr, T, HEIGHT, NZ, N_SI**2), parity="even").solve(k, 4, tol=1e-10, maxiter=400)[0]

    fd = (f_band(fill + h * d) - f_band(fill - h * d)) / (2 * h)
    adj = float(np.sum(grad * d))
    print(f"band 0 at M = {f0:.3f} THz: adjoint {adj:.6e}, FD {fd:.6e}, rel diff {abs(adj - fd) / abs(fd):.1e}")


if __name__ == "__main__":
    main()
