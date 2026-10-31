"""
Finite-difference check of the optical band sensitivities ∂f/∂fraction in omc.optics_pwe.

The fill of the blade cell is perturbed along a smooth random direction and along a direction
concentrated on one hole edge. The finite differences rebuild the cell, so they include the
change of the interface normals, which the analytic sensitivity holds fixed; the difference
measures that approximation.

    pixi run python validation/optics_gradient_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omc.geometry import HexLattice, blade_hole, solid_fraction  # noqa: E402
from omc.optics_pwe import PWESolver, slab_cell  # noqa: E402

A, T, HEIGHT, N_SI = 0.448, 0.22, 1.5, 3.48
N, NZ = 24, 64
BANDS = range(4)
STEP = 1e-3


def bands_at(frac: np.ndarray, k: np.ndarray, lattice: HexLattice) -> tuple[np.ndarray, PWESolver]:
    solver = PWESolver(slab_cell(lattice, frac, T, HEIGHT, NZ, N_SI**2), parity="even")
    f = solver.solve(k, n_bands=len(BANDS), tol=1e-4, maxiter=300)
    return f, solver


def main() -> None:
    lattice = HexLattice(A)
    ks, corners = lattice.gamma_m_k_path(15)
    frac = solid_fraction(lattice, [blade_hole(0.172, 0.093, 0.025)], N)
    rng = np.random.default_rng(0)
    smooth = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((N, N))) * np.exp(-0.5 * np.add.outer(
        np.fft.fftfreq(N) ** 2, np.fft.fftfreq(N) ** 2) * N**2 / 4)))
    smooth /= np.abs(smooth).max()
    edge = ((frac > 0.05) & (frac < 0.95)).astype(float) * rng.standard_normal((N, N))
    directions = {"smooth": smooth, "edge": edge}

    for name, k in (("M", ks[corners[1]]), ("K", ks[corners[2]]), ("mid", ks[7])):
        f0, solver = bands_at(frac, k, lattice)
        grad = solver.sensitivities(BANDS)
        for dname, d in directions.items():
            fp, _ = bands_at(frac + STEP * d, k, lattice)
            fm, _ = bands_at(frac - STEP * d, k, lattice)
            fd = (fp - fm) / (2 * STEP)
            an = (grad * d).sum(axis=(1, 2))
            rel = np.abs(an - fd) / np.maximum(np.abs(fd), 1e-12)
            print(
                f"k={name:3s} {dname:6s}  f = {np.round(f0, 2)}  FD {np.round(fd, 4)}  adjoint {np.round(an, 4)}  "
                f"rel err {np.array2string(rel, formatter={'float_kind': lambda x: f'{x:.1e}'})}",
                flush=True,
            )


if __name__ == "__main__":
    main()
