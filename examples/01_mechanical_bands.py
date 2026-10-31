"""
Mechanical Bloch bands of a patterned Si membrane and their gradient with respect to the pattern.

Tripod (blade) holes in a triangular lattice, a = 448 nm, in a 220 nm cubic Si slab. Both
z-parities on a short Γ-M-K-Γ path, the full phononic gap, then ∂f/∂fraction of one band at M
from mech_adjoint.sensitivities, checked against a central difference along a random direction.

    pixi run python examples/01_mechanical_bands.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wavesolver.geometry import HexLattice, blade_hole, solid_fraction  # noqa: E402
from wavesolver.materials import SI  # noqa: E402
from wavesolver.mech_adjoint import sensitivities  # noqa: E402
from wavesolver.mech_bloch import BlochSolver  # noqa: E402

A, T, N = 0.448, 0.22, 32  # lattice constant, thickness (µm), elements per lattice vector


def main() -> None:
    lattice = HexLattice(A)
    fraction = solid_fraction(lattice, [blade_hole(r=0.172, w=0.093, fillet=0.025)], N)
    ks, corners = lattice.gamma_m_k_path(6)

    bands = {}
    for parity in ("even", "odd"):
        solver = BlochSolver(lattice, fraction, SI, model="slab_p", thickness=T, parity=parity)
        t0 = time.perf_counter()
        bands[parity] = solver.bands(ks, n_bands=8)  # (n_k, n_bands), Hz
        print(f"{parity:>4}: {solver.ndof} DOF, {(time.perf_counter() - t0) / len(ks):.2f} s per k-point")

    # Full gap: the largest gap in the merged spectrum of both parities over the path.
    f = np.sort(np.hstack([bands["even"], bands["odd"]]), axis=1)
    lo, hi = f.max(axis=0)[:-1], f.min(axis=0)[1:]
    b = int(np.argmax(hi - lo))
    print(f"widest full gap on the path: {lo[b] / 1e9:.2f}-{hi[b] / 1e9:.2f} GHz" if hi[b] > lo[b] else "no full gap")

    # Gradient of the even bands at M with respect to the element fractions.
    solver = BlochSolver(lattice, fraction, SI, model="slab_p", thickness=T, parity="even")
    k = ks[corners[1]]
    sens = sensitivities(solver, k, n_bands=6)
    band = next(int(c[0]) for c in sens.clusters if len(c) == 1 and sens.freqs[c[0]] > 1e9)
    grad = sens.dfreq[0, band]  # (N, N) Hz per unit fraction

    rng = np.random.default_rng(0)
    d = rng.standard_normal(fraction.shape) * (fraction > 0.01) * (fraction < 0.99)  # edge elements only
    h = 1e-4

    def f_band(fr):
        return BlochSolver(lattice, fr, SI, model="slab_p", thickness=T, parity="even").solve(k, 6)[band]

    fd = (f_band(fraction + h * d) - f_band(fraction - h * d)) / (2 * h)
    adj = float(np.sum(grad * d))
    print(f"band {band} at M = {sens.freqs[band] / 1e9:.4f} GHz: adjoint {adj:.6e}, FD {fd:.6e}, rel diff {abs(adj - fd) / abs(fd):.1e}")


if __name__ == "__main__":
    main()
