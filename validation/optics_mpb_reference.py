"""
MPB reference: TE-like (z-even) bands of the blade cell in a 220 nm Si slab.

Runs in its own environment, as MPB is not part of the project environment:

    ~/.pixi/bin/pixi exec --spec "pymeep=*=mpi_mpich*" --spec python=3.11 --spec shapely \
        python validation/optics_mpb_reference.py

Writes validation/out/optics_mpb_blade.npz with frequencies in THz on the same Γ-M-K-Γ
path (15 points per leg) as the other benchmarks.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import meep as mp
import numpy as np
from meep import mpb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wavesolver.geometry import HexLattice, blade_hole, solid_fraction  # noqa: E402

A = 0.448
T = 0.22
HEIGHT = 2.0
N_SI = 3.48
RESOLUTION = 32
N_BANDS = 8
C_THZ_UM = 299.792458
OUT = ROOT / "validation/out"


def main() -> None:
    lattice = HexLattice(A)
    ks, _ = lattice.gamma_m_k_path(15)
    hole = blade_hole(0.172, 0.093, 0.025)

    geometry_lattice = mp.Lattice(
        size=mp.Vector3(1, 1, HEIGHT / A),
        basis1=mp.Vector3(1, 0),
        basis2=mp.Vector3(0.5, math.sqrt(3) / 2),
    )
    # MPB places objects in lattice coordinates; convert the Cartesian outline.
    verts = [
        mp.cartesian_to_lattice(mp.Vector3(x / A, y / A, 0), geometry_lattice)
        for x, y in list(hole.exterior.coords)[:-1]
    ]
    si = mp.Medium(epsilon=N_SI**2)
    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, T / A), material=si),
        mp.Prism(verts, height=HEIGHT / A, axis=mp.Vector3(0, 0, 1), center=mp.Vector3(), material=mp.air),
    ]
    # k in units of the reciprocal lattice vectors: c_i = k · a_i / 2π.
    k_points = [mp.Vector3(*(lattice.vectors @ k / (2 * np.pi))) for k in ks]
    ms = mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        k_points=k_points,
        resolution=RESOLUTION,
        num_bands=N_BANDS,
    )
    t0 = time.perf_counter()
    ms.run_zeven()
    dt = time.perf_counter() - t0
    freqs = np.array(ms.all_freqs) * C_THZ_UM / A

    # Check the geometry against our rasterisation: Si fill at the slab mid-plane.
    eps = np.asarray(ms.get_epsilon())
    # MPB's grid does not start at the slab centre; take the z layer with the most Si.
    mid = eps[:, :, np.argmax(eps.mean(axis=(0, 1)))] if eps.ndim == 3 else eps
    fill_mpb = float(np.mean((mid - 1) / (N_SI**2 - 1)))
    fill_ours = float(solid_fraction(lattice, [hole], 64).mean())
    print(f"MPB: {len(ks)} k-points in {dt:.0f} s; mid-plane Si fill {fill_mpb:.4f} (ours {fill_ours:.4f})")

    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "optics_mpb_blade.npz", ks=ks, freqs=freqs, fill=fill_mpb, resolution=RESOLUTION, height=HEIGHT)


if __name__ == "__main__":
    main()
