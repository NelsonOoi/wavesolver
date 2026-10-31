"""
3D slab mechanical bands of the blade unit cell against COMSOL, both z-parities.

Reference: Blade_UC_mech_bands_results_20240911T084423.mat (COMSOL export, not distributed), from the
3D blade unit cell (isotropic Si, t = 220 nm, blade r = 172 nm, w = 93 nm, 25 nm fillet,
a = 448 nm): 451 k-points Γ-M-K-Γ, 10 symmetric then 10 antisymmetric bands. Every
STRIDE-th k-point is compared.

    python validation/blade_mech_3d.py [path/to/Blade_UC_mech_bands_results_20240911T084423.mat]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wavesolver.geometry import HexLattice, blade_hole, solid_fraction  # noqa: E402
from wavesolver.materials import SI_ISOTROPIC  # noqa: E402
from wavesolver.mech_bloch import BlochSolver  # noqa: E402

# The COMSOL band export is not distributed with the repository; pass its path as the first argument.
REFERENCE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "validation/reference/Blade_UC_mech_bands_results_20240911T084423.mat"
OUT = ROOT / "validation/out"
N_BANDS = 10
STRIDE = 10
MESHES = ((64, 4), (96, 4))
WORKERS = 4


def main() -> None:
    ref = sio.loadmat(REFERENCE)
    sel = np.arange(0, ref["freqs"].shape[0], STRIDE)
    f_ref = np.real(ref["freqs"][sel])
    ks = ref["ks"].T[sel] * 1e-6
    lattice = HexLattice(float(ref["a"].squeeze()) * 1e6)
    hole = blade_hole(r=0.172, w=0.093, fillet=0.025)
    t = float(ref["t"].squeeze()) * 1e6

    OUT.mkdir(exist_ok=True)
    results = {}
    for n, order in MESHES:
        frac = solid_fraction(lattice, [hole], n)
        for parity, cols in (("even", slice(0, N_BANDS)), ("odd", slice(N_BANDS, 2 * N_BANDS))):
            t0 = time.perf_counter()
            solver = BlochSolver(lattice, frac, SI_ISOTROPIC, model="slab_p", thickness=t, order=order, parity=parity)
            t1 = time.perf_counter()
            f = solver.bands(ks, N_BANDS, workers=WORKERS)
            t2 = time.perf_counter()
            results[n, order, parity] = f
            r = f_ref[:, cols]
            mask = r > 0.5e9
            rel = np.abs(f[mask] - r[mask]) / r[mask]
            print(
                f"n={n:3d} p={order} {parity:4s} dof={solver.ndof:6d}  setup {t1 - t0:4.1f} s  "
                f"{len(ks)} k in {t2 - t1:5.1f} s ({(t2 - t1) / len(ks):.2f} s/k)  "
                f"vs COMSOL: mean {100 * rel.mean():.2f} %  max {100 * rel.max():.2f} %",
                flush=True,
            )

    n, order = MESHES[-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    x = sel
    corners = [0, 150, 300, 450]
    for ax, (parity, cols, title) in zip(
        axes,
        (("even", slice(0, N_BANDS), "Symmetric"), ("odd", slice(N_BANDS, 2 * N_BANDS), "Antisymmetric")),
    ):
        ax.plot(np.arange(ref["freqs"].shape[0]), np.real(ref["freqs"][:, cols]) / 1e9, "k-", lw=0.8)
        ax.plot(x, results[n, order, parity] / 1e9, "C3o", ms=3, mfc="none")
        ax.plot([], [], "k-", lw=0.8, label="COMSOL")
        ax.plot([], [], "C3o", ms=3, mfc="none", label=f"this solver (n={n}, p={order})")
        ax.set_xticks(corners, ["Γ", "M", "K", "Γ"])
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Frequency (GHz)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Blade cell, 220 nm Si slab, a = 448 nm, r = 172 nm, w = 93 nm")
    fig.tight_layout()
    fig.savefig(OUT / "blade_mech_3d.png", dpi=150)
    np.savez(OUT / "blade_mech_3d.npz", ks=ks, f_ref=f_ref, **{f"n{n}_p{p}_{par}": v for (n, p, par), v in results.items()})


if __name__ == "__main__":
    main()
