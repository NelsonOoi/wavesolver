"""
2D mechanical bands of the blade unit cell against COMSOL.

Reference: Blade_UC_mech_bands_results_20240913T163535.mat (COMSOL export, not distributed), from
Blade2D_unitcell_20240501_isotropic_with_fillet.mph: isotropic Si, plane strain, blade
r = 172 nm, w = 93 nm, 25 nm fillet, a = 448 nm, 43 k-points Γ-M-K-Γ, 10 bands.

    python validation/blade_mech_2d.py [path/to/Blade_UC_mech_bands_results_20240913T163535.mat]
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

from omc.geometry import HexLattice, blade_hole, solid_fraction  # noqa: E402
from omc.materials import SI_ISOTROPIC  # noqa: E402
from omc.mech_bloch import BlochSolver  # noqa: E402

# The COMSOL band export is not distributed with the repository; pass its path as the first argument.
REFERENCE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "validation/reference/Blade_UC_mech_bands_results_20240913T163535.mat"
OUT = ROOT / "validation/out"
N_BANDS = 10
WORKERS = os.cpu_count()


def main() -> None:
    ref = sio.loadmat(REFERENCE)
    f_ref = np.real(ref["freqs"][:, :N_BANDS])
    ks_ref = ref["ks"].T * 1e-6  # 1/m → 1/µm
    lattice = HexLattice(float(ref["a"].squeeze()) * 1e6)
    ks, corners = lattice.gamma_m_k_path(15)
    assert np.allclose(ks, ks_ref), "k-path differs from the COMSOL run"
    hole = blade_hole(r=0.172, w=0.093, fillet=0.025)

    OUT.mkdir(exist_ok=True)
    results = {}
    for model in ("plane_strain", "plane_stress"):
        for n in (48, 96, 144):
            t0 = time.perf_counter()
            frac = solid_fraction(lattice, [hole], n)
            solver = BlochSolver(lattice, frac, SI_ISOTROPIC, model=model)
            t1 = time.perf_counter()
            f = solver.bands(ks, N_BANDS, workers=WORKERS)
            t2 = time.perf_counter()
            results[model, n] = f
            msg = f"{model:13s} n={n:3d}  dof={solver.ndof:6d}  setup {t1 - t0:5.2f} s  bands {t2 - t1:5.2f} s"
            if model == "plane_strain":
                mask = f_ref > 0.5e9
                rel = np.abs(f[mask] - f_ref[mask]) / f_ref[mask]
                msg += f"  vs COMSOL: mean {100 * rel.mean():.2f} %  max {100 * rel.max():.2f} %"
            print(msg)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(ks))
    ax.plot(x, f_ref / 1e9, "k.", ms=7, label="COMSOL (plane strain)")
    ax.plot(x, results["plane_strain", 144] / 1e9, "C0-", lw=1, label="this solver, plane strain")
    ax.plot(x, results["plane_stress", 144] / 1e9, "C3--", lw=1, label="this solver, plane stress")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), fontsize=8)
    ax.set_xticks(corners, ["Γ", "M", "K", "Γ"])
    ax.set_ylabel("Frequency (GHz)")
    ax.set_title("Blade cell, a = 448 nm, r = 172 nm, w = 93 nm")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "blade_mech_2d.png", dpi=150)
    np.savez(OUT / "blade_mech_2d.npz", ks=ks, f_ref=f_ref, **{f"{m}_{n}": v for (m, n), v in results.items()})


if __name__ == "__main__":
    main()
