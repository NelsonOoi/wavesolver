"""
TE-like bands of the blade cell in a 220 nm Si slab, against MPB.

Run validation/optics_mpb_reference.py first (in its own environment) to write the reference;
without it the bands are computed and plotted alone.

    pixi run python validation/optics_blade.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omc.geometry import HexLattice, blade_hole, solid_fraction  # noqa: E402
from omc.optics_pwe import PWESolver, light_line_thz, slab_cell  # noqa: E402

A, T, HEIGHT, N_SI = 0.448, 0.22, 2.0, 3.48
N_BANDS = 6
OUT = ROOT / "validation/out"
REFERENCE = next((p for p in (OUT / "optics_mpb_blade.npz", ROOT / "validation/reference/optics_mpb_blade.npz") if p.exists()), OUT / "optics_mpb_blade.npz")
GRIDS = ((32, 128),)


def te_gap(f: np.ndarray, light: np.ndarray, band: int) -> tuple[float, float]:
    """Gap between band and band+1 (0-based), using only points below the light line."""
    below = f < light[:, None]
    top = np.max(np.where(below[:, band], f[:, band], -np.inf))
    bottom = np.min(np.where(below[:, band + 1], f[:, band + 1], np.inf))
    return top, bottom


def main() -> None:
    lattice = HexLattice(A)
    ks, corners = lattice.gamma_m_k_path(15)
    light = light_line_thz(ks)
    hole = blade_hole(0.172, 0.093, 0.025)
    ref = np.load(REFERENCE) if REFERENCE.exists() else None

    results = {}
    for n, nz in GRIDS:
        cell = slab_cell(lattice, solid_fraction(lattice, [hole], n), T, HEIGHT, nz, N_SI**2)
        solver = PWESolver(cell, parity="even")
        t0 = time.perf_counter()
        f = solver.bands(ks, N_BANDS)
        dt = time.perf_counter() - t0
        results[n, nz] = f
        top, bottom = te_gap(f, light, 0)
        msg = f"N={n} nz={nz}: {len(ks)} k in {dt:.0f} s ({dt / len(ks):.1f} s/k); band 1-2 gap {top:.1f}-{bottom:.1f} THz"
        if ref is not None:
            fr = ref["freqs"][:, :N_BANDS]
            guided = (fr < light[:, None]) & (f < light[:, None])
            rel = np.abs(f - fr)[guided] / fr[guided]
            msg += f"; vs MPB below light line: mean {100 * rel.mean():.2f} %  max {100 * rel.max():.2f} %"
        print(msg, flush=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(ks))
    if ref is not None:
        ax.plot(x, ref["freqs"][:, :N_BANDS], "k.", ms=5)
        ax.plot([], [], "k.", label=f"MPB (res {int(ref['resolution'])}/a)")
    f = results[GRIDS[-1]]
    ax.plot(x, f, "C3-", lw=1)
    ax.plot([], [], "C3-", label=f"this solver (N={GRIDS[-1][0]}, nz={GRIDS[-1][1]})")
    ax.fill_between(x, light, f.max() * 1.05, color="0.85", label="above light line")
    ax.axhline(193.4, color="C0", ls=":", lw=1, label="1550 nm")
    ax.set_xticks(corners, ["Γ", "M", "K", "Γ"])
    ax.set_ylim(0, f.max() * 1.05)
    ax.set_ylabel("Frequency (THz)")
    ax.set_title("TE-like bands, blade cell in 220 nm Si slab")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "optics_blade.png", dpi=150)
    np.savez(OUT / "optics_blade.npz", ks=ks, **{f"N{n}_nz{nz}": v for (n, nz), v in results.items()})


if __name__ == "__main__":
    main()
