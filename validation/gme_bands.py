"""
GME band accuracy and cost against the plane-wave references.

blade:  TE-like bands of the blade cell (a = 448 nm) below the light line against MPB
        (validation/reference/optics_mpb_blade.npz, 32/a, 0.1 % from this project's PWE): points with
        100 THz < f < 0.97 × light line; mean, max and mean signed error vs gmax and guided orders.
line:   localised (dagger-row) bands of the GPT07 line-defect supercell (defect and mirror cells)
        at X against PWE (defect edge 187.72 THz at 6 rows, o24×300×64, and 187.74 THz at 4 rows,
        o16×146×64; mirror edge 204.8 THz, 6 rows), σ_y = −1 sector (the defect band's).

    pixi run python validation/gme_bands.py blade
    pixi run python validation/gme_bands.py line --rows 4
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wavesolver.geometry import HexLattice, blade_hole, gpt_omc, line_defect_supercell, solid_fraction  # noqa: E402
from wavesolver.optics_gme import GMESolver, light_line_thz  # noqa: E402

OUT = ROOT / "validation/out"
PWE_REF = {("defect", 4): 187.74, ("defect", 6): 187.72, ("mirror", 6): 204.8, ("mirror", 4): 204.8}


def blade(args) -> None:
    A = 0.448
    lat = HexLattice(A)
    ref = np.load(next(p for p in (OUT / "optics_mpb_blade.npz", ROOT / "validation/reference/optics_mpb_blade.npz") if p.exists()))
    ks, fr = ref["ks"], ref["freqs"][:, :4]
    light = light_line_thz(ks)
    sel = (fr < 0.97 * light[:, None]) & (fr > 100)
    fill = solid_fraction(lat, [blade_hole(0.172, 0.093, 0.025)], 128, supersample=4)
    print(f"blade cell, {sel.sum()} band points below the light line; raster 128²")
    print("  rule     g  orders      N   mean|err|  max|err|  mean err   s/k")
    out = []
    for rule in args.rules.split(","):
        for g in [float(v) for v in args.g.split(",")]:
            for nte, ntm in ((1, 0), (1, 1), (2, 2)):
                s = GMESolver(lat, fill.shape, gmax=2 * np.pi * g / A, n_te=nte, n_tm=ntm, rule=rule)
                s.solve(fill, ks[1], 4)  # compile
                t0 = time.perf_counter()
                f = np.array([s.solve(fill, k, 4, keep_lu=False).freqs for k in ks])
                dt = (time.perf_counter() - t0) / len(ks)
                rel = ((f - fr) / fr)[sel]
                print(f"  {rule:7s} {g:3.0f}  TE{nte} TM{ntm}  {s.last.basis.n_real:5d}   {100 * np.abs(rel).mean():6.2f} %  {100 * np.abs(rel).max():6.2f} %"
                      f"  {100 * rel.mean():+6.2f} %  {dt:5.2f}", flush=True)
                out.append((rule, g, nte, ntm, np.abs(rel).mean(), np.abs(rel).max(), rel.mean(), dt))
    np.savez(OUT / "gme_bands_blade.npz", rows=np.array(out, dtype=object))  # object rows: np.load(..., allow_pickle=True)


def line(args) -> None:
    drawn = gpt_omc.Crystal().drawn()
    hw = drawn.row_offset + drawn.row_pitch / 2
    out = []
    for cell in ("defect", "mirror"):
        lat, holes = line_defect_supercell(cell=cell, shield_rows=args.rows)
        a = lat.vectors[0, 0]
        kX = np.array([np.pi / a, 0.0])
        n2 = 2 * int(round(64 * lat.vectors[1, 1] / a / 2))
        fill = solid_fraction(lat, holes, (64, n2), supersample=2)
        ref = PWE_REF[(cell, args.rows)]
        print(f"{cell} cell, {args.rows} shield rows, raster 64×{n2}; PWE band edge {ref} THz")
        for g in [float(v) for v in args.g.split(",")]:
            for nte, ntm in ((1, 0), (1, 1), (2, 2)):
                if nte == 2 and g > args.gmax_multi:
                    continue
                s = GMESolver(lat, fill.shape, gmax=2 * np.pi * g / a, n_te=nte, n_tm=ntm, sym_y=-1)
                t0 = time.perf_counter()
                r = s.solve(fill, kX, 8, target_thz=ref)
                dt = time.perf_counter() - t0
                loc = np.array([s.y_fraction(r, j, hw) for j in range(len(r.freqs))])
                cand = np.flatnonzero(loc > 0.6)
                j = cand[np.argmin(np.abs(r.freqs[cand] - ref))]
                print(f"  g {g:3.0f} TE{nte} TM{ntm}: N {r.timing['size']:5d}, edge {r.freqs[j]:8.3f} THz ({100 * (r.freqs[j] / ref - 1):+5.2f} %), "
                      f"{dt:5.1f} s (assemble {r.timing['assemble']:.1f}, eig {r.timing['eig']:.1f}); localised: "
                      + " ".join(f"{f:.1f}" for f in r.freqs[cand]), flush=True)
                out.append((cell, args.rows, g, nte, ntm, r.freqs[j], ref, dt, r.timing["size"]))
    np.savez(OUT / f"gme_bands_line_r{args.rows}.npz", rows=np.array(out, dtype=object))  # object rows: np.load(..., allow_pickle=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["blade", "line"])
    ap.add_argument("--g", default=None)
    ap.add_argument("--rules", default="inverse,direct")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--gmax_multi", type=float, default=6)
    args = ap.parse_args()
    if args.g is None:
        args.g = "4,5,6,8" if args.case == "blade" else "5,6,7,8"
    blade(args) if args.case == "blade" else line(args)


if __name__ == "__main__":
    main()
