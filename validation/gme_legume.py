"""
legume cross-check of wavesolver.optics_gme (legume is NOT in the project environment).

1. Export the geometry from the project environment (polygons, lattice, k-points):

       pixi run python validation/gme_legume.py export

   writes validation/out/gme_legume_blade.npz (blade cell, MPB k-path) and
   validation/out/gme_legume_cavity_c-8_8_r2.npz (reduced GPT07 cavity, rectangular cell).

2. Run legume in an isolated environment, e.g.

       uv venv /tmp/legume && VIRTUAL_ENV=/tmp/legume uv pip install legume-gme
       /tmp/legume/bin/python -I validation/gme_legume.py blade validation/out/gme_legume_blade.npz
       /tmp/legume/bin/python -I validation/gme_legume.py cavity validation/out/gme_legume_cavity_c-8_8_r2.npz 4 177.27

   Settings matched to GMESolver(truncate="G", n_te=1, n_tm=0 → gmode_inds=[0]; n_tm=1 → [0, 3]),
   eps_eff = average. legume uses the exact polygon Fourier transform; wavesolver.optics_gme the pixel
   raster, so the raster must be fine (≥ 128 px/a) for sub-0.1 % agreement.

Results (2026-10-27): blade bands 0.11 % at 128 px/a (0.30 % at 64); blade Q_rad of leaky bands
(Q 30–1.6e5) 1–2 %; GPT07 cavity −8..8, 2 rows, g = 3, 144 px/a: f within 0.05–0.07 %, Q within
0.1–11 % for Q ≤ 2e5, the Q ≈ 1e7 cavity mode within 2.5× (raster-limited).
"""

from __future__ import annotations

import sys
import time

import numpy as np

C = 299.792458


def export() -> None:
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from shapely.geometry.polygon import orient

    from wavesolver.cavity import gpt07_cavity
    from wavesolver.geometry import HexLattice, blade_hole

    out = ROOT / "validation/out"
    lat = HexLattice(0.448)
    ks = np.load(next(p for p in (out / "optics_mpb_blade.npz", ROOT / "validation/reference/optics_mpb_blade.npz") if p.exists()))["ks"]
    out.mkdir(exist_ok=True)
    p = np.array(orient(blade_hole(0.172, 0.093, 0.025), 1.0).exterior.coords)[:-1]
    np.savez(out / "gme_legume_blade.npz", A=lat.vectors, n=1, p0=p, ks=ks, a=0.448)
    try:
        dom = gpt07_cavity(columns=(-8, 8), shield_rows=2, skew=False)
    except ImportError as e:
        print(f"exported the blade case only (cavity case: {e})")
        return
    polys = [np.array(orient(h, 1.0).exterior.coords)[:-1] for h in dom.holes]
    np.savez(out / "gme_legume_cavity_c-8_8_r2.npz", A=dom.lattice.vectors, n=len(polys), a=dom.pitch, **{f"p{i}": q for i, q in enumerate(polys)})
    print("exported")


def crystal(d):
    import legume

    phc = legume.PhotCryst(legume.Lattice(d["A"][0], d["A"][1]))
    phc.add_layer(d=0.22, eps_b=3.48**2)
    for i in range(int(d["n"])):
        q = d[f"p{i}"]
        phc.add_shape(legume.Poly(eps=1.0, x_edges=q[:, 0], y_edges=q[:, 1]))
    return phc


def blade(path: str) -> None:
    import legume

    d = np.load(path)
    for g in (3, 4, 5, 6):
        for inds in ([0], [0, 3]):
            gme = legume.GuidedModeExp(crystal(d), gmax=g / float(d["a"]), truncate_g="abs")
            t0 = time.time()
            gme.run(kpoints=d["ks"].T, gmode_inds=inds, numeig=8, compute_im=True, verbose=False)
            np.savez(path.replace(".npz", f"_g{g}_{'_'.join(map(str, inds))}.npz"), f=gme.freqs * C, fi=gme.freqs_im * C)
            print(g, inds, f"{time.time() - t0:.1f} s", flush=True)


def cavity(path: str, g: float, target: float) -> None:
    import legume

    d = np.load(path)
    gme = legume.GuidedModeExp(crystal(d), gmax=g / float(d["a"]), truncate_g="abs")
    t0 = time.time()
    gme.run(kpoints=np.zeros((2, 1)), gmode_inds=[0], numeig=10, compute_im=True, eig_solver="eigsh", eig_sigma=target / C, verbose=False)
    f, fi = gme.freqs[0] * C, gme.freqs_im[0] * C
    print(f"legume g{g}: {gme.gvec.shape[1]} plane waves, {time.time() - t0:.0f} s")
    for x, y in sorted(zip(f, fi)):
        print(f"  {x:9.3f} THz  Q {x / (2 * abs(y)) if y else np.inf:10.4g}")


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "export":
        export()
    elif kind == "blade":
        blade(sys.argv[2])
    else:
        cavity(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
