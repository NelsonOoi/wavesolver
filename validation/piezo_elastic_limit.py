"""
Piezo solver with e = 0 against the elastic BlochSolver ("stack" model), and symmetry of the
assembled piezoelectric system.

Geometry: blade unit cell (a = 448 nm, r = 172 nm, w = 93 nm), 220 nm Si with 250 nm X-cut LN
on top, the LN patterned as the Si with an extra 60 nm erosion (LN ⊆ Si). With e = 0 the
potential decouples and the frequencies must agree with BlochSolver to round-off. Then with
the real e: the assembled system is Hermitian at any Bloch k (real symmetric at Γ), and the
short-circuit admittance matrix of two electrodes is reciprocal (Y12 = Y21).

    pixi run python validation/piezo_elastic_limit.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wavesolver.geometry import HexLattice, blade_hole, solid_fraction  # noqa: E402
from wavesolver.materials import LN_PIEZO, SI, SI_EPS_R, Piezoelectric, x_cut  # noqa: E402
from wavesolver.mech_bloch import BlochSolver, Layer  # noqa: E402
from wavesolver.piezo import Electrode, PiezoLayer, PiezoSolver  # noqa: E402

N = 24
ORDER = 2


def main() -> None:
    lat = HexLattice(0.448)
    f_si = solid_fraction(lat, [blade_hole(0.172, 0.093, 0.025)], N)
    f_ln = solid_fraction(lat, [blade_hole(0.172 + 0.06, 0.093 + 0.12, 0.025)], N)
    f_ln = np.minimum(f_ln, f_si)
    ln = LN_PIEZO.rotated(x_cut(30.0))
    ln0 = Piezoelectric(ln.name, ln.C, ln.rho, 0 * ln.e, ln.eps_r)
    ks = [np.zeros(2), np.array([2.0, 0.7]), np.pi / lat.a * np.array([1.0, 1 / np.sqrt(3)])]

    bloch = BlochSolver(lat, model="stack", order=ORDER, layers=[Layer(SI, 0.22, f_si), Layer(ln.elastic, 0.25, f_ln)])
    layers = [
        PiezoLayer(0.22, f_si, SI, eps_r=SI_EPS_R, order=ORDER),
        PiezoLayer(0.25, f_ln, ln0, order=ORDER),
    ]
    t0 = time.perf_counter()
    pz = PiezoSolver(lat.vectors, layers, bc=("periodic", "periodic"), origin=-(lat.vectors[0] + lat.vectors[1]) / 2)
    print(f"setup {time.perf_counter() - t0:.2f} s, u dof {pz.K.shape[0]}, phi dof {pz.Kpp.shape[0]} (full grid)")
    worst = 0.0
    for k in ks:
        fb = bloch.solve(k, 12, shift_hz=-0.5e9)
        t0 = time.perf_counter()
        fp = pz.eigenmodes(12, -0.5e9, k=k).real
        dt = time.perf_counter() - t0
        sel = fb > 1e8
        rel = np.abs(fp[sel] - fb[sel]) / fb[sel]
        worst = max(worst, rel.max())
        print(f"k = {np.round(k, 3)}: max rel diff {rel.max():.2e} ({dt:.2f} s)  f/GHz = {np.round(fp[:8] / 1e9, 4)}")
    print(f"(a) e = 0 vs BlochSolver: worst relative difference {worst:.2e}")

    # (c) symmetry of the assembled piezo system and reciprocity.
    layers[1] = PiezoLayer(0.25, f_ln, ln, order=ORDER)
    mask_a = np.zeros((N, N), bool)
    mask_b = np.zeros((N, N), bool)
    mask_a[2:8, 2:8] = True
    mask_b[14:20, 12:18] = True
    el = [Electrode("A", mask_a & (f_ln > 0.5), 2), Electrode("B", mask_b & (f_ln > 0.5), 2)]
    pz = PiezoSolver(lat.vectors, layers, bc=("periodic", "periodic"), origin=-(lat.vectors[0] + lat.vectors[1]) / 2, electrodes=el)
    for k in ks[:2]:
        A, _, _ = pz.reduced(0.3, k)
        herm = abs(A - A.conj().T).max() / abs(A).max()
        print(f"(c) k = {np.round(k, 3)}: |A - A^H| / |A| = {herm:.1e}")
    for f in (3.0e9, 5.5e9):
        Y = pz.admittance([f], ["A", "B"])[0]
        print(f"(c) f = {f / 1e9} GHz: Y = {np.array2string(Y, precision=3)}  |Y12 - Y21|/|Y12| = {abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1]):.1e}")
    C = pz.capacitance(["A", "B"])
    print(f"    clamped capacitance matrix (aF): {np.array2string(C * 1e18, precision=4)}")


if __name__ == "__main__":
    main()
