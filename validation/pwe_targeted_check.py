"""
Targeted PWE (PWESolver.solve_near, Jacobi–Davidson about a frequency) against references.

  1. unit cell (blade, a = 448 nm) at M: the modes nearest a frequency inside the band stack
     against LOBPCG's lowest bands (frequencies, eigenvector overlap), and the adjoint of a
     targeted band against the LOBPCG one;
  2. --cavity: the GPT07 finite cavity (columns c0..c1, shield rows) as one supercell at Γ,
     against GME (193.88 THz, ±1 %) and meep (192.73 THz) in validation/out/gme_cavity_*.log.

    pixi run python validation/pwe_targeted_check.py [--cavity --c0 -8 --c1 8 --rows 2 --px 16 --nz 32]
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

from omc.cavity import gpt07_cavity  # noqa: E402
from omc.geometry import HexLattice, blade_hole, solid_fraction  # noqa: E402
from omc.optics_pwe import PWESolver, slab_cell  # noqa: E402

EPS_SI = 3.48**2
T = 0.22


def unit_cell_check() -> None:
    lat = HexLattice(0.448)
    cell = slab_cell(lat, solid_fraction(lat, [blade_hole(0.172, 0.093)], 24), T, 1.5, 32, EPS_SI)
    M = np.pi / lat.a * np.array([1.0, 1 / np.sqrt(3)])
    ref = PWESolver(cell, "even")
    t0 = time.perf_counter()
    f_ref = ref.solve(M, 12, tol=1e-9, maxiter=800)
    t_ref = time.perf_counter() - t0
    print(f"LOBPCG, 12 lowest bands at M ({t_ref:.1f} s): " + " ".join(f"{f:.4f}" for f in f_ref), flush=True)
    target = (f_ref[8] + f_ref[9]) / 2 + 0.1
    s = PWESolver(cell, "even")
    t0 = time.perf_counter()
    f = s.solve_near(M, target, n_bands=3, tol=1e-9, verbose=True)
    t_jd = time.perf_counter() - t0
    near = np.argsort(np.abs(f_ref - target))[:3]
    print(f"solve_near({target:.3f} THz), 3 modes ({t_jd:.1f} s, {s.jd_info}): " + " ".join(f"{v:.6f}" for v in f), flush=True)
    for j, fj in enumerate(f):
        i = int(np.argmin(np.abs(f_ref - fj)))
        ov = abs(np.vdot(ref.last_vectors[:, i], s.last_vectors[:, j]))
        print(f"   {fj:.6f} THz vs LOBPCG band {i} {f_ref[i]:.6f}: Δf/f {abs(fj - f_ref[i]) / f_ref[i]:.1e}, |overlap| {ov:.8f}")
    assert sorted(near) == sorted(int(np.argmin(np.abs(f_ref - fj))) for fj in f), "not the nearest modes"
    # adjoint of a targeted band against the same band from LOBPCG (any right-hand side)
    j = 1
    i = int(np.argmin(np.abs(f_ref - f[j])))
    rng = np.random.default_rng(1)
    rhs = rng.standard_normal(s.last_vectors.shape[0]) + 1j * rng.standard_normal(s.last_vectors.shape[0])
    w_t, info_t = s.adjoint_solve(j, rhs, tol=1e-10)
    w_r, info_r = ref.adjoint_solve(i, rhs, tol=1e-10)
    # eigenvector phases differ: compare a phase-free functional, Re(xᴴ ...) of the response to dΘ ∝ 1
    ph = np.vdot(ref.last_vectors[:, i], s.last_vectors[:, j])
    ph /= abs(ph)
    err = np.linalg.norm(w_t - w_r) / np.linalg.norm(w_r)
    print(f"adjoint of band {i}: MINRES {info_t['iterations']} (targeted) vs {info_r['iterations']} (LOBPCG) iterations, "
          f"relative difference {err:.1e} (phase of the eigenvectors {np.angle(ph):+.2f} rad)", flush=True)


def envelope_start(dom, n2: int, args, s: PWESolver) -> tuple[np.ndarray, float]:
    """
    Start vectors for the cavity: the defect column's X-point band-edge mode (one-column PWE on
    the same grid), tiled along the cavity with its Bloch phase e^{iπx/a}, times Gaussian
    envelopes centred on column 0 (widths 1.5, 2.5, 4 columns) and one odd envelope; and that
    band edge (THz), from which the cavity target is set (GPT07: the mode lies 3–6 THz above).
    The edge is the row-localised X band nearest args.target − 5 THz (args.target may be a guess).
    """
    a = dom.pitch
    edge_dom = gpt07_cavity(columns=(0, 0), shield_rows=args.rows, skew=False)
    edge_fill = solid_fraction(edge_dom.lattice, edge_dom.holes, (args.px, n2), supersample=2)
    se = PWESolver(slab_cell(edge_dom.lattice, edge_fill, T, args.height, args.nz, EPS_SI), "even")
    fX = se.solve(np.array([np.pi / a, 0.0]), 8 + 4 * args.rows, tol=1e-6, maxiter=300)  # folded shield bands grow with the rows
    y = (np.arange(n2) + 0.5) / n2 * edge_dom.lattice.vectors[1, 1] - edge_dom.lattice.vectors[1, 1] / 2
    in_row = []
    for j in range(len(fX)):
        h = (np.abs(se.h_field(j)) ** 2).sum(axis=(0, 3))
        in_row.append(h[:, np.abs(y) < dom.half_width].sum() / h.sum())
    cand = [j for j in range(len(fX)) if in_row[j] > 0.6 and 165 < fX[j] < args.target + 5]
    band = min(cand, key=lambda j: abs(fX[j] - (args.target - 5)))
    print("defect column at X: " + " ".join(f"{f:.2f}({r:.2f})" for f, r in zip(fX, in_row)) + f" → band-edge mode {fX[band]:.3f} THz", flush=True)
    h = se.h_field(band)  # periodic part (3, px, n2, nz)
    hc = np.concatenate([h] * dom.n_columns, axis=1)  # (3, n1, n2, nz)
    n1 = hc.shape[1]
    x = (np.arange(n1) + 0.5) / n1 * dom.lattice.vectors[0, 0] + (dom.columns[0] - 0.5) * a  # lab x, column n centred at n·a
    bloch = hc * np.exp(1j * np.pi * x / a)[None, :, None, None]
    cols = []
    for env in (np.exp(-0.5 * (x / (1.5 * a)) ** 2), np.exp(-0.5 * (x / (2.5 * a)) ** 2), np.exp(-0.5 * (x / (4 * a)) ** 2),
                (x / a) * np.exp(-0.5 * (x / (2.5 * a)) ** 2)):
        H = np.fft.fftn(bloch * env[None, :, None, None], axes=(1, 2, 3)) / np.prod(s.shape)
        cols.append(s._flat(s._project(s._amplitudes(H[None])))[:, 0])
    return np.column_stack(cols), float(fX[band])


def cavity_check(args) -> None:
    dom = gpt07_cavity(columns=(args.c0, args.c1), shield_rows=args.rows, skew=False)
    a = dom.pitch
    V = dom.lattice.vectors
    n1 = args.px * dom.n_columns
    n2 = 2 * int(round(V[1, 1] / (a / args.px) / 2))
    fill = solid_fraction(dom.lattice, dom.holes, (n1, n2), supersample=2)
    cell = slab_cell(dom.lattice, fill, T, args.height, args.nz, EPS_SI)
    print(f"GPT07 cavity columns {args.c0}..{args.c1}, {args.rows} shield rows: cell {V[0, 0]:.2f} × {V[1, 1]:.2f} µm, "
          f"grid {n1}×{n2}×{args.nz} ({2 * n1 * n2 * args.nz / 1e6:.1f} M unknowns)", flush=True)
    s = PWESolver(cell, "even")
    s._set_k(np.zeros(2))
    x0 = None if args.random_start else envelope_start(dom, n2, args, s)[0]
    t0 = time.perf_counter()
    f = s.solve_near(np.zeros(2), args.target, n_bands=args.nb, x0=x0, tol=args.tol, inner=args.inner, verbose=True)
    print(f"solve_near({args.target} THz): {time.perf_counter() - t0:.0f} s, {s.jd_info}", flush=True)
    for j, fj in enumerate(f):
        h = (np.abs(s.h_field(j)) ** 2).sum(axis=(0, 3))  # (n1, n2)
        per_col = h.reshape(dom.n_columns, args.px, n2).sum(axis=(1, 2))
        y = (np.arange(n2) + 0.5) / n2 * V[1, 1] - V[1, 1] / 2
        in_row = h[:, np.abs(y) < dom.half_width].sum() / h.sum()
        cols = np.arange(args.c0, args.c1 + 1)
        neff = per_col.sum() / per_col.max()
        print(f"  {fj:.3f} THz: |H|² in the row {in_row:.2f}, in columns −3..3 {per_col[np.abs(cols) <= 3].sum() / per_col.sum():.2f}, "
              f"N_eff {neff:.1f} columns; residual {s.residuals[j]:.1e}")
    print("  references: GME 193.884 THz (g 6, ±1 %), meep 192.73 THz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cavity", action="store_true")
    ap.add_argument("--c0", type=int, default=-8)
    ap.add_argument("--c1", type=int, default=8)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--px", type=int, default=16, help="optical grid points per column along x (and the same spacing in y)")
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--height", type=float, default=1.5)
    ap.add_argument("--target", type=float, default=193.5)
    ap.add_argument("--nb", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--inner", type=int, default=40)
    ap.add_argument("--random-start", action="store_true", help="cavity: random start vectors instead of the envelope guess")
    args = ap.parse_args()
    if args.cavity:
        cavity_check(args)
    else:
        unit_cell_check()


if __name__ == "__main__":
    main()
