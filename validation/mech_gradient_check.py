"""
Finite-difference checks of the mechanical Bloch band sensitivities (wavesolver.mech_adjoint).

1. Fraction level: ∂f_band/∂fraction along a random direction of element fractions, for
   plane stress, both slab_p parities and a two-layer LN/Si stack (one direction per layer),
   at a generic k. Also the identity Σ_e f_e ∂λ/∂f_e = 0 (K and M scale together).
2. Shape parameters: ∂f_band/∂θ for every field of a `TripodHole` with all features on
   (arms, hub, end pads, corner radii, junction holes), through the smoothed-raster
   Jacobian of wavesolver.parametrize, both slab_p parities.
4. Free-form density (C3v node orbits → cone filter → tanh projection, β = 8): band
   gradient along a random direction of the design vector, and the Zhou length-scale
   measures' gradients, against FD.
3. Degenerate bands at Γ and K (split by ~1e-3 by the skewed mesh): the cluster mean
   against FD, and the full cluster matrix A (whose eigenvalues are the subspace
   derivatives) against central differences of the assembled K(k), M(k) projected onto
   the cluster's eigenvectors. (FD of the individual bands cannot test the subspace
   formula: the mesh splitting makes them non-degenerate, and second-order coupling to
   neighbouring bands is of the same size as the split at resolvable steps.)

Central differences; the relative error is |FD - adjoint| / max(|adjoint| over the
parameters of that band), so a parameter that barely moves a band is not divided by ~0.

    pixi run python validation/mech_gradient_check.py
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wavesolver.geometry import HexLattice, blade_hole, solid_fraction  # noqa: E402
from wavesolver.materials import LN, SI, SI_ISOTROPIC, x_cut  # noqa: E402
from wavesolver.mech_adjoint import cluster_matrix, sensitivities  # noqa: E402
from wavesolver.mech_bloch import BlochSolver, Layer  # noqa: E402
from wavesolver.parametrize import DensityCell, ParametricCell, TripodHole  # noqa: E402

OUT = ROOT / "validation/out"
LAT = HexLattice(0.448)
K_GENERIC = np.array([3.1, 1.2])  # 1/µm, inside the IBZ, away from crossings
NB = 8
RTOL_CLUSTER = 5e-3  # the skewed mesh splits symmetry-degenerate bands by up to ~3e-3 at n = 48


def rel_err(fd: np.ndarray, ad: np.ndarray) -> np.ndarray:
    """Per band (rows) and parameter (columns), normalised by the band's largest gradient."""
    scale = np.abs(ad).max(axis=-1, keepdims=True)
    return np.abs(fd - ad) / np.where(scale > 0, scale, 1)


def nondegenerate(freqs: np.ndarray, rtol: float = 1e-3) -> np.ndarray:
    gaps = np.diff(freqs) / freqs[1:]
    ok = np.ones(len(freqs), bool)
    ok[1:] &= gaps > rtol
    ok[:-1] &= gaps > rtol
    return ok & (freqs > 1e8)


def check_fraction_level(results: dict) -> None:
    n = 32
    frac = solid_fraction(LAT, [blade_hole(0.172, 0.093, 0.025)], n)
    rng = np.random.default_rng(1)
    cases = {
        "plane_stress": dict(fraction=frac, material=SI_ISOTROPIC, model="plane_stress"),
        "slab_p_even": dict(fraction=frac, material=SI_ISOTROPIC, model="slab_p", thickness=0.22, parity="even"),
        "slab_p_odd": dict(fraction=frac, material=SI_ISOTROPIC, model="slab_p", thickness=0.22, parity="odd"),
    }
    n_st = 24
    fs = solid_fraction(LAT, [blade_hole(0.172, 0.093, 0.025)], n_st)
    fl = np.minimum(fs, solid_fraction(LAT, [blade_hole(0.19, 0.12, 0.025)], n_st))  # LN pattern ⊂ Si
    stack_layers = lambda a, b: [Layer(SI, 0.22, a), Layer(LN.rotated(x_cut(0.0)), 0.25, b)]
    print("\n1. Fraction level, random direction, h = 1e-5")
    for name, kw in [*cases.items(), ("stack_LN_Si", None)]:
        if kw is None:
            base = [fs, fl]
            make = lambda fr: BlochSolver(LAT, model="stack", order=2, layers=stack_layers(*fr))
        else:
            base = [kw["fraction"]]
            make = lambda fr, kw=kw: BlochSolver(LAT, **{**kw, "fraction": fr[0]})
        solver = make(base)
        t0 = time.perf_counter()
        s = sensitivities(solver, K_GENERIC, NB, degenerate="raw")
        t_sens = time.perf_counter() - t0
        dirs = [rng.uniform(-1, 1, f.shape) * (f > 1e-3) for f in base]
        h = 1e-5
        fp = make([f + h * d for f, d in zip(base, dirs)]).solve(K_GENERIC, NB)
        fm = make([f - h * d for f, d in zip(base, dirs)]).solve(K_GENERIC, NB)
        fd = (fp - fm) / (2 * h)
        ad = sum(np.einsum("bij,ij->b", s.dfreq[p], d) for p, d in enumerate(dirs))
        ok = nondegenerate(s.freqs)
        err = np.abs(fd - ad) / np.abs(ad)
        # Σ_e f_e ∂f/∂f_e = 0: scaling every fraction leaves the spectrum unchanged.
        homog = sum(np.einsum("bij,ij->b", s.dfreq[p], f) for p, f in enumerate(base))
        hom_rel = np.abs(homog) / np.abs(s.dfreq).sum(axis=(0, 2, 3))
        print(
            f"  {name:13s} ndof={solver.ndof:6d} parts={len(solver.parts)}  sens+solve {t_sens:5.2f} s  "
            f"max rel err (non-degenerate) {err[ok].max():.1e}  homogeneity {hom_rel[ok].max():.1e}"
        )
        results[f"frac_{name}_err"] = err
        results[f"frac_{name}_freqs"] = s.freqs


SHAPE = TripodHole(r=0.18, w=0.105, hub=0.06, pad_t=0.11, channel=0.11, rr=0.03, blend=0.03, junction=0.03)
FREE = ("r", "w", "hub", "pad_t", "channel", "rr", "blend", "junction")


def check_shape_parameters(results: dict) -> None:
    n = 48
    cell = ParametricCell(LAT, SHAPE, FREE)
    th = cell.theta0()
    h = 2e-5  # µm
    print(f"\n2. Shape parameters through the smoothed raster (n = {n}), h = {h * 1e3:g} nm")
    for parity in ("even", "odd"):
        make = lambda t: BlochSolver(
            LAT, cell.fraction(t, n), SI_ISOTROPIC, model="slab_p", thickness=0.22, parity=parity, order=4
        )
        t0 = time.perf_counter()
        s = sensitivities(make(th), K_GENERIC, NB, degenerate="raw")
        ad = cell.fraction_vjp(th, s.dfreq[0], n)  # (nb, nθ)
        t_ad = time.perf_counter() - t0
        fd = np.zeros_like(ad)
        crossed = 0
        for i in range(len(th)):
            e = np.zeros(len(th))
            e[i] = h
            fp, fm = cell.fraction(th + e, n), cell.fraction(th - e, n)
            crossed += int(((fp > 1e-3) != (fm > 1e-3)).sum())  # elements entering/leaving the mesh
            fd[:, i] = (make(th + e).solve(K_GENERIC, NB) - make(th - e).solve(K_GENERIC, NB)) / (2 * h)
        err = rel_err(fd, ad)
        ok = nondegenerate(s.freqs)
        print(f"  {parity}: adjoint (solve + sensitivities + VJP) {t_ad:.2f} s; FD {2 * len(th)} solves; elements crossing min_fraction: {crossed}")
        print("    band  f (GHz)   " + " ".join(f"{p:>9s}" for p in FREE) + "   max err")
        for b in range(NB):
            tag = "" if ok[b] else " (degenerate/acoustic: excluded)"
            print(f"    {b + 1:4d} {s.freqs[b] / 1e9:8.4f}  " + " ".join(f"{x:9.2e}" for x in ad[b] / 1e9) + f"   {err[b].max():.1e}{tag}")
        print(f"    max rel err over non-degenerate bands: {err[ok].max():.1e}  (gradients in GHz/µm)")
        results[f"shape_{parity}_ad"] = ad
        results[f"shape_{parity}_fd"] = fd
        results[f"shape_{parity}_freqs"] = s.freqs


def check_degenerate(results: dict) -> None:
    n = 48
    cell = ParametricCell(LAT, SHAPE, FREE)
    th = cell.theta0()
    J = cell.fraction_jacobian(th, n)
    print("\n3. Degenerate clusters (slab_p even, n = 48)")
    G = np.zeros(2)
    K = np.pi / LAT.a * np.array([4 / 3, 0.0])
    make = lambda t: BlochSolver(LAT, cell.fraction(t, n), SI_ISOTROPIC, model="slab_p", thickness=0.22, parity="even")
    rng = np.random.default_rng(3)
    direction = rng.uniform(-1, 1, len(th))
    direction /= np.linalg.norm(direction)
    dfrac = [J @ direction]
    for name, k in (("Γ", G), ("K", K)):
        s = sensitivities(make(th), k, NB, rtol=RTOL_CLUSTER, keep_vectors=True)
        for c in [c for c in s.clusters if len(c) > 1]:
            split = np.ptp(s.freqs[c]) / s.freqs[c].mean()
            # Cluster mean: every band of the cluster carries the mean gradient.
            mean_ad = float(np.einsum("ij,ij->", s.dfreq[0, c[0]], dfrac[0]))
            h = 1e-5
            mean_fd = (make(th + h * direction).solve(k, NB)[c].mean() - make(th - h * direction).solve(k, NB)[c].mean()) / (2 * h)
            # Cluster matrix A_ij = x_iᴴ δK x_j - ½(λ_i + λ_j) x_iᴴ δM x_j (off-diagonals included)
            # against central differences of the assembled K(k), M(k) projected on the cluster.
            solver0 = make(th)
            A = cluster_matrix(solver0, s, c, dfrac)
            X = s.X[:, c]
            Mx = solver0.matrices(k)[1] @ X
            X = X @ np.linalg.inv(np.linalg.cholesky(X.conj().T @ Mx)).conj().T
            h = 1e-5
            Kp, Mp = make(th + h * direction).matrices(k)
            Km, Mm = make(th - h * direction).matrices(k)
            dK = X.conj().T @ ((Kp - Km) @ X) / (2 * h)
            dM = X.conj().T @ ((Mp - Mm) @ X) / (2 * h)
            lam = s.lam[c]
            A_fd = dK - 0.5 * (lam[:, None] + lam[None, :]) * dM
            sub = np.linalg.eigvalsh(A) * s.freqs[c].mean() / (2 * lam.mean()) / 1e9
            print(
                f"  {name}: bands {c[0] + 1}-{c[-1] + 1} at {s.freqs[c].mean() / 1e9:.4f} GHz, mesh splitting {split:.1e}\n"
                f"     cluster mean: adjoint {mean_ad / 1e9:+.6f}, FD {mean_fd / 1e9:+.6f} GHz/µm (rel err {abs(mean_ad - mean_fd) / abs(mean_ad):.1e})\n"
                f"     cluster matrix vs FD of projected K, M: rel err {np.abs(A - A_fd).max() / np.abs(A).max():.1e} "
                f"(|A_12|/|A_11| = {abs(A[0, 1]) / abs(A[0, 0]):.2f}); subspace slopes {np.round(sub, 3)} GHz/µm"
            )
            results[f"degen_{name}_{c[0]}"] = np.array([mean_ad, mean_fd])


def check_density(results: dict) -> None:
    n = 48
    dc = DensityCell(LAT, n, R=0.03, symmetry="C3v", beta=8.0)
    # Start from the parametric cell's raster on the nodes, so the density is a real design.
    cell = ParametricCell(LAT, SHAPE, ())
    nodes = np.roll(np.roll(cell.fraction(np.zeros(0), n), 0, 0), 0, 1)
    x = np.clip(dc.from_fraction_nodes(nodes), 0, 1)
    rng = np.random.default_rng(5)
    v = rng.uniform(-1, 1, dc.n_vars)
    print(f"\n4. Free-form density: {dc.n_vars} C3v orbit variables on {n}×{n} nodes, R = 30 nm, β = 8")
    make = lambda x: BlochSolver(LAT, dc.fraction(x), SI_ISOTROPIC, model="slab_p", thickness=0.22, parity="even")
    s = sensitivities(make(x), K_GENERIC, NB, degenerate="raw")
    ad = dc.fraction_vjp(x, s.dfreq[0]) @ v
    h = 1e-5
    fp, fm = dc.fraction(x + h * v), dc.fraction(x - h * v)
    crossed = int(((fp > 1e-3) != (fm > 1e-3)).sum())
    fd = (make(x + h * v).solve(K_GENERIC, NB) - make(x - h * v).solve(K_GENERIC, NB)) / (2 * h)
    ok = nondegenerate(s.freqs)
    err = np.abs(fd - ad) / np.abs(ad)
    print(f"  bands: max rel err (non-degenerate) {err[ok].max():.1e}; elements crossing min_fraction {crossed}")
    g, dg = dc.length_scale(x)
    fd_ls = (dc.length_scale(x + h * v)[0] - dc.length_scale(x - h * v)[0]) / (2 * h)
    print(f"  length-scale measures (solid, void) {g}: rel err {np.abs(fd_ls - dg @ v) / np.abs(dg @ v)}")
    results["density_err"] = err


def main() -> None:
    OUT.mkdir(exist_ok=True)
    results: dict = {}
    t0 = time.perf_counter()
    check_fraction_level(results)
    check_shape_parameters(results)
    check_degenerate(results)
    check_density(results)
    np.savez(OUT / "mech_adjoint_gradcheck.npz", **results)
    print(f"\ntotal {time.perf_counter() - t0:.0f} s; saved {OUT / 'mech_adjoint_gradcheck.npz'}")


if __name__ == "__main__":
    main()
