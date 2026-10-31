"""
Validation of wavesolver.coupling (the two first-order terms of g0) and of PWESolver.voxel_sensitivity.

0. voxel_sensitivity summed over z with g(z) against the existing in-plane `sensitivities`.
1. strain() of a slab_p mode against central differences of displacement() (x, y, z).
a. Moving boundary on the blade cell (a = 448 nm): δF from `advected_fill` (−u·∇f on an m×
   supersampled raster, exact face motion) and from the coarse −u·∇F on the optical voxels,
   against finite differences that rebuild the geometry on the same grid (exact-area
   rasterisation, so tiny steps are clean):
   - in-plane dilation of the hole about the cell centre, u = ε(x, y, 0), vs the hole scaled
     by 1 ± ε (the hole lies inside the cell, so u = ε r is continuous where ∇F ≠ 0);
   - thickness scaling u_z = ε z vs the slab rebuilt with t(1 ± ε); and the same for the
     unpatterned slab against the analytic TE0 dispersion;
   - rigid translations (x, y, z): physically no shift. The FD rebuild is not exactly zero on
     a coarse grid (the rasterised cell is not translation invariant); δF converges to it.
   The FD difference of the exact fills, δF_exact, is also contracted with ∂λ/∂F: that tests
   voxel_sensitivity alone, and δF = −u·∇F vs δF_exact is then the representation error.
b. Photoelastic shift of the unpatterned 220 nm slab (TE0, E ∥ y, k ∥ x) under uniform strain,
   against the analytic TE0 frequency with n_y changed by Δ(1/n²)_yy = (p S)_yy, for Si with
   [100] and [110] along x.

    pixi run python validation/g0_validate.py
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import shapely
from scipy.optimize import brentq
from shapely import affinity
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wavesolver.coupling import (  # noqa: E402
    advected_fill,
    advected_fill_coarse,
    block_mean,
    displacement,
    mech_mode,
    moving_boundary_shift,
    optical_mode,
    photoelastic_shift,
    strain,
    voxel_points,
)
from wavesolver.geometry import HexLattice, blade_hole, solid_fraction  # noqa: E402
from wavesolver.materials import SI_ISOTROPIC, SI_P, rot_z, rotate_stiffness  # noqa: E402
from wavesolver.mech_bloch import BlochSolver  # noqa: E402
from wavesolver.optics_pwe import C_UM_PER_PS, PWESolver, slab_cell  # noqa: E402

A, T, N_SI = 0.448, 0.22, 3.48
OUT = ROOT / "validation/out"


def exact_fraction(lattice, holes, n: int) -> np.ndarray:
    """Solid fraction of each lattice-aligned grid cell from exact polygon areas (as solid_fraction's layout)."""
    V = lattice.vectors
    s = np.arange(n + 1) / n - 0.5
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    corners = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    pts = np.stack([s[i + c1][..., None] * V[0] + s[j + c2][..., None] * V[1] for c1, c2 in corners], axis=2)
    cells = shapely.polygons(pts.reshape(-1, 5, 2))
    shape = unary_union([affinity.translate(h, *(p * V[0] + q * V[1])) for h in holes for p, q in itertools.product((-1, 0, 1), repeat=2)])
    shapely.prepare(shape)
    area = abs(np.linalg.det(V)) / n**2
    return 1.0 - (shapely.area(shapely.intersection(cells, shape)) / area).reshape(n, n)


def te0_thz(beta: float, n: float = N_SI, t: float = T) -> float:
    def f(k0):
        kappa = np.sqrt(n**2 * k0**2 - beta**2)
        gamma = np.sqrt(beta**2 - k0**2)
        return kappa * np.tan(kappa * t / 2) - gamma

    lo = beta / n * (1 + 1e-12)
    hi = min(beta, np.sqrt((np.pi / t) ** 2 + beta**2) / n) * (1 - 1e-9)
    return brentq(f, lo, hi, xtol=1e-15) * C_UM_PER_PS / (2 * np.pi)


def solve(lattice, frac, k, nb, thickness=T, height=1.5, nz=64, tol=1e-6):
    s = PWESolver(slab_cell(lattice, frac, thickness, height, nz, N_SI**2), parity="even")
    f = s.solve(k, nb, tol=tol, maxiter=400)
    return f, s


def cart(lattice, s1, s2):
    V = lattice.vectors
    return s1 * V[0, 0] + s2 * V[1, 0], s1 * V[0, 1] + s2 * V[1, 1]


def check_voxel_sensitivity(lattice, frac, k):
    f, s = solve(lattice, frac, k, 2, tol=1e-5)
    g = s.cell.z_cover
    ref = s.sensitivities(range(2))
    for b in range(2):
        lam = (2 * np.pi * f[b] / C_UM_PER_PS) ** 2
        mine = (s.voxel_sensitivity(b) * g).sum(axis=2) * C_UM_PER_PS / (4 * np.pi * np.sqrt(lam))
        print(f"  band {b}: Σ_z g ∂λ/∂F vs sensitivities(): max rel diff {np.abs(mine - ref[b]).max() / np.abs(ref[b]).max():.1e}")


def check_strain(lattice, frac):
    solver = BlochSolver(lattice, frac, SI_ISOTROPIC, model="slab_p", thickness=T, parity="even")
    k = np.array([2.0, 1.1])
    freqs, X = solver.solve(k, 6, vectors=True)
    mode = mech_mode(solver, k, freqs[4], X[:, 4], real=False)
    rng = np.random.default_rng(0)
    n = frac.shape[0]
    # Points inside kept elements, away from element edges (Q1 gradients jump across them).
    i, j = rng.integers(0, n, 40), rng.integers(0, n, 40)
    keep = frac[i, j] > 0.5
    s1 = (i[keep] + rng.uniform(0.2, 0.8, keep.sum())) / n - 0.5
    s2 = (j[keep] + rng.uniform(0.2, 0.8, keep.sum())) / n - 0.5
    z = np.array([-0.08, 0.03, 0.1])
    S = strain(mode, s1, s2, z)
    h = 1e-6
    rb = lattice.reciprocal / (2 * np.pi)
    U = lambda a, b, c: displacement(mode, a, b, c, renormalise=False)
    dx = (U(s1 + h * rb[0, 0], s2 + h * rb[1, 0], z) - U(s1 - h * rb[0, 0], s2 - h * rb[1, 0], z)) / (2 * h)
    dy = (U(s1 + h * rb[0, 1], s2 + h * rb[1, 1], z) - U(s1 - h * rb[0, 1], s2 - h * rb[1, 1], z)) / (2 * h)
    dz = (U(s1, s2, z + h) - U(s1, s2, z - h)) / (2 * h)
    fd = np.array([dx[0], dy[1], dz[2], dz[1] + dy[2], dz[0] + dx[2], dy[0] + dx[1]])
    print(f"  {keep.sum()} points × {len(z)} heights: max |S − FD| / max|S| = {np.abs(S - fd).max() / np.abs(S).max():.1e}; "
          f"m_eff = {mode.mass:.3e} kg, x_zpf = {mode.x_zpf:.3e} m at {mode.freq / 1e9:.3f} GHz")


def check_moving_boundary(lattice, hole, results):
    n, nz, height = 24, 64, 1.5
    k = np.pi / A * np.array([1.0, 1 / np.sqrt(3)])  # M
    nb = 2
    supersample = (4, 8, 16)
    fine = {m: exact_fraction(lattice, [hole], n * m) for m in supersample}
    frac = exact_fraction(lattice, [hole], n)
    ring = np.concatenate([frac[0], frac[-1], frac[:, 0], frac[:, -1]])
    print(f"  grid {n}×{n}×{nz}; fill on the cell boundary min {ring.min():.3f} (1 → hole inside the cell); "
          f"|block mean of fine raster − fraction| {max(np.abs(block_mean(f, n, n) - frac).max() for f in fine.values()):.1e}")
    f0, s0 = solve(lattice, frac, k, nb)
    opts = [optical_mode(s0, b) for b in range(nb)]

    def field(kind):
        def u_fn(s1, s2, z):
            x, y = cart(lattice, s1, s2)
            one = np.ones(np.shape(x) + np.shape(z))
            zc = np.clip(z, -T / 2, T / 2) * one
            return {
                "dilation xy": np.array([x[..., None] * one, y[..., None] * one, 0 * one]),
                "thickness z": np.array([0 * one, 0 * one, zc]),
                "translation x": np.array([one, 0 * one, 0 * one]),
                "translation y": np.array([0 * one, one, 0 * one]),
                "translation z": np.array([0 * one, 0 * one, one]),
            }[kind]
        return u_fn

    kinds = ("dilation xy", "thickness z", "translation x", "translation y", "translation z")
    s1, s2, z = voxel_points(s0.cell)
    s1, s2 = np.broadcast_arrays(s1, s2)
    pred = {}
    for kind in kinds:
        coarse = advected_fill_coarse(s0.cell, field(kind)(s1, s2, z))
        pred[kind] = {1: np.array([moving_boundary_shift(o, coarse) for o in opts])}
        for m in supersample:
            dF = advected_fill(s0.cell, fine[m], field(kind))
            pred[kind][m] = np.array([moving_boundary_shift(o, dF) for o in opts])
    eps = 2e-3
    F = lambda fr, t: fr[:, :, None] * slab_cell(lattice, fr, t, height, nz, N_SI**2).z_cover[None, None, :]
    fd, exact_dF = {}, {}
    for name, build in (
        ("dilation xy", lambda e: (exact_fraction(lattice, [affinity.scale(hole, 1 + e, 1 + e, origin=(0, 0))], n), T)),
        ("thickness z", lambda e: (frac, T * (1 + e))),
        ("translation x", lambda e: (exact_fraction(lattice, [affinity.translate(hole, e, 0)], n), T)),
        ("translation y", lambda e: (exact_fraction(lattice, [affinity.translate(hole, 0, e)], n), T)),
    ):
        (fp_, tp), (fm_, tm) = build(eps), build(-eps)
        fp = solve(lattice, fp_, k, nb, thickness=tp)[0]
        fm = solve(lattice, fm_, k, nb, thickness=tm)[0]
        fd[name] = (fp - fm) / (2 * eps * f0)
        dF = (F(fp_, tp) - F(fm_, tm)) / (2 * eps)
        exact_dF[name] = np.array([(o.dlam_dF * dF).sum() / (2 * o.lam) for o in opts])
    scale = np.abs(fd["dilation xy"])
    fmt = lambda a: np.array2string(np.asarray(a), precision=5)
    print(f"  bands at M: {np.round(f0, 3)} THz. δω/ω per unit ε (translations: per µm, errors relative to |dilation|)")
    for kind in kinds:
        ref = fd.get(kind)
        den = scale if kind.startswith("translation") else (np.abs(ref) if ref is not None else None)
        print(f"    {kind}: FD rebuild {fmt(ref) if ref is not None else '— (physically 0)'}"
              + (f"; ∂λ/∂F·δF_exact err {np.array2string(np.abs(exact_dF[kind] - ref) / den, precision=1)}" if ref is not None else ""))
        for m, v in pred[kind].items():
            err = np.abs(v - ref) / den if ref is not None else np.abs(v) / scale
            label = "coarse −u·∇F" if m == 1 else f"advected_fill m={m:2d}"
            print(f"        {label:22s} {fmt(v)}  {'rel err' if ref is not None else '|·|/|dilation|'} {np.array2string(err, precision=4)}")
        results[f"mb_{kind}"] = np.array([pred[kind][m] for m in (1, *supersample)] + [ref if ref is not None else np.zeros(nb)])

    # Unpatterned slab: thickness scaling against the analytic TE0 dispersion (face term only).
    lat8 = HexLattice(A)
    for nz_, h_ in ((64, 2.0), (128, 2.0)):
        f0s, ss = solve(lat8, np.ones((8, 8)), np.array([6.0, 0.0]), 1, height=h_, nz=nz_)
        p = moving_boundary_shift(optical_mode(ss, 0), advected_fill(ss.cell, np.ones((32, 32)), field("thickness z")))
        dt = 1e-6
        exact = (te0_thz(6.0, t=T * (1 + dt)) - te0_thz(6.0, t=T * (1 - dt))) / (2 * dt * te0_thz(6.0))
        print(f"  unpatterned slab, β = 6/µm, dz = {1000 * h_ / nz_:.1f} nm: (t/ω) ∂ω/∂t {p:.5f}, analytic {exact:.5f}, rel err {abs(p - exact) / abs(exact):.2e}")
        results[f"mb_slab_nz{nz_}"] = np.array([p, exact])


def check_photoelastic(results):
    lat8 = HexLattice(A)
    beta = 6.0
    f0, s = solve(lat8, np.ones((8, 8)), np.array([beta, 0.0]), 1, height=2.0, nz=128)
    opt = optical_mode(s, 0)
    print(f"  TE0 at β = {beta}/µm: PWE {f0[0]:.3f} THz, analytic {te0_thz(beta):.3f} THz; |E_x|²+|E_z|² share {((np.abs(opt.E[0])**2 + np.abs(opt.E[2])**2).sum() / (np.abs(opt.E)**2).sum()):.1e}")
    dn = 1e-6
    dlnw_dn = (te0_thz(beta, N_SI + dn) - te0_thz(beta, N_SI - dn)) / (2 * dn * te0_thz(beta))
    for label, R in (("[100] ∥ x", np.eye(3)), ("[110] ∥ x", rot_z(45))):
        p = rotate_stiffness(SI_P, R)
        for comp, name in ((1, "S_yy"), (0, "S_xx"), (2, "S_zz"), (5, "S_xy (shear)")):
            S = np.zeros((6,) + s.cell.eps.shape)
            S[comp] = 1.0
            pred = photoelastic_shift(opt, S, p)
            d_eta = p[1, comp] if abs(p[1, comp]) > 1e-12 else 0.0  # Δ(1/n²)_yy per unit strain
            exact = dlnw_dn * (-N_SI**3 * d_eta / 2)
            tag = f"rel err {abs(pred - exact) / abs(exact):.2e}" if abs(exact) > 0 else f"(exact 0; |pred| = {abs(pred):.1e})"
            print(f"    {label} {name:13s} p_2{comp + 1} = {d_eta:+.4f}: δω/ω per unit strain {pred:+.5f}, analytic {exact:+.5f}  {tag}")
            results[f"pe_{label[:5]}_{name[:4]}"] = np.array([pred, exact])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    results: dict = {}
    t0 = time.perf_counter()
    lattice = HexLattice(A)
    hole = blade_hole(0.172, 0.093, 0.025)
    print("0. voxel_sensitivity vs sensitivities (blade cell, 24×24×64, k = M)")
    check_voxel_sensitivity(lattice, solid_fraction(lattice, [hole], 24), np.pi / A * np.array([1.0, 1 / np.sqrt(3)]))
    print("1. strain() vs FD of displacement() (blade cell, slab_p even, n = 24)")
    check_strain(lattice, solid_fraction(lattice, [hole], 24))
    print("a. moving boundary")
    check_moving_boundary(lattice, hole, results)
    print("b. photoelastic, unpatterned slab, uniform strain (Lz = 2 µm, dz = 15.6 nm)")
    check_photoelastic(results)
    np.savez(OUT / "g0_validate.npz", **results)
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
