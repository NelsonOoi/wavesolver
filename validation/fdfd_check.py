"""
wavesolver.optics_fdfd on cases with a known answer, smallest first. Run with MKL_THREADING_LAYER=TBB for threads.

  strip   : n_eff of a 500 × 220 nm Si strip against optics_pwe on a one-cell supercell at the same k.
  launch  : a mode launched into a uniform strip ending in the absorber: power below the plane (1), power above it, reflection from the end.
  operator: A is symmetric, the solve meets its residual, and a discrete mode solves A in the strip.
  disk    : Q and f of a whispering-gallery mode of a Si disk as the z grid is refined: how few z cells the radiation needs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wavesolver.optics_fdfd import C_UM_THZ, SlabFDFD, ZGrid  # noqa: E402


def strip_raster(sol: SlabFDFD, m: int, width: float) -> np.ndarray:
    """Fill raster of x ≥ 0: a strip |x| < width/2, whole length."""
    F1, F2 = sol.nx * m, sol.ny * m
    x = (np.arange(F1) + 0.5) * sol.dx / m
    return (np.clip((width / 2 - x) / (sol.dx / m) + 0.5, 0, 1))[:, None] * np.ones((F1, F2))


def setup(sol: SlabFDFD, raster: np.ndarray, m: int, f: float, q_d: float = np.inf) -> None:
    sol.set_frequency(f, q_d)
    sol.set_eps(*[np.asarray(v) for v in sol.eps_maps(raster, m)])


def strip(args):
    f = 195.0
    k0 = 2 * np.pi * f / C_UM_THZ
    for dx in (0.03884, 0.0259, 0.01942):
        nx = int(round(1.5 / dx))
        sol = SlabFDFD(2 * nx * dx, 40 * dx, nx, 40, ZGrid.make(n_slab=args.n_slab), threads=args.threads)
        setup(sol, strip_raster(sol, 4, 0.5), 4, f)
        mode = sol.strip_mode(20)
        print(f"FDFD Δx {dx * 1e3:.1f} nm: n_eff {mode['n_eff']:.4f}, τ/k0 {abs(mode['tau']) / k0:.4f}; z cells {sol.nz}", flush=True)
    # PWE reference: one period along y of the strip
    from wavesolver.geometry import Lattice2D
    from wavesolver.optics_pwe import PWESolver, slab_cell

    Lx, a, N = 3.0, 0.1, 240
    ny = 8
    lat = Lattice2D(np.array([[Lx, 0.0], [0.0, a]]))
    x = (np.arange(N) + 0.5) / N * Lx - Lx / 2
    frac = (np.clip((0.25 - np.abs(x)) / (Lx / N) + 0.5, 0, 1))[:, None] * np.ones((N, ny))
    cell = slab_cell(lat, frac, 0.22, 2.0, 64, 3.48**2)
    s = PWESolver(cell, "even")
    fs = []
    for n_try in (2.40, 2.50):
        k = np.array([0.0, n_try * k0])
        fb = s.bands(k[None], 2)[0]
        fs.append((n_try, fb[0]))
        print(f"PWE k_y/k0(195) {n_try:.2f}: f {fb[0]:.3f} THz; bands {np.round(fb, 2)}", flush=True)
    (n0, f0), (n1, f1) = fs
    n_eff = n0 + (195.0 - f0) * (n1 - n0) / (f1 - f0)
    print(f"PWE n_eff at 195 THz by interpolation: {n_eff:.4f}", flush=True)


def launch(args):
    dx = args.dx
    nx, ny = int(round(1.5 / dx)), int(round(8.0 / dx))
    sol = SlabFDFD(2 * nx * dx, ny * dx, nx, ny, ZGrid.make(n_slab=args.n_slab), npml=args.npml, sigma_xy=args.sigma, threads=args.threads)
    setup(sol, strip_raster(sol, 4, 0.5), 4, 195.0, args.q_d)
    print(f"{sol.n} unknowns, {sol.nz} z cells, reach {sol.z_edges[-1]:.2f} µm", flush=True)
    t0 = time.perf_counter()
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in sol.factor().items()}, f"{time.perf_counter() - t0:.1f} s", flush=True)
    j0 = ny - args.npml - 4
    b, mode = sol.port_source(j0)
    x = sol.solve(b)
    print(f"n_eff {mode['n_eff']:.4f}; residual {sol.residual(x, b):.1e}; |b| rows {np.count_nonzero(np.abs(b) > 1e-12 * np.abs(b).max())}", flush=True)
    for j in (j0 - 100, j0 - 40, j0 - 5, j0 + 2):
        print(f"  y-segment {j}: flux {sol.flux_y(x, j):+.5f}", flush=True)
    if np.isfinite(args.q_d):
        # energy balance: with k² = k0²(1 + i/Q_d) the flux falls along the strip by (k0/Q_d) ∫ ε|E|² dV, which is what ties U/P to the launched power
        j1, j2 = j0 - 100, j0 - 40
        cx, cy = np.zeros((sol.nx, sol.ny)), np.zeros((sol.nx, sol.ny))
        m = np.zeros((sol.nx, sol.ny))
        m[:, j1 + 1 : j2 + 1] = 1.0  # the cells between the segments j1 + ½ and j2 + ½ … evaluated on the edges in that span
        ex, ey, ez = sol.split(x)
        Wx = np.real(sol.mass_x)
        Wy = np.real(sol.mass_y)
        Wz = np.real(sol.mass_z)
        u = (Wx * sol.eps_x * np.abs(ex) ** 2)[:, j1:j2].sum() + (Wy * sol.eps_y * np.abs(ey) ** 2)[:, j1 + 1 : j2 + 1].sum() + (Wz * sol.eps_z * np.abs(ez) ** 2)[:, j1:j2].sum()
        p1, p2 = sol.flux_y(x, j1), sol.flux_y(x, j2)
        print(f"  energy balance: flux change {abs(p2) - abs(p1):+.6f}, (k0/Q_d) ∫ε|E|²dV {sol.k0 / args.q_d * u:+.6f}, ratio {(abs(p2) - abs(p1)) / (sol.k0 / args.q_d * u):.5f}", flush=True)
    # the mode against the field below the plane, and what comes back from the far absorber
    Ex, _, _ = sol.fields(x)
    jm = (j0 - 150, j0 - 100, j0 - 50)
    prof = mode["Ex"]
    amp = [np.vdot(prof, Ex[:, j - 1, :]) / np.vdot(prof, prof) for j in jm]
    decay = np.exp(1j * mode["theta"] * 50)
    print(f"  mode amplitude along the strip: {np.abs(amp)}; per 50 cells {abs(amp[1] / amp[0]):.5f}, expected {abs(np.exp(1j * mode['theta'] * 50)):.5f}; phase error {np.angle(amp[1] / amp[0] * np.conj(decay)):+.2e} rad", flush=True)
    th = mode["theta"]
    # forward plus backward at three points: a standing-wave ratio gives the reflection from the −y absorber
    A = np.array([[np.exp(1j * th * (j - j0)), np.exp(-1j * th * (j - j0))] for j in jm])
    c, *_ = np.linalg.lstsq(A, np.array(amp), rcond=None)
    print(f"  fit: incident amplitude {abs(c[0]):.5f}, reflected from the absorber {abs(c[1]):.2e} (|r| {abs(c[1] / c[0]):.2e})", flush=True)


def operator(args):
    dx = 0.03884
    nx, ny = 30, 40
    sol = SlabFDFD(2 * nx * dx, ny * dx, nx, ny, ZGrid.make(n_slab=args.n_slab), threads=args.threads)
    setup(sol, strip_raster(sol, 4, 0.5), 4, 195.0, 200.0)
    rng = np.random.default_rng(0)
    a, c = rng.standard_normal(sol.n) + 1j * rng.standard_normal(sol.n), rng.standard_normal(sol.n) + 1j * rng.standard_normal(sol.n)
    print(f"symmetry: |aᵀAc − cᵀAa|/|aᵀAc| = {abs(a @ sol.apply(c) - c @ sol.apply(a)) / abs(a @ sol.apply(c)):.1e}", flush=True)
    sol.factor()
    b = rng.standard_normal(sol.n) + 0j
    print(f"residual of a random right-hand side: {sol.residual(sol.solve(b), b):.1e}", flush=True)
    mode = sol.strip_mode(20)
    b, _ = sol.port_source(20, mode, reach=8)
    th = mode["theta"]
    ex = np.zeros(sol.shape_x, complex)
    ey = np.zeros(sol.shape_y, complex)
    ez = np.zeros(sol.shape_z, complex)
    for j in range(1, ny):
        ex[:, j - 1, :] = mode["Ex"] * sol.lxS[:, None] * np.exp(1j * th * (j - 20))
        ez[:, j - 1, :] = mode["Ez"] * sol.lzS[None, :] * np.exp(1j * th * (j - 20))
    for j in range(ny):
        ey[:, j, :] = mode["Ey"] * sol.lyS[j] * np.exp(1j * th * (j + 0.5 - 20))
    e = np.r_[ex.ravel(), ey.ravel(), ez.ravel()]
    r = sol.apply(e)
    rx, ry, rz = sol.split(r)
    mid = (np.abs(rx[:, 14:24, :]).max(), np.abs(ry[:, 14:24, :]).max(), np.abs(rz[:, 14:24, :]).max())
    print(f"the discrete mode in A, away from the y absorbers: |A e| max {max(mid):.1e} against |C'RC e| {np.abs(sol.C.T @ (sol.R * (sol.C @ e))).max():.1e}", flush=True)


def disk(args):
    """A Si disk of radius r on the grid; the mode of the given sector nearest f: Q as the z grid changes. The in-plane absorber is far from the disk."""
    r, dx = args.r, args.dx
    half = r + 1.6
    nx = int(round(half / dx))
    ny = 2 * nx
    grids = {"coarse 2 slab": dict(n_slab=2, dz_air=0.055, growth=1.5, dz_max=0.3, reach=0.6, n_pml=4), "3 slab (default)": dict(), "3 slab, 6 pml": dict(n_pml=6), "3 slab, reach 1.0": dict(reach=1.0),
             "3 slab, dz_max 0.15": dict(dz_max=0.15, n_pml=8), "5 slab, growth 1.25": dict(n_slab=5, dz_air=0.022, growth=1.25, dz_max=0.15, reach=1.0, n_pml=8), "8 slab, growth 1.15": dict(n_slab=8, dz_air=0.014, growth=1.15, dz_max=0.1, reach=1.2, n_pml=12)}
    f0 = args.f
    m = 4
    F1, F2 = nx * m, ny * m
    X = (np.arange(F1) + 0.5) * dx / m
    Y = (np.arange(F2) + 0.5) * dx / m - ny * dx / 2
    d = np.sqrt(X[:, None] ** 2 + Y[None, :] ** 2)
    raster = np.clip((r - d) / (dx / m) + 0.5, 0, 1)
    for name, kw in grids.items():
        zg = ZGrid.make(**kw)
        sol = SlabFDFD(2 * nx * dx, ny * dx, nx, ny, zg, npml=args.npml, sigma_xy=args.sigma, sigma_z=args.sigma_z, threads=args.threads)
        setup(sol, raster, m, f0, 300.0)
        t0 = time.perf_counter()
        sol.factor()
        t1 = time.perf_counter()
        f, Q, _ = sol.modes(f0, n=args.n_modes)
        s = ", ".join(f"{a:.3f} THz Q {b:.0f}" for a, b in zip(f[:3], Q[:3]))
        print(f"{name}: {sol.nz} z cells, reach {sol.z_edges[-1]:.2f}; factor {t1 - t0:.0f} s, modes {time.perf_counter() - t1:.0f} s; {s}; {sol.lu.stats['peak_gb']:.1f} GB", flush=True)
        sol.lu.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["strip", "launch", "operator", "disk"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--n_slab", type=int, default=3)
    ap.add_argument("--dx", type=float, default=0.03884)
    ap.add_argument("--npml", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=40.0)
    ap.add_argument("--sigma_z", type=float, default=12.0)
    ap.add_argument("--q_d", type=float, default=np.inf)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--f", type=float, default=195.0)
    ap.add_argument("--n_modes", type=int, default=4)
    args = ap.parse_args()
    {"strip": strip, "launch": launch, "operator": operator, "disk": disk}[args.what](args)
