"""
Fast g0 of a finite cavity: the optical cavity mode by targeted PWE (PWESolver.solve_near) in a
window of the mechanical grid, against a fixed mechanical mode.

The optical mode is localised to a few columns and decays ×~0.1 per shield row, so the optics
needs only a block of the mechanical domain (`GridWindow`: element columns i0 … i0 + n1w − 1,
rows j0 … j0 + n2w − 1, centred in y). The window is periodic in the PWE; the pattern is cut at
its edges, where the field is negligible. Its fill is the block of the mechanical domain's
fine raster, so the optical voxels are blocks of mechanical elements and the strain sub-points
stay on element centroids.

The cold start (`local_bloch_start`) is the defect cell's X-point band-edge mode tiled over the
window with the Bloch phase e^{iπx/a}, under Gaussian envelopes (GPT07: ~8 Jacobi–Davidson
iterations to g0 within 1 %). In an optimisation the previous design's mode is the start
(`CavityOptics.solve(x0=...)`): 2 iterations after a 2 nm change of every hole.

g0 is a functional of the localised part of the optical mode; the slowly converging residual
components are delocalised (air and waveguide-side) supercell modes near the cavity frequency,
so g0 converges well before the global residual (GPT07: g0 within ~1 % of meep at a relative
residual of 3e-2; research/cavity_gpt07_g0_pwe.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .coupling import (
    MechMode,
    advected_fill,
    block_mean,
    displacement,
    moving_boundary_shift,
    optical_mode,
    photoelastic_shift,
    strain_on_voxels,
)
from .geometry import Lattice2D
from .materials import SI_P
from .optics_pwe import PWESolver, slab_cell


@dataclass
class GridWindow:
    """A block of the mechanical grid (n1m × n2m elements over a rectangular cell L1 × L2)."""

    i0: int
    n1w: int
    j0: int
    n2w: int
    n1m: int
    n2m: int
    L1: float
    L2: float

    @classmethod
    def around(cls, n1m: int, n2m: int, L1: float, L2: float, x_lo: float, x_hi: float, half_height: float) -> GridWindow:
        """Window over centred x ∈ [x_lo, x_hi) (snapped to elements) and |y| < half_height (symmetric)."""
        dx, dy = L1 / n1m, L2 / n2m
        i0 = int(round((x_lo + L1 / 2) / dx))
        i1 = int(round((x_hi + L1 / 2) / dx))
        half = min(int(round(half_height / dy)), n2m // 2)
        j0 = n2m // 2 - half if n2m % 2 == 0 else (n2m - 2 * half) // 2
        return cls(i0, i1 - i0, j0, 2 * half, n1m, n2m, L1, L2)

    @property
    def lattice(self) -> Lattice2D:
        return Lattice2D(np.array([[self.n1w * self.L1 / self.n1m, 0.0], [0.0, self.n2w * self.L2 / self.n2m]]))

    @property
    def centre(self) -> tuple[float, float]:
        """Window centre in the mechanical cell's centred coordinates (µm)."""
        return ((self.i0 + self.n1w / 2) / self.n1m - 0.5) * self.L1, ((self.j0 + self.n2w / 2) / self.n2m - 0.5) * self.L2

    def fine(self, fine_full: np.ndarray) -> np.ndarray:
        m1, m2 = fine_full.shape[0] // self.n1m, fine_full.shape[1] // self.n2m
        return fine_full[self.i0 * m1 : (self.i0 + self.n1w) * m1, self.j0 * m2 : (self.j0 + self.n2w) * m2]

    def to_mode(self, s1, s2):
        """Window centred lattice coordinates → the mechanical cell's."""
        lw = self.lattice.vectors
        xc, yc = self.centre
        return (s1 * lw[0, 0] + xc) / self.L1, (s2 * lw[1, 1] + yc) / self.L2


def defect_mode_at_x(frac_cell: np.ndarray, pitch: float, height: float, thickness: float, slab_height: float, nz: int, eps_si: float,
                     edge_guess: float, in_row_half: float, n_near: int = 4):
    """
    The defect cell's row-localised X-point band-edge mode nearest edge_guess (THz): targeted PWE
    on the one-column cell (fraction (px, n2), periodic along x), n_near modes. Returns
    (f THz, periodic H (3, px, n2, nz)).
    """
    lat = Lattice2D(np.array([[pitch, 0.0], [0.0, height]]))
    s = PWESolver(slab_cell(lat, frac_cell, thickness, slab_height, nz, eps_si), "even")
    f = s.solve_near(np.array([np.pi / pitch, 0.0]), edge_guess, n_bands=n_near, tol=1e-6)
    n2 = frac_cell.shape[1]
    y = (np.arange(n2) + 0.5) / n2 * height - height / 2
    best = None
    for j in range(len(f)):
        h = s.h_field(j)
        w = (np.abs(h) ** 2).sum(axis=(0, 3))
        loc = w[:, np.abs(y) < in_row_half].sum() / w.sum()
        if loc >= 0.5 and (best is None or abs(f[j] - edge_guess) < abs(best[0] - edge_guess)):
            best = (float(f[j]), h)
    if best is None:
        raise RuntimeError("no row-localised X-point mode of the defect cell near the guess")
    return best


def local_bloch_start(s: PWESolver, frac_cell: np.ndarray, win: GridWindow, pitch: float, centre_col: int, thickness: float, slab_height: float,
                      nz: int, eps_si: float, edge_guess: float, in_row_half: float, widths=(1.5, 2.5, 4.0),
                      verbose: bool = False, envelopes=None) -> tuple[np.ndarray, float]:
    """
    Start vectors (flat, for s.solve_near) and the defect band edge (THz): the defect cell's X-point
    band-edge mode tiled over the window's columns with its Bloch phase, under Gaussian envelopes.
    frac_cell: the defect cell on its own (px, n2) grid, n2 the window's; centre_col: the window
    column of the defect cell. (Column blocks of the cavity raster are not unit cells when holes
    cross column boundaries, as GPT07's daggers do, so per-column modes are not used.)
    envelopes: functions of ξ = x / pitch (x = 0 at the defect column) replacing the Gaussians,
    e.g. Hermite–Gauss envelopes for the higher-order cavity modes.
    """
    n1, n2 = s.shape[:2]
    ncol = int(round(win.lattice.vectors[0, 0] / pitch))
    px = n1 // ncol
    if px * ncol != n1 or frac_cell.shape != (px, n2):
        raise ValueError("the window must hold a whole number of columns of the defect cell's px voxels")
    t0 = time.perf_counter()
    f_edge, h_c = defect_mode_at_x(frac_cell, pitch, win.lattice.vectors[1, 1], thickness, slab_height, nz, eps_si, edge_guess, in_row_half)
    if verbose:
        print(f"    start: defect band edge {f_edge:.3f} THz ({time.perf_counter() - t0:.0f} s)", flush=True)
    hc = np.concatenate([h_c] * ncol, axis=1)  # (3, n1, n2, nz)
    x = (np.arange(n1) + 0.5) / n1 * win.lattice.vectors[0, 0] - (centre_col + 0.5) * pitch  # 0 at the defect column centre
    bloch = hc * np.exp(1j * np.pi * x / pitch)[None, :, None, None]
    s._set_k(np.zeros(2))
    cols = []
    if envelopes is not None:
        envs = [np.asarray(fn(x / pitch), float) for fn in envelopes]
    else:
        envs = [np.exp(-0.5 * (x / (w * pitch)) ** 2) for w in widths] + [(x / pitch) * np.exp(-0.5 * (x / (widths[min(1, len(widths) - 1)] * pitch)) ** 2)]
    for env in envs:
        H = np.fft.fftn(bloch * env[None, :, None, None], axes=(1, 2, 3)) / np.prod(s.shape)
        cols.append(s._flat(s._project(s._amplitudes(H[None])))[:, 0])
    return np.column_stack(cols), f_edge


@dataclass
class CavityG0:
    f_o: float  # THz
    g_mb: float  # Hz (g/2π)
    g_pe: float
    residual: float
    outer: int
    inner: int
    seconds: float
    solver: PWESolver

    @property
    def g0(self) -> float:
        return self.g_mb + self.g_pe


class CavityOptics:
    """
    The optical side of g0 for a fixed mechanical mode on a rectangular cavity cell: build once
    per design (`set_design`), then `solve` (cold from the local Bloch start, or warm from a
    previous mode).
    """

    def __init__(self, mode: MechMode, win: GridWindow, pitch: float, centre_col: int, thickness: float = 0.22, nz: int = 32,
                 slab_height: float = 1.5, eps_si: float = 3.48**2, in_row_half: float | None = None, p_si: np.ndarray = SI_P):
        self.mode, self.win, self.pitch, self.centre_col = mode, win, pitch, centre_col
        self.t, self.nz, self.H, self.eps_si, self.p = thickness, nz, slab_height, eps_si, p_si
        self.in_row_half = in_row_half if in_row_half is not None else 0.5 * pitch

    def set_design(self, fine_full: np.ndarray, ratio: int = 1) -> None:
        """fine_full: the mechanical domain's supersampled Si raster; ratio: mechanical elements per optical voxel side."""
        w = self.win
        fine = w.fine(fine_full)
        n1, n2 = w.n1w // ratio, w.n2w // ratio
        self.fine, self.ratio = fine, ratio
        self.frac = block_mean(fine, n1, n2)
        self.cell = slab_cell(w.lattice, self.frac, self.t, self.H, self.nz, self.eps_si)
        u_fn = lambda a, b, z: displacement(self.mode, *w.to_mode(a, b), z)  # noqa: E731
        self.dF = advected_fill(self.cell, fine, u_fn)
        self.S = strain_on_voxels(self.mode, self.cell, m=ratio, to_mode=w.to_mode)

    def g0_of(self, s: PWESolver, band: int = 0) -> tuple[float, float]:
        opt = optical_mode(s, band)
        k = opt.omega * 1e6 * self.mode.x_zpf / (2 * np.pi)
        return moving_boundary_shift(opt, self.dF) * k, photoelastic_shift(opt, self.S, self.p) * k

    def solve(self, f_guess: float, x0: np.ndarray | None = None, defect_cell: np.ndarray | None = None, above: float = 4.0,
              tol: float = 2e-2, inner: int = 40, maxiter: int = 40, callback=None, verbose: bool = False,
              edge_guess: float | None = None, widths=(1.5, 2.5, 4.0), min_iter: int = 1) -> CavityG0:
        """
        Cold (x0 None: start from defect_cell, the defect cell's fill on its own (px, n2) grid;
        its X-point band edge nearest edge_guess (default f_guess − 5: the GPT07 cavity lies a few
        THz above its edge); target = that edge + above) or warm (x0: the previous design's mode,
        target f_guess).
        """
        t0 = time.perf_counter()
        s = PWESolver(self.cell, "even")
        s._set_k(np.zeros(2))
        if x0 is None:
            x0, edge = local_bloch_start(s, defect_cell, self.win, self.pitch, self.centre_col, self.t, self.H, self.nz, self.eps_si,
                                         f_guess - 5 if edge_guess is None else edge_guess, self.in_row_half, widths=widths, verbose=verbose)
            target = edge + above
        else:
            target = f_guess

        def cb(it, f, rel, u):
            if callback is not None:
                callback(it, f, rel, u, self)

        f = s.solve_near(np.zeros(2), target, n_bands=1, x0=x0, tol=tol, inner=inner, inner_fine=inner, maxiter=maxiter, callback=cb,
                         min_iter=min_iter)
        mb, pe = self.g0_of(s)
        return CavityG0(float(f[0]), mb, pe, float(s.residuals[0]) if len(s.residuals) else np.nan, s.jd_info["outer"],
                        s.jd_info["inner"], time.perf_counter() - t0, s)
