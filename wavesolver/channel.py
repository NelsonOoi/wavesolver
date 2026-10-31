"""
Phononic channel cells inside the shared shield crystal, for the two mechanical-port options, and
the analysis that rates them: guided mechanical bands through f_m (BlochROM), group velocity,
parity and localisation, optical guiding near the cavity frequency (GME), and the evanescent decay
through the shield / mirror cells that sets the cavity–channel coupling.

Option A — `VerticalChannel`: a channel along +y, perpendicular to the line defect. With a1 ∥ x the
y axis is the Γ–M direction of the hexagonal lattice; the shortest lattice vector along it is
2a2 − a1 = (0, √3a) (two rows, h = √3a/2 each), so the channel is periodic with √3a and its cell is
the rectangular supercell a1' = (Lx, 0), a2' = (0, √3a), Lx = n_cols·a, holding two sites per column
(rows 2k and 2k + 1, the second shifted by a/2). Alternatives and why not:
- a skewed cell a2 = (a/2, h) has period h but tilts the channel by 30° from y (it follows a2);
- a2 itself (60° from x) is not perpendicular;
- a rectangular cell is also what the solvers take (BlochROM/BlochSolver on Lattice2D, GME with σ
  symmetries off), its seam along x is a plain crystal seam (Lx a multiple of a), and two rows per
  period is the smallest cell in which the channel is straight. The price is a folded band diagram
  (period √3a: 2 rows per period).
Along y the optical light line at the zone edge is c/(2√3a) ≈ 132 THz for a = 654 nm, so no optical
mode at 193 THz can be guided below the light line along this channel: light that reaches it can only
leak out of plane. The optical requirement for A is therefore that the cavity field reaching the
channel is negligible (the TE gap decay over the n_sep shield rows), not a mode gap.

The channel modifies the holes on the sites within |x_site − x_axis| < band/2 ("channel sites"):
scale (s = 0 removes the hole: a solid strip), plus optional extra slots. x_axis = a/4 puts one site
of every row in a band of width a (alternately at −a/4 and +a/4: a zig-zag of removed holes, the
Γ–M analogue of a W1 waveguide).

Option B — `LineChannel`: a cell of the line-defect supercell (wavesolver.linedefect.LineDefectLayout, one
period a along x, the defect region |y| < W/2), given as a fine raster as the cavity chain's cells.
Along x the light line at X is c/(2a) ≈ 229 THz > 193 THz, so a B channel *must* have a TE mode gap
at the cavity frequency (it continues the cavity's far mirror), while carrying the σ_y-even, z-even
breathing band through f_m (the GPT07 "waveguide" cells).

Units: µm, Hz (mechanics), THz (optics).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, box

from .geometry import Lattice2D, solid_fraction
from .layout import ChannelA, Crystal
from .mech_adjoint import element_forms, element_vectors

SQRT3 = math.sqrt(3)
F_M = 4.0e9
F_O = 193.4  # THz
C_UM_THZ = 299.792458


# --- option A geometry ------------------------------------------------------------------------------


@dataclass
class VerticalChannel:
    """
    Option-A channel cell (see the module docstring). crystal: wavesolver.layout.Crystal (shield, W,
    x_shift). n_cols: supercell width in lattice constants (shield on both sides). x_axis: lab x of
    the channel axis. band: width of the band of channel sites (default a: one site per row when
    x_axis is a quarter period off a site). scale: hole scale on the channel sites (0 removes them).
    slot: optional (width, length) of a slot along y cut through the channel axis per period (None:
    none) — a knob that softens the strip mechanically. The cell spans rows 0 and 1 of the upper
    shield (y from W/2 to W/2 + 2h), centred on (x_axis, W/2 + h).
    """

    crystal: Crystal
    n_cols: int = 5
    x_axis: float | None = None
    band: float | None = None
    scale: float = 0.0
    slot: tuple[float, float] | None = None
    extra: list[Polygon] = field(default_factory=list)  # extra air shapes in cell-centred coords
    beam: tuple[float, float, float, int] | None = None  # (w_beam, w_trench, w_tether, tethers per period)

    def __post_init__(self):
        cr = self.crystal
        self.a, self.h = cr.a, cr.h
        if self.x_axis is None:
            self.x_axis = cr.x_shift * cr.a + cr.a / 4
        if self.band is None:
            self.band = cr.a
        self.Lx = self.n_cols * self.a
        self.Ly = 2 * self.h
        self.centre = (self.x_axis, cr.W / 2 + self.h)
        self.lattice = Lattice2D(np.array([[self.Lx, 0.0], [0.0, self.Ly]]))

    def is_channel_site(self, x: float) -> bool:
        return abs(x - self.x_axis) < self.band / 2 - 1e-9

    def holes(self) -> list[Polygon]:
        """Air polygons of one period in cell-centred coordinates (each hole once; solid_fraction adds the images)."""
        cr = self.crystal
        xc, yc = self.centre
        out = []
        for i, j, x, y in cr.sites(xc - self.Lx / 2, xc + self.Lx / 2, 2):
            for p in cr.holes:
                q = affinity.translate(p, x, y)
                if self.is_channel_site(x):
                    if self.scale <= 0:
                        continue
                    q = affinity.scale(q, self.scale, self.scale, origin=(x, y))
                out.append(affinity.translate(q, -xc, -yc))
        if self.slot is not None:
            w, l = self.slot
            out.append(box(-w / 2, -l / 2, w / 2, l / 2))
        out += list(self.extra)
        return out

    def fraction(self, n1: int, n2: int, ss: int = 4) -> np.ndarray:
        """
        Solid fraction on an n1 × n2 grid. With `beam`: a trench of width w_trench along the axis (cut through
        the crystal) holding a straight Si beam of width w_beam, tied to the trench walls by `tethers per
        period` pairs of tethers (width w_tether along y, evenly spaced, the first at the cell centre).
        """
        if self.beam is None:
            return solid_fraction(self.lattice, self.holes(), (n1, n2), supersample=ss)
        wb, wt, wteth, nt = self.beam
        f = solid_fraction(self.lattice, self.holes(), (n1 * ss, n2 * ss), supersample=1)
        x = ((np.arange(n1 * ss) + 0.5) / (n1 * ss) - 0.5) * self.Lx
        y = ((np.arange(n2 * ss) + 0.5) / (n2 * ss) - 0.5) * self.Ly
        X, Y = np.meshgrid(x, y, indexing="ij")
        f[np.abs(X) < wt / 2] = 0.0
        solid = np.abs(X) < wb / 2
        for q in range(nt):
            yq = (q / nt + 0.5) % 1.0 - 0.5
            dy = (Y / self.Ly - yq + 0.5) % 1.0 - 0.5
            solid |= (np.abs(dy * self.Ly) < wteth / 2) & (np.abs(X) < wt / 2 + 0.02)
        f[solid] = 1.0
        return f.reshape(n1, ss, n2, ss).mean(axis=(1, 3))

    def grid(self, per_a: int = 32) -> tuple[int, int]:
        """(n1, n2) with per_a elements per a along x and the nearest even count over √3a along y."""
        return self.n_cols * per_a, 2 * int(round(self.Ly / (self.a / per_a) / 2))

    def channel_mask(self, n1: int, n2: int, half_width: float | None = None) -> np.ndarray:
        """Pixels within |x − x_axis| < half_width (default: band/2 + a/2) of the axis."""
        hw = self.band / 2 + self.a / 2 if half_width is None else half_width
        x = ((np.arange(n1) + 0.5) / n1 - 0.5) * self.Lx
        return np.repeat((np.abs(x) < hw)[:, None], n2, axis=1)

    def layout_channel(self, n_sep: int, per_a: int = 64, half_width: float | None = None) -> ChannelA:
        """The wavesolver.layout.ChannelA of this cell (fine raster per_a per a)."""
        n1, n2 = self.grid(per_a)
        hw = self.band / 2 + self.a / 2 if half_width is None else half_width
        return ChannelA(self.fraction(n1, n2), self.Lx, self.x_axis, hw, n_sep)


# --- option B geometry ------------------------------------------------------------------------------


@dataclass
class LineChannel:
    """
    Option-B channel cell: a fine raster (F1, F2) of the line-defect supercell `layout`
    (wavesolver.linedefect.LineDefectLayout). `mech(n_cell)` block-means it to the mechanical grid.
    """

    layout: object
    fine: np.ndarray
    name: str = "channel"

    def mech(self) -> np.ndarray:
        from .coupling import block_mean

        n1, n2 = self.layout.shape("mech")
        return block_mean(self.fine, n1, n2)

    def defect_mask(self, n1: int, n2: int, extra_rows: float = 0.5) -> np.ndarray:
        L = self.layout
        y = ((np.arange(n2) + 0.5) / n2 - 0.5) * L.Ly
        return np.repeat((np.abs(y) < L.W / 2 + extra_rows * L.h)[None, :], n1, axis=0)


# --- mechanical band analysis ----------------------------------------------------------------------------


@dataclass
class ChannelBands:
    """
    Bands of a channel supercell in a frequency window along a k-path. ks (nk, 2) in 1/µm; per k a list
    of frequencies (Hz) and localisation (fraction of ∫ρ|u|² in the channel mask) and parity labels.
    """

    ks: np.ndarray
    kpar: np.ndarray  # k along the channel (1/µm)
    freqs: list[np.ndarray]
    loc: list[np.ndarray]
    parity: str
    sym: list[np.ndarray] | None = None
    timing: dict = field(default_factory=dict)

    def guided(self, loc_min: float = 0.5) -> list[np.ndarray]:
        return [f[l > loc_min] for f, l in zip(self.freqs, self.loc)]

    def crossings(self, f0: float = F_M, loc_min: float = 0.5) -> list[dict]:
        """
        Guided bands crossing f0: from consecutive k-points with a localised mode on each side of f0,
        matched by order; returns k at the crossing (linear interpolation) and v_g = 2π Δf/Δk (m/s).
        """
        out = []
        g = self.guided(loc_min)
        for n in range(len(g) - 1):
            a, b = np.sort(g[n]), np.sort(g[n + 1])
            for fa in a:
                # nearest partner at the next k
                if len(b) == 0:
                    continue
                fb = b[np.argmin(np.abs(b - fa))]
                if (fa - f0) * (fb - f0) <= 0 and fa != fb and abs(fb - fa) < 0.25e9:
                    t = (f0 - fa) / (fb - fa)
                    k = self.kpar[n] + t * (self.kpar[n + 1] - self.kpar[n])
                    vg = 2 * np.pi * (fb - fa) / ((self.kpar[n + 1] - self.kpar[n]) * 1e6)
                    out.append(dict(k=float(k), vg=float(vg), f=(float(fa), float(fb))))
        return out


def element_mass(solver, k: np.ndarray, X: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    """∫ρ|u|² per fraction-map pixel (n1, n2, nmodes) of modes X at k (all parts summed)."""
    n1, n2 = solver.shape
    theta = solver.lattice.vectors @ np.asarray(k, float)
    out = np.zeros((n1 * n2, X.shape[1]))
    for part in solver.parts:
        _, eM = element_forms(part, element_vectors(part, theta, X))
        w = fraction.ravel()[part.cell][:, None] * eM
        for v in range(X.shape[1]):
            out[:, v] += np.bincount(part.cell, weights=w[:, v], minlength=n1 * n2)
    return out.reshape(n1, n2, -1)


def rom_bands(lattice, fraction: np.ndarray, material, ks: np.ndarray, kpar: np.ndarray, mask: np.ndarray,
              window_hz: tuple[float, float], parity: str = "even", order: int = 4, sigma_hz: float = F_M,
              n_residual=(240, 60), n_fixed: int = 100, n_interior: int = 100, rom=None, verbose: bool = True):
    """
    Bands of a supercell in window_hz by Bloch mode synthesis (wavesolver.mech_rom.BlochROM shifted to
    sigma_hz), with the localisation of each mode in `mask` (pixels). Returns (ChannelBands, rom).
    """
    from .mech_rom import BlochROM

    t0 = time.perf_counter()
    if rom is None:
        rom = BlochROM(lattice, fraction, material, n_fixed=n_fixed, n_iface=0, n_residual=n_residual, n_interior_residual=n_interior,
                       sigma0_hz=sigma_hz, f_ref_hz=window_hz[1], model="slab_p", thickness=0.22, parity=parity, order=order)
    t_red = time.perf_counter() - t0
    freqs, locs = [], []
    t0 = time.perf_counter()
    for k in ks:
        f, q = rom.solve(k, vectors=True, window_hz=window_hz)
        if len(f):
            X = rom.reconstruct(k, q)
            em = element_mass(rom.solver, k, X, fraction)
            tot = em.sum(axis=(0, 1))
            loc = (em * mask[..., None]).sum(axis=(0, 1)) / np.maximum(tot, 1e-300)
        else:
            loc = np.zeros(0)
        freqs.append(np.asarray(f))
        locs.append(loc)
    t_sweep = time.perf_counter() - t0
    if verbose:
        print(f"    BlochROM {parity}: reduced {rom.n_reduced} in {t_red:.0f} s, {len(ks)} k in {t_sweep:.0f} s", flush=True)
    return ChannelBands(np.asarray(ks), np.asarray(kpar), freqs, locs, parity, timing=dict(reduce=t_red, sweep=t_sweep)), rom


# --- evanescent decay ----------------------------------------------------------------------------------


def tb_complex_k(f: float, f_edge: float, f_far: float, period: float) -> float:
    """
    Im k (1/µm) at frequency f in the gap above (or below) a band f(k) = e + 2t cos(k·period) with
    band edge f_edge at the zone edge or centre and the other end f_far: cosh(κ·period) = |f − e|/(2|t|)
    measured from the band edge side (the tight-binding continuation of wavesolver.linedefect.complex_k).
    """
    e = (f_edge + f_far) / 2
    t = abs(f_edge - f_far) / 4
    x = abs(f - e) / (2 * t)
    if x <= 1:
        return 0.0
    return float(np.arccosh(x) / period)


def fit_decay(profile: np.ndarray, positions: np.ndarray, skip: int = 1) -> tuple[float, float]:
    """
    Exponential fit of a positive per-cell (or per-row) energy profile: E_n ≈ E_0 exp(−2κ x_n); returns
    (2κ per unit length, r²) from a least-squares line through ln E over the points after `skip`.
    """
    p = np.asarray(profile, float)
    x = np.asarray(positions, float)
    sel = (p > 0)
    sel[:skip] = False
    A = np.column_stack([np.ones(sel.sum()), x[sel]])
    y = np.log(p[sel])
    c, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (res[0] / ss if len(res) else 0.0) if ss > 0 else 1.0
    return float(-c[1]), float(r2)


# --- coupling model --------------------------------------------------------------------------------


def transducer_share(gamma_ex: float, L_t: float, v_g: float, detuning_factor: float = 1.0) -> float:
    """
    Energy share P_t of the cavity mode in a transducer section of length L_t (µm) at the end of a
    lossless channel of group velocity v_g (m/s), given the cavity's decay rate into the channel when
    the channel is matched, γ_ex (rad/s). A matched channel holds the leaked power flux γ_ex·U over the
    transit time L_t/v_g: P_t = γ_ex L_t / v_g. A closed (reflecting) channel of FSR Δ = π v_g/L_ch
    with the cavity mid-way between two channel modes gives P_t ≈ (π²/8)·(γ_ex L_t /(2 v_g))·... ≈ γ_ex
    L_t/(2 v_g) up to a detuning factor F (≈ 1 mid-way, ≫ 1 near a channel resonance). Returns F·γ_ex
    L_t/(2 v_g) for a closed channel (F = detuning_factor).
    """
    return detuning_factor * gamma_ex * (L_t * 1e-6) / (2 * v_g)


def g_em_from_share(g_period: float, n_periods: int, share: float) -> float:
    """g_em = g_period √(N P): N transducer periods holding a total energy share P, coherently (README)."""
    return g_period * math.sqrt(n_periods * share)
