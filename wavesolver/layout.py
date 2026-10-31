"""
Parametric layout of the whole piezo-optomechanical transducer: shield crystal, cavity chain
(line defect + mirror tapers), end-fire optical port, phononic channel (option A: along +y through
the shield; option B: along +x past the cavity's far mirror), LN patch transducer with finger
electrodes, routing to the microwave-resonator pad, open windows and the release region.

Coordinates (µm, as gpt_omc and wavesolver.linedefect). The line defect runs along x with its defect
cell (column 0) centred on the origin; column n is centred at x = n·a. The defect region is
|y| < W/2 (W = defect_rows·h, h = √3a/2). The upper shield is the hexagonal crystal with sites
(x_shift·a + i·a + j·a/2, y₁ + j·h), j ≥ 0, y₁ = W/2 + h/2 (wavesolver.linedefect.LineDefectLayout); the
lower shield is its mirror image y → −y. Row j of the upper shield owns the band
W/2 + j·h ≤ y < W/2 + (j + 1)·h.

Geometry model. Everything is polygons (shapely), assembled in this order:
1. Shield: the holes owned by one lattice site (contoured once from the shield's level function at
   ~2 nm, `site_holes`), placed whole on every site of the crystal block. Sites are kept by their
   position (columns x_lo ≤ x_site < x_hi, rows 0 … n_rows − 1), so the crystal edge is made of whole
   holes, as in gpt_omc.
2. Raster regions (`RasterRegion`), in priority order: the cavity chain (per-cell fine rasters of
   the line-defect supercell, as wavesolver.cavity.chain_domain / wavesolver.linedefect take them) and the option-A
   channel (its rectangular supercell raster, tiled along y). Each region's air (fraction < ½,
   contoured with linear interpolation, i.e. the boundary the solvers' fraction maps imply) replaces
   whatever lies inside the region's rectangle: etch = (etch − rect) ∪ (air ∩ rect).
3. Open windows (gpt_omc's M1_Pos_end) around the transducer plate and around the suspended part of
   the optical waveguide, with the Si standing inside them (gpt_omc's LN_Si_plate: output strip,
   plate under the LN patch, 1D-shield tethers, waveguide strip and its tethers).
4. Etched trenches defining the optical waveguide on the unreleased Si up to the facet, with an
   inverse taper.
5. LN patch (LN kept; top face drawn, the bottom face is wider by t_LN/tan(sidewall)), metal
   (fingers, bus bars, leads along the tethers, pads), release region (all etched features grown
   by the HF undercut; informational).

Layers (DXF names). gpt_omc's three layers keep their GPT07 meaning: LAYER_ETCH = "M1_Pos_BDs"
(holes and trenches), LAYER_WINDOW = "M1_Pos_end" (open windows), LAYER_PLATE = "LN_Si_plate"
(Si standing inside the windows). The fracturing software combines them as on GPT07 (etch =
holes ∪ windows − plate); `Device.dxf_layers(union=True)` writes that combination on M1_Pos_BDs
instead. New layers, names to be confirmed for the mask: LAYER_LN = "LN_patch" (LN kept),
LAYER_METAL = "M2_metal", LAYER_RELEASE = "Release" (informational, not written by default).

Solver rasters. `Device.si_fraction(rect, n1, n2)` rasterises the drawn Si on any rectangle with the
wavesolver.cavity / wavesolver.linedefect pixel convention (pixel (i, j) centred at x0 + (i + ½)·dx, y0 + (j + ½)·dy);
`ln_fraction` gives the LN fraction of a sub-layer at a height (sidewall included), and
`metal_mask` the electrode masks per net, as wavesolver.piezo.PiezoLayer / Electrode take them.
`Device.regions` holds the named rectangles (cavity, channel, transducer, optical_port, ...).

    from wavesolver.layout import DeviceSpec, build_device
    dev = build_device(DeviceSpec(option="B", ...))
    dev.write_dxf("layout_B.dxf")
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient

from .geometry import gpt_omc

SQRT3 = math.sqrt(3)
LAYER_ETCH = gpt_omc.LAYER  # "M1_Pos_BDs"
LAYER_WINDOW = gpt_omc.WINDOW_LAYER  # "M1_Pos_end"
LAYER_PLATE = gpt_omc.PLATE_LAYER  # "LN_Si_plate"
LAYER_LN = "LN_patch"
LAYER_METAL = "M2_metal"
LAYER_RELEASE = "Release"
GRID = 1e-6  # µm: polygon coordinates are snapped to 1 pm before unions


# --- polygon helpers ---------------------------------------------------------------------------


def _polys(geom) -> list[Polygon]:
    """The polygons of any shapely geometry (empty / lines dropped)."""
    if geom is None or geom.is_empty:
        return []
    out = []
    for g in shapely.get_parts(geom):
        if isinstance(g, Polygon) and g.area > 0:
            out.append(orient(g))
        elif isinstance(g, MultiPolygon) or g.geom_type == "GeometryCollection":
            out += _polys(g)
    return out


def _union(geoms) -> Polygon | MultiPolygon:
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return Polygon()
    return shapely.union_all(geoms, grid_size=GRID)


def _override(pieces: list[Polygon], area: Polygon, new: list[Polygon]) -> list[Polygon]:
    """Air pieces with everything inside `area` replaced by `new` ∩ area (pieces stay a list; no global union)."""
    shapely.prepare(area)
    out = []
    for p in pieces:
        if not area.intersects(p):
            out.append(p)
        elif not area.contains(p):
            out += _polys(p.difference(area))
    for q in new:
        if area.contains(q):
            out.append(q)
        elif area.intersects(q):
            out += _polys(q.intersection(area))
    return out


def contour_air(frac: np.ndarray, x0: float, y0: float, dx: float, dy: float, level: float = 0.5) -> list[Polygon]:
    """
    Air polygons (fraction < level) of a pixel raster frac (n1, n2) whose pixel (i, j) is centred
    at (x0 + (i + ½)dx, y0 + (j + ½)dy): marching squares with linear interpolation (contourpy), the
    boundary the fraction map implies. Air touching the raster edge is closed along the edge
    (edge-padded), so the result can be clipped to the raster rectangle.
    """
    import contourpy

    n1, n2 = frac.shape
    f = np.pad(np.asarray(frac, float), 1, mode="edge")
    xs = x0 + (np.arange(-1, n1 + 1) + 0.5) * dx
    ys = y0 + (np.arange(-1, n2 + 1) + 0.5) * dy
    # the padded edge is moved onto the raster edge so the polygons close exactly there
    xs[0], xs[-1] = x0, x0 + n1 * dx
    ys[0], ys[-1] = y0, y0 + n2 * dy
    gen = contourpy.contour_generator(xs, ys, f.T, fill_type=contourpy.FillType.OuterOffset)
    points, offsets = gen.filled(-1.0, level)
    polys = []
    for pts, off in zip(points, offsets):
        rings = [pts[off[k] : off[k + 1]] for k in range(len(off) - 1)]
        if len(rings[0]) < 4:
            continue
        p = Polygon(rings[0], [r for r in rings[1:] if len(r) >= 4]).buffer(0)
        polys += _polys(p)
    return polys


def _rect(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


# --- shield --------------------------------------------------------------------------------------


def shield_level(shield, xy: np.ndarray) -> np.ndarray:
    """
    Continuous level of the shield crystal at crystal-frame points xy (..., 2): > 0 in Si, < 0 in air,
    zero on the boundary. TripodShield: the signed distance of wavesolver.parametrize; DensityShield: the
    filtered density minus its threshold η; anything else with `.solid(xy)`: a 3×3-supersampled
    indicator minus ½ (staircase, ~res/3).
    """
    xy = np.asarray(xy, float)
    flat = xy.reshape(-1, 2)
    if hasattr(shield, "cell") and hasattr(shield.cell, "periodic_sdf"):
        import jax.numpy as jnp

        from .linedefect import _reduce_hex

        p = _reduce_hex(flat, shield.cell.lattice.vectors)
        out = np.empty(len(p))
        th = jnp.zeros(0)
        for i in range(0, len(p), 65536):
            out[i : i + 65536] = np.asarray(shield.cell.periodic_sdf(jnp.asarray(p[i : i + 65536]), th))
        return out.reshape(xy.shape[:-1])
    if hasattr(shield, "dc") and hasattr(shield.dc, "field_at"):
        st = flat @ np.linalg.inv(shield.dc.lattice.vectors)
        st = (st + 0.5) % 1.0 - 0.5
        return (shield.dc.field_at(shield.x, st) - shield.eta).reshape(xy.shape[:-1])
    return shield.solid(xy).astype(float) - 0.5


def hex_vectors(a: float) -> np.ndarray:
    return a * np.array([[1.0, 0.0], [0.5, SQRT3 / 2]])


def site_holes(shield, res: float = 0.002) -> list[Polygon]:
    """
    The air polygons owned by one lattice site, in coordinates centred on the site: the filled
    contours of `shield_level` over 2 × 2 cells whose centroid falls in the site's unit cell
    (lattice coordinates in [−½, ½)²). Placing them on every site rebuilds the crystal without
    clipping (each hole whole, exactly once).
    """
    A = hex_vectors(shield.a)
    span = 2.0 * shield.a
    n = int(math.ceil(span / res)) + 1
    xs = np.linspace(-span, span, 2 * n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    lev = shield_level(shield, np.stack([X, Y], -1))
    dx = xs[1] - xs[0]
    frac = np.clip(0.5 + lev / dx, 0.0, 1.0)  # a level → fraction ramp one pixel wide (contour at ½ = level 0)
    air = contour_air(frac, xs[0] - dx / 2, xs[0] - dx / 2, dx, dx)
    Ainv = np.linalg.inv(A)
    out = []
    for p in air:
        st = np.array(p.centroid.coords[0]) @ Ainv
        inside = p.bounds[0] > xs[0] + dx and p.bounds[2] < xs[-1] - dx and p.bounds[1] > xs[0] + dx and p.bounds[3] < xs[-1] - dx
        if np.all(st >= -0.5) and np.all(st < 0.5) and p.area > 1e-6:
            if not inside:
                raise ValueError("a hole reaches the edge of the 2 × 2 cell sampling window")
            out.append(orient(p.simplify(res / 4)))
    return out


@dataclass
class Crystal:
    """Geometry of the shield crystal around a line defect along x (see the module docstring)."""

    shield: object
    defect_rows: float = 2.0
    x_shift: float = 0.0
    res: float = 0.002

    def __post_init__(self):
        self.a = float(self.shield.a)
        self.h = SQRT3 / 2 * self.a
        self.W = self.defect_rows * self.h
        self.y1 = self.W / 2 + self.h / 2
        self.holes = site_holes(self.shield, self.res)

    def site(self, i: int, j: int) -> tuple[float, float]:
        """Lab position of upper-shield site (i, j) (j ≥ 0: row; i: index along x)."""
        return self.x_shift * self.a + i * self.a + j * self.a / 2, self.y1 + j * self.h

    def row_band(self, j: int) -> tuple[float, float]:
        return self.W / 2 + j * self.h, self.W / 2 + (j + 1) * self.h

    def sites(self, x_lo: float, x_hi: float, rows: int) -> list[tuple[int, int, float, float]]:
        """Upper-shield sites with x_lo ≤ x < x_hi in rows 0 … rows − 1."""
        out = []
        for j in range(rows):
            i0 = math.floor((x_lo - self.x_shift * self.a - j * self.a / 2) / self.a) - 1
            i1 = math.ceil((x_hi - self.x_shift * self.a - j * self.a / 2) / self.a) + 1
            for i in range(i0, i1 + 1):
                x, y = self.site(i, j)
                if x_lo <= x < x_hi:
                    out.append((i, j, x, y))
        return out

    def hole_polygons(self, x_lo: float, x_hi: float, rows: int, sides=(1, -1), skip: Callable | None = None) -> list[Polygon]:
        """Whole holes of the shield block (both halves by default); skip(i, j, x, y) → True drops a site."""
        out = []
        for i, j, x, y in self.sites(x_lo, x_hi, rows):
            if skip is not None and skip(i, j, x, y):
                continue
            for p in self.holes:
                q = affinity.translate(p, x, y)
                if 1 in sides:
                    out.append(q)
                if -1 in sides:
                    out.append(affinity.scale(q, 1.0, -1.0, origin=(0, 0)))
        return out

    def level(self, xy: np.ndarray) -> np.ndarray:
        """Shield level at lab points (upper crystal; lower = mirror image)."""
        xy = np.asarray(xy, float)
        cp = np.stack([xy[..., 0] - self.x_shift * self.a, np.abs(xy[..., 1]) - self.y1], -1)
        return shield_level(self.shield, cp)


# --- raster regions ------------------------------------------------------------------------------


@dataclass
class RasterRegion:
    """
    A rectangle whose pattern is given by a fine fraction raster (Si fraction per pixel), pixel
    (i, j) centred at (x0 + (i + ½)·dx, y0 + (j + ½)·dy); inside `clip` (default: the whole raster
    rectangle) the raster's air replaces the layout below it.
    """

    name: str
    fraction: np.ndarray
    x0: float
    y0: float
    dx: float
    dy: float
    clip: Polygon | None = None

    @property
    def rect(self) -> Polygon:
        n1, n2 = self.fraction.shape
        return _rect(self.x0, self.y0, self.x0 + n1 * self.dx, self.y0 + n2 * self.dy)

    @property
    def area(self) -> Polygon:
        return self.rect if self.clip is None else self.rect.intersection(self.clip)

    def air(self) -> list[Polygon]:
        return contour_air(self.fraction, self.x0, self.y0, self.dx, self.dy)


@dataclass
class CavityChain:
    """
    The cavity as cells along x (wavesolver.cavity.chain_domain convention): fine fraction rasters
    (F1, F2), one per column first_column … first_column + N − 1, each spanning
    [n·a − a/2, n·a + a/2] × [−height/2, height/2]. labels: region name per cell ("mirror",
    "taper", "defect", "channel", ...).
    """

    cells: list[np.ndarray]
    pitch: float
    height: float
    first_column: int
    labels: list[str] | None = None

    @property
    def columns(self) -> tuple[int, int]:
        return self.first_column, self.first_column + len(self.cells) - 1

    def append(self, cells: list[np.ndarray], label: str) -> "CavityChain":
        lab = (self.labels or ["cavity"] * len(self.cells)) + [label] * len(cells)
        return CavityChain(self.cells + list(cells), self.pitch, self.height, self.first_column, lab)

    def region(self) -> RasterRegion:
        f = np.concatenate(self.cells, axis=0)
        F1, F2 = self.cells[0].shape
        c0, _ = self.columns
        return RasterRegion("cavity", f, (c0 - 0.5) * self.pitch, -self.height / 2, self.pitch / F1, self.height / F2)


# --- option A channel and transducer specs -----------------------------------------------------------


@dataclass
class ChannelA:
    """
    Option A channel: a periodic structure along +y (period 2h = √3a, two crystal rows) centred on
    x = x_axis, given as one period's fine raster (F1, F2) spanning [x_axis − Lx/2, x_axis + Lx/2]
    × [W/2 + 2k·h, W/2 + (2k + 2)·h] for period k (rows 2k and 2k + 1). It starts at row n_sep
    (the shield rows 0 … n_sep − 1 between the defect region and the channel stay crystal) and runs
    to the top of the crystal block; inside |x − x_axis| < half_width it replaces the crystal.
    `wavesolver.channel.VerticalChannel.layout_raster()` makes it.
    """

    fraction: np.ndarray
    Lx: float
    x_axis: float = 0.0
    half_width: float | None = None
    n_sep: int = 4


@dataclass
class OpticalPort:
    """
    End-fire port on the −x side: n_wg crystal columns of solid line defect (|y| < W/2, a W-type
    waveguide; placeholder for the 2c coupler cells), then an open window of length window_len with
    the suspended strip (width w_wg) held by tether pairs (width tether_w, every tether_pitch), then
    trench-defined waveguide on the unreleased Si to the facet at x = facet_x, with an inverse taper
    to tip_w over taper_len.
    """

    n_wg: int = 2
    w_wg: float = 0.50
    window_len: float = 6.0
    window_half: float | None = None
    tether_w: float = 0.10
    tether_pitch: float = 2.5
    trench: float = 2.0
    facet_x: float = -60.0
    taper_len: float = 15.0
    tip_w: float = 0.12


@dataclass
class Transducer:
    """
    LN patch transducer at the channel end, in its local frame (propagation +x', aperture along y',
    channel exit at the origin). Layout along x': output strip (w_strip × strip_len) → Si plate
    carrying the LN patch (top face patch_len × aperture; sidewall angle from the vertical as the
    65° fab sidewall: bottom face wider by t_ln/tan(sidewall) per side) → free end. Fingers: n_pairs
    hot/ground pairs at pitch (period of one pair, the acoustic wavelength), width finger_w, across
    the LN top and down to the bus bars on the plate (hot: +y', ground: −y'); finger_gap from the
    opposite bus. The plate is held at ±y' by 1D-shield tethers (gpt_omc.Shield1D scaled to 4 GHz)
    that carry the leads (lead_w) to the pads past the window. pad: the microwave-resonator pad (hot)
    and ground pad, pad_size each, pad_gap past the window edge.
    """

    pitch: float = 0.63
    n_pairs: int = 8
    finger_w: float = 0.16
    finger_gap: float = 0.15
    aperture: float = 1.6
    t_ln: float = 0.25
    sidewall_deg: float = 65.0
    margin: float = 0.4  # LN beyond the outer fingers along x'
    w_strip: float = 0.60
    strip_len: float = 2.0
    bus_w: float = 0.30
    clearance: float = 0.10
    tether: gpt_omc.Shield1D = field(default_factory=lambda: gpt_omc.scaled(gpt_omc.Shield1D(cells=8), 1.75))
    lead_w: float = 0.10
    window_margin: float = 1.0
    pad_size: float = 8.0
    pad_gap: float = 3.0

    @property
    def ln_out(self) -> float:
        """Bottom-face outset of the LN patch (µm per side)."""
        return self.t_ln / math.tan(math.radians(self.sidewall_deg))

    @property
    def patch_len(self) -> float:
        return self.n_pairs * self.pitch + 2 * self.margin

    def plate_bounds(self) -> tuple[float, float, float]:
        """(x'0, x'1, half-width) of the Si plate."""
        x0 = self.strip_len
        L = self.patch_len + 2 * (self.ln_out + self.clearance)
        hw = self.aperture / 2 + self.ln_out + self.clearance + self.bus_w + self.clearance
        return x0, x0 + L, hw

    def patch_top(self) -> Polygon:
        x0, x1, _ = self.plate_bounds()
        c = (x0 + x1) / 2
        return _rect(c - self.patch_len / 2, -self.aperture / 2, c + self.patch_len / 2, self.aperture / 2)

    def finger_x(self) -> list[tuple[float, str]]:
        x0, x1, _ = self.plate_bounds()
        c = (x0 + x1) / 2
        start = c - self.n_pairs * self.pitch / 2 + self.pitch / 4
        return [(start + k * self.pitch / 2, "hot" if k % 2 == 0 else "gnd") for k in range(2 * self.n_pairs)]


@dataclass
class DeviceSpec:
    """
    Everything that defines one device. shield: an wavesolver.linedefect shield (.a, .solid); chain: the
    CavityChain (cells left → right; column 0 the defect). option "A" or "B". B: channel_b cells
    (line-defect supercell rasters, same grid as the chain) appended after the chain's last column.
    A: channel_a (ChannelA). rows: shield rows per side (A: also the channel's top end).
    right_pad: extra columns of plain shield past the chain on the +x side (A only).
    """

    shield: object
    chain: CavityChain
    option: str = "B"
    channel_b: list[np.ndarray] | None = None
    channel_a: ChannelA | None = None
    defect_rows: float = 2.0
    x_shift: float = 0.0
    rows: int = 8
    right_pad: int = 2
    port: OpticalPort = field(default_factory=OpticalPort)
    transducer: Transducer = field(default_factory=Transducer)
    undercut: float = 1.5
    res: float = 0.002


# --- the built device ----------------------------------------------------------------------------


@dataclass
class Device:
    spec: DeviceSpec
    crystal: Crystal
    holes: list[Polygon]  # M1_Pos_BDs: holes and trenches (not merged with the windows)
    windows: list[Polygon]  # M1_Pos_end
    plate: list[Polygon]  # LN_Si_plate: Si standing inside the windows
    ln: list[Polygon]  # LN kept (top face)
    metal: dict[str, list[Polygon]]  # per net: "hot", "gnd"
    regions: dict[str, Polygon]
    frame: Polygon  # outline of the drawn area
    info: dict = field(default_factory=dict)

    # combined etch (holes ∪ windows − plate)
    def etch(self) -> Polygon | MultiPolygon:
        if "_etch" not in self.info:
            self.info["_etch"] = _union([_union(self.holes), _union(self.windows).difference(_union(self.plate))])
        return self.info["_etch"]

    def release(self) -> Polygon | MultiPolygon:
        """Released (undercut) region: every etched feature grown by spec.undercut (informational)."""
        return self.etch().buffer(self.spec.undercut, join_style="round", resolution=8)

    def ln_bottom(self) -> list[Polygon]:
        out = self.spec.transducer.ln_out
        return [p.buffer(out, join_style="mitre") for p in self.ln]

    # rasters for the solvers
    def si_fraction(self, rect: tuple[float, float, float, float], n1: int, n2: int, ss: int = 4) -> np.ndarray:
        """Drawn Si fraction on the n1 × n2 pixel grid of rect = (x0, y0, x1, y1) (ss² sub-points per pixel)."""
        x0, y0, x1, y1 = rect
        dx, dy = (x1 - x0) / n1, (y1 - y0) / n2
        xs = x0 + (np.arange(n1 * ss) + 0.5) * dx / ss
        ys = y0 + (np.arange(n2 * ss) + 0.5) * dy / ss
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        g = self.etch().intersection(_rect(x0 - dx, y0 - dy, x1 + dx, y1 + dy))
        shapely.prepare(g)
        air = shapely.contains_xy(g, X, Y)
        return 1.0 - air.reshape(n1, ss, n2, ss).mean(axis=(1, 3))

    def ln_fraction(self, rect, n1: int, n2: int, z: float = 0.5, ss: int = 4) -> np.ndarray:
        """LN fraction at relative height z ∈ [0, 1] of the LN layer (0 = bottom face), sidewall included."""
        t = self.spec.transducer
        shapes = _union([p.buffer((1 - z) * t.ln_out, join_style="mitre") for p in self.ln])
        return _fraction(shapes, rect, n1, n2, ss)

    def metal_mask(self, rect, n1: int, n2: int, net: str, ss: int = 4) -> np.ndarray:
        return _fraction(_union(self.metal.get(net, [])), rect, n1, n2, ss)

    # output
    def dxf_layers(self, union: bool = False, release: bool = False) -> dict[str, list[np.ndarray]]:
        """
        Rings per DXF layer. union=False: as on GPT07 (holes, windows and plate on their own layers,
        for the fracturing software to combine); union=True: the combined etch on M1_Pos_BDs.
        Interior rings (none expected) are written as further rings on the same layer.
        """
        def rings(geoms):
            out = []
            for p in _polys(_union(geoms)) if geoms else []:
                out.append(np.array(p.exterior.coords)[:-1])
                out += [np.array(r.coords)[:-1] for r in p.interiors]
            return out

        layers = {}
        if union:
            layers[LAYER_ETCH] = rings([self.etch()])
        else:
            layers[LAYER_ETCH] = rings(self.holes)
            layers[LAYER_WINDOW] = rings(self.windows)
            layers[LAYER_PLATE] = rings(self.plate)
        layers[LAYER_LN] = rings(self.ln)
        layers[LAYER_METAL] = rings([g for v in self.metal.values() for g in v])
        if release:
            layers[LAYER_RELEASE] = rings([self.release()])
        return layers

    def write_dxf(self, path: str | Path, unit: str = "um", union: bool = False, release: bool = False) -> Path:
        """Write (and read back to check) a DXF with gpt_omc.export (ezdxf, LWPOLYLINE per ring)."""
        return gpt_omc.export(Path(path), self.dxf_layers(union, release), unit)

    def check(self, min_si: float = 0.08, min_ln: float = 0.2) -> dict:
        """
        Drawn-geometry checks: Si islands (Si pieces not connected to the frame); the parts of Si, of the
        etched gaps and of the LN that an opening with a disc of the minimum feature does not reach, as
        area and as depth (largest distance of such a part from the opened set: a corner of radius < R
        gives a few nm, a too-narrow bridge or gap gives about its missing half-width or more).
        """
        etch = self.etch()
        si = self.frame.difference(etch)
        parts = _polys(si)
        main = max(parts, key=lambda p: p.area) if parts else None
        islands = [p for p in parts if p is not main and p.area > 1e-6]

        def thin(geom, w):
            r = w / 2 - 1e-4
            opened = geom.buffer(-r).buffer(r)
            rest = _polys(geom.difference(opened))
            if not rest:
                return 0.0, 0.0
            pts = np.concatenate([np.array(p.exterior.coords) for p in rest])
            pts = pts[shapely.contains_xy(geom.buffer(1e-4), pts[:, 0], pts[:, 1])]
            depth = float(shapely.distance(opened, shapely.points(pts)).max()) if len(pts) else 0.0
            return float(sum(p.area for p in rest)), depth

        a_si, d_si = thin(si, min_si)
        a_gap, d_gap = thin(etch, min_si)
        a_ln, d_ln = thin(_union(self.ln), min_ln)
        return dict(islands=len(islands), island_area_um2=float(sum(p.area for p in islands)), thin_si_um2=a_si, thin_si_depth_nm=1e3 * d_si,
                    thin_gap_um2=a_gap, thin_gap_depth_nm=1e3 * d_gap, thin_ln_um2=a_ln, thin_ln_depth_nm=1e3 * d_ln, n_etch=len(_polys(etch)))


def _fraction(geom, rect, n1, n2, ss):
    x0, y0, x1, y1 = rect
    dx, dy = (x1 - x0) / n1, (y1 - y0) / n2
    xs = x0 + (np.arange(n1 * ss) + 0.5) * dx / ss
    ys = y0 + (np.arange(n2 * ss) + 0.5) * dy / ss
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    if geom.is_empty:
        return np.zeros((n1, n2))
    shapely.prepare(geom)
    return shapely.contains_xy(geom, X, Y).reshape(n1, ss, n2, ss).mean(axis=(1, 3))


# --- transducer polygons (local frame) ------------------------------------------------------------------


def transducer_parts(t: Transducer, window_half: float) -> dict:
    """
    Polygons of the transducer in its local frame (channel exit at x' = 0, propagation +x'):
    window, plate (strip + plate + tethers), LN top face, metal per net, and the tether/pad layout.
    window_half: half-width of the open window along y' (at least the tether reach).
    """
    x0, x1, hw = t.plate_bounds()
    s = t.tether
    strip = _rect(-0.05, -t.w_strip / 2, x0 + 1e-3, t.w_strip / 2)  # overlaps the channel end by 50 nm
    plate = _rect(x0, -hw, x1, hw)
    # tethers: 1D shield chains from the plate's ±y' edges to the window edge, at the plate centre
    xc = (x0 + x1) / 2
    n_cells = max(1, int(math.ceil((window_half - hw) / s.period)))
    reach = hw + n_cells * s.period
    upper = [gpt_omc.shield_1d_cell(s, xc, hw + i * s.period) for i in range(n_cells)]
    upper.append(_rect(xc - s.neck_width / 2, hw + n_cells * s.period - 1e-3, xc + s.neck_width / 2, reach + t.window_margin + 1e-3))
    lower = [affinity.scale(c, 1.0, -1.0, origin=(0, 0)) for c in upper]
    win = _rect(0.0, -reach, x1 + t.window_margin, reach)
    # metal: fingers, bus bars, leads, pads
    top = t.patch_top()
    ln_bot_hw = t.aperture / 2 + t.ln_out
    bus_y = ln_bot_hw + t.clearance + t.bus_w / 2
    hot, gnd = [], []
    for x, net in t.finger_x():
        if net == "hot":
            hot.append(_rect(x - t.finger_w / 2, -t.aperture / 2 + t.finger_gap, x + t.finger_w / 2, bus_y))
        else:
            gnd.append(_rect(x - t.finger_w / 2, -bus_y, x + t.finger_w / 2, t.aperture / 2 - t.finger_gap))
    xf = [x for x, _ in t.finger_x()]
    bx0, bx1 = min(xf) - t.finger_w / 2, max(xf) + t.finger_w / 2
    hot.append(_rect(bx0, bus_y - t.bus_w / 2, bx1, bus_y + t.bus_w / 2))
    gnd.append(_rect(bx0, -bus_y - t.bus_w / 2, bx1, -bus_y + t.bus_w / 2))
    y_out = reach + t.window_margin
    hot.append(_rect(xc - t.lead_w / 2, bus_y, xc + t.lead_w / 2, y_out + t.pad_gap))
    gnd.append(_rect(xc - t.lead_w / 2, -y_out - t.pad_gap, xc + t.lead_w / 2, -bus_y))
    P = t.pad_size
    hot.append(_rect(xc - P / 2, y_out + t.pad_gap, xc + P / 2, y_out + t.pad_gap + P))
    gnd.append(_rect(xc - P / 2, -y_out - t.pad_gap - P, xc + P / 2, -y_out - t.pad_gap))
    return dict(window=win, plate=[strip, plate] + upper + lower, ln=[top], hot=hot, gnd=gnd,
                rect=(0.0, -hw, x1, hw), reach=reach, plate_rect=(x0, -hw, x1, hw))


def _clean_window_edges(holes, windows, plate, reach: float, min_feature: float, seams=()):
    """
    Where a window edge cuts through the crystal it leaves slivers: Si pieces between the edge and a hole,
    or etched gaps between the edge and Si, narrower than min_feature. Within `reach` of a window edge the
    Si pieces an opening with a min_feature disc does not reach are etched (added to the holes) and the
    gaps it does not reach are filled (removed from the holes, added to the plate inside windows).
    """
    win = _union(windows)
    edge = _union([win.boundary.buffer(reach)] + [g.buffer(reach) for g in seams])
    etch = _union([_union(holes), win.difference(_union(plate))]).intersection(edge.buffer(1.0))
    box_ = box(*edge.buffer(1.0).bounds)
    si = box_.difference(etch)
    r = min_feature / 2 - 1e-4
    thin_si = si.difference(si.buffer(-r).buffer(r)).intersection(edge)
    thin_gap = etch.difference(etch.buffer(-r).buffer(r)).intersection(edge)
    holes = _polys(_union(holes + _polys(thin_si)).difference(thin_gap))
    plate = _polys(_union(plate + _polys(thin_gap.intersection(win))).difference(thin_si))
    return holes, plate


def _place(geoms, angle: float, origin: tuple[float, float]):
    def f(g):
        return affinity.translate(affinity.rotate(g, angle, origin=(0, 0)), *origin)

    return [f(g) for g in geoms] if isinstance(geoms, list) else f(geoms)


# --- the generator -----------------------------------------------------------------------------------


def build_device(spec: DeviceSpec) -> Device:
    """Assemble all layers of the device (see the module docstring)."""
    cr = Crystal(spec.shield, spec.defect_rows, spec.x_shift, spec.res)
    a, h, W = cr.a, cr.h, cr.W
    chain = spec.chain
    if spec.option == "B":
        # channel_b: line-defect channel cells after the far mirror (GPT07 "waveguide" cells); empty: the
        # line defect exits straight into the transducer window as the bare output strip (GPT07 exit → plate)
        if spec.channel_b:
            chain = chain.append(spec.channel_b, "channel")
    elif spec.option != "A":
        raise ValueError(spec.option)
    c0, c1 = chain.columns
    port, td = spec.port, spec.transducer
    # crystal block: optical-port W-waveguide columns on the left; plain shield past the chain on the right (A)
    x_lo = (c0 - port.n_wg - 0.5) * a
    x_hi = (c1 + 0.5) * a + (spec.right_pad * a if spec.option == "A" else 0.0)
    rows = spec.rows
    y_top = W / 2 + rows * h
    pieces = cr.hole_polygons(x_lo, x_hi, rows)
    regions = {"crystal": _rect(x_lo, -y_top, x_hi, y_top)}

    # cavity chain (and B channel) raster
    reg = chain.region()
    pieces = _override(pieces, reg.area, reg.air())
    regions["cavity"] = _rect((spec.chain.columns[0] - 0.5) * a, -chain.height / 2, (spec.chain.columns[1] + 0.5) * a, chain.height / 2)
    if spec.option == "B" and spec.channel_b:
        nb = len(spec.channel_b)
        regions["channel"] = _rect((c1 - nb + 0.5) * a, -chain.height / 2, (c1 + 0.5) * a, chain.height / 2)

    # option A channel: tiled along +y from row n_sep to the top of the block
    if spec.option == "A":
        ch = spec.channel_a
        F1, F2 = ch.fraction.shape
        if (rows - ch.n_sep) % 2:
            raise ValueError("rows − n_sep must be even (whole channel periods up to the top of the block)")
        n_per = (rows - ch.n_sep) // 2
        hw = ch.Lx / 2 if ch.half_width is None else ch.half_width
        y_start = W / 2 + ch.n_sep * h
        # the period raster spans rows (2k, 2k+1) of the global channel; it starts at row n_sep (any parity):
        # for odd n_sep the first half period is dropped, so the pattern keeps its phase with the lattice
        if F2 % 2:
            raise ValueError("the channel raster needs an even number of pixels along y")
        f = np.concatenate([ch.fraction] * (n_per + 1), axis=1)
        off = (ch.n_sep % 2) * (F2 // 2)
        f = f[:, off : off + n_per * F2]
        regA = RasterRegion("channel", f, ch.x_axis - ch.Lx / 2, y_start, ch.Lx / F1, 2 * h / F2,
                            clip=_rect(ch.x_axis - hw, y_start, ch.x_axis + hw, y_top + h))
        pieces = _override(pieces, regA.area, regA.air())
        regions["channel"] = regA.area

    # transducer at the channel end (local frame rotated into place)
    if spec.option == "B":
        angle, origin = 0.0, ((c1 + 0.5) * a, 0.0)
        win_half = max(y_top + h / 2, 0.0)
    else:
        angle, origin = 90.0, (spec.channel_a.x_axis, y_top)
        win_half = 6.0
    tp = transducer_parts(td, win_half)
    windows = [_place(tp["window"], angle, origin)]
    plate = _place(tp["plate"], angle, origin)
    ln = _place(tp["ln"], angle, origin)
    metal = {"hot": _place(tp["hot"], angle, origin), "gnd": _place(tp["gnd"], angle, origin)}
    regions["transducer"] = _place(_rect(*tp["rect"]), angle, origin)
    regions["plate"] = _place(_rect(*tp["plate_rect"]), angle, origin)

    # optical port on the −x side
    wh = port.window_half if port.window_half is not None else y_top + h / 2
    xw1, xw0 = x_lo, x_lo - port.window_len
    windows.append(_rect(xw0, -wh, xw1, wh))
    strip = _rect(xw0 - 1e-3, -port.w_wg / 2, xw1 + 0.5 * a, port.w_wg / 2)
    teth = []
    n_t = max(1, int(port.window_len // port.tether_pitch))
    for k in range(n_t):
        xt = xw1 - (k + 0.5) * port.window_len / n_t
        teth.append(_rect(xt - port.tether_w / 2, -wh - 1e-3, xt + port.tether_w / 2, wh + 1e-3))
    plate += [strip] + [t.difference(_rect(xw0, -port.w_wg / 2, xw1, port.w_wg / 2)) for t in teth] + teth
    # trench-defined waveguide on the unreleased Si, inverse taper to the facet
    xt0 = port.facet_x
    xt1 = port.facet_x + port.taper_len
    core = Polygon([(xw0 + 1e-3, port.w_wg / 2), (xt1, port.w_wg / 2), (xt0, port.tip_w / 2), (xt0, -port.tip_w / 2),
                    (xt1, -port.w_wg / 2), (xw0 + 1e-3, -port.w_wg / 2)])
    tr = port.trench
    trench = _rect(xt0, -port.w_wg / 2 - tr, xw0 + 1e-3, port.w_wg / 2 + tr).difference(core)
    regions["optical_port"] = _rect(xt0, -wh, (c0 - 0.5) * a, wh)

    holes_out = _polys(_union(pieces)) + _polys(trench)
    # slivers where window edges and the ends of the chain raster cut through holes
    ce = [LineString([(x, -chain.height / 2), (x, chain.height / 2)]) for x in ((c0 - 0.5) * a, (c1 + 0.5) * a)]
    if spec.option == "A":
        ce.append(regions["channel"])  # the whole channel region: its trench walls cut through shield holes
    holes_out, plate = _clean_window_edges(holes_out, windows, _polys(_union(plate)), 0.3, 0.08, seams=ce)
    frame_geoms = [_rect(xt0, -y_top - 2 * h, x_hi + 2 * h, y_top + 2 * h)] + windows + metal["hot"] + metal["gnd"]
    frame = box(*_union(frame_geoms).bounds)
    dev = Device(spec, cr, holes_out, windows, _polys(_union(plate)), _polys(_union(ln)), metal, regions, frame)
    # Si islands (pieces the windows or the sliver clean-up cut loose) would float away at release: etch them
    si = frame.difference(dev.etch())
    parts = _polys(si)
    if len(parts) > 1:
        main = max(parts, key=lambda q: q.area)
        isl = [q for q in parts if q is not main]
        dev.holes = _polys(_union(dev.holes + isl))
        dev.plate = _polys(_union(dev.plate).difference(_union(isl)))
        dev.info.pop("_etch", None)
        dev.info["removed_islands"] = [dict(centre=[round(v, 3) for v in q.centroid.coords[0]], area_um2=q.area) for q in isl]
    dev.info.update(chain=chain, a=a, h=h, W=W, y_top=y_top, x_lo=x_lo, x_hi=x_hi, transducer_frame=(angle, origin))
    return dev


# --- convenience: the cavity chain from the line-defect agent's stored designs ------------------------------


def linedefect_chain(design: str = "linedefect_opt_E1_x.npy", mirror: str = "eta:0.6", n_left: tuple[int, int] = (8, 4),
                     n_right: tuple[int, int] = (8, 8), Nt: float = 6.0, p: float = 2.0, rows: int = 3, beta: float = 32.0,
                     sharp: bool = False, n1: int = 32):
    """
    Cavity chain from an wavesolver.linedefect design (research/out/<design>) and its MorphFamily mirror:
    left (optical side) n_left = (taper half-length, mirror cells), right likewise; the morph s(n) =
    min(1, (|n|/N_t)^p) inside each taper half, 1 in the mirror cells (as linedefect.taper_profile, but
    with the two sides independent). Returns (CavityChain, LineDefectLayout, MorphFamily, s-profile).
    """
    from .linedefect import OUT, LineDefectLayout, MorphFamily, default_shield, density_for

    L = LineDefectLayout(default_shield(), rows=rows, n1=n1)
    x_d = np.load(OUT / design)
    dd = density_for(L, x_d, beta=beta)
    fam = MorphFamily(dd, x_d, mirror, sharp=sharp)
    (hl, ml), (hr, mr) = n_left, n_right
    n = np.arange(-(hl + ml), hr + mr + 1)
    s = np.minimum(1.0, (np.abs(n) / Nt) ** p) if Nt > 0 else np.ones(len(n))
    s[(n < 0) & (-n > hl)] = 1.0
    s[(n > 0) & (n > hr)] = 1.0
    cells = [fam.fine(float(si)) for si in s]
    labels = ["mirror" if si >= 1 else ("defect" if k == 0 else "taper") for k, si in zip(n, s)]
    return CavityChain(cells, L.a, L.Ly, int(n[0]), labels), L, fam, (n, s)


def plot_device(dev: Device, ax, xlim=None, ylim=None, show_release: bool = False, labels: bool = True, lw: float = 0.0):
    """Si (grey), etched (white), LN (teal, top face; bottom face outlined), metal (gold) on a matplotlib axis."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    def patch(geom, **kw):
        for p in _polys(geom):
            verts = [np.array(p.exterior.coords)] + [np.array(r.coords) for r in p.interiors]
            codes = []
            for v in verts:
                c = np.full(len(v), MPath.LINETO)
                c[0] = MPath.MOVETO
                c[-1] = MPath.CLOSEPOLY
                codes.append(c)
            ax.add_patch(PathPatch(MPath(np.concatenate(verts), np.concatenate(codes)), **kw))

    view = dev.frame if xlim is None else _rect(xlim[0], ylim[0], xlim[1], ylim[1])
    si_c, ln_c, m_c = "#9aa3ad", "#2a9d8f", "#e9b949"
    if show_release:
        patch(dev.release().intersection(view), facecolor="#f3e6e6", edgecolor="none", zorder=0)
    patch(view.difference(dev.etch()), facecolor=si_c, edgecolor="none", lw=lw, zorder=1)
    patch(_union(dev.ln_bottom()).intersection(view), facecolor="none", edgecolor=ln_c, lw=0.6, ls="--", zorder=2)
    patch(_union(dev.ln).intersection(view), facecolor=ln_c, alpha=0.75, edgecolor="none", zorder=2)
    for net, col in (("hot", m_c), ("gnd", "#c98a1b")):
        patch(_union(dev.metal[net]).intersection(view), facecolor=col, edgecolor="none", alpha=0.95, zorder=3)
    b = view.bounds
    ax.set_xlim(b[0], b[2])
    ax.set_ylim(b[1], b[3])
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    if labels:
        from matplotlib.patches import Patch

        ax.legend(handles=[Patch(color=si_c, label="Si (220 nm)"), Patch(color="white", ec="k", lw=0.5, label="etched / open"),
                           Patch(color=ln_c, alpha=0.75, label="LN patch (250 nm, top face; dashed: bottom)"),
                           Patch(color=m_c, label="metal: hot (resonator)"), Patch(color="#c98a1b", label="metal: ground")]
                  + ([Patch(color="#f3e6e6", label="released (HF undercut)")] if show_release else []),
                  loc="lower left", fontsize=7, framealpha=0.9)
