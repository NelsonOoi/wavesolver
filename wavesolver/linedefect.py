"""
Free-form line defects in a shared phoxonic crystal: geometry, swappable solver backends and a
per-supercell evaluation (bands + g0 + gradients) for codesign stage 2, step 1 (the cavity alone).

Geometry (`LineDefectLayout`). A rectangular supercell a1 = (a, 0), a2 = (0, L_y): one period
of the shield crystal along x (the Γ–K direction of the hexagonal lattice), with `defect_rows`
crystal rows (height W = defect_rows·h, h = √3 a / 2) replaced by the free-form defect region
|y| < W/2, and `rows` shield rows on each side. The upper shield is the crystal with its first
row at y₁ = W/2 + h/2 (sites at x = x_shift·a + i a); the lower shield is its mirror image
y → −y, so σ_y is an exact symmetry of every structure (also of a chiral shield, whose lower half
is then the other enantiomer; cubic Si with [100] ∥ x is σ_y invariant). At y = ±L_y/2 the two
outer rows meet as mirror images (a seam rows·h from the defect, far enough for modes in the
gaps). On the lattice-aligned (here Cartesian) grids σ_y maps pixels and nodes exactly.

Shield (`TripodShield`, `DensityShield`: anything with `.a` and `.solid(xy)`). The shield is an
input: the parametric (twisted) tripod of research/unitcell_codesign.py, or a free-form
DensityCell of research/unitcell_freeform.py.

Design (`DefectDensity`). Design values on the nodes of the mechanical grid inside the defect
region, one per σ_y orbit (optionally σ_x too). The node field (design values inside, the shield's
Si indicator outside, so the filter sees the neighbouring shield) is filtered by a periodic cone
of radius R, interpolated bilinearly to sub-points of a fine raster, projected (tanh, β, η) and
averaged per fine pixel. Inside the region the fine fraction is the projected design, outside it
is the shield's own supersampled fraction (unfiltered). The fine raster is the moving-boundary
raster of wavesolver.coupling; the optical and mechanical fractions are its block means, which is the
convention of coupling_adjoint.fine_total_gradient. Minimum length scale: the Zhou et al. (2015)
indicators on the filtered node field (as parametrize.DensityCell), plus a raster opening check
of the thresholded design (`feature_check`).

Backends. Optics: `PWEOptics` (wavesolver.optics_pwe at the X point, k = π/a). Mechanics: `FEMMech`
(wavesolver.mech_bloch slab_p, even z-parity, reduced exactly to the σ_y-even or σ_y-odd subspace).
Both expose the small interface used by `evaluate` (solve, mode selection, frequency
sensitivities, the solver objects for coupling_adjoint.g0_gradient), so a GME or ROM backend can
replace them.

Units: µm, Hz (mechanics), THz (optics).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import HexLattice, Lattice2D
from .linalg import factorize_indefinite
from .mech_adjoint import element_forms, element_vectors
from .mech_bloch import OMEGA_UNIT, BlochSolver

jax.config.update("jax_enable_x64", True)

SQRT3 = math.sqrt(3)
T_SI = 0.22
EPS_SI = 3.48**2
C_UM_THZ = 299.792458
ROOT = Path(__file__).resolve().parents[1]
# Saved designs (unit-cell results, defect and mirror vectors) are read from here.
OUT = Path(os.environ.get("WAVESOLVER_DATA_DIR", ROOT / "research/out"))


# --- shields -----------------------------------------------------------------------------------


def _reduce_hex(xy: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Points mapped into the centred unit cell of the lattice with rows A."""
    st = xy @ np.linalg.inv(A)
    st = (st + 0.5) % 1.0 - 0.5
    return st @ A


@dataclass
class TripodShield:
    """Parametric tripod crystal (wavesolver.parametrize.ParametricCell without free parameters)."""

    cell: object
    name: str = "tripod"

    @property
    def a(self) -> float:
        return float(self.cell.lattice.a)

    def solid(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float)
        p = _reduce_hex(xy.reshape(-1, 2), self.cell.lattice.vectors)
        out = np.empty(len(p), bool)
        th = jnp.zeros(0)
        for i in range(0, len(p), 65536):
            out[i : i + 65536] = np.asarray(self.cell.periodic_sdf(jnp.asarray(p[i : i + 65536]), th)) > 0
        return out.reshape(xy.shape[:-1])

    @classmethod
    def from_result(cls, path: str | Path = OUT / "unitcell_codesign_twist_B_amax700_result.json") -> "TripodShield":
        from .parametrize import ParametricCell, TripodHole

        r = json.loads(Path(path).read_text())
        return cls(ParametricCell(HexLattice(r["a"]), TripodHole(**r["shape_um"]), ()), Path(path).stem.replace("_result", ""))


@dataclass
class DensityShield:
    """Free-form shared cell (wavesolver.parametrize.DensityCell + design vector), thresholded at η."""

    dc: object
    x: np.ndarray
    eta: float = 0.5
    name: str = "freeform"

    @property
    def a(self) -> float:
        return float(self.dc.lattice.a)

    def solid(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float)
        st = xy.reshape(-1, 2) @ np.linalg.inv(self.dc.lattice.vectors)
        st = (st + 0.5) % 1.0 - 0.5
        return (self.dc.field_at(self.x, st) > self.eta).reshape(xy.shape[:-1])

    @classmethod
    def from_result(cls, name: str, eta: float = 0.5) -> "DensityShield":
        """A research/unitcell_freeform.py result by tag (e.g. 'freeform_twist_B_amax700')."""
        from .parametrize import DensityCell

        r = json.loads((OUT / f"{name}_result.json").read_text())
        dc = DensityCell(HexLattice(r.get("a_ref", r["a"])), r["n"], r["R"], r["symmetry"], beta=r["betas"][-1])
        dc.lattice = HexLattice(r["a"])  # the filter kernel stays that of a_ref: the design lives in lattice coordinates
        return cls(dc, np.load(OUT / f"{name}_x.npy"), eta, name)


def default_shield() -> TripodShield | DensityShield:
    """
    The shield crystal of the line defect. OMC_SHIELD selects it: unset or "tripod" → the parametric
    twisted tripod (unitcell_codesign_twist_B_amax700); "freeform:<tag>[:<eta>]" → a
    research/unitcell_freeform.py result (e.g. freeform:freeform_param_pads_refined).
    """
    spec = os.environ.get("OMC_SHIELD", "tripod")
    if spec == "tripod":
        return TripodShield.from_result()
    kind, _, rest = spec.partition(":")
    if kind != "freeform" or not rest:
        raise ValueError(f"OMC_SHIELD={spec!r}: expected 'tripod' or 'freeform:<tag>[:<eta>]'")
    tag, _, eta = rest.partition(":")
    return DensityShield.from_result(tag, float(eta) if eta else 0.5)


# --- layout ------------------------------------------------------------------------------------


@dataclass
class LineDefectLayout:
    """
    Rectangular line-defect supercell around a shield crystal (see the module docstring).
    n1: mechanical/design grid per period a (nodes = design nodes); n2 follows for near-square
    elements, rounded to a multiple of 4 (so the optical grid n2/2 is even). mf: fine raster
    factor over the mechanical grid; of: optical coarsening (optical grid = mechanical / of).
    """

    shield: object
    rows: int = 3
    # defaults overridable by OMC_DEFECT_ROWS / OMC_X_SHIFT (defect width W = defect_rows·h; shield offset along x in units of a)
    defect_rows: float = field(default_factory=lambda: float(os.environ.get("OMC_DEFECT_ROWS", "2.0")))
    x_shift: float = field(default_factory=lambda: float(os.environ.get("OMC_X_SHIFT", "0.0")))
    n1: int = 32
    mf: int = 2
    of: int = 2
    n2: int | None = None
    periods: int = 1

    def __post_init__(self):
        a = self.a
        self.Lx = self.periods * a  # supercell length along x (periods × the shield period)
        self.h = SQRT3 / 2 * a
        self.W = self.defect_rows * self.h
        self.Ly = self.W + 2 * self.rows * self.h
        if self.n2 is None:
            self.n2 = 4 * int(round(self.Ly / (a / self.n1) / 4))
        self.lattice = Lattice2D(np.array([[self.Lx, 0.0], [0.0, self.Ly]]))
        self.y1 = self.W / 2 + self.h / 2

    @property
    def a(self) -> float:
        return self.shield.a

    # grids
    def shape(self, which: str) -> tuple[int, int]:
        m1 = self.n1 * self.periods
        if which == "mech":
            return m1, self.n2
        if which == "fine":
            return m1 * self.mf, self.n2 * self.mf
        if which == "optics":
            return m1 // self.of, self.n2 // self.of
        raise ValueError(which)

    def pixel_xy(self, n1: int, n2: int, ss: int = 1) -> np.ndarray:
        """Lab (x, y) of the ss × ss sub-points of every pixel of an (n1, n2) grid: (n1·ss, n2·ss, 2)."""
        s1 = (np.arange(n1 * ss) + 0.5) / (n1 * ss) - 0.5
        s2 = (np.arange(n2 * ss) + 0.5) / (n2 * ss) - 0.5
        S1, S2 = np.meshgrid(s1, s2, indexing="ij")
        return np.stack([S1 * self.Lx, S2 * self.Ly], axis=-1)

    def node_xy(self, n1: int, n2: int) -> np.ndarray:
        s1 = np.arange(n1) / n1 - 0.5
        s2 = np.arange(n2) / n2 - 0.5
        S1, S2 = np.meshgrid(s1, s2, indexing="ij")
        return np.stack([S1 * self.Lx, S2 * self.Ly], axis=-1)

    def crystal_point(self, xy: np.ndarray) -> np.ndarray:
        """Shield-crystal coordinates of lab points (upper shield; lower = mirror image)."""
        x, y = xy[..., 0], np.abs(xy[..., 1])
        return np.stack([x - self.x_shift * self.a, y - self.y1], axis=-1)

    def shield_solid(self, xy: np.ndarray) -> np.ndarray:
        return self.shield.solid(self.crystal_point(xy))

    def shield_fraction(self, n1: int, n2: int, ss: int = 4) -> np.ndarray:
        """Supersampled solid fraction of the continued shield crystal on an (n1, n2) pixel grid (whole cell)."""
        s = self.shield_solid(self.pixel_xy(n1, n2, ss))
        return s.reshape(n1, ss, n2, ss).mean(axis=(1, 3))

    def region_pixels(self, n1: int, n2: int) -> np.ndarray:
        """Pixels (n1, n2) whose centre lies in the defect region |y| < W/2."""
        return np.abs(self.pixel_xy(n1, n2)[..., 1]) < self.W / 2

    def defect_window(self, extra: float = 0.5) -> float:
        """|y| bound used to tag defect-localised modes: the region plus `extra` rows."""
        return self.W / 2 + extra * self.h

    def describe(self) -> str:
        return (f"a = {self.a * 1e3:.1f} nm" + (f" × {self.periods} periods" if self.periods > 1 else "") + f", h = {self.h * 1e3:.1f} nm, W = {self.W * 1e3:.0f} nm, rows {self.rows}, "
                f"L_y = {self.Ly:.3f} µm; mech {self.shape('mech')}, fine {self.shape('fine')}, optics {self.shape('optics')}")


# --- free-form defect density --------------------------------------------------------------------


def _interp_matrix(n_nodes: int, n_sub_per_node: int) -> np.ndarray:
    """Periodic linear interpolation from n nodes (node i at s = i/n − ½) to n·m sub-points at the pixel-sub centres."""
    m = n_sub_per_node
    p = np.arange(n_nodes * m)
    u = (p + 0.5) / m
    i0 = np.floor(u).astype(int)
    t = u - i0
    Mx = np.zeros((n_nodes * m, n_nodes))
    Mx[p, i0 % n_nodes] += 1 - t
    Mx[p, (i0 + 1) % n_nodes] += t
    return Mx


class DefectDensity:
    """
    Free-form defect: design vector x (one value per symmetry orbit of the design nodes) →
    fine solid fraction (n1·mf, n2·mf). See the module docstring. beta is mutable (continuation).
    """

    def __init__(self, layout: LineDefectLayout, R: float = 0.09, beta: float = 8.0, eta: float = 0.5,
                 sym_x: bool = False, ss: int = 2, shield_ss: int = 4, band: float = 0.0, tethers: bool = False,
                 w_range: tuple[float, float] = (0.10, 0.25), tether_reach: float | None = None, tether_x0: float = 0.0):
        self.layout = L = layout
        self.R, self.beta, self.eta, self.sym_x, self.ss, self.band = R, beta, eta, sym_x, ss, band
        self.tethers, self.w_range, self.tether_x0 = tethers, w_range, tether_x0
        n1, n2 = L.shape("mech")
        self.n1, self.n2 = n1, n2
        nxy = L.node_xy(n1, n2)
        # design_reach (TF 2026-10-29, env OMC_DESIGN_REACH, default 0): the design nodes extend this far (µm) beyond W/2 into
        # the first shield row, so the optimiser can etch away trefoil finger tips that a gap would otherwise leave floating
        self.design_reach = float(os.environ.get("OMC_DESIGN_REACH", "0.0"))
        self.design_nodes = np.abs(nxy[..., 1]) < L.W / 2 + self.design_reach - 1e-9
        # orbits under σ_y (j → −j mod n2) and optionally σ_x (i → −i mod n1)
        I, J = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
        idx = [I * n2 + J, I * n2 + (-J) % n2]
        if sym_x:
            idx += [((-I) % n1) * n2 + J, ((-I) % n1) * n2 + (-J) % n2]
        rep = np.min(np.stack(idx), axis=0)
        lab = np.full(n1 * n2, -1)
        reps = np.unique(rep[self.design_nodes])
        lab_of_rep = {r: k for k, r in enumerate(reps)}
        flat = rep.ravel()
        dn = self.design_nodes.ravel()
        lab[dn] = [lab_of_rep[r] for r in flat[dn]]
        self.orbits = lab.reshape(n1, n2)
        self.n_dens = len(reps)
        # explicit tether: a solid bar across the defect rows (σ_y symmetric: one tether per side per
        # period), joined to the density before the filter. Two extra variables in [0, 1]: width
        # w = w_lo + u (w_hi − w_lo) and position x_t = x0 + (u − ½) a. It reaches |y| < W/2 + tether_reach
        # (default: the band, at least h/2), i.e. into the Si of the first shield row.
        self.n_vars = self.n_dens + (2 if tethers else 0)
        self.reach = (max(band, L.h / 2) if tether_reach is None else tether_reach)
        # background node values: shield Si indicator averaged over the node-centred pixel
        ss_b = 4
        off = (np.arange(ss_b) + 0.5) / ss_b - 0.5
        dx, dy = L.Lx / n1, L.Ly / n2
        pts = nxy[:, :, None, None, :] + np.stack(np.meshgrid(off * dx, off * dy, indexing="ij"), -1)[None, None]
        self.background = L.shield_solid(pts).mean(axis=(2, 3)).astype(float)
        self.background[self.design_nodes] = 0.0
        # cone filter on the periodic node grid
        d1 = ((np.arange(n1) + n1 // 2) % n1 - n1 // 2) * dx
        d2 = ((np.arange(n2) + n2 // 2) % n2 - n2 // 2) * dy
        D = np.sqrt(d1[:, None] ** 2 + d2[None, :] ** 2)
        w = np.maximum(R - D, 0)
        self.kernel_hat = jnp.asarray(np.fft.fft2(w / w.sum()))
        # interpolation to fine sub-points and the fixed parts of the fine raster
        mf = L.mf
        self.Ix = jnp.asarray(_interp_matrix(n1, mf * ss))
        self.Iy = jnp.asarray(_interp_matrix(n2, mf * ss))
        F1, F2 = L.shape("fine")
        self.fine_shape = (F1, F2)
        # projected zone: the design region plus a band of the (filtered, projected) shield, so the
        # design meets the shield through the filter rather than at a hard cut
        self.region_fine = (np.abs(L.pixel_xy(F1, F2)[..., 1]) < L.W / 2 + band).astype(float)
        self.ls_nodes = np.abs(nxy[..., 1]) < L.W / 2 + band - 1e-9
        self.shield_fine = L.shield_fraction(F1, F2, shield_ss)
        self._orb = jnp.asarray(np.where(self.design_nodes, self.orbits, 0))
        self._dn = jnp.asarray(self.design_nodes.astype(float))
        self._lsn = jnp.asarray(self.ls_nodes.astype(float))
        self._nx = jnp.asarray(nxy[..., 0])
        self._tz = jnp.asarray((np.abs(nxy[..., 1]) < L.W / 2 + self.reach).astype(float))
        self._bg = jnp.asarray(self.background)
        self._rf = jnp.asarray(self.region_fine)
        self._sf = jnp.asarray(self.shield_fine)
        self._fns: dict = {}

    # maps
    def tether_params(self, x) -> tuple[float, float]:
        """(width µm, centre x µm) of the tether."""
        u = np.asarray(x, float)[self.n_dens :]
        return self.w_range[0] + u[0] * (self.w_range[1] - self.w_range[0]), self.tether_x0 + (u[1] - 0.5) * self.layout.a

    def tether_nodes(self, x):
        x = jnp.asarray(x)
        w = self.w_range[0] + x[self.n_dens] * (self.w_range[1] - self.w_range[0])
        xt = self.tether_x0 + (x[self.n_dens + 1] - 0.5) * self.layout.a
        a = self.layout.a
        d = jnp.abs((self._nx - xt + a / 2) % a - a / 2)
        delta = a / self.n1 / 2
        return self._tz * 0.5 * (1 + jnp.tanh((w / 2 - d) / delta))

    def nodes(self, x):
        x = jnp.asarray(x)
        rho = self._dn * x[: self.n_dens][self._orb] + self._bg
        if self.tethers:
            rho = 1 - (1 - rho) * (1 - self.tether_nodes(x))
        return rho

    def filtered(self, x):
        return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(self.nodes(x)) * self.kernel_hat))

    def project(self, rt, beta, eta):
        return (jnp.tanh(beta * eta) + jnp.tanh(beta * (rt - eta))) / (jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)))

    def _fine(self, x, beta, eta):
        rt = self.filtered(x)
        sub = self.Ix @ rt @ self.Iy.T
        pb = self.project(sub, beta, eta)
        F1, F2 = self.fine_shape
        s = self.ss
        pix = pb.reshape(F1, s, F2, s).mean(axis=(1, 3))
        return self._rf * pix + (1 - self._rf) * self._sf

    def fine(self, x, eta: float | None = None, beta: float | None = None) -> np.ndarray:
        if "fine" not in self._fns:
            self._fns["fine"] = jax.jit(self._fine)
        return np.asarray(self._fns["fine"](jnp.asarray(x, float), self.beta if beta is None else beta, self.eta if eta is None else eta))

    def fine_vjp(self, x, G: np.ndarray, eta: float | None = None) -> np.ndarray:
        """Σ G[..., i, j] ∂fine_ij/∂x for G (..., F1, F2) → (..., n_vars)."""
        if "vjp" not in self._fns:
            self._fns["vjp"] = jax.jit(lambda x, G, beta, eta: jax.vmap(jax.vjp(lambda x: self._fine(x, beta, eta), x)[1])(G)[0])
        G = np.asarray(G, float)
        out = self._fns["vjp"](jnp.asarray(x, float), jnp.asarray(G.reshape(-1, *self.fine_shape)), self.beta, self.eta if eta is None else eta)
        return np.asarray(out).reshape(*G.shape[:-2], self.n_vars)

    def sharp_fine(self, x, eta: float | None = None, ss: int = 4) -> np.ndarray:
        """Fine fraction of the thresholded design (β → ∞), ss × ss sub-points per pixel."""
        L = self.layout
        rt = np.asarray(self.filtered(jnp.asarray(x, float)))
        Ix = _interp_matrix(self.n1, L.mf * ss)
        Iy = _interp_matrix(self.n2, L.mf * ss)
        solid = (Ix @ rt @ Iy.T) > (self.eta if eta is None else eta)
        F1, F2 = self.fine_shape
        pix = solid.reshape(F1, ss, F2, ss).mean(axis=(1, 3))
        return self.region_fine * pix + (1 - self.region_fine) * self.shield_fine

    def from_node_field(self, rho: np.ndarray) -> np.ndarray:
        """Design vector = orbit mean of a node field (n1, n2) over the design nodes."""
        o = self.orbits[self.design_nodes]
        xd = np.bincount(o, weights=rho[self.design_nodes], minlength=self.n_dens) / np.bincount(o, minlength=self.n_dens)
        return self.with_tether(xd) if self.tethers else xd

    def with_tether(self, x_dens: np.ndarray, width: float = 0.12, xt: float | None = None) -> np.ndarray:
        """Append tether variables (width µm, centre x µm; default x0) to a density vector."""
        xt = self.tether_x0 if xt is None else xt
        u_w = (width - self.w_range[0]) / (self.w_range[1] - self.w_range[0])
        a = self.layout.a
        return np.concatenate([np.asarray(x_dens, float)[: self.n_dens], [u_w, ((xt - self.tether_x0) / a + 0.5) % 1.0]])

    def from_solid_fn(self, solid_fn, ss: int = 4) -> np.ndarray:
        """Design vector from a lab-coordinate Si indicator solid_fn(xy) (e.g. a starting geometry)."""
        L = self.layout
        nxy = L.node_xy(self.n1, self.n2)
        off = (np.arange(ss) + 0.5) / ss - 0.5
        pts = nxy[:, :, None, None, :] + np.stack(np.meshgrid(off * L.Lx / self.n1, off * L.Ly / self.n2, indexing="ij"), -1)[None, None]
        val = solid_fn(pts)
        if self.design_reach > 0:  # beyond W/2 the start keeps the shield (candidates describe the defect region only)
            val = np.where(np.abs(pts[..., 1]) < L.W / 2, val, L.shield_solid(pts))
        return self.from_node_field(val.mean(axis=(2, 3)))

    # length scale (Zhou et al. 2015), on the design nodes and their filter neighbourhood
    def _length_scale(self, x, beta, eta_e, eta_d, c):
        rt = self.filtered(x)
        rb = self.project(rt, beta, self.eta)
        dx, dy = self.layout.Lx / self.n1, self.layout.Ly / self.n2
        gx = (jnp.roll(rt, -1, 0) - jnp.roll(rt, 1, 0)) / (2 * dx)
        gy = (jnp.roll(rt, -1, 1) - jnp.roll(rt, 1, 1)) / (2 * dy)
        ind = jnp.exp(-c * (gx**2 + gy**2))
        m = self._lsn
        gs = jnp.sum(m * rb * ind * jnp.minimum(rt - eta_e, 0) ** 2) / jnp.sum(m)
        gv = jnp.sum(m * (1 - rb) * ind * jnp.minimum(eta_d - rt, 0) ** 2) / jnp.sum(m)
        return jnp.stack([gs, gv])

    def length_scale(self, x, eta_e: float = 0.75, eta_d: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
        """Solid and void length-scale measures (≤ ε to satisfy) and gradients (2, n_vars)."""
        h = self.layout.Lx / self.n1
        c = self.R**4 / h**2  # as parametrize.DensityCell (gradients per µm)
        key = ("ls", eta_e, eta_d)
        if key not in self._fns:
            f = lambda x, beta: self._length_scale(x, beta, eta_e, eta_d, c)
            self._fns[key] = (jax.jit(f), jax.jit(jax.jacrev(f)))
        v, g = self._fns[key]
        xj = jnp.asarray(x, float)
        return np.asarray(v(xj, self.beta)), np.asarray(g(xj, self.beta))


def morph_fine(dd: DefectDensity, x_a: np.ndarray, x_b: np.ndarray, s: float, eta: float | None = None, sharp: bool = False) -> np.ndarray:
    """Fine raster of the cell morphed linearly in design space: x(s) = (1 − s) x_a + s x_b, then filtered and projected."""
    x = (1 - s) * np.asarray(x_a) + s * np.asarray(x_b)
    return dd.sharp_fine(x, eta) if sharp else dd.fine(x, eta)


# --- fabrication check ---------------------------------------------------------------------------


def feature_check(layout: LineDefectLayout, fine_sharp: np.ndarray, min_feature: float = 0.08, window: float | None = None,
                  min_void: float | None = None) -> dict:
    """
    Raster opening check of a thresholded fine raster (fraction > 0.5 = Si) for the minimum
    feature 2R = min_feature, for Si and void, in |y| < window (default: defect region + 1 row).
    Returns per phase the depth (µm) of the worst uncovered point beyond R and the uncovered
    area fraction, and the number of floating Si islands (components not touching the shield).
    """
    from scipy.ndimage import distance_transform_edt, label

    F1, F2 = fine_sharp.shape
    dx, dy = layout.Lx / F1, layout.Ly / F2
    solid = fine_sharp > 0.5
    # tile 3× along x for periodicity
    S = np.concatenate([solid] * 3, axis=0)
    d = (distance_transform_edt(S, sampling=(dx, dy)) - distance_transform_edt(~S, sampling=(dx, dy)))
    R = min_feature / 2
    R_void = R if min_void is None else min_void / 2  # asymmetric rule (TF 2026-10-29): void minimum may differ from Si
    win = layout.W / 2 + layout.h if window is None else window
    y = layout.pixel_xy(F1, F2)[..., 1]
    centre = np.zeros_like(S)
    centre[F1 : 2 * F1] = (np.abs(y) < win)
    out = {}
    for name, phase, Rp in (("si", d, R), ("void", -d, R_void)):
        core = phase >= Rp
        dist = distance_transform_edt(~core, sampling=(dx, dy))
        thin = (phase > 0) & (dist > Rp + 0.5 * max(dx, dy))
        out[f"{name}_depth_nm"] = float((dist[thin & centre] - Rp).max() * 1e3) if (thin & centre).any() else 0.0
        out[f"{name}_area"] = float((thin & centre).sum() / max(((phase > 0) & centre).sum(), 1))
    lab, nlab = label(S)
    touching = np.unique(lab[:, np.abs(np.concatenate([y] * 3, 0)[0]) > layout.W / 2 + layout.h])
    out["islands"] = int(len(np.setdiff1d(np.arange(1, nlab + 1), touching)))
    return out


# --- optics backend (PWE) ------------------------------------------------------------------------


@dataclass
class OpticalSolve:
    freqs: np.ndarray
    loc: np.ndarray
    solver: object
    k: np.ndarray
    time: float


class PWEOptics:
    """TE-like PWE (wavesolver.optics_pwe) on the optical grid of a layout, at in-plane k (default X)."""

    def __init__(self, layout: LineDefectLayout, nz: int = 48, height: float = 1.5, n_bands: int = 16, tol: float = 1e-4, maxiter: int = 400):
        self.layout, self.nz, self.height, self.n_bands, self.tol, self.maxiter = layout, nz, height, n_bands, tol, maxiter
        self._prev = {}

    @property
    def kX(self) -> np.ndarray:
        L = self.layout  # π/a folded into the supercell zone (Γ for an even number of periods)
        k = np.pi / L.a
        G = 2 * np.pi / L.Lx
        return np.array([(k + G / 2) % G - G / 2, 0.0])

    def solve(self, f_o: np.ndarray, k: np.ndarray | None = None, n_bands: int | None = None, warm: str | None = "X") -> OpticalSolve:
        from .optics_pwe import PWESolver, slab_cell

        k = self.kX if k is None else np.asarray(k, float)
        t0 = time.perf_counter()
        cell = slab_cell(self.layout.lattice, f_o, T_SI, self.height, self.nz, EPS_SI)
        s = PWESolver(cell, parity="even")
        nb = n_bands or self.n_bands
        if warm and warm in self._prev and self._prev[warm].shape[0] == nb + 2 and self._prev[warm].shape[2:] == s.shape:
            s._prev = self._prev[warm]
        f = s.solve(k, nb, tol=self.tol, maxiter=self.maxiter)
        if warm:
            self._prev[warm] = s._prev
        return OpticalSolve(f, self.localisation(s, nb), s, k, time.perf_counter() - t0)

    def localisation(self, s, nb: int) -> np.ndarray:
        n1, n2, _ = s.shape
        y = ((np.arange(n2) + 0.5) / n2 - 0.5) * self.layout.Ly
        inside = (np.abs(y) < self.layout.defect_window())[None, :, None]
        out = []
        for b in range(nb):
            w = (np.abs(s.h_field(b)) ** 2).sum(axis=0)
            out.append((w * inside).sum() / w.sum())
        return np.array(out)

    def translation_parity(self, so: OpticalSolve, band: int) -> float:
        """Re⟨H(x), H(x + a)⟩/|H|² of a band on a multi-period supercell: −1 for the folded X-point band, +1 for Γ."""
        H = so.solver.h_field(band)
        shift = H.shape[1] // self.layout.periods
        return float(np.real(np.vdot(H, np.roll(H, -shift, axis=1))) / np.real(np.vdot(H, H)))

    def h_overlap(self, s_a, band_a: int, s_b, band_b: int) -> float:
        """|⟨H_a, H_b⟩| / (|H_a||H_b|) of two solves on the same grid (mode tracking)."""
        Ha, Hb = s_a.h_field(band_a), s_b.h_field(band_b)
        return float(abs(np.vdot(Ha, Hb)) / (np.linalg.norm(Ha) * np.linalg.norm(Hb)))

    @staticmethod
    def sensitivity(sol: OpticalSolve, band: int) -> np.ndarray:
        """∂f/∂f_o (THz per unit fill) on the optical grid."""
        return sol.solver.sensitivities([band])[0]


# --- mechanics backend (slab_p FEM with exact σ_y reduction) --------------------------------------


def sym_y_basis(solver: BlochSolver, parity: int = 1) -> sp.csr_matrix:
    """
    Orthonormal basis of the σ_y-even (parity +1) or -odd (−1) DOF subspace of a solver on a
    rectangular, y-symmetric grid: σ_y u(x, y) = (u_x, −u_y, u_z)(x, −y). parity +1 is
    wavesolver.cavity._sym_y_basis.
    """
    n1, n2 = solver.shape
    dpn = len(solver.basis)
    g = solver.dofs
    node, b = g // dpn, g % dpn
    i, j = node // n2, node % n2
    partner = (i * n2 + (-j) % n2) * dpn + b
    sign = np.where(np.array([c for c, _ in solver.basis])[b] == 1, -1.0, 1.0) * parity
    pos = np.full(g.max() + 1, -1)
    pos[g] = np.arange(len(g))
    q = pos[partner]
    if np.any(q < 0):
        raise ValueError("DOFs are not mirror symmetric")
    me = np.arange(len(g))
    self_pair = q == me
    keep_self = self_pair & (sign > 0)
    first = (~self_pair) & (me < q)
    ns, nf = int(keep_self.sum()), int(first.sum())
    c = ns + np.arange(nf)
    rows = np.concatenate([me[keep_self], me[first], q[first]])
    cols = np.concatenate([np.arange(ns), c, c])
    vals = np.concatenate([np.ones(ns), np.full(nf, 1 / np.sqrt(2)), sign[first] / np.sqrt(2)])
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(g), ns + nf))


@dataclass
class MechSolve:
    freqs: np.ndarray
    X: np.ndarray  # full-DOF eigenvectors (ndof, nm), M-normalised
    loc: np.ndarray
    parity: int
    solver: BlochSolver
    k: np.ndarray
    time: float


class FEMMech:
    """slab_p (even z-parity) FEM on the mechanical grid, σ_y-reduced, shift-invert near a target."""

    def __init__(self, layout: LineDefectLayout, material=None, order: int = 4, min_fraction: float = 1e-9):
        from .materials import SI

        self.layout, self.material, self.order, self.min_fraction = layout, (material or SI), order, min_fraction

    def build(self, f_m: np.ndarray) -> BlochSolver:
        return BlochSolver(self.layout.lattice, f_m, self.material, model="slab_p", thickness=T_SI, parity="even",
                           order=self.order, min_fraction=self.min_fraction)

    def solve(self, f_m: np.ndarray, target_hz: float = 4e9, n_modes: int = 10, parity: int = 1, kx: float = 0.0,
              solver: BlochSolver | None = None) -> MechSolve:
        t0 = time.perf_counter()
        s = solver or self.build(f_m)
        k = np.array([kx, 0.0])
        K, M = s.matrices(k)
        P = sym_y_basis(s, parity)
        real = kx == 0
        if real:
            K, M = K.real, M.real
        Kr, Mr = (P.T @ K @ P).tocsc(), (P.T @ M @ P).tocsc()
        sigma = (2 * np.pi * target_hz / OMEGA_UNIT) ** 2
        solve = factorize_indefinite(Kr - sigma * Mr)
        op = spla.LinearOperator(Kr.shape, matvec=solve, dtype=float if real else complex)
        lam, Q = spla.eigsh(Kr, k=n_modes, M=Mr, sigma=sigma, which="LM", OPinv=op, tol=1e-10)
        o = np.argsort(lam)
        lam, Q = lam[o], Q[:, o]
        X = P @ Q
        f = OMEGA_UNIT * np.sqrt(np.clip(lam.real, 0, None)) / (2 * np.pi)
        loc = self.localisation(s, X, f_m, k)
        return MechSolve(f, X, loc, parity, s, k, time.perf_counter() - t0)

    def localisation(self, s: BlochSolver, X: np.ndarray, f_m: np.ndarray, k: np.ndarray, window: float | None = None) -> np.ndarray:
        part = s.parts[0]
        theta = s.lattice.vectors @ k
        _, eM = element_forms(part, element_vectors(part, theta, X))
        n1, n2 = s.shape
        j = part.cell % n2
        y = ((j + 0.5) / n2 - 0.5) * self.layout.Ly
        w = f_m.ravel()[part.cell][:, None] * eM
        win = self.layout.defect_window() if window is None else window
        return (w * (np.abs(y) < win)[:, None]).sum(0) / w.sum(0)

    @staticmethod
    def sensitivity(sol: MechSolve, b: int) -> np.ndarray:
        """∂f_b/∂f_m (Hz per unit fraction) on the mechanical grid (non-degenerate mode)."""
        s = sol.solver
        part = s.parts[0]
        theta = s.lattice.vectors @ sol.k
        x = sol.X[:, [b]]
        eK, eM = element_forms(part, element_vectors(part, theta, x))
        lam = (2 * np.pi * sol.freqs[b] / OMEGA_UNIT) ** 2
        _, M = s.matrices(sol.k)
        norm = np.real(np.vdot(x[:, 0], M @ x[:, 0]))
        n1, n2 = s.shape
        dlam = np.bincount(part.cell, weights=(eK[:, 0] - lam * eM[:, 0]), minlength=n1 * n2) / norm
        return (sol.freqs[b] / (2 * lam) * dlam).reshape(n1, n2)

    @staticmethod
    def overlap(sol_a: MechSolve, a: int, sol_b: MechSolve, b: int) -> float:
        """|x_aᵀ M_b x_b| / norms, on the same grid and DOF set (mode tracking); 0 if the DOF sets differ."""
        if sol_a.solver.ndof != sol_b.solver.ndof or not np.array_equal(sol_a.solver.dofs, sol_b.solver.dofs):
            # different element sets: Euclidean overlap on the union of the DOF indices
            n = max(sol_a.solver.dofs.max(), sol_b.solver.dofs.max()) + 1
            va, vb = np.zeros(n, complex), np.zeros(n, complex)
            va[sol_a.solver.dofs], vb[sol_b.solver.dofs] = sol_a.X[:, a], sol_b.X[:, b]
            return float(abs(np.vdot(va, vb)) / (np.linalg.norm(va) * np.linalg.norm(vb)))
        _, M = sol_b.solver.matrices(sol_b.k)
        xa, xb = sol_a.X[:, a], sol_b.X[:, b]
        return float(abs(np.vdot(xa, M @ xb)) / np.sqrt(abs(np.vdot(xa, M @ xa)) * abs(np.vdot(xb, M @ xb))))


__all__ = [
    "DefectDensity",
    "DensityShield",
    "FEMMech",
    "LineDefectLayout",
    "MechSolve",
    "OpticalSolve",
    "PWEOptics",
    "ThermalAnchor",
    "TripodShield",
    "default_shield",
    "feature_check",
    "tether_section",
    "morph_fine",
    "sym_y_basis",
]


# --- per-supercell evaluation: bands, g0 and gradients -------------------------------------------


@dataclass
class Targets:
    """Design targets and constraint scales of the defect cell (supercell level)."""

    f_opt: float = 193.4  # THz, at X
    tol_opt: float = 0.003  # relative half-width of the allowed optical window
    f_mech: float = 4.0e9  # Hz, at Γ
    tol_mech: float = 0.005
    sep_mech: float = 0.15e9  # min distance to any other σ_y-even mode at Γ (not seam-localised)
    sep_opt: float = 4.0  # THz, min distance to any other defect-localised optical band at X
    loc_opt: float = 0.6  # min |H|² fraction in the defect window for the optical defect band
    loc_mech: float = 0.3
    ls_eps: float = 2e-4  # length-scale measures ≤ ls_eps (fraction of the design nodes, Zhou)
    # Zhou thresholds on the filtered field: with the R = 90 nm cone a bar whose filtered peak is
    # 0.833 is 100 nm wide nominal and 80 nm eroded (η 0.6, −9 nm per side); 0.75/0.25 ≈ R nominal only
    eta_e: float = 0.75
    eta_d: float = 0.25
    g_ref: float = 1e6  # Hz, objective scale (g0/2π)


def seam_fraction_mech(mech: FEMMech, sol: MechSolve) -> np.ndarray:
    """ρ|u|² fraction within one row of the supercell seam (|y| > L_y/2 − h)."""
    L = mech.layout
    inner = mech.localisation(sol.solver, sol.X, sol.solver_fraction, sol.k, window=L.Ly / 2 - L.h)
    return 1 - inner


class AxisBridge:
    """
    Slot-closure constraint (TF branch, 2026-10-29; off unless passed to an evaluator). A slot along the
    line axis splits the defect into two halves joined only through the shield; this measures how much Si
    BRIDGES the axis in the eroded design (threshold eta, default 0.6, i.e. ~-9 nm per edge): for each fine
    column i, c_i = 1 - (mean_j (1 - rho_ij)^p)^(1/p) over the rows |y| < band (a soft minimum: c_i ~ 1 only if
    the column is Si across the whole band around y = 0), and W = sum_i c_i dx / periods, the bridged length
    per period. Constraint value c_bridge = W / w_min - 1 >= 0 (w_min: 0.08 um rule + margin). With the Zhou
    length-scale constraints active, the bridged length forms a bar >= 80 nm wide rather than slivers.
    __call__(x) -> (W, c_bridge, dc/dx).
    """

    def __init__(self, dd: "DefectDensity", w_min: float = 0.09, band: float = 0.04, eta: float = 0.6, p: float = 16.0,
                 beta: float | None = None):
        L = dd.layout
        F1, F2 = dd.fine_shape
        y = L.pixel_xy(F1, F2)[0, :, 1]
        self.rows = np.flatnonzero(np.abs(y) < band)
        if len(self.rows) == 0:
            raise ValueError("AxisBridge: no fine rows within the band")
        self.dd, self.w_min, self.band, self.eta, self.p = dd, w_min, band, eta, p
        self.dx = L.Lx / F1 / L.periods
        # beta: projection sharpness of the measured design (None: the density's current beta). A low beta (e.g. 4) keeps
        # the gradient alive where the axis is deep void (the beta-16/32 projection saturates there and dW/dx -> 0)
        self.beta = beta
        self._vjp = None

    def value(self, fine_e: np.ndarray):
        r = 1.0 - np.clip(fine_e[:, self.rows], 0.0, 1.0)
        n, p = len(self.rows), self.p
        S = np.mean(r**p, axis=1) + 1e-30
        c = 1.0 - S ** (1.0 / p)
        W = float(np.sum(c) * self.dx)
        dc_dr = S[:, None] ** (1.0 / p - 1.0) * r ** (p - 1.0) / n  # d(S^(1/p))/dr; dc/drho = +this
        G = np.zeros_like(fine_e)
        G[:, self.rows] = dc_dr * self.dx
        return W, G

    def __call__(self, x: np.ndarray, gradient: bool = True):
        if self.beta is None:
            fe = self.dd.fine(x, self.eta)
        else:
            fe = self.dd.fine(x, self.eta, self.beta)
        W, G = self.value(fe)
        c = W / self.w_min - 1.0
        if not gradient:
            return W, c, None
        if self.beta is None:
            return W, c, self.dd.fine_vjp(x, G[None] / self.w_min, self.eta)[0]
        if self._vjp is None:
            dd, b, e = self.dd, float(self.beta), float(self.eta)
            self._vjp = jax.jit(lambda xx, GG: jax.vjp(lambda z: dd._fine(z, b, e), xx)[1](GG)[0])
        return W, c, np.asarray(self._vjp(jnp.asarray(x, float), jnp.asarray(G / self.w_min)))


class Evaluator:
    """
    Values and x-gradients of the defect-cell objective and constraints for a DefectDensity:
    optical defect band at X (optics backend), σ_y-even mechanical mode at Γ (mechanics backend),
    their per-supercell g0 (coupling_adjoint.g0_gradient), the separations from neighbouring
    bands, and the length-scale measures. Modes are tracked between calls by field overlap.
    """

    def __init__(self, dd: DefectDensity, optics: PWEOptics, mech: FEMMech, targets: Targets = Targets(), p=None,
                 n_mech: int = 10, minres_tol: float = 1e-8, pick_window: float = 0.2, anchor: "ThermalAnchor | None" = None,
                 J_max: float = 2.0, eta_erode: float = 0.6, bridge: "AxisBridge | None" = None, disp_min: float = 0.0,
                 disp_k: float = 0.25):
        from .materials import SI_P

        self.dd, self.optics, self.mech, self.t = dd, optics, mech, targets
        self.p = SI_P if p is None else p
        self.n_mech = n_mech
        self.minres_tol = minres_tol
        self.pick_window = pick_window
        self.anchor, self.J_max, self.eta_erode = anchor, J_max, eta_erode
        self.bridge = bridge  # optional slot-closure constraint (AxisBridge)
        # optional mechanical-dispersion constraint (TF 2026-10-29): |f_m(Γ) − f_m(k = disp_k·π/a)| ≥ disp_min (Hz) for the
        # tracked mode's band, so that a finite cavity forms one coherent mode (a flat band, e.g. uncoupled tongues, gives
        # N_m ≈ 1 and a crowd of nearly degenerate cavity modes)
        self.disp_min, self.disp_k = disp_min, disp_k
        self.prev_opt = None  # (solver, band)
        self.prev_mech = None  # (MechSolve, index)

    # mode selection
    def pick_optical(self, so: OpticalSolve) -> int:
        cand = [b for b in range(len(so.freqs)) if so.loc[b] > self.t.loc_opt]
        if self.dd.layout.periods > 1 and self.prev_opt is None:  # only bands folded from X
            cand = [b for b in cand if self.optics.translation_parity(so, b) < 0]
        if not cand:
            raise RuntimeError("no defect-localised optical band at X")
        if self.prev_opt is not None:
            ov = [self.optics.h_overlap(self.prev_opt[0], self.prev_opt[1], so.solver, b) for b in cand]
            if max(ov) > 0.5:
                return cand[int(np.argmax(ov))]
        return min(cand, key=lambda b: abs(so.freqs[b] - self.t.f_opt))

    def pick_mech(self, sm: MechSolve, seam: np.ndarray, so: OpticalSolve, band: int, fine, f_m) -> int:
        cand = [j for j in range(len(sm.freqs)) if sm.loc[j] > self.t.loc_mech and seam[j] < 0.5]
        if not cand:
            raise RuntimeError("no defect-localised σ_y-even mechanical mode near the target")
        if self.prev_mech is not None:
            ov = [FEMMech.overlap(self.prev_mech[0], self.prev_mech[1], sm, j) for j in cand]
            if np.all(np.isfinite(ov)) and max(ov) > 0.5:
                return cand[int(np.argmax(ov))]
        # first call (or tracking lost): the strongest coupling within ±pick_window of the target
        from .coupling import g0 as g0_fwd, mech_mode, optical_mode

        om = optical_mode(so.solver, band)
        _, M = sm.solver.matrices(np.zeros(2))
        best, gbest = None, -1.0
        for j in cand:
            if abs(sm.freqs[j] / self.t.f_mech - 1) > self.pick_window:
                continue
            mm = mech_mode(sm.solver, np.zeros(2), sm.freqs[j], sm.X[:, j], M=M, fraction=f_m)
            g = abs(g0_fwd(om, mm, fine, self.p).g0)
            if g > gbest:
                best, gbest = j, g
        if best is None:
            best = min(cand, key=lambda j: abs(sm.freqs[j] - self.t.f_mech))
        return best

    def __call__(self, x: np.ndarray, gradient: bool = True, eta: float | None = None) -> dict:
        from .coupling import block_mean
        from .coupling_adjoint import block_mean_adjoint, fine_total_gradient, g0_gradient

        t0 = time.perf_counter()
        dd, L, t = self.dd, self.dd.layout, self.t
        fine = dd.fine(x, eta)
        no, nm = L.shape("optics"), L.shape("mech")
        f_o, f_m = block_mean(fine, *no), block_mean(fine, *nm)
        so = self.optics.solve(f_o)
        b = self.pick_optical(so)
        sm = self.mech.solve(f_m, t.f_mech, self.n_mech, parity=1)
        sm.solver_fraction = f_m
        seam = seam_fraction_mech(self.mech, sm)
        j = self.pick_mech(sm, seam, so, b, fine, f_m)
        t1 = time.perf_counter()
        res = g0_gradient(so.solver, b, sm.solver, sm.freqs[j], sm.X[:, j], fine, f_m, self.p, mech_freqs=sm.freqs,
                          minres_tol=self.minres_tol, meff=False)
        t2 = time.perf_counter()
        fo, fm = float(so.freqs[b]), float(sm.freqs[j])
        # neighbours
        others_m = [i for i in range(len(sm.freqs)) if i != j and seam[i] < 0.5]
        up_m = min([i for i in others_m if sm.freqs[i] > fm], key=lambda i: sm.freqs[i], default=None)
        dn_m = max([i for i in others_m if sm.freqs[i] < fm], key=lambda i: sm.freqs[i], default=None)
        others_o = [i for i in range(len(so.freqs)) if i != b and so.loc[i] > 0.5]
        up_o = min([i for i in others_o if so.freqs[i] > fo], key=lambda i: so.freqs[i], default=None)
        dn_o = max([i for i in others_o if so.freqs[i] < fo], key=lambda i: so.freqs[i], default=None)
        sgn = np.sign(res.g0) or 1.0
        g0_hz = abs(res.g0) / (2 * np.pi)
        ls, d_ls = dd.length_scale(x, t.eta_e, t.eta_d)
        vals = dict(
            obj=-g0_hz / t.g_ref,
            c_opt_lo=(fo / t.f_opt - (1 - t.tol_opt)) / t.tol_opt,
            c_opt_hi=((1 + t.tol_opt) - fo / t.f_opt) / t.tol_opt,
            c_mech_lo=(fm / t.f_mech - (1 - t.tol_mech)) / t.tol_mech,
            c_mech_hi=((1 + t.tol_mech) - fm / t.f_mech) / t.tol_mech,
        )
        if up_m is not None:
            vals["c_sep_m_up"] = (sm.freqs[up_m] - fm) / t.sep_mech - 1
        if dn_m is not None:
            vals["c_sep_m_dn"] = (fm - sm.freqs[dn_m]) / t.sep_mech - 1
        if up_o is not None:
            vals["c_sep_o_up"] = (so.freqs[up_o] - fo) / t.sep_opt - 1
        if dn_o is not None:
            vals["c_sep_o_dn"] = (fo - so.freqs[dn_o]) / t.sep_opt - 1
        vals["c_ls_solid"] = 1 - ls[0] / t.ls_eps
        vals["c_ls_void"] = 1 - ls[1] / t.ls_eps
        if self.anchor is not None:  # connectivity / thermal anchoring of the eroded design
            fine_e = dd.fine(x, self.eta_erode)
            J_h, gJ = self.anchor(block_mean(fine_e, *self.anchor.shape))
            vals["c_heat"] = 1 - J_h / self.J_max
        if self.bridge is not None:
            W_b, c_b, g_b = self.bridge(x, gradient)
            vals["c_bridge"] = c_b
        if self.disp_min > 0:
            smk = self.mech.solve(f_m, fm, 6, parity=1, kx=self.disp_k * np.pi / L.a, solver=sm.solver)
            jk = int(np.argmax([FEMMech.overlap(sm, j, smk, i) for i in range(len(smk.freqs))]))
            disp = fm - float(smk.freqs[jk])
            vals["c_disp"] = abs(disp) / self.disp_min - 1
        info = dict(f_opt=fo, f_mech=fm, g0_hz=g0_hz,
                    g_pe_hz=res.g_pe / (2 * np.pi) * sgn, m_eff=res.m_eff, band=b, mode=j, loc_opt=float(so.loc[b]),
                    loc_mech=float(sm.loc[j]), opt_bands=so.freqs.tolist(), opt_loc=so.loc.tolist(), mech_freqs=sm.freqs.tolist(),
                    mech_loc=sm.loc.tolist(), mech_seam=seam.tolist(), fill=float(fine.mean()), ls=ls.tolist(),
                    t_opt=so.time, t_mech=sm.time, t_g0=t2 - t1)
        if self.anchor is not None:
            info["J_heat"] = J_h
        if self.bridge is not None:
            info["bridge_um"] = W_b
        if self.disp_min > 0:
            info["disp_hz"] = disp
        if dd.tethers:
            info["tether_w"], info["tether_x"] = (float(v) for v in dd.tether_params(x))
        info["g_mb_hz"] = res.g_mb / (2 * np.pi) * sgn
        out = dict(values=vals, info=info, fine=fine, so=so, sm=sm, band=b, mode=j)
        if gradient:
            F = dd.fine_shape
            G = {"obj": -sgn * fine_total_gradient(res) / (2 * np.pi) / t.g_ref}
            d_fo = block_mean_adjoint(self.optics.sensitivity(so, b), F)
            d_fm = block_mean_adjoint(res.d_mech_freq, F)
            G["c_opt_lo"] = d_fo / t.f_opt / t.tol_opt
            G["c_opt_hi"] = -G["c_opt_lo"]
            G["c_mech_lo"] = d_fm / t.f_mech / t.tol_mech
            G["c_mech_hi"] = -G["c_mech_lo"]
            if up_m is not None:
                G["c_sep_m_up"] = (block_mean_adjoint(FEMMech.sensitivity(sm, up_m), F) - d_fm) / t.sep_mech
            if dn_m is not None:
                G["c_sep_m_dn"] = (d_fm - block_mean_adjoint(FEMMech.sensitivity(sm, dn_m), F)) / t.sep_mech
            if up_o is not None:
                G["c_sep_o_up"] = (block_mean_adjoint(self.optics.sensitivity(so, up_o), F) - d_fo) / t.sep_opt
            if dn_o is not None:
                G["c_sep_o_dn"] = (d_fo - block_mean_adjoint(self.optics.sensitivity(so, dn_o), F)) / t.sep_opt
            keys = list(G)
            gx = dd.fine_vjp(x, np.stack([G[k] for k in keys]), eta)
            grads = dict(zip(keys, gx))
            if self.anchor is not None:
                grads["c_heat"] = -dd.fine_vjp(x, block_mean_adjoint(gJ, F)[None], self.eta_erode)[0] / self.J_max
            grads["c_ls_solid"] = -d_ls[0] / t.ls_eps
            grads["c_ls_void"] = -d_ls[1] / t.ls_eps
            if self.bridge is not None:
                grads["c_bridge"] = g_b
            if self.disp_min > 0:
                d_disp = block_mean_adjoint(res.d_mech_freq - FEMMech.sensitivity(smk, jk), F)
                grads["c_disp"] = np.sign(disp) * dd.fine_vjp(x, d_disp[None], eta)[0] / self.disp_min
            out["grads"] = grads
        self.prev_opt = (so.solver, b)
        self.prev_mech = (sm, j)
        info["t_total"] = time.perf_counter() - t0
        return out


# --- optimiser: sequential linearised steps with a move limit (SLP/SQP) ----------------------------


def sqp_step(g_obj: np.ndarray, cons: dict, grads: dict, x: np.ndarray, move, rho: float = 1.0, penalty: float = 50.0, w=None):
    """
    One subproblem: min_d g·d + w/2 |d|² (w = ρ |g|_∞ / move) + penalty Σ s_i  s.t. c_i + ∇c_i·d + s_i ≥ 0, s ≥ 0,
    max(−move, −x) ≤ d ≤ min(move, 1 − x). Separable in d, so it is solved through its dual: for
    multipliers λ ∈ [0, penalty]^m (the elastic slacks bound them), d(λ) = clip((Jᵀλ − g)/w, lo, hi)
    and the concave dual is maximised by L-BFGS-B (m is small).
    move and w may be per-variable arrays (w given: used as is, e.g. scaled per group of variables
    whose gradients differ by orders of magnitude).
    Returns d and the predicted objective change, slacks and linearised constraints.
    """
    from scipy.optimize import minimize

    keys = list(cons)
    m = len(keys)
    c = np.array([cons[k] for k in keys])
    J = np.array([grads[k] for k in keys])
    lo = np.maximum(-move, -x)
    hi = np.minimum(move, 1 - x)
    # proximal weight scaled to the objective gradient: unconstrained, the steepest component would
    # move by move/ρ (a fixed ρ/move² makes the steps vanish when |g| ≪ 1)
    if w is None:
        w = rho * max(np.abs(g_obj).max(), 1e-12) / move

    def d_of(lam):
        return np.clip((J.T @ lam - g_obj) / w, lo, hi)

    def neg_dual(lam):
        d = d_of(lam)
        r = c + J @ d
        val = g_obj @ d + 0.5 * w * d @ d - lam @ r
        return -val, r  # ∂(dual)/∂λ = −r (envelope theorem)

    r = minimize(neg_dual, np.zeros(m), jac=True, bounds=[(0, penalty)] * m, method="L-BFGS-B", options=dict(maxiter=500, gtol=1e-10))
    d = d_of(r.x)
    lin = c + J @ d
    return d, dict(pred_obj=float(g_obj @ d), slack=float(np.maximum(-lin, 0).sum()), success=bool(r.success), nit=int(r.nit),
                   pred_cons=dict(zip(keys, lin.tolist())), lam=dict(zip(keys, r.x.tolist())))


def merit(vals: dict, penalty: float = 50.0, dead: float = 0.0) -> float:
    """Objective + penalty · Σ violations beyond a dead band (constraints are scaled to O(1) windows)."""
    return vals["obj"] + penalty * sum(max(0.0, -v - dead) for k, v in vals.items() if k.startswith("c_"))


# --- envelope (tight-binding) model of the tapered cavity -----------------------------------------


def tb_parameters(f_edge, f_off, k0: str, dka: float):
    """
    Nearest-neighbour band f(k) = e + 2t cos(ka) through the band-edge value f_edge at k0 ("gamma":
    k = 0, "x": k = π/a) and f_off at a distance dka = δk·a from it (signed t: at Γ t > 0 is a band
    maximum; at X t > 0 a band minimum). Returns (e, t).
    """
    f_edge, f_off = np.asarray(f_edge, float), np.asarray(f_off, float)
    if k0 == "gamma":
        t = -(f_off - f_edge) / (2 * (1 - np.cos(dka)))
        return f_edge - 2 * t, t
    t = (f_off - f_edge) / (2 * (1 - np.cos(dka)))
    return f_edge + 2 * t, t


def tb_modes(e: np.ndarray, t: np.ndarray):
    """Eigenmodes of H = diag(e_n) + t_{n+½} (bonds: mean of the neighbouring cells), ψ = 0 past the ends."""
    e, t = np.asarray(e, float), np.asarray(t, float)
    th = (t[1:] + t[:-1]) / 2
    return np.linalg.eigh(np.diag(e) + np.diag(th, 1) + np.diag(th, -1))


def tb_cavity_mode(e: np.ndarray, t: np.ndarray, k0: str, centre: int | None = None):
    """
    The cavity mode of a tapered chain: the TB eigenmode with the most weight in the three central
    cells among those whose frequency lies inside the central cell's band (bound below the mirror
    band edge). Returns (frequency, ψ (signed, unit norm), envelope |ψ| with the X-point sign
    alternation removed).
    """
    w, V = tb_modes(e, t)
    n = len(e)
    c = n // 2 if centre is None else centre
    # the fundamental cavity mode is the extreme eigenvalue on the band-edge side of the central
    # cell (the top of the spectrum for a band maximum at k0, the bottom for a minimum)
    tc = np.asarray(t, float)[c]
    edge_is_max = (tc > 0) if k0 == "gamma" else (tc < 0)
    order = np.argsort(w)[::-1] if edge_is_max else np.argsort(w)
    i = int(order[0])
    wgt = (V[max(c - 1, 0) : c + 2] ** 2).sum(axis=0)
    if wgt[i] < 1e-3 * wgt.max():  # the extreme state lives elsewhere (e.g. a deeper well in the taper): take the most central of the first few
        i = int(max(order[:5], key=lambda j: wgt[j]))
    psi = V[:, i]
    env = psi * (np.where(np.arange(n) % 2, -1.0, 1.0) if k0 == "x" else 1.0)
    env = env * np.sign(env[c] or 1.0)
    return float(w[i]), psi, env


def complex_k(f: float, e: float, t: float, k0: str) -> float:
    """Decay per cell κ (Im k·a) of the TB band at frequency f outside the band (0 inside): mirror strength."""
    c = (f - e) / (2 * t)  # cos(ka) for a real k
    if abs(c) <= 1:
        return 0.0
    return float(np.arccosh(abs(c)))


def envelope_g0(env_o: np.ndarray, env_m: np.ndarray, g_cell: np.ndarray | float) -> float:
    """
    Cavity g0 from the envelopes: Σ_n g_n w_n ψm_n / √(Σ ψm_n²), w_n = |ψo_n|² / Σ|ψo|² the optical
    weight per cell, ψm the mechanical envelope (signed, Bloch mode sign aligned with the cell
    used for g_n), g_n the per-supercell g0 of cell n (assumes equal per-cell m_eff).
    """
    w = env_o**2 / np.sum(env_o**2)
    return float(np.sum(np.asarray(g_cell) * w * env_m) / np.sqrt(np.sum(env_m**2)))


def overlap_factor(env_o: np.ndarray, env_m: np.ndarray) -> float:
    """g0_cavity / g0_cell for uniform cells: Σ w ψm / √Σψm² (≤ 1, = 1/√N for flat envelopes over N cells)."""
    return envelope_g0(env_o, env_m, 1.0)


def participation(env: np.ndarray, power: int = 2) -> float:
    """N_eff = Σ|ψ|² / max|ψ|² (power 2) of an envelope."""
    a = np.abs(env) ** power
    return float(a.sum() / a.max())


def light_cone_fraction(env_o: np.ndarray, a: float, f_thz: float, n_k: int = 4096) -> float:
    """
    Fraction of the optical envelope's spectral power (DTFT of the X-point envelope, i.e. the
    field's Fourier components near ±π/a) with |k_x| < ω/c: a 1D proxy for the radiation into the
    light cone (k_y spread ignored, so it overestimates the light-cone content).
    """
    n = np.arange(len(env_o))
    q = np.linspace(-np.pi, np.pi, n_k, endpoint=False)  # q a (offset from X)
    spec = np.abs(np.exp(-1j * np.outer(q, n)) @ env_o) ** 2
    k_lc = 2 * np.pi * f_thz / C_UM_THZ * a  # (ω/c)·a
    inside = np.abs(np.pi + q) < k_lc
    return float(spec[inside].sum() / spec.sum())


# --- morph family and cavity chain ------------------------------------------------------------------


class MorphFamily:
    """
    Cells x(s) between the defect cell (s = 0) and a mirror cell (s = 1), as fine rasters: mirror
    "eta:<η_M>" (same design thresholded at η(s) = η₀ + s(η_M − η₀): a uniform erosion/dilation) or
    "x:<file.npy>" (a second design; linear in design space, then filter and projection).
    """

    def __init__(self, dd: DefectDensity, x_d: np.ndarray, mirror: str, sharp: bool = False):
        self.dd, self.x_d, self.mirror, self.sharp = dd, np.asarray(x_d, float), mirror, sharp
        kind, _, arg = mirror.partition(":")
        self.kind = kind
        if kind == "eta":
            self.eta_m = float(arg)
        elif kind == "x":
            self.x_m = np.load(OUT / arg)
        else:
            raise ValueError(mirror)
        self._cache: dict = {}

    def fine(self, s: float) -> np.ndarray:
        key = round(float(s), 10)
        if key not in self._cache:
            dd = self.dd
            if self.kind == "eta":
                eta = dd.eta + s * (self.eta_m - dd.eta)
                f = dd.sharp_fine(self.x_d, eta) if self.sharp else dd.fine(self.x_d, eta)
            else:
                x = (1 - s) * self.x_d + s * self.x_m
                f = dd.sharp_fine(x) if self.sharp else dd.fine(x)
            self._cache[key] = f
        return self._cache[key]


def taper_profile(n_half: int, n_mirror: int, Nt: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell indices n = −(n_half + n_mirror) … and morph s(n) = min(1, (|n|/N_t)^p), 1 beyond n_half."""
    n = np.arange(-(n_half + n_mirror), n_half + n_mirror + 1)
    s = np.minimum(1.0, (np.abs(n) / Nt) ** p) if Nt > 0 else np.ones(len(n))
    s[np.abs(n) > n_half] = 1.0
    return n, s


def layout_like(base: LineDefectLayout, rows: int) -> LineDefectLayout:
    """The layout of `base` with another number of shield rows and the same node spacing (n2 scaled)."""
    n2 = base.n2 * (base.defect_rows + 2 * rows) / (base.defect_rows + 2 * base.rows)
    if abs(n2 - round(n2)) > 1e-9 or round(n2) % 2:
        raise ValueError(f"rows {rows}: n2 = {n2} is not an even integer")
    return LineDefectLayout(base.shield, rows=rows, defect_rows=base.defect_rows, x_shift=base.x_shift, n1=base.n1, mf=base.mf, of=base.of,
                            n2=int(round(n2)), periods=base.periods)


def reembed(x: np.ndarray, dd_src: DefectDensity, dd_dst: DefectDensity) -> np.ndarray:
    """A design vector of dd_src on dd_dst (same defect region and node spacing, more or fewer shield rows)."""
    if dd_src.n1 != dd_dst.n1 or abs(dd_src.layout.Ly / dd_src.n2 - dd_dst.layout.Ly / dd_dst.n2) > 1e-12:
        raise ValueError("reembed needs the same node grid spacing")
    rho = np.zeros((dd_src.n1, dd_src.n2))
    rho[dd_src.design_nodes] = np.asarray(x)[: dd_src.n_dens][dd_src.orbits[dd_src.design_nodes]]
    off = (dd_dst.n2 - dd_src.n2) // 2
    big = np.zeros((dd_dst.n1, dd_dst.n2))
    if off >= 0:
        big[:, off : off + dd_src.n2] = rho
    else:
        big = rho[:, -off : -off + dd_dst.n2]
    out = dd_dst.from_node_field(big)
    if dd_dst.tethers:
        out = np.concatenate([out[: dd_dst.n_dens], np.asarray(x)[dd_src.n_dens :][:2]]) if dd_src.tethers else dd_dst.with_tether(out)
    return out


# --- connectivity / thermal anchoring: virtual temperature ---------------------------------------


class ThermalAnchor:
    """
    Virtual-temperature connectivity constraint (heat-conduction analogy, e.g. Li et al. 2016) on
    an (n1, n2) pixel grid of the supercell: conductivity k = k_min + (1 − k_min) f³ of the solid
    fraction f, heat source q = f per unit area in the defect region |y| < W/2, temperature 0 in
    the shield beyond |y| > W/2 + sink_rows·h, periodic in x (finite volumes, arithmetic-mean face
    conductance). The measure J = (Σ q T^p / Σ q)^{1/p} (≈ max T over the heated Si, µm²·(q/k))
    is small when every Si part of the defect conducts to the membrane, ~1/k_min for a floating
    part: a differentiable connectivity constraint J ≤ J_max and a direct proxy for the thermal
    anchoring of the defect. __call__(f) → (J, ∂J/∂f) by one adjoint solve.
    """

    def __init__(self, layout: LineDefectLayout, shape: tuple[int, int], kmin: float = 1e-4, p: float = 10.0, sink_rows: float = 1.0):
        self.layout, self.shape, self.kmin, self.p = layout, shape, kmin, p
        n1, n2 = shape
        self.dx, self.dy = layout.Lx / n1, layout.Ly / n2
        y = ((np.arange(n2) + 0.5) / n2 - 0.5) * layout.Ly
        dom_j = np.abs(y) < layout.W / 2 + sink_rows * layout.h
        self.src = np.broadcast_to((np.abs(y) < layout.W / 2)[None, :], shape).astype(float)
        idx = -np.ones(shape, int)
        cells = np.argwhere(np.broadcast_to(dom_j[None, :], shape))
        idx[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
        self.idx, self.n = idx, len(cells)
        I, J = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
        faces = []
        for di, dj, c in ((1, 0, self.dy / self.dx), (0, 1, self.dx / self.dy)):
            A = (I.ravel(), J.ravel())
            B = ((I.ravel() + di) % n1, J.ravel() + dj)
            ok = B[1] < n2
            A, B = (A[0][ok], A[1][ok]), (B[0][ok], B[1][ok])
            ia, ib = idx[A], idx[B]
            keep = (ia >= 0) | (ib >= 0)
            faces.append((np.ravel_multi_index((A[0][keep], A[1][keep]), shape), np.ravel_multi_index((B[0][keep], B[1][keep]), shape),
                          ia[keep], ib[keep], np.full(keep.sum(), c)))
        self.fa, self.fb, self.ia, self.ib, self.fc = (np.concatenate([f[k] for f in faces]) for k in range(5))

    def __call__(self, f: np.ndarray):
        f = np.asarray(f, float).ravel()
        k = self.kmin + (1 - self.kmin) * f**3
        dk = 3 * (1 - self.kmin) * f**2
        g = self.fc * (k[self.fa] + k[self.fb]) / 2
        n = self.n
        rows, cols, vals = [], [], []
        for i, j in ((self.ia, self.ib), (self.ib, self.ia)):
            m = i >= 0
            rows.append(i[m]); cols.append(i[m]); vals.append(g[m])
            mm = m & (j >= 0)
            rows.append(i[mm]); cols.append(j[mm]); vals.append(-g[mm])
        K = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        cell = np.flatnonzero(self.idx.ravel() >= 0)
        area = self.dx * self.dy
        w = (f * self.src.ravel())[cell]
        q = w * area
        lu = spla.splu(K)
        T = lu.solve(q)
        p = self.p
        Tp = np.maximum(T, 0) ** p
        S = (w @ Tp) / w.sum()
        Jv = S ** (1 / p)
        dJ_dT = S ** (1 / p - 1) * w * np.maximum(T, 0) ** (p - 1) / w.sum()
        dJ_dw = S ** (1 / p - 1) / p * (Tp - S) / w.sum()
        lam = lu.solve(dJ_dT)  # K symmetric
        grad = np.zeros(f.size)
        grad[cell] += (lam * area + dJ_dw) * self.src.ravel()[cell]
        Ta = np.where(self.ia >= 0, T[np.maximum(self.ia, 0)], 0.0)
        Tb = np.where(self.ib >= 0, T[np.maximum(self.ib, 0)], 0.0)
        La = np.where(self.ia >= 0, lam[np.maximum(self.ia, 0)], 0.0)
        Lb = np.where(self.ib >= 0, lam[np.maximum(self.ib, 0)], 0.0)
        c = -self.fc / 2 * (La - Lb) * (Ta - Tb)  # −λᵀ ∂K/∂k_a T, per face, for each side
        np.add.at(grad, self.fa, c * dk[self.fa])
        np.add.at(grad, self.fb, c * dk[self.fb])
        Tfull = np.zeros(f.size)
        Tfull[cell] = T
        self.last_T = Tfull.reshape(self.shape)
        return float(Jv), grad.reshape(self.shape)


def tether_section(layout: LineDefectLayout, fine_sharp: np.ndarray, y_lo: float | None = None, y_hi: float | None = None) -> dict:
    """
    Narrowest horizontal cut between the defect centre and the shield (upper half; σ_y gives the
    lower): min over y in [y_lo, y_hi] (default W/2 − 0.3 µm … W/2 + h/2) of the Si length along x
    per period, i.e. the total tether width per side per period, and its cross-section (× 220 nm).
    """
    F1, F2 = fine_sharp.shape
    y = ((np.arange(F2) + 0.5) / F2 - 0.5) * layout.Ly
    lo = layout.W / 2 - 0.3 if y_lo is None else y_lo
    hi = layout.W / 2 + layout.h / 2 if y_hi is None else y_hi
    sel = (y > lo) & (y < hi)
    widths = (fine_sharp[:, sel] > 0.5).sum(axis=0) * layout.Lx / F1 / layout.periods  # per shield period
    j = int(np.argmin(widths))
    col = fine_sharp[:, np.flatnonzero(sel)[j]] > 0.5
    xs = ((np.arange(F1) + 0.5) / F1 - 0.5) * layout.Lx
    edges = np.flatnonzero(np.diff(np.concatenate([[0], col.astype(int), [0]])))
    intervals = [(float(xs[s0] - layout.Lx / F1 / 2), float(xs[s1 - 1] + layout.Lx / F1 / 2)) for s0, s1 in zip(edges[::2], edges[1::2])]
    return dict(width_um=float(widths[j]), y_um=float(y[sel][j]), section_um2=float(widths[j] * T_SI), intervals=intervals)


def density_for(layout: LineDefectLayout, x: np.ndarray, R: float = 0.09, beta: float = 32.0, band: float = 0.2834,
                tether_x0: float = 0.5) -> DefectDensity:
    """The DefectDensity a stored design vector belongs to (tethers if it carries the two extra variables; band, x0 as in the tethered runs)."""
    dd = DefectDensity(layout, R=R, beta=beta, band=band)
    if len(x) == dd.n_dens + 2:
        dd = DefectDensity(layout, R=R, beta=beta, band=band, tethers=True, tether_x0=tether_x0 * layout.a)
    elif len(x) != dd.n_dens:
        raise ValueError(f"design length {len(x)} does not fit this layout ({dd.n_dens} density variables)")
    return dd


def tile_periods(x: np.ndarray, dd1: DefectDensity, ddp: DefectDensity) -> np.ndarray:
    """A one-period design repeated on a `periods`-period supercell (same node spacing; aligned so the patterns coincide)."""
    p = ddp.layout.periods
    rho = np.zeros((dd1.n1, dd1.n2))
    rho[dd1.design_nodes] = np.asarray(x)[: dd1.n_dens][dd1.orbits[dd1.design_nodes]]
    # node i of the p-period cell sits at x = (i/(p n) − ½) p a, node j of the one-period cell at (j/n − ½) a
    shift = (dd1.n1 * (p - 1) // 2) % dd1.n1
    return ddp.from_node_field(np.roll(np.concatenate([rho] * p, axis=0), shift, axis=0))


# --- mirror cell ----------------------------------------------------------------------------------


class MirrorEvaluator:
    """
    Mirror-cell design: stay as close as possible to the defect cell (objective ‖x − x_D‖²/n, so the
    morph between them is gentle) while every defect-localised band of the cell avoids a window
    around the cavity frequencies: TE-like bands at X outside f_oc ± Δo, σ_y-even mechanical bands
    (localised, not seam) outside f_mc ± Δm at each k of `mech_ks` (fractions of π/a). With those
    windows empty the cell is a mirror for both modes; the band offsets are its strength. Plus the
    connectivity (virtual temperature, eroded design) and Zhou length-scale constraints.
    """

    def __init__(self, dd: DefectDensity, optics: PWEOptics, mech: FEMMech, x_ref: np.ndarray, f_oc: float, d_o: float, f_mc: float,
                 d_m: float, mech_ks=(0.0, 0.5, 1.0), anchor: "ThermalAnchor | None" = None, J_max: float = 2.0, targets: Targets = Targets(),
                 n_mech: int = 10, loc_o: float = 0.5, loc_m: float = 0.3, robust: dict | None = None, bridge: "AxisBridge | None" = None):
        # robust: {η: f_mc(η)} — mechanical windows also for the eroded/dilated mirror, centred on the
        # defect cell's band edge at the same η (fab variation shifts both cells together)
        self.robust = robust or {}
        self.dd, self.optics, self.mech, self.x_ref = dd, optics, mech, np.asarray(x_ref, float)
        self.f_oc, self.d_o, self.f_mc, self.d_m, self.mech_ks = f_oc, d_o, f_mc, d_m, mech_ks
        self.anchor, self.J_max, self.t, self.n_mech, self.loc_o, self.loc_m = anchor, J_max, targets, n_mech, loc_o, loc_m
        self.bridge = bridge  # optional slot-closure constraint (AxisBridge): the mirror cell bridges the axis

    def __call__(self, x: np.ndarray, gradient: bool = True) -> dict:
        from .coupling import block_mean
        from .coupling_adjoint import block_mean_adjoint

        t0 = time.perf_counter()
        dd, L = self.dd, self.dd.layout
        F = dd.fine_shape
        fine = dd.fine(x)
        f_o, f_m = block_mean(fine, *L.shape("optics")), block_mean(fine, *L.shape("mech"))
        n = len(x)
        vals = {"obj": float(np.sum((x - self.x_ref) ** 2) / n)}
        G = {}
        grads = {"obj": 2 * (x - self.x_ref) / n}
        info = {}
        so = self.optics.solve(f_o)
        loc = [b for b in range(len(so.freqs)) if so.loc[b] > self.loc_o]
        up = min([b for b in loc if so.freqs[b] >= self.f_oc], key=lambda b: so.freqs[b], default=None)
        dn = max([b for b in loc if so.freqs[b] < self.f_oc], key=lambda b: so.freqs[b], default=None)
        info["opt"] = [(float(so.freqs[b]), float(so.loc[b])) for b in range(len(so.freqs))]
        if up is not None:
            vals["c_o_up"] = (so.freqs[up] - self.f_oc) / self.d_o - 1
            if gradient:
                G["c_o_up"] = block_mean_adjoint(self.optics.sensitivity(so, up), F) / self.d_o
        if dn is not None:
            vals["c_o_dn"] = (self.f_oc - so.freqs[dn]) / self.d_o - 1
            if gradient:
                G["c_o_dn"] = -block_mean_adjoint(self.optics.sensitivity(so, dn), F) / self.d_o
        info["f_o_up"] = None if up is None else float(so.freqs[up])
        info["f_o_dn"] = None if dn is None else float(so.freqs[dn])
        s = self.mech.build(f_m)
        for kk in self.mech_ks:
            sm = self.mech.solve(f_m, self.f_mc, self.n_mech, kx=kk * np.pi / L.a, solver=s)
            sm.solver_fraction = f_m
            seam = seam_fraction_mech(self.mech, sm)
            ok = [j for j in range(len(sm.freqs)) if sm.loc[j] > self.loc_m and seam[j] < 0.5]
            up = min([j for j in ok if sm.freqs[j] >= self.f_mc], key=lambda j: sm.freqs[j], default=None)
            dn = max([j for j in ok if sm.freqs[j] < self.f_mc], key=lambda j: sm.freqs[j], default=None)
            info[f"mech_k{kk:g}"] = [(float(f), float(l), float(sf)) for f, l, sf in zip(sm.freqs, sm.loc, seam)]
            for name, j, sgn in ((f"c_m{kk:g}_up", up, 1), (f"c_m{kk:g}_dn", dn, -1)):
                if j is None:
                    continue
                vals[name] = sgn * (sm.freqs[j] - self.f_mc) / self.d_m - 1
                info[name.replace("c_", "f_")] = float(sm.freqs[j])
                if gradient:
                    G[name] = sgn * block_mean_adjoint(FEMMech.sensitivity(sm, j), F) / self.d_m
        Gr = {}
        for eta_r, f_r in self.robust.items():
            f_mr = block_mean(dd.fine(x, eta_r), *L.shape("mech"))
            sr = self.mech.build(f_mr)
            for kk in self.mech_ks:
                sm = self.mech.solve(f_mr, f_r, self.n_mech, kx=kk * np.pi / L.a, solver=sr)
                sm.solver_fraction = f_mr
                seam = seam_fraction_mech(self.mech, sm)
                ok = [j for j in range(len(sm.freqs)) if sm.loc[j] > self.loc_m and seam[j] < 0.5]
                up = min([j for j in ok if sm.freqs[j] >= f_r], key=lambda j: sm.freqs[j], default=None)
                dn = max([j for j in ok if sm.freqs[j] < f_r], key=lambda j: sm.freqs[j], default=None)
                for name, j, sgn in ((f"c_m{kk:g}_up_e{eta_r:g}", up, 1), (f"c_m{kk:g}_dn_e{eta_r:g}", dn, -1)):
                    if j is None:
                        continue
                    vals[name] = sgn * (sm.freqs[j] - f_r) / self.d_m - 1
                    info[name.replace("c_", "f_")] = float(sm.freqs[j])
                    if gradient:
                        Gr.setdefault(eta_r, {})[name] = sgn * block_mean_adjoint(FEMMech.sensitivity(sm, j), F) / self.d_m
        ls, d_ls = dd.length_scale(x, self.t.eta_e, self.t.eta_d)
        vals["c_ls_solid"] = 1 - ls[0] / self.t.ls_eps
        vals["c_ls_void"] = 1 - ls[1] / self.t.ls_eps
        grads["c_ls_solid"] = -d_ls[0] / self.t.ls_eps
        grads["c_ls_void"] = -d_ls[1] / self.t.ls_eps
        if self.anchor is not None:
            J_h, gJ = self.anchor(block_mean(dd.fine(x, 0.6), *self.anchor.shape))
            vals["c_heat"] = 1 - J_h / self.J_max
            info["J_heat"] = J_h
            if gradient:
                grads["c_heat"] = -dd.fine_vjp(x, block_mean_adjoint(gJ, F)[None], 0.6)[0] / self.J_max
        if self.bridge is not None:
            W_b, c_b, g_b = self.bridge(x, gradient)
            vals["c_bridge"] = c_b
            info["bridge_um"] = W_b
            if gradient:
                grads["c_bridge"] = g_b
        if gradient and G:
            keys = list(G)
            gx = dd.fine_vjp(x, np.stack([G[k] for k in keys]))
            grads.update(dict(zip(keys, gx)))
        for eta_r, Ge in Gr.items():
            keys = list(Ge)
            gx = dd.fine_vjp(x, np.stack([Ge[k] for k in keys]), eta_r)
            grads.update(dict(zip(keys, gx)))
        info["t_total"] = time.perf_counter() - t0
        return dict(values=vals, grads=grads, info=info, fine=fine)
