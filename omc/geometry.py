"""
Lattices, hole shapes and their rasterisation onto lattice-aligned grids.

Lengths are in µm. The tripod (blade) hole is defined here. The line-defect supercell and the
finite-cavity builders in omc.cavity and omc.layout take their hole shapes from an external fab
generator, `gpt_omc.py`, which is not part of this repository: it is imported on first use, from
the Python path, from the directory in GPT_OMC_PATH, or from research/ in the repository.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

SQRT3 = math.sqrt(3)

_ROOT = Path(__file__).resolve().parents[1]
ARM_DIRECTIONS = np.array([(0.0, 1.0), (SQRT3 / 2, -0.5), (-SQRT3 / 2, -0.5)])
CROTCH_DIRECTIONS = np.array([(SQRT3 / 2, 0.5), (0.0, -1.0), (-SQRT3 / 2, 0.5)])
QUAD_SEGS = 32


class _LazyGptOmc:
    """Stands in for the `gpt_omc` module and imports it on first attribute access."""

    _module = None

    def __getattr__(self, name: str):
        if _LazyGptOmc._module is None:
            for path in (os.environ.get("GPT_OMC_PATH"), _ROOT / "research"):
                if path and (Path(path) / "gpt_omc.py").exists() and str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            try:
                import gpt_omc
            except ImportError as e:
                raise ImportError(
                    "this builder needs the external fab generator gpt_omc.py: put it on the Python path, "
                    "in research/, or set GPT_OMC_PATH to its directory"
                ) from e
            _LazyGptOmc._module = gpt_omc
        return getattr(_LazyGptOmc._module, name)


gpt_omc = _LazyGptOmc()


@dataclass(frozen=True)
class HexLattice:
    """Triangular lattice with a1 along x, as the blade shield in gpt_omc."""

    a: float

    @property
    def vectors(self) -> np.ndarray:
        """Rows a1, a2."""
        return self.a * np.array([[1.0, 0.0], [0.5, SQRT3 / 2]])

    @property
    def reciprocal(self) -> np.ndarray:
        """Rows b1, b2 with a_i · b_j = 2π δ_ij."""
        return 2 * np.pi * np.linalg.inv(self.vectors).T

    def gamma_m_k_path(self, n_per_leg: int) -> tuple[np.ndarray, np.ndarray]:
        """
        k-points Γ → M → K → Γ in 1/µm, with the corners shared between legs, as in the
        COMSOL band scripts. Returns the points (N, 2) and the indices of Γ, M, K, Γ.
        """
        G = np.zeros(2)
        M = np.pi / self.a * np.array([1.0, 1 / SQRT3])
        K = np.pi / self.a * np.array([4 / 3, 0.0])
        legs = [(G, M), (M, K), (K, G)]
        t = np.linspace(0, 1, n_per_leg)
        points = [p + t[:, None] * (q - p) for p, q in legs]
        ks = np.concatenate([points[0]] + [leg[1:] for leg in points[1:]])
        corners = np.array([0, n_per_leg - 1, 2 * n_per_leg - 2, 3 * n_per_leg - 3])
        return ks, corners


@dataclass(frozen=True)
class Lattice2D:
    """A general 2D lattice (or supercell), with the same interface as HexLattice."""

    vectors: np.ndarray

    @property
    def reciprocal(self) -> np.ndarray:
        return 2 * np.pi * np.linalg.inv(self.vectors).T


def tripod(r: float, w: float) -> np.ndarray:
    """
    Outline of three arms of width w reaching r from the centre, at 90°, -30° and 210°: the two
    corners at the end of each arm, then the crotch where it meets the next arm, on the bisector
    of the two arms at w/√3 from the centre (nine vertices, clockwise).
    """
    points = []
    for direction, crotch in zip(ARM_DIRECTIONS, CROTCH_DIRECTIONS):
        normal = np.array([-direction[1], direction[0]])
        points += [r * direction + w / 2 * normal, r * direction - w / 2 * normal, w / SQRT3 * crotch]
    return np.array(points)


def round_corners(shape: Polygon, radius: float) -> Polygon:
    """Round every corner to at least `radius` by a closing followed by an opening."""
    if radius == 0:
        return shape
    rounded = (
        shape.buffer(radius, quad_segs=QUAD_SEGS)
        .buffer(-2 * radius, quad_segs=QUAD_SEGS)
        .buffer(radius, quad_segs=QUAD_SEGS)
    )
    if not isinstance(rounded, Polygon) or rounded.is_empty:
        raise ValueError("the fillet split or erased the hole; use a smaller radius")
    return orient(rounded)


def blade_hole(r: float, w: float, fillet: float = 0.0) -> Polygon:
    """Blade (tripod) hole centred on the origin, with every corner rounded to `fillet`."""
    return round_corners(Polygon(tripod(r, w)), fillet)


def line_defect_supercell(
    crystal: gpt_omc.Crystal | None = None, cell: str = "defect", shield_rows: int = 6, fillet: float = 0.0
) -> tuple[Lattice2D, list[Polygon]]:
    """
    One period of the GPT dagger line defect: the dagger row at y = ±row_offset (the lower
    row the mirror image of the upper, as in gpt_omc) and shield_rows rows of blades on each
    side, all as drawn on the mask (Crystal.drawn(): scale applied, no SEM correction).
    cell picks the dagger shape: "mirror", "defect" or "waveguide".

    The supercell is a1 = (a, 0), a2 = (a/2, L_y), with L_y one row pitch more than the
    distance between the outer blade rows. The a/2 offset keeps the triangular stacking across
    the boundary, where the next supercell's (mirrored) blade rows begin.
    """
    c = (crystal or gpt_omc.Crystal()).drawn()
    shape = {"mirror": c.mirror, "defect": c.defect, "waveguide": c.waveguide}[cell]
    a, rp = c.pitch, c.row_pitch
    top = [affinity.translate(p, 0.0, c.row_offset) for p in gpt_omc.dagger(shape)]
    for r in range(1, shield_rows + 1):
        x = a / 2 if r % 2 else 0.0
        top += [affinity.translate(p, x, c.row_offset + r * rp) for p in gpt_omc.blade(c.shield)]
    holes = top + [affinity.scale(p, 1.0, -1.0, origin=(0, 0)) for p in top]
    if fillet:
        holes = [round_corners(p, fillet) for p in holes]
    height = 2 * (c.row_offset + shield_rows * rp) + rp
    return Lattice2D(np.array([[a, 0.0], [a / 2, height]])), holes


def solid_fraction(
    lattice: HexLattice | Lattice2D, holes: list[Polygon], n: int | tuple[int, int], supersample: int = 4
) -> np.ndarray:
    """
    Fraction of solid in each cell of an n1 × n2 grid along a1, a2 (n may be one int for a
    square grid), with hole coordinates measured from the unit cell's centre. Holes from
    neighbouring cells are included, so shapes may cross the cell boundary. Returned as f[i, j]
    for the cell spanning [i, i+1]/n1 along a1 and [j, j+1]/n2 along a2, measured from the
    cell corner at -(a1 + a2)/2.
    """
    n1, n2 = (n, n) if np.isscalar(n) else n
    s1 = (np.arange(n1 * supersample) + 0.5) / (n1 * supersample) - 0.5
    s2 = (np.arange(n2 * supersample) + 0.5) / (n2 * supersample) - 0.5
    S, T = np.meshgrid(s1, s2, indexing="ij")
    A = lattice.vectors
    xy = S[..., None] * A[0] + T[..., None] * A[1]
    shape = unary_union(holes)
    shapely.prepare(shape)
    inside = np.zeros(S.shape, dtype=bool)
    for i, j in itertools.product((-1, 0, 1), repeat=2):
        shifted = xy - (i * A[0] + j * A[1])
        inside |= shapely.contains_xy(shape, shifted[..., 0], shifted[..., 1])
    return 1.0 - inside.reshape(n1, supersample, n2, supersample).mean(axis=(1, 3))
