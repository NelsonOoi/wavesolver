"""
Differentiable maps from design parameters to the element solid fractions of `BlochSolver`.

Both maps return f[i, j] on the n × n lattice-aligned grid of `geometry.solid_fraction` and a
vector-Jacobian product, so band gradients ∂f_band/∂fraction from `omc.mech_adjoint` chain to
the parameters. They are written in JAX (float64).

Parametric holes (`TripodHole`, `ParametricCell`): each hole is a signed distance function
(negative inside) built from rounded boxes, discs and half-planes, joined by smooth minima
and maxima that round the corners. The fraction of an element is the mean of a C² smoothed
Heaviside H(sdf/δ) over supersample points, with δ the supersample spacing: a smoothed area
coverage, exactly 0 or 1 away from the boundary, so fully void elements stay void (and are
dropped by the solver). The same SDF gives the hole outlines (`ParametricCell.polygons`,
contoured finely) for `geometry.solid_fraction`, GDS and min-feature checks, and the Si
bridge widths between distinct holes: dist(A, B) = min_x sdf_A(x) + sdf_B(x), whose
parameter gradient is ∂_θ(sdf_A + sdf_B) at the minimiser (envelope theorem).

Free-form density (`DensityCell`): design values on the grid nodes, which, unlike element
centres, map onto each other under the 60° rotations and mirrors of the hexagonal cell, so
C3v or C6v symmetry is imposed exactly by sharing one variable per node orbit. Then a
periodic cone filter of radius R (physical distance on the skewed grid), a tanh projection
of sharpness β about η (raised by continuation), and element fraction = mean of the four
corner nodes. Minimum length scale: the geometric constraints of Zhou et al. (2015), which
penalise solid where the filtered field is below η_e, and void where it is above η_d, at
the structural skeleton (where the filtered field is flat); with a cone filter of radius R and η = 0.5,
η_e ≈ 0.75 and η_d ≈ 0.25 give a minimum length scale of about R.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import HexLattice

jax.config.update("jax_enable_x64", True)

SQRT3 = math.sqrt(3)


# -- signed distance primitives (negative inside) ----------------------------------------


def sd_disc(p: jnp.ndarray, c, r) -> jnp.ndarray:
    return jnp.linalg.norm(p - c, axis=-1) - r


def sd_box(p: jnp.ndarray, c, half_u, half_v, u, rr) -> jnp.ndarray:
    """Box of half-sizes (half_u, half_v) along (u, u⊥), centred at c, corners rounded by rr."""
    d = p - c
    du = jnp.abs(d @ u)
    dv = jnp.abs(d @ jnp.array([-u[1], u[0]]))
    qu, qv = du - (half_u - rr), dv - (half_v - rr)
    outside = jnp.sqrt(jnp.maximum(qu, 0) ** 2 + jnp.maximum(qv, 0) ** 2 + 1e-30)
    return outside + jnp.minimum(jnp.maximum(qu, qv), 0) - rr


def smin(a: jnp.ndarray, b: jnp.ndarray, k) -> jnp.ndarray:
    """Polynomial smooth minimum (union), blending over |a - b| < k; min(a, b) for k = 0."""
    k = jnp.maximum(k, 1e-9)
    h = jnp.maximum(k - jnp.abs(a - b), 0) / k
    return jnp.minimum(a, b) - h * h * k / 4


def smooth_step(x: jnp.ndarray) -> jnp.ndarray:
    """C² step on [-1, 1]: 0 below, 1 above."""
    x = jnp.clip(x, -1, 1)
    return 0.5 + x * (15 - 10 * x**2 + 3 * x**4) / 16


# -- parametric cells ---------------------------------------------------------------------


@dataclass
class TripodHole:
    """
    C3v tripod hole on each lattice site, generalising the blade: three arms of width w
    reaching r from the centre at 90°, 210°, 330° (as gpt_omc.tripod), pointing at the
    down-triangle centroids J (distance D = a/√3), where three arm ends of neighbouring
    holes meet.

    Pads: the last pad_t of each arm widens to a half-width pad_l, and the arm end is cut
    by sides parallel to the bisectors between neighbouring arms at J, so the arm ends of
    the three holes around J leave a triangular Si node at J (inradius D - r) joined to the
    masses by three straight Si channels of width `channel`: compliant tethers whose width
    is a parameter and whose length grows with pad_l and pad_t. Without pads the arm ends
    leave short notches (the blade).

    hub: central disc; junction: disc hole at J; rr: convex-corner radius; blend: fillet
    radius of the inner corners. twist (rad): rotates the arms about the hole centre, which
    breaks the mirrors (C3 only): the arm ends then pass beside J and overlap their
    neighbours side by side, so the Si between them forms long spiral tethers. Twisted
    arms keep their pads but not the channel sides, which assume arms pointing at J. Lengths in µm; a feature of size 0 is absent. The defaults
    are the blade.
    """

    r: float = 0.172
    w: float = 0.093
    hub: float = 0.0
    pad_t: float = 0.0
    pad_l: float = 0.1
    channel: float = 0.1
    rr: float = 0.025
    blend: float = 0.025
    junction: float = 0.0
    twist: float = 0.0


ARM_ANGLES = np.radians([90.0, 210.0, 330.0])


BLEND_K = 0.6  # smin width per unit fillet radius: matches gpt_omc.fillet's closing (fill to 1e-5)


def tripod_sdf(p: jnp.ndarray, th: dict) -> jnp.ndarray:
    """SDF of one tripod hole centred on the origin (without the junction hole)."""
    # The three arms at once: p in each arm's frame, with the arm along +y.
    rot = jnp.asarray(ARM_ANGLES - np.pi / 2) + th.get("twist", 0.0)
    c, s = jnp.cos(rot), jnp.sin(rot)
    R = jnp.stack([jnp.stack([c, s], -1), jnp.stack([-s, c], -1)], -2)  # (3, 2, 2), rotates by -rot
    q = jnp.einsum("aij,...j->a...i", R, p)
    u = jnp.array([0.0, 1.0])
    k = BLEND_K * th["blend"]
    d = sd_box(q, u * th["r"] / 2, th["r"] / 2, th["w"] / 2, u, jnp.minimum(th["rr"], th["w"] / 2 - 1e-4))
    if "pad_t" in th:
        pad_rr = jnp.minimum(th["rr"], jnp.minimum(th["pad_t"] / 2, th["pad_l"]) - 1e-4)
        d = smin(d, sd_box(q, u * (th["r"] - th["pad_t"] / 2), th["pad_t"] / 2, th["pad_l"], u, pad_rr), k)
        # Channel sides: the arm end stays channel/2 from the bisector lines through J that
        # run between this arm and its neighbours (x = ±√3 (D - y) - channel).
        x, y = q[..., 0], q[..., 1]
        for sign in (1.0, -1.0) if "twist" not in th else ():
            h = th["channel"] / 2 - (-sign * x / 2 - SQRT3 / 2 * (y - th["D"]))
            d = -smin(-d, -h, BLEND_K * th["rr"])
    d = smin(smin(d[0], d[1], k), d[2], k)
    if "hub" in th:
        d = smin(d, sd_disc(p, jnp.zeros(2), th["hub"]), k)
    return d


@dataclass
class ParametricCell:
    """
    A hexagonal cell with a `TripodHole`, some of whose fields are free. theta orders the
    names in `free`; the rest stay at their values in `base`. Bounds per name in `bounds`.
    """

    lattice: HexLattice
    base: TripodHole
    free: tuple[str, ...]
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        self.down = np.array([0.0, self.lattice.a / SQRT3])  # down-triangle centroid
        self._fns: dict = {}

    # parameter vector <-> named values
    def values(self, theta: np.ndarray) -> dict:
        v = {f.name: getattr(self.base, f.name) for f in fields(self.base)}
        v.update(dict(zip(self.free, theta)))
        return v

    def theta0(self) -> np.ndarray:
        return np.array([getattr(self.base, k) for k in self.free])

    def shape(self, theta: np.ndarray) -> TripodHole:
        return TripodHole(**{k: float(x) for k, x in self.values(theta).items()})

    def _th(self, theta) -> dict:
        v = {f.name: getattr(self.base, f.name) for f in fields(self.base)}
        th = {k: jnp.asarray(x, dtype=float) for k, x in v.items()}
        for i, k in enumerate(self.free):
            th[k] = theta[i]
        th["D"] = self.lattice.a / SQRT3
        if self.base.pad_t == 0 and "pad_t" not in self.free:
            th.pop("pad_t"), th.pop("pad_l"), th.pop("channel")
        if self.base.hub == 0 and "hub" not in self.free:
            th.pop("hub")
        if self.base.twist == 0 and "twist" not in self.free:
            th.pop("twist")
        return th

    @property
    def has_junction(self) -> bool:
        return self.base.junction > 0 or "junction" in self.free

    def component_sdfs(self, p: jnp.ndarray, theta) -> list[jnp.ndarray]:
        """SDFs of the distinct holes of the cell at the origin: tripod, then junction."""
        th = self._th(theta)
        out = [tripod_sdf(p, th)]
        if self.has_junction:
            out.append(sd_disc(p, jnp.asarray(self.down), th["junction"]))
        return out

    def periodic_sdf(self, p: jnp.ndarray, theta) -> jnp.ndarray:
        """SDF of all holes of the crystal (3 × 3 images suffice for points in the cell)."""
        A = self.lattice.vectors
        offsets = jnp.asarray([i * A[0] + j * A[1] for i in (-1, 0, 1) for j in (-1, 0, 1)])
        pp = p[None] - offsets.reshape(9, *([1] * (p.ndim - 1)), 2)
        return jnp.min(jnp.stack(self.component_sdfs(pp, theta)).min(axis=0), axis=0)

    # element fractions
    def fraction_fn(self, n: int, supersample: int = 4):
        """Jitted theta ↦ f (n, n), as `geometry.solid_fraction` with a smoothed boundary."""
        key = ("frac", n, supersample)
        if key not in self._fns:
            pts = jnp.asarray(sample_points(self.lattice, n, supersample))
            delta = self.lattice.a / (n * supersample)

            def frac(theta):
                h = smooth_step(self.periodic_sdf(pts, theta) / delta)
                return h.reshape(n, supersample, n, supersample).mean(axis=(1, 3))

            self._fns[key] = jax.jit(frac)
        return self._fns[key]

    def fraction(self, theta: np.ndarray, n: int, supersample: int = 4) -> np.ndarray:
        return np.asarray(self.fraction_fn(n, supersample)(jnp.asarray(theta, dtype=float)))

    def fraction_jacobian(self, theta: np.ndarray, n: int, supersample: int = 4) -> np.ndarray:
        """∂f/∂θ (n, n, len(theta)) by forward mode: cheap for a few parameters."""
        key = ("jac", n, supersample)
        if key not in self._fns:
            self._fns[key] = jax.jit(jax.jacfwd(self.fraction_fn(n, supersample)))
        return np.asarray(self._fns[key](jnp.asarray(theta, dtype=float)))

    def fraction_vjp(self, theta: np.ndarray, G: np.ndarray, n: int, supersample: int = 4) -> np.ndarray:
        """Σ_ij G[..., i, j] ∂f_ij/∂θ for G (..., n, n): shape (..., len(theta))."""
        return np.einsum("...ij,ijp->...p", G, self.fraction_jacobian(theta, n, supersample))

    # geometry outputs
    def polygons(self, theta: np.ndarray, spacing: float = 0.001) -> list:
        """Shapely outlines of the cell's holes (tripod, junction), contoured from the SDF."""
        import contourpy
        from shapely import affinity
        from shapely.geometry import Polygon

        a = self.lattice.a
        g = np.arange(-0.75 * a, 0.75 * a + spacing / 2, spacing)
        P = np.stack(np.meshgrid(g, g, indexing="xy"), axis=-1)
        theta = jnp.asarray(theta, dtype=float)
        centres = [np.zeros(2), self.down][: len(self.component_sdfs(jnp.zeros((1, 2)), theta))]
        polys = []
        for ci, c in enumerate(centres):
            d = np.asarray(self.component_sdfs(jnp.asarray(P + c), theta)[ci])
            lines = contourpy.contour_generator(g, g, d).lines(0.0)
            if lines:
                polys.append(affinity.translate(Polygon(max(lines, key=len)), *c))
        return polys

    # minimum features
    def pairs(self) -> list[tuple[int, np.ndarray, int, np.ndarray]]:
        """
        Hole pairs (component a, offset a, component b, offset b) bounding the Si bridges,
        one per symmetry-distinct centre distance below 1.2 a: tripod-tripod, and with a
        junction hole, tripod-junction (arm end and flank) and junction-junction.
        """
        A = self.lattice.vectors
        centres = [np.zeros(2), self.down] if self.has_junction else [np.zeros(2)]
        out, seen = [], set()
        for ca in range(len(centres)):
            for cb in range(ca, len(centres)):
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        R = i * A[0] + j * A[1]
                        d = np.linalg.norm(centres[cb] + R - centres[ca])
                        key = (ca, cb, round(d / self.lattice.a, 6))
                        if 1e-9 < d < 1.2 * self.lattice.a and key not in seen:
                            seen.add(key)
                            out.append((ca, np.zeros(2), cb, R))
        return out

    def bridge_fn(self):
        """
        Jitted (theta, X) ↦ (sdf_A + sdf_B, sdf_A) per pair at points X (npairs, m, 2), and
        the θ-Jacobian of sdf_A + sdf_B at one point x (npairs, 2) per pair.
        """
        if "bridge" not in self._fns:
            pairs = self.pairs()

            def g(theta, X):
                sa = jnp.stack([self.component_sdfs(x - oa, theta)[ca] for (ca, oa, _, _), x in zip(pairs, X)])
                sb = jnp.stack([self.component_sdfs(x - ob, theta)[cb] for (_, _, cb, ob), x in zip(pairs, X)])
                return sa + sb, sa

            self._fns["bridge"] = (jax.jit(g), jax.jit(jax.jacrev(lambda th, x: g(th, x[:, None, :])[0][:, 0])))
        return self._fns["bridge"]

    def bridges(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Si bridge widths of the distinct hole pairs (µm; negative where the holes merge) and
        their θ-gradients (npairs, len(theta)): min of sdf_A + sdf_B over Si points, by a
        coarse grid search and two finer local grids.
        """
        theta = jnp.asarray(theta, dtype=float)
        a = self.lattice.a
        npair = len(self.pairs())
        fn, jac = self.bridge_fn()
        g = np.arange(-1.2 * a, 1.2 * a, 0.004)
        P = np.stack(np.meshgrid(g, g, indexing="xy"), axis=-1).reshape(-1, 2)
        X = np.broadcast_to(P, (npair, *P.shape))
        for span in (None, 0.008, 0.0004):
            if span is not None:
                t = np.linspace(-span, span, 41)
                L = np.stack(np.meshgrid(t, t, indexing="xy"), axis=-1).reshape(-1, 2)
                X = best[:, None, :] + L[None]
            v, sa = (np.asarray(x) for x in fn(theta, jnp.asarray(X)))
            si = (sa > 0) & (v - sa > 0)  # Si points: outside both holes
            v = np.where(si.any(axis=1, keepdims=True) & ~si, np.inf, v)
            best = X[np.arange(npair), np.argmin(v, axis=1)]
        x = jnp.asarray(best)
        return np.asarray(fn(theta, x[:, None, :])[0])[:, 0], np.asarray(jac(theta, x))


def feature_check(cell: ParametricCell, theta: np.ndarray, R: float = 0.05, spacing: float = 0.002) -> dict:
    """
    Raster check of the minimum feature 2R for Si and for holes, independent of the pair
    list: a point is covered if some disc of radius R inside the same phase contains it
    (morphological opening, via distance transforms on a Cartesian grid of the periodic
    SDF). Returns, per phase, the largest distance (µm) of an uncovered point beyond R
    ("depth": ~0 for a clean design, a few nm at fillets tighter than R, about half the
    length of any bridge or slot narrower than 2R) and the uncovered area fraction, the
    number and area (µm²) of floating Si islands, and the grid and masks for plotting.
    """
    a = cell.lattice.a
    g = np.arange(-a, a, spacing)
    X, Y = np.meshgrid(g, g, indexing="xy")
    xy = np.stack([X, Y], axis=-1)
    st = xy @ np.linalg.inv(cell.lattice.vectors)  # lattice coordinates
    st = (st + 0.5) % 1.0 - 0.5
    d = np.asarray(cell.periodic_sdf(jnp.asarray(st @ cell.lattice.vectors), jnp.asarray(theta, dtype=float)))
    return _opening_check(d, g, a, R, spacing)


def _opening_check(d: np.ndarray, g: np.ndarray, a: float, R: float, spacing: float) -> dict:
    """The checks of `feature_check` from the signed distance d (Si > 0) on the grid g × g."""
    from scipy.ndimage import distance_transform_edt, label

    X, Y = np.meshgrid(g, g, indexing="xy")
    out = {"x": g, "sdf": d}
    centre = (np.abs(X) < a / 2) & (np.abs(Y) < a / 2)
    for name, phase in (("si", d), ("void", -d)):
        core = phase >= R  # centres of discs of radius R inside the phase
        dist = distance_transform_edt(~core, sampling=spacing)
        thin = (phase > 0) & (dist > R + spacing)
        out[f"{name}_mask"] = thin
        out[f"{name}_depth"] = float((dist[thin & centre] - R).max()) if (thin & centre).any() else 0.0
        out[f"{name}_area"] = float(thin[centre].sum() / max((phase > 0)[centre].sum(), 1))
    # Floating Si: components of the Si raster that do not reach the edge of the 2a × 2a box
    # (the connected membrane spans every cell).
    lab, nlab = label(d > 0)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    islands = np.setdiff1d(np.arange(1, nlab + 1), edge)
    out["islands"] = int(len(islands))
    out["island_area"] = float(np.isin(lab, islands).sum() * spacing**2)
    return out


def sample_points(lattice: HexLattice, n: int, supersample: int) -> np.ndarray:
    """Supersample points (n·s, n·s, 2) in µm, ordered as in `geometry.solid_fraction`."""
    m = n * supersample
    s = (np.arange(m) + 0.5) / m - 0.5
    S, T = np.meshgrid(s, s, indexing="ij")
    A = lattice.vectors
    return S[..., None] * A[0] + T[..., None] * A[1]


# -- free-form density ----------------------------------------------------------------------


def node_orbits(n: int, symmetry: str = "C3v") -> np.ndarray:
    """
    Orbit label of every node (n, n) of the cell grid under the point group about the cell
    centre (node n/2, n/2): "C3v" (rotations by 120°, mirrors x → -x and its rotations, as
    the tripod), "C6v", their chiral subgroups "C3" (the twisted tripod) and "C6", or "none".
    """
    assert n % 2 == 0
    p, q = np.meshgrid(np.arange(n) - n // 2, np.arange(n) - n // 2, indexing="ij")

    def rot60(p, q):  # a1 → a2, a2 → a2 - a1
        return -q, p + q

    def mirror(p, q):  # x → -x
        return -p - q, q

    ops = [(p, q)]
    if symmetry != "none":
        step = 1 if symmetry.startswith("C6") else 2
        cur = (p, q)
        rots = [cur]
        for _ in range(5):
            cur = rot60(*cur)
            rots.append(cur)
        rots = rots[::step]
        ops = rots + ([mirror(*r) for r in rots] if symmetry.endswith("v") else [])
    idx = np.stack([((pp + n // 2) % n) * n + (qq + n // 2) % n for pp, qq in ops])  # (nops, n, n)
    # The operations form a group, so the images of a node are its whole orbit.
    _, inv = np.unique(idx.min(axis=0), return_inverse=True)
    return inv.reshape(n, n)


@dataclass
class DensityCell:
    """
    Free-form density on the node grid: x (one value per symmetry orbit) → filter (cone,
    radius R µm) → projection (β, η) → element fractions. beta is mutable for continuation.
    """

    lattice: HexLattice
    n: int
    R: float
    symmetry: str = "C3v"
    beta: float = 1.0
    eta: float = 0.5

    def __post_init__(self):
        self.orbits = node_orbits(self.n, self.symmetry)
        self.n_vars = int(self.orbits.max()) + 1
        n, A = self.n, self.lattice.vectors
        d = (np.arange(n) + n // 2) % n - n // 2  # minimal-image offsets
        D1, D2 = np.meshgrid(d, d, indexing="ij")
        dist = np.linalg.norm((D1[..., None] * A[0] + D2[..., None] * A[1]) / n, axis=-1)
        w = np.maximum(self.R - dist, 0)
        self.kernel_hat = jnp.asarray(np.fft.fft2(w / w.sum()))
        self._fns: dict = {}

    def nodes(self, x):
        return jnp.asarray(x)[self.orbits]

    def filtered(self, x):
        return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(self.nodes(x)) * self.kernel_hat))

    def project(self, rt, beta, eta=None):
        eta = self.eta if eta is None else eta
        return (jnp.tanh(beta * eta) + jnp.tanh(beta * (rt - eta))) / (jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)))

    def _frac(self, x, beta, eta):
        rb = self.project(self.filtered(x), beta, eta)
        return (rb + jnp.roll(rb, -1, 0) + jnp.roll(rb, -1, 1) + jnp.roll(jnp.roll(rb, -1, 0), -1, 1)) / 4

    def fraction(self, x: np.ndarray, eta: float | None = None) -> np.ndarray:
        """Element fractions (n, n); eta other than self.eta gives an eroded (> η) or dilated design."""
        if "frac" not in self._fns:
            self._fns["frac"] = jax.jit(self._frac)
        eta = self.eta if eta is None else eta
        return np.asarray(self._fns["frac"](jnp.asarray(x, dtype=float), self.beta, eta))

    def fraction_vjp(self, x: np.ndarray, G: np.ndarray, eta: float | None = None) -> np.ndarray:
        """Σ_ij G[..., i, j] ∂f_ij/∂x for G (..., n, n): shape (..., n_vars)."""
        if "vjp" not in self._fns:
            self._fns["vjp"] = jax.jit(lambda x, G, beta, eta: jax.vmap(jax.vjp(lambda x: self._frac(x, beta, eta), x)[1])(G)[0])
        G = np.asarray(G, dtype=float)
        eta = self.eta if eta is None else eta
        out = self._fns["vjp"](jnp.asarray(x, dtype=float), jnp.asarray(G.reshape(-1, self.n, self.n)), self.beta, eta)
        return np.asarray(out).reshape(*G.shape[:-2], self.n_vars)

    # sharp geometry: the filtered field, bilinear between nodes, thresholded at η (β → ∞)
    def field_at(self, x: np.ndarray, st: np.ndarray) -> np.ndarray:
        """Filtered field at lattice coordinates st (..., 2) measured from the cell centre."""
        from scipy.ndimage import map_coordinates

        rt = np.asarray(self.filtered(jnp.asarray(x, dtype=float)))
        u = (np.asarray(st) + 0.5) * self.n  # node i sits at s = i/n - 1/2
        return map_coordinates(rt, np.moveaxis(u, -1, 0), order=1, mode="grid-wrap")

    def sharp_fraction(self, x: np.ndarray, n: int, eta: float | None = None, supersample: int = 4) -> np.ndarray:
        """Element fractions (n, n) of the thresholded design, on any grid (e.g. finer for verification)."""
        m = n * supersample
        s = (np.arange(m) + 0.5) / m - 0.5
        st = np.stack(np.meshgrid(s, s, indexing="ij"), axis=-1)
        solid = self.field_at(x, st) > (self.eta if eta is None else eta)
        return solid.reshape(n, supersample, n, supersample).mean(axis=(1, 3))

    def feature_check(self, x: np.ndarray, R: float, eta: float | None = None, spacing: float = 0.002) -> dict:
        """`feature_check` of the thresholded design (signed distance from its raster)."""
        from scipy.ndimage import distance_transform_edt

        a = self.lattice.a
        g = np.arange(-a, a, spacing)
        X, Y = np.meshgrid(g, g, indexing="xy")
        st = np.stack([X, Y], axis=-1) @ np.linalg.inv(self.lattice.vectors)
        solid = self.field_at(x, (st + 0.5) % 1.0 - 0.5) > (self.eta if eta is None else eta)
        d = (distance_transform_edt(solid) - distance_transform_edt(~solid)) * spacing
        return _opening_check(d, g, a, R, spacing)

    def _length_scale(self, x, beta, eta_e, eta_d, c):
        """Zhou et al. (2015) solid and void length-scale measures (≤ ε to satisfy)."""
        rt = self.filtered(x)
        rb = self.project(rt, beta)
        A = self.lattice.vectors / self.n
        g1 = (jnp.roll(rt, -1, 0) - jnp.roll(rt, 1, 0)) / 2  # ∂/∂(grid index 1)
        g2 = (jnp.roll(rt, -1, 1) - jnp.roll(rt, 1, 1)) / 2
        Jinv = np.linalg.inv(np.column_stack([A[0], A[1]]))  # grid index per µm
        gx = g1 * Jinv[0, 0] + g2 * Jinv[1, 0]
        gy = g1 * Jinv[0, 1] + g2 * Jinv[1, 1]
        ind = jnp.exp(-c * (gx**2 + gy**2))
        gs = jnp.mean(rb * ind * jnp.minimum(rt - eta_e, 0) ** 2)
        gv = jnp.mean((1 - rb) * ind * jnp.minimum(eta_d - rt, 0) ** 2)
        return jnp.stack([gs, gv])

    def length_scale(self, x: np.ndarray, eta_e: float = 0.75, eta_d: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
        """Solid and void length-scale measures and their gradients (2, n_vars)."""
        h = self.lattice.a / self.n
        c = self.R**4 / h**2  # Zhou's c = R⁴ in grid units: the indicator is ~1 only on the skeleton
        f = lambda x: self._length_scale(x, self.beta, eta_e, eta_d, c)
        x = jnp.asarray(x, dtype=float)
        return np.asarray(f(x)), np.asarray(jax.jacrev(f)(x))

    def from_fraction_nodes(self, rho_nodes: np.ndarray) -> np.ndarray:
        """Design vector whose node field is the orbit average of rho_nodes (n, n)."""
        return np.bincount(self.orbits.ravel(), weights=rho_nodes.ravel()) / np.bincount(self.orbits.ravel())
