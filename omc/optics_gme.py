"""
Optical modes of a patterned slab by guided-mode expansion (GME), after L. C. Andreani and
D. Gerace, PRB 73, 235114 (2006); `legume` (Minkov et al., ACS Photonics 2020) is the reference
implementation used for cross-checks.

The slab is one patterned layer of thickness d (in-plane permittivity raster ε(ρ), e.g. Si
with air holes) between two identical homogeneous claddings ε_c (air for the suspended
membrane). H is expanded on the guided modes of the *effective* homogeneous slab (core
permittivity ε̄, by default the mean of ε(ρ)) at every in-plane wavevector g = k + G:

    H(r) = Σ_{G,α} c_{G,α} e^{i(k+G)·ρ} h_α(z; |k+G|) / √S,

with α running over the TE_m and TM_m slab modes kept (m = order), each normalised to
∫|h|² dz = 1, so the basis is orthonormal and the master equation ∇×(ε⁻¹∇×H) = (ω/c)² H
becomes the Hermitian eigenproblem

    M c = (ω/c)² c,   M_{μν} = ∫ ε⁻¹(r) (∇×H_μ)*·(∇×H_ν) d³r
                              = η(G_μ, G_ν) I_core(μ, ν) + δ_{G_μ G_ν} ε_c⁻¹ I_clad(μ, ν).

I_core, I_clad are the z-overlaps of the curls over the core and the two claddings; they are
analytic (products of cos/sin in the core, of decaying exponentials outside). η(G, G') is
the in-plane Fourier matrix of ε⁻¹ in the core:

- rule "inverse" (Ho et al.; default, as Andreani and legume): η = [ε(G − G')]⁻¹, the
  inverse of the Toeplitz matrix of ε over the basis G's;
- rule "direct": η = the Fourier coefficients of 1/ε(ρ).

The raster is read as pixels (piecewise constant), so its Fourier coefficients
ε̂(m) = DFT · sinc(m1/n1) sinc(m2/n2) · phase are exact for every m: the raster resolution
and the plane-wave cutoff are independent.

z-symmetry. With identical claddings σ_z separates the basis: parity "even" (TE-like, as
omc.optics_pwe: σ_z H = H, in-plane E even in z) keeps TE_m with m even and TM_m with m odd;
"odd" (TM-like) the others. n_te, n_tm are the number of TE and TM orders kept in the sector
(TE-like: TE0, TE2, … and TM1, TM3, …).

Radiation loss. A mode above the light line of some G (|k+G| < √ε_c ω/c) couples to the
radiative modes of the effective slab at that in-plane wavevector. To first order (Fermi's
golden rule on the Hermitian problem, Andreani & Gerace Eq. 21)

    Im (ω/c)² = −π Σ_{G, pol} ρ(|k+G|, ω) |⟨rad_{G,pol}|Θ|ψ⟩|²,   ρ = ε_c / (2 k_z),

with the radiative modes normalised to δ(k_z − k_z'). Since guided and radiative modes are
eigenmodes of the effective-slab operator at the same g, ⟨rad|Θ|ψ⟩ only involves
(η − ε̄⁻¹) in the core. Q = Re(ω²)/(−Im ω²) = ω/(2 |Im ω|). Both claddings are equivalent, so
half the loss goes up and half down; the σ_z partner radiative modes do not couple.

Mirror σ_y (lattices mapped onto themselves by y → −y: rectangular cells and the skewed
line-defect supercell; k_y = 0): `sym_y=±1` keeps the combinations |G,α⟩ ± σ_y|G,α⟩
(σ_y|G,TE⟩ = −|σG,TE⟩, σ_y|G,TM⟩ = +|σG,TM⟩), halving the problem; the scalar Toeplitz matrix
of the inverse rule splits the same way. ε̂ is symmetrised in Fourier space, so the raster need
not be exactly symmetric. sym_y = ±1 is meep's Mirror(Y, phase=±1): −1 holds the modes with
E_y even and E_x odd in y, which include the GPT07 line-defect band and cavity mode (checked
numerically on the field).

Time reversal. At k ≡ −k (Γ, e.g. every cavity supercell) K|G,α⟩ = ζ_α|−G,α⟩ (ζ = +1 TE,
−1 TM; a signed permutation of the σ_y combinations) makes M and the Toeplitz blocks real
symmetric in a K-invariant basis (`real_transform`): real LU/inverses, ~4× fewer flops and half
the memory (time_reversal="auto"). Not available at X of the skewed line-defect supercell,
whose a2 = (a/2, L_y) gives k = (π/a, 0) a phase i across a2, so −k ≢ k.

Eigen-solvers. "dense" (LAPACK, lowest n bands) or "shift-invert" (LU of M − σ, ARPACK) for a
few bands next to a target frequency, which is what the line-defect band (an interior band)
and cavity modes need. All jitted kernels take the basis as padded traced arrays, so they
compile once per size bucket (first call per bucket: ~5–20 s), not per k or per design step.

Gradients (JAX). M, the radiative couplings and the effective-slab profiles are jitted JAX
functions of the fill raster (ε̄ included: the slab dispersion is solved by bisection and a
final differentiable Newton step, which gives the implicit derivative). Band frequencies:
Hellmann–Feynman, dλ = cᴴ dM c, one reverse pass through the assembly. Q_rad: Im λ also
depends on the eigenvector, so one adjoint solve (M − λ) w = P ∂(Im λ)/∂c* (deflated GMRES
preconditioned by the shift-invert LU, or a dense solve), then one reverse pass for
∂/∂fill [Im λ + (∂Im λ/∂λ) cᴴMc − 2 Re wᴴ M c].

Fields. `curl_h_on_cell` evaluates e = ∇×H (∝ D, periodic part, as PWESolver.curl_h) on the
voxels of an omc.optics_pwe.SlabCell, for omc.coupling.optical_mode_from_field. In voxels cut
by the slab faces e_∥ is ⟨ε⟩_z E_∥ (E_∥ continuous), as the Kottke tensor of the cell expects.
`fields_on_cell` also returns GME's own E = η D. `curl_h` evaluates on any lattice grid.

Accuracy (validation/gme_bands.py, gme_cavity.py; numbers for the 220 nm Si membrane).
- Two separate errors. (i) In-plane truncation: the inverse rule converges slowly and from
  below; the GPT07 dagger features need |G| ≲ 6–8 · 2π/a (the blade shield alone ≳ 4).
  (ii) The effective-slab z-basis: with the in-plane part converged it puts TE-like bands
  1.3–2.5 % high on the blade cell and ~5 % high on the dagger line-defect band edges (fields in
  the air slots), shrinking only slowly with more guided orders or several ε̄ families. At
  moderate gmax the two partly cancel, which is why GME (and legume) land within ~0.3–1 % of
  PWE there (blade, inverse rule, g = 6–8 with TE0, TE2, TM1, TM3: 0.26–0.36 % mean; GPT07
  defect edge at X, g = 6, TE0: −0.1 %). Treat absolute frequencies as ±1 %, calibrate against
  PWE at checkpoints, and keep the basis settings fixed within an optimisation.
- Q_rad matches legume to 1–6 % at equal settings for Q ≲ 4e6 (blade bands, GPT07 cavity
  modes; the pixel raster against legume's polygons). Above ~1e6 Q depends on the raster
  (pixel staircase): use ≥ 96–144 px/a. The loss is first order in the coupling and ignores the
  radiative frequency shift.
- Boundary sensitivities of slot-like modes are too large: on the GPT07 defect band at X the
  response to a dagger dilation is 1.6× PWE's at g = 6 (FD-exact within GME), and g0 from GME
  fields is +18–23 % (moving boundary +35–50 %, photoelastic −10–20 %) for every basis tried.
  Fill gradients are exact derivatives of the GME eigenvalue, but of an approximate model.

Units: µm; k, G in rad/µm; ω written as k0 = ω/c (rad/µm); eigenvalue λ = k0²; frequencies in
THz.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

jax.config.update("jax_enable_x64", True)

C_UM_PER_PS = 299.792458
N_SI = 3.48
BIG = 1e6  # eigenvalue given to the padding basis functions


def thz(k0):
    return C_UM_PER_PS * np.asarray(k0) / (2 * np.pi)


def k0_of_thz(f):
    return 2 * np.pi * np.asarray(f) / C_UM_PER_PS


def light_line_thz(ks: np.ndarray, eps_clad: float = 1.0) -> np.ndarray:
    return C_UM_PER_PS * np.linalg.norm(ks, axis=1) / (2 * np.pi * np.sqrt(eps_clad))


def _sinc(x):
    """sin(x)/x, differentiable at 0."""
    return jnp.sinc(x / jnp.pi)


def _bucket(n: int, q: int) -> int:
    """Padded size: n rounded up to a multiple of q (so jitted kernels compile once per bucket)."""
    return max(q, int(np.ceil(n / q)) * q)


# --- time reversal --------------------------------------------------------------------------------


def real_transform(pi: np.ndarray, tau: np.ndarray) -> dict:
    """
    A unitary V whose columns are invariant under the antiunitary K e_a = τ_a e_{π(a)} (K² = 1):
    pairs a ≠ π(a) give (e_a + τ_a e_πa)/√2 and i (e_a − τ_a e_πa)/√2, fixed points e_a (τ = 1)
    or i e_a (τ = −1). A K-symmetric Hermitian A is real symmetric in this basis (V^H A V).
    Returned as column descriptors (I, J, al, be: v_p = al_p e_I + be_p e_J) and row descriptors
    (rp (n, 2), rc (n, 2): V[i, rp[i, s]] = rc[i, s]).
    """
    n = len(pi)
    if np.any(pi[pi] != np.arange(n)) or np.any(np.abs(tau * tau[pi] - 1) > 1e-12):
        raise ValueError("not an involution")
    I, J, al, be = np.zeros(n, int), np.zeros(n, int), np.zeros(n, complex), np.zeros(n, complex)
    rp, rc = np.zeros((n, 2), int), np.zeros((n, 2), complex)
    seen = np.zeros(n, bool)
    p = 0
    h = 1 / np.sqrt(2)
    for a in range(n):
        if seen[a]:
            continue
        b = pi[a]
        if b == a:
            c = 1.0 if tau[a] > 0 else 1j
            I[p], J[p], al[p], be[p] = a, a, c, 0.0
            rp[a], rc[a] = (p, p), (c, 0.0)
            p += 1
        else:
            t = tau[a]
            I[p : p + 2], J[p : p + 2] = a, b
            al[p : p + 2] = h, 1j * h
            be[p : p + 2] = t * h, -1j * t * h
            rp[a], rc[a] = (p, p + 1), (h, 1j * h)
            rp[b], rc[b] = (p, p + 1), (t * h, -1j * t * h)
            p += 2
        seen[a] = seen[b] = True
    return dict(I=I, J=J, al=al, be=be, rp=rp, rc=rc)


def to_real(A, V, xp=np):
    """Re(V^H A V) from the descriptors of real_transform (xp: numpy or jax.numpy)."""
    I, J, al, be = V["I"], V["J"], V["al"], V["be"]
    out = (xp.conj(al)[:, None] * al[None, :] * A[I[:, None], I[None, :]] + xp.conj(al)[:, None] * be[None, :] * A[I[:, None], J[None, :]]
           + xp.conj(be)[:, None] * al[None, :] * A[J[:, None], I[None, :]] + xp.conj(be)[:, None] * be[None, :] * A[J[:, None], J[None, :]])
    return xp.real(out)


def from_real(R, V, xp=np):
    """V R V^H (complex) from the row descriptors of real_transform."""
    rp, rc = V["rp"], V["rc"]
    out = 0
    for s_ in range(2):
        for t_ in range(2):
            out = out + rc[:, s_][:, None] * xp.conj(rc[:, t_])[None, :] * R[rp[:, s_][:, None], rp[:, t_][None, :]]
    return out


def apply_v(y, V, xp=np):
    """V y for vectors (rows of y indexed by the real basis)."""
    rp, rc = V["rp"], V["rc"]
    sh = (-1,) + (1,) * (y.ndim - 1)
    return rc[:, 0].reshape(sh) * y[rp[:, 0]] + rc[:, 1].reshape(sh) * y[rp[:, 1]]


def apply_vh(c, V, xp=np):
    """V^H c."""
    I, J, al, be = V["I"], V["J"], V["al"], V["be"]
    sh = (-1,) + (1,) * (c.ndim - 1)
    return xp.conj(al).reshape(sh) * c[I] + xp.conj(be).reshape(sh) * c[J]


# --- basis -------------------------------------------------------------------------------------


@dataclass
class Basis:
    """
    Plane waves and guided orders at one k. G = mG @ B (rows of B = b1, b2); kg = k + G.
    Basis function μ of the full list: plane wave gi[μ], polarisation pol[μ] (0 TE, 1 TM), slab
    order order[μ]. With sym_y the eigenproblem is on σ_y combinations of rep[i] (index into the
    full list) with its mirror image mirror_of[rep[i]]: weight w (1/√2, or 1/2 on the axis
    k_y + G_y = 0), s = sign of σ_y on the representative (−1 TE, +1 TM).

    `packed` holds the same data as padded JAX arrays (sizes rounded up, padding masked) for the
    jitted kernels; eigenvectors live on the padded problem space (`size`), padding entries 0.
    """

    k: np.ndarray
    mG: np.ndarray
    kg: np.ndarray
    gi: np.ndarray
    pol: np.ndarray
    order: np.ndarray
    eps_bar: float
    rep: np.ndarray | None = None
    mirror_of: np.ndarray | None = None
    w: np.ndarray | None = None
    s: np.ndarray | None = None
    sym_y: int = 0
    packed: dict | None = None
    tr: dict | None = None  # real_transform of the padded problem space (time reversal), if used

    @property
    def n_g(self) -> int:
        return len(self.mG)

    @property
    def n_full(self) -> int:
        return len(self.gi)

    @property
    def n_real(self) -> int:
        """Unpadded dimension of the eigenproblem."""
        return self.n_full if self.rep is None else len(self.rep)

    @property
    def size(self) -> int:
        """Padded dimension of the eigenproblem."""
        return int(self.packed["mask"].shape[0] if self.rep is None else self.packed["smask"].shape[0])

    def expand(self, c: np.ndarray) -> np.ndarray:
        """Coefficients on the padded full (unsymmetrised) basis from problem-space vectors (rows)."""
        if self.rep is None:
            return np.asarray(c)
        P = self.packed
        c = np.asarray(c)
        out = np.zeros((P["mask"].shape[0],) + c.shape[1:], dtype=complex)
        sh = (-1,) + (1,) * (c.ndim - 1)
        w = np.asarray(P["w"]).reshape(sh)
        np.add.at(out, np.asarray(P["rep"]), w * c)
        np.add.at(out, np.asarray(P["mir"]), w * self.sym_y * np.asarray(P["s"]).reshape(sh) * c)
        return out


@dataclass
class GMEResult:
    """Modes at one k: frequencies (THz), λ = k0², eigenvectors (basis.size × n), the basis and the (symmetrised) fill."""

    k: np.ndarray
    freqs: np.ndarray
    lam: np.ndarray
    vectors: np.ndarray
    basis: Basis
    fill: np.ndarray
    eps_bar: float
    timing: dict = field(default_factory=dict)
    _lu: object = None
    _sigma: float | None = None
    _M: np.ndarray | None = None
    _eta: tuple | None = None


class GMESolver:
    """
    Guided-mode expansion for one patterned slab layer on a 2D lattice (HexLattice, Lattice2D,
    or anything with `.vectors` rows a1, a2), with the in-plane solid fraction raster on a
    lattice-aligned (n1, n2) grid as geometry.solid_fraction makes it (pixel (i, j) spans
    [i, i+1]/n1 × [j, j+1]/n2 measured from the cell corner −(a1 + a2)/2).

    gmax: plane-wave cutoff in rad/µm (|k+G| ≤ gmax with truncate="k+G", |G| ≤ gmax with "G",
    as legume). A unit-cell cutoff of g·2π/a is `gmax=2*np.pi*g/a`. n_te, n_tm: number of TE
    and TM slab orders in the parity sector (TE-like: TE0, TE2, … and TM1, TM3, …).
    eps_eff: fixed effective-slab permittivity (default: the mean of ε over the raster, which is
    then also differentiated). sym_y: 0, or ±1 to keep one σ_y sector (meep's Mirror(Y) phase).
    time_reversal: "auto" uses real arithmetic whenever k ≡ −k.

        sol = GMESolver(lattice, fill.shape, gmax=2 * np.pi * 6 / a, n_te=1, n_tm=1, sym_y=-1)
        res = sol.solve(fill, k, n_bands=6, target_thz=188.0)  # bands nearest 188 THz
        rad = sol.radiation(res)                                # Q_rad per band
        f, df = sol.gradient(res, j, "freq")                    # ∂f/∂fill (THz per unit fill)
        lq, dlq = sol.gradient(res, j, "logQ")                  # ∂ ln Q_rad / ∂fill
        e = sol.curl_h_on_cell(res, j, cell)                    # → coupling.optical_mode_from_field
    """

    def __init__(
        self,
        lattice,
        shape: tuple[int, int],
        gmax: float,
        thickness: float = 0.22,
        eps_slab: float = N_SI**2,
        eps_clad: float = 1.0,
        n_te: int = 1,
        n_tm: int = 1,
        parity: str = "even",
        rule: str = "inverse",
        truncate: str = "k+G",
        eps_eff: float | None = None,
        sym_y: int = 0,
        time_reversal: bool | str = "auto",
        min_chi_d: float = 1e-3,
        pad: tuple[int, int] = (32, 32),
    ):
        self.lattice = lattice
        self.A = np.asarray(lattice.vectors, float)
        self.B = 2 * np.pi * np.linalg.inv(self.A).T
        self.shape = tuple(int(n) for n in shape)
        self.gmax = float(gmax)
        self.d = float(thickness)
        self.eps_slab = float(eps_slab)
        self.eps_clad = float(eps_clad)
        self.n_te, self.n_tm = int(n_te), int(n_tm)
        if parity not in ("even", "odd"):
            raise ValueError("parity must be 'even' (TE-like) or 'odd' (TM-like)")
        self.parity = parity
        self.sigma_z = 1 if parity == "even" else -1
        if rule not in ("inverse", "direct"):
            raise ValueError("rule must be 'inverse' or 'direct'")
        self.rule = rule
        if truncate not in ("k+G", "G"):
            raise ValueError("truncate must be 'k+G' or 'G'")
        self.truncate = truncate
        self.eps_eff = eps_eff
        self.sym_y = int(sym_y)
        self.time_reversal = time_reversal
        # σ_y on reciprocal indices: G = m B → σ_y G = m S B with S = B diag(1, −1) B⁻¹ (integer
        # when σ_y maps the lattice onto itself: rectangular cells, and the skewed line-defect
        # supercell a2 = (a/2, L_y) → a1 − a2)
        S = self.B @ np.diag([1.0, -1.0]) @ np.linalg.inv(self.B)
        self.mirror_matrix = np.round(S).astype(int) if np.allclose(S, np.round(S), atol=1e-9) else None
        if self.sym_y and self.mirror_matrix is None:
            raise ValueError("sym_y needs a lattice mapped onto itself by y → −y")
        self.min_chi_d = min_chi_d
        self.pad = pad
        self.area = abs(np.linalg.det(self.A))
        p0 = 0 if parity == "even" else 1
        self.te_orders = [p0 + 2 * i for i in range(self.n_te)]
        self.tm_orders = [(1 - p0) + 2 * i for i in range(self.n_tm)]
        self._Binv = np.linalg.inv(self.B)
        self._basis_cache: dict = {}
        self.last: GMEResult | None = None

    # --- permittivity ------------------------------------------------------------------------

    def eps_of(self, fill):
        return self.eps_clad + fill * (self.eps_slab - self.eps_clad)

    def eps_bar_of(self, fill) -> float:
        return float(self.eps_eff) if self.eps_eff is not None else float(np.mean(self.eps_of(np.asarray(fill))))


    # --- basis -------------------------------------------------------------------------------

    def _plane_waves(self, k: np.ndarray):
        kmax = self.gmax + np.linalg.norm(k)
        n_max = [int(np.ceil(kmax * np.linalg.norm(self._Binv[:, i]))) + 1 for i in range(2)]
        m1, m2 = np.meshgrid(np.arange(-n_max[0], n_max[0] + 1), np.arange(-n_max[1], n_max[1] + 1), indexing="ij")
        mG = np.stack([m1.ravel(), m2.ravel()], axis=1)
        G = mG @ self.B
        kg = G + k[None, :]
        norm = np.linalg.norm(kg if self.truncate == "k+G" else G, axis=1)
        keep = norm <= self.gmax * (1 + 1e-12)
        mG, kg = mG[keep], kg[keep]
        order = np.lexsort((mG[:, 1], mG[:, 0], np.round(np.linalg.norm(kg, axis=1), 9)))
        return mG[order], kg[order]

    def _exists(self, g: np.ndarray, m: int, eps_bar: float) -> np.ndarray:
        """Order m guided at in-plane |g| in the effective slab (g d √((ε̄ − ε_c)/ε_c) > mπ), not too close to cutoff."""
        r = self.eps_clad / eps_bar
        lo = m * np.pi / self.d
        chi_lo2 = g**2 * (1 - r) - r * lo**2
        return (chi_lo2 > (self.min_chi_d / self.d) ** 2) & (g > 1e-9)

    def basis(self, k, eps_bar: float) -> Basis:
        k = np.asarray(k, float)
        key = (tuple(np.round(k, 12)), round(float(eps_bar), 8))
        if key in self._basis_cache:
            return self._basis_cache[key]
        mG, kg = self._plane_waves(k)
        g = np.linalg.norm(kg, axis=1)
        gi, pol, order = [], [], []
        for p, orders in ((0, self.te_orders), (1, self.tm_orders)):
            for m in orders:
                ok = np.flatnonzero(self._exists(g, m, eps_bar))
                gi.append(ok)
                pol.append(np.full(len(ok), p))
                order.append(np.full(len(ok), m))
        gi, pol, order = (np.concatenate(v).astype(int) for v in (gi, pol, order))
        b = Basis(k, mG, kg, gi, pol, order, float(eps_bar))
        if self.sym_y:
            self._symmetrise_basis(b)
        self._pack(b)
        self._time_reversal(b)
        if len(self._basis_cache) > 64:
            self._basis_cache.clear()
        self._basis_cache[key] = b
        return b

    def _symmetrise_basis(self, b: Basis) -> None:
        if abs(b.k[1]) > 1e-12:
            raise ValueError("sym_y needs k_y = 0")
        idx = {(int(m1), int(m2)): i for i, (m1, m2) in enumerate(b.mG)}
        mm = b.mG @ self.mirror_matrix
        gmir = np.array([idx[(int(m1), int(m2))] for m1, m2 in mm])
        fidx = {(int(g), int(p), int(o)): i for i, (g, p, o) in enumerate(zip(b.gi, b.pol, b.order))}
        mirror_of = np.array([fidx[(int(gmir[g]), int(p), int(o))] for g, p, o in zip(b.gi, b.pol, b.order)])
        s = np.where(b.pol == 0, -1.0, 1.0)
        rep, w = [], []
        for i in range(len(b.gi)):
            if mirror_of[i] == i:
                if self.sym_y * s[i] > 0:
                    rep.append(i)
                    w.append(0.5)
            elif i < mirror_of[i]:
                rep.append(i)
                w.append(1 / np.sqrt(2))
        rep = np.array(rep, int)
        b.rep, b.mirror_of, b.w, b.s, b.sym_y = rep, mirror_of, np.array(w), s[rep], self.sym_y

    def _kappa(self, b: Basis) -> np.ndarray | None:
        """Index of −(k+G) for every plane wave, if the list is closed under it (k ≡ −k), else None."""
        key = {tuple(np.round(v, 9)): i for i, v in enumerate(b.kg)}
        out = [key.get(tuple(np.round(-v, 9) + 0.0)) for v in b.kg]
        return None if any(o is None for o in out) else np.array(out)

    def _time_reversal(self, b: Basis) -> None:
        """Real arithmetic at time-reversal-invariant k: K|G,α⟩ = ζ_α|−G,α⟩ (ζ = +1 TE, −1 TM)."""
        if self.time_reversal is False:
            return
        kap = self._kappa(b)
        if kap is None:
            if self.time_reversal is True:
                raise ValueError("time_reversal=True needs k ≡ −k and a plane-wave set closed under G → −2k − G")
            return
        P = {k_: np.asarray(v) for k_, v in b.packed.items()}
        fidx = {(int(g), int(p), int(o)): i for i, (g, p, o) in enumerate(zip(b.gi, b.pol, b.order))}
        kf = np.array([fidx[(int(kap[g]), int(p), int(o))] for g, p, o in zip(b.gi, b.pol, b.order)])
        zeta = np.where(b.pol == 0, 1.0, -1.0)
        if b.rep is None:
            N = len(P["mask"])
            pi = np.r_[kf, np.arange(len(kf), N)]
            tau = np.r_[zeta, np.ones(N - len(kf))]
        else:
            ns, NS = len(b.rep), len(P["smask"])
            where = {int(r): a for a, r in enumerate(b.rep)}
            pi, tau = np.arange(NS), np.ones(NS)
            for a, r in enumerate(b.rep):
                t = kf[r]
                if t in where:
                    pi[a], tau[a] = where[t], zeta[r]
                else:
                    a2 = where[int(b.mirror_of[t])]
                    pi[a], tau[a] = a2, zeta[r] * self.sym_y * b.s[a]
        b.tr = real_transform(pi, tau)
        # scalar plane waves (inverse-rule blocks): K e_G = e_{−G}
        NG = len(P["gmask"])
        kapp = np.r_[kap, np.arange(len(kap), NG)]
        extra = {}
        if b.rep is None:
            extra["full"] = real_transform(kapp, np.ones(NG))
        else:
            for tag in ("p", "m"):
                reps = P[f"{tag}_rep"]
                nb = len(reps)
                real = P[f"{tag}_mask"] > 0
                where = {int(r): a for a, r in enumerate(reps) if real[a]}
                pi, tau = np.arange(nb), np.ones(nb)
                for a in np.flatnonzero(real):
                    t = int(kapp[reps[a]])
                    if t in where:
                        pi[a] = where[t]
                    else:
                        pi[a] = where[int(P["gmir"][t])]
                        tau[a] = 1.0 if tag == "p" else -1.0
                extra[tag] = real_transform(pi, tau)
        for tag, V in extra.items():
            for k_, v in V.items():
                b.packed[f"tr_{tag}_{k_}"] = jnp.asarray(v)

    def _pack(self, b: Basis) -> None:
        qg, qn = self.pad
        ng, n = b.n_g, b.n_full
        NG, N = _bucket(ng, qg), _bucket(n, qn)
        mG = np.concatenate([b.mG, np.repeat(b.mG[:1], NG - ng, axis=0)])
        gmask = np.r_[np.ones(ng), np.zeros(NG - ng)]
        pad_i = np.zeros(N - n, int)
        gi = np.r_[b.gi, b.gi[pad_i]]
        pol = np.r_[b.pol, b.pol[pad_i]]
        order = np.r_[b.order, b.order[pad_i]]
        mask = np.r_[np.ones(n), np.zeros(N - n)]
        P = dict(mG=mG, gmask=gmask, gi=gi, pol=pol, order=order, mask=mask, kg=b.kg)
        if b.rep is not None:
            ns = len(b.rep)
            NS = _bucket(ns, qn)
            rep = np.r_[b.rep, np.zeros(NS - ns, int)]
            P.update(rep=rep, mir=b.mirror_of[rep], w=np.r_[b.w, np.zeros(NS - ns)], s=np.r_[b.s, np.ones(NS - ns)],
                     smask=np.r_[np.ones(ns), np.zeros(NS - ns)])
            P.update(self._scalar_blocks(b, NG, qg))
        b.packed = {k_: jnp.asarray(v) for k_, v in P.items()}

    def _scalar_blocks(self, b: Basis, NG: int, q: int) -> dict:
        """
        σ_y-even/odd scalar plane-wave combinations for the block inverse of T: + block reps
        (pairs i < σi and fixed points, weight 1/√2 or ½), − block reps (pairs, 1/√2); and, per
        G index, the coefficients β± and block columns c± with |G_i⟩ = Σ± β±(i) e±_{c±(i)}.
        Padding plane waves are fixed points (+ block, identity rows).
        """
        ng = b.n_g
        idx = {(int(m1), int(m2)): i for i, (m1, m2) in enumerate(b.mG)}
        gmir = np.array([idx[(int(m1), int(m2))] for m1, m2 in b.mG @ self.mirror_matrix])
        gmir = np.r_[gmir, np.arange(ng, NG)]
        i = np.arange(NG)
        fixed, pair = gmir == i, i < gmir
        p_rep = np.flatnonzero(fixed | pair)
        m_rep = np.flatnonzero(pair)
        NP, NM = _bucket(len(p_rep), q), _bucket(max(len(m_rep), 1), q)
        pw = np.where(fixed[p_rep], 0.5, 1 / np.sqrt(2))
        bp = np.where(fixed, 1.0, 1 / np.sqrt(2))
        cp = np.zeros(NG, int)
        cp[p_rep] = np.arange(len(p_rep))
        cp[gmir[p_rep]] = np.arange(len(p_rep))
        bm = np.where(fixed, 0.0, np.where(pair, 1 / np.sqrt(2), -1 / np.sqrt(2)))
        cm = np.zeros(NG, int)
        cm[m_rep] = np.arange(len(m_rep))
        cm[gmir[m_rep]] = np.arange(len(m_rep))
        real_g = np.r_[np.ones(ng), np.zeros(NG - ng)]
        pad = lambda a, n, v=0: np.r_[a, np.full(n - len(a), v)]
        return dict(gmir=gmir, p_rep=pad(p_rep, NP), p_w=pad(pw, NP), p_mask=pad(real_g[p_rep], NP),
                    m_rep=pad(m_rep, NM), m_w=pad(np.full(len(m_rep), 1 / np.sqrt(2)), NM), m_mask=pad(real_g[m_rep], NM),
                    bp=bp, cp=cp, bm=bm, cm=cm)

    # --- JAX pieces (pure functions of the fill and the packed basis) --------------------------

    def _ft(self, fill, dm1, dm2, inverse_eps: bool):
        """
        Exact Fourier coefficients of the pixel raster of ε (or 1/ε) at integer index arrays
        (dm1, dm2): DFT (periodic in m) × pixel sinc × phase of the pixel centres
        s = (i + ½)/n − ½. With sym_y, the σ_y-symmetric part ½ (ε̂(m) + ε̂(m S)).
        """
        n1, n2 = self.shape
        e = self.eps_of(fill)
        if inverse_eps:
            e = 1.0 / e
        D = jnp.fft.fft2(e) / (n1 * n2)

        def at(m1, m2):
            ph = jnp.exp(-2j * jnp.pi * (m1 * (0.5 / n1 - 0.5) + m2 * (0.5 / n2 - 0.5)))
            return D[m1 % n1, m2 % n2] * ph * jnp.sinc(m1 / n1) * jnp.sinc(m2 / n2)

        if self.sym_y:
            S = self.mirror_matrix
            return 0.5 * (at(dm1, dm2) + at(dm1 * S[0, 0] + dm2 * S[1, 0], dm1 * S[0, 1] + dm2 * S[1, 1]))
        return at(dm1, dm2)

    def _eta_blocks(self, fill, P):
        """
        The arrays that define η: the inverse of T = [ε(G − G')] (padding: identity), with sym_y as
        two σ_y blocks T± (a quarter of the work, the full η is never formed); for the direct
        rule just the fill.
        """
        mG, gm = P["mG"], P["gmask"]
        if self.rule == "direct":
            return (fill,)

        def T_of(r, c):
            return self._ft(fill, mG[r][:, None, 0] - mG[c][None, :, 0], mG[r][:, None, 1] - mG[c][None, :, 1], False)

        def inverse(Tm, tag):
            Tm = 0.5 * (Tm + jnp.conj(Tm.T))
            if f"tr_{tag}_I" not in P:
                return jnp.linalg.inv(Tm)
            V = {k_: P[f"tr_{tag}_{k_}"] for k_ in ("I", "J", "al", "be", "rp", "rc")}
            R = to_real(Tm, V, jnp)
            return from_real(jnp.linalg.inv(0.5 * (R + R.T)), V, jnp)

        if not self.sym_y:
            idx = jnp.arange(mG.shape[0])
            T = jnp.where(gm[:, None] * gm[None, :] > 0, T_of(idx, idx), 0.0) + jnp.diag(1.0 - gm)
            return (inverse(T, "full"),)
        blocks = []
        for tag, sign in (("p", 1.0), ("m", -1.0)):
            r, w, msk = P[f"{tag}_rep"], P[f"{tag}_w"], P[f"{tag}_mask"]
            Tb = 2 * (w[:, None] * w[None, :]) * (T_of(r, r) + sign * T_of(r, P["gmir"][r]))
            Tb = jnp.where(msk[:, None] * msk[None, :] > 0, Tb, 0.0) + jnp.diag(1.0 - msk)
            blocks.append(inverse(Tb, tag))
        return tuple(blocks)

    def _eta_fn(self, blocks, P):
        """η(G_gr, G_gc) for index arrays into the padded plane-wave list, from `_eta_blocks`."""
        mG = P["mG"]
        if self.rule == "direct":
            fill = blocks[0]
            return lambda gr, gc: self._ft(fill, mG[gr, 0] - mG[gc, 0], mG[gr, 1] - mG[gc, 1], True)
        if not self.sym_y:
            X = blocks[0]
            return lambda gr, gc: X[gr, gc]
        Xp, Xm = blocks
        bp, cp, bm, cm = P["bp"], P["cp"], P["bm"], P["cm"]
        return lambda gr, gc: (bp[gr] * bp[gc] * Xp[cp[gr], cp[gc]] + bm[gr] * bm[gc] * Xm[cm[gr], cm[gc]])

    def _eta(self, fill, P):
        return self._eta_fn(self._eta_blocks(fill, P), P)

    def _eps_bar_j(self, fill):
        if self.eps_eff is not None:
            return jnp.asarray(float(self.eps_eff))
        return jnp.mean(self.eps_of(fill))

    def _guided_q(self, g, m, rho, eps_bar):
        """Core wavenumber q of order m at in-plane g: F(q) = q sin θ − ρ χ cos θ = 0, θ = qd/2 − mπ/2 (bisection + one differentiable Newton step)."""
        d, ec = self.d, self.eps_clad

        def chi_of(q, eb):
            r = ec / eb
            return jnp.sqrt(jnp.maximum(g**2 * (1 - r) - r * q**2, 0.0))

        def F(q, eb, rh):
            th = q * d / 2 - m * jnp.pi / 2
            return q * jnp.sin(th) - rh * chi_of(q, eb) * jnp.cos(th)

        eb0 = jax.lax.stop_gradient(eps_bar)
        rh0 = jax.lax.stop_gradient(rho)
        qmax = g * jnp.sqrt((eb0 - ec) / ec)
        lo = m * jnp.pi / d + 0.0 * g
        hi = jnp.minimum((m + 1) * jnp.pi / d, qmax)

        def body(_, lh):
            lo, hi = lh
            mid = 0.5 * (lo + hi)
            neg = F(mid, eb0, rh0) < 0
            return jnp.where(neg, mid, lo), jnp.where(neg, hi, mid)

        lo, hi = jax.lax.fori_loop(0, 60, body, (lo, hi))
        q0 = jax.lax.stop_gradient(0.5 * (lo + hi))
        r = ec / eb0
        chi0 = jnp.maximum(chi_of(q0, eb0), 1e-12)
        th = q0 * d / 2 - m * jnp.pi / 2
        dF = jnp.sin(th) + (q0 * d / 2) * jnp.cos(th) + rh0 * r * q0 * jnp.cos(th) / chi0 + rh0 * chi0 * (d / 2) * jnp.sin(th)
        return q0 - F(q0, eps_bar, rho) / jax.lax.stop_gradient(dF)

    @staticmethod
    def _units(kg):
        g = jnp.sqrt(kg[:, 0] ** 2 + kg[:, 1] ** 2)
        gs = jnp.where(g > 1e-12, g, 1.0)
        ghat = jnp.stack([jnp.where(g > 1e-12, kg[:, 0] / gs, 1.0), jnp.where(g > 1e-12, kg[:, 1] / gs, 0.0)], axis=1)
        return g, ghat, jnp.stack([-ghat[:, 1], ghat[:, 0]], axis=1)

    def _profiles(self, fill, P):
        """
        Curl amplitudes of the (padded, full) guided basis: in-plane unit vector u (N, 2); core
        amplitudes a_c (in-plane, times F_ip(qz) = cos (TE-like) or sin), b_c (z, times F_z, the
        other one); cladding amplitudes at the faces a_l, b_l (times e^{−χ(|z|−d/2)}, with the
        component's parity); q, χ, k0.
        """
        d, ec, sz = self.d, self.eps_clad, self.sigma_z
        eps_bar = self._eps_bar_j(fill)
        kg = P["kg"][P["gi"]]
        pol, order = P["pol"], P["order"]
        g, ghat, ehat = self._units(kg)
        te = pol == 0
        rho = jnp.where(te, 1.0, eps_bar / ec)
        q = self._guided_q(g, order, rho, eps_bar)
        k0 = jnp.sqrt((q**2 + g**2) / eps_bar)
        chi = jnp.sqrt(jnp.maximum(g**2 - ec * k0**2, 1e-30))
        even_T = order % 2 == 0
        x = q * d / 2
        Tf = jnp.where(even_T, jnp.cos(x), jnp.sin(x))
        S = d / 2 * _sinc(q * d)
        intT2 = jnp.where(even_T, d / 2 + S, d / 2 - S)
        intTp2 = jnp.where(even_T, d / 2 - S, d / 2 + S)
        nte = (q**2 * intTp2 + g**2 * intT2 + (chi**2 + g**2) * Tf**2 / chi) / k0**2
        ntm = intT2 + Tf**2 / chi
        N = 1 / jnp.sqrt(jnp.where(te, nte, ntm))
        a_c = jnp.where(te, -1j * k0 * eps_bar * N, -sz * N * q + 0j)
        a_l = jnp.where(te, -1j * k0 * ec * N * Tf, chi * N * Tf + 0j)
        b_c = jnp.where(te, 0j, 1j * g * N)
        b_l = jnp.where(te, 0j, 1j * g * N * Tf)
        u = jnp.where(te[:, None], ehat, ghat)
        return dict(u=u, a_c=a_c, a_l=a_l, b_c=b_c, b_l=b_l, q=q, chi=chi, k0=k0)

    def _rad_profiles(self, kg, k0, eps_bar):
        """Radiative modes (TE rows then TM rows) of the effective slab at k0 and in-plane kg, normalised to δ(k_z − k_z'); and the DOS ρ."""
        d, ec, sz = self.d, self.eps_clad, self.sigma_z
        n = kg.shape[0]
        g, ghat, ehat = self._units(jnp.concatenate([kg, kg], axis=0))
        te = jnp.concatenate([jnp.ones(n, bool), jnp.zeros(n, bool)])
        q = jnp.sqrt(jnp.maximum(eps_bar * k0**2 - g**2, 1e-30))
        kz = jnp.sqrt(jnp.maximum(ec * k0**2 - g**2, 1e-30))
        cosT = jnp.where(te, sz > 0, sz < 0)  # T = cos: TE in the even sector, TM in the odd one
        x = q * d / 2
        Tf = jnp.where(cosT, jnp.cos(x), jnp.sin(x))
        Tp = jnp.where(cosT, jnp.sin(x), jnp.cos(x))
        Nte = 1 / jnp.sqrt(jnp.pi * ec * (Tf**2 + (q / kz) ** 2 * Tp**2))
        Ntm = 1 / jnp.sqrt(jnp.pi * (Tf**2 + (ec * q / (eps_bar * kz)) ** 2 * Tp**2))
        N = jnp.where(te, Nte, Ntm)
        a_c = jnp.where(te, -1j * k0 * eps_bar * N, -sz * N * q + 0j)
        b_c = jnp.where(te, 0j, 1j * g * N)
        u = jnp.where(te[:, None], ehat, ghat)
        return dict(u=u, a_c=a_c, b_c=b_c, q=q), ec / (2 * kz)

    def _core_overlap(self, Pm, Qm):
        """∫_core (curl_μ)*·(curl_ν) dz (without η), descriptor sets Pm (rows), Qm (cols)."""
        d, sz = self.d, self.sigma_z
        qm, qn = Pm["q"][:, None], Qm["q"][None, :]
        sm = _sinc((qm - qn) * d / 2)
        sp = _sinc((qm + qn) * d / 2)
        uu = Pm["u"] @ Qm["u"].T
        return (uu * jnp.conj(Pm["a_c"])[:, None] * Qm["a_c"][None, :] * (d / 2) * (sm + sz * sp)
                + jnp.conj(Pm["b_c"])[:, None] * Qm["b_c"][None, :] * (d / 2) * (sm - sz * sp))

    def _clad_overlap(self, Pm, Qm):
        """Both claddings, 2 a*a'/(χ + χ') (used for equal G only)."""
        cs = 2 / (Pm["chi"][:, None] + Qm["chi"][None, :])
        uu = Pm["u"] @ Qm["u"].T
        return (uu * jnp.conj(Pm["a_l"])[:, None] * Qm["a_l"][None, :] + jnp.conj(Pm["b_l"])[:, None] * Qm["b_l"][None, :]) * cs

    def _block(self, eta, prof, gi, rows, cols):
        Pm = {k_: v[rows] for k_, v in prof.items()}
        Qm = {k_: v[cols] for k_, v in prof.items()}
        gr, gc = gi[rows], gi[cols]
        M = eta(gr[:, None], gc[None, :]) * self._core_overlap(Pm, Qm)
        return M + jnp.where(gr[:, None] == gc[None, :], self._clad_overlap(Pm, Qm) / self.eps_clad, 0.0)

    def _matrix(self, fill, P, blocks=None):
        """The Hermitian GME matrix on the padded problem space (symmetrised with sym_y)."""
        eta = self._eta_fn(self._eta_blocks(fill, P) if blocks is None else blocks, P)
        prof = self._profiles(fill, P)
        gi = P["gi"]
        if self.sym_y == 0:
            idx = jnp.arange(gi.shape[0])
            mask = P["mask"]
            M = self._block(eta, prof, gi, idx, idx) * (mask[:, None] * mask[None, :])
        else:
            rep, mir, w, mask = P["rep"], P["mir"], P["w"], P["smask"]
            M = 2 * (w[:, None] * w[None, :]) * (self._block(eta, prof, gi, rep, rep)
                                                 + self.sym_y * P["s"][None, :] * self._block(eta, prof, gi, rep, mir))
            M = M * (mask[:, None] * mask[None, :])
        M = M + jnp.diag(BIG * (1.0 - mask))
        return 0.5 * (M + jnp.conj(M.T))

    def _expand_j(self, P, c):
        if self.sym_y == 0:
            return c * P["mask"]
        out = jnp.zeros(P["gi"].shape[0], dtype=complex)
        out = out.at[P["rep"]].add(P["w"] * c)
        return out.at[P["mir"]].add(P["w"] * self.sym_y * P["s"] * c)

    def _im_lam_multi(self, fill, P, lams, C, ch, cmask, blocks=None):
        """Im λ of several modes (columns of C, eigenvalues lams) sharing η, profiles and the coupling rows."""
        eta = self._eta_fn(self._eta_blocks(fill, P) if blocks is None else blocks, P)
        prof = self._profiles(fill, P)
        eps_bar = self._eps_bar_j(fill)
        gr = jnp.concatenate([ch, ch])
        gi = P["gi"]
        coup = eta(gr[:, None], gi[None, :]) - jnp.where(gr[:, None] == gi[None, :], 1.0, 0.0) / eps_bar
        g2 = jnp.sum(P["kg"][gr] ** 2, axis=1)
        ims, terms = [], []
        for j in range(C.shape[1]):
            k0 = jnp.sqrt(lams[j])
            R, dos = self._rad_profiles(P["kg"][ch], k0, eps_bar)
            cm = jnp.concatenate([cmask, cmask]) * (g2 < self.eps_clad * lams[j])
            V = (coup * self._core_overlap(R, prof)) @ self._expand_j(P, C[:, j])
            t = jnp.where(cm > 0, -jnp.pi * dos * jnp.abs(V) ** 2, 0.0)
            ims.append(jnp.sum(t))
            terms.append(t)
        return jnp.stack(ims), jnp.stack(terms)

    def _im_lam(self, fill, P, lam, c, ch, cmask, eta=None, prof=None):
        """Im λ (≤ 0) and its terms per channel (TE rows, then TM), for problem-space vector c at real λ."""
        if eta is None:
            eta = self._eta(fill, P)
        if prof is None:
            prof = self._profiles(fill, P)
        eps_bar = self._eps_bar_j(fill)
        k0 = jnp.sqrt(lam)
        R, dos = self._rad_profiles(P["kg"][ch], k0, eps_bar)
        gr = jnp.concatenate([ch, ch])
        cm = jnp.concatenate([cmask, cmask])
        gi = P["gi"]
        coup = eta(gr[:, None], gi[None, :]) - jnp.where(gr[:, None] == gi[None, :], 1.0, 0.0) / eps_bar
        V = (coup * self._core_overlap(R, prof)) @ self._expand_j(P, c)
        g2 = jnp.sum(P["kg"][gr] ** 2, axis=1)
        terms = jnp.where((cm > 0) & (g2 < self.eps_clad * lam), -jnp.pi * dos * jnp.abs(V) ** 2, 0.0)
        return jnp.sum(terms), terms

    # --- solve -------------------------------------------------------------------------------

    def prepare(self, fill, k) -> tuple[np.ndarray, Basis]:
        """(fill, basis at k for the fill's ε̄)."""
        fill = np.asarray(fill, float)
        return fill, self.basis(np.asarray(k, float), self.eps_bar_of(fill))

    def matrix(self, fill, k) -> np.ndarray:
        fill, b = self.prepare(fill, k)
        return np.asarray(_k_matrix(_Static(self), jnp.asarray(fill), b.packed))

    def solve(self, fill, k, n_bands: int = 8, target_thz: float | None = None, method: str | None = None,
              keep_lu: bool = True) -> GMEResult:
        """
        Bands at in-plane k (rad/µm). Without target_thz: the lowest n_bands (dense eigh). With
        target_thz: the n_bands eigenvalues nearest the target, by shift-invert (dense LU of
        M − σ and ARPACK), or by a full dense eigh if method="dense" or the problem is small.
        """
        t0 = time.perf_counter()
        fill, b = self.prepare(fill, k)
        blocks = _k_eta_blocks(_Static(self), jnp.asarray(fill), b.packed)
        M = np.asarray(_k_matrix_from(_Static(self), jnp.asarray(fill), b.packed, blocks))
        t1 = time.perf_counter()
        n = M.shape[0]
        nb = min(n_bands, b.n_real)
        lu = sigma = None
        A = M if b.tr is None else to_real(M, b.tr)  # real symmetric with time reversal
        if target_thz is None:
            lam, X = sla.eigh(A, subset_by_index=[0, nb - 1], driver="evr")
        elif method == "dense" or b.n_real <= 600:
            sigma = float(k0_of_thz(target_thz)) ** 2
            lam_all, X_all = sla.eigh(A)
            sel = np.sort(np.argsort(np.abs(lam_all - sigma))[:nb])
            lam, X = lam_all[sel], X_all[:, sel]
        else:
            sigma = float(k0_of_thz(target_thz)) ** 2
            lu = sla.lu_factor(A - sigma * np.eye(n), check_finite=False)
            op = spla.LinearOperator((n, n), matvec=lambda v: sla.lu_solve(lu, v, check_finite=False), dtype=A.dtype)
            mu, X = spla.eigsh(op, k=nb, which="LM", tol=1e-13, ncv=max(2 * nb + 1, 24))
            H = X.conj().T @ (A @ X)  # Rayleigh–Ritz on the converged subspace
            lam, V = np.linalg.eigh(0.5 * (H + H.conj().T))
            X = X @ V
        if b.tr is not None:
            X = apply_v(X.astype(complex), b.tr)
        t2 = time.perf_counter()
        res = GMEResult(np.asarray(k, float), thz(np.sqrt(np.clip(lam, 0, None))), lam, X, b, fill, b.eps_bar,
                        dict(assemble=t1 - t0, eig=t2 - t1, size=b.n_real, n_g=b.n_g), lu if keep_lu else None, sigma, M)
        res._eta = blocks if keep_lu else None
        self.last = res
        return res

    def bands(self, fill, ks, n_bands: int = 8, **kw) -> np.ndarray:
        return np.array([self.solve(fill, k, n_bands, keep_lu=False, **kw).freqs for k in ks])

    # --- radiation ---------------------------------------------------------------------------

    def _channels(self, b: Basis, lam: float):
        g = np.linalg.norm(b.kg, axis=1)
        ch = np.flatnonzero(g < np.sqrt(self.eps_clad * lam) * (1 - 1e-12))
        if len(ch) == 0:
            return None, None
        nc = _bucket(len(ch), 8)
        return jnp.asarray(np.r_[ch, np.full(nc - len(ch), ch[0])]), jnp.asarray(np.r_[np.ones(len(ch)), np.zeros(nc - len(ch))])

    def radiation(self, res: GMEResult, band: int | None = None):
        """
        First-order radiation loss of one band (default: all bands of the solve): dict with Im λ,
        Im f (THz, negative), Q_rad and the loss per channel (rows TE, TM; columns `channels`,
        indices into basis.kg). η and the profiles are computed once for all bands.
        """
        bands = list(range(len(res.lam))) if band is None else [band]
        b = res.basis
        lams = np.asarray(res.lam)[bands]
        ch, cm = self._channels(b, float(lams.max()))
        out = []
        if ch is None:
            out = [dict(im_lam=0.0, im_thz=0.0, Q=np.inf, channels=np.array([], int), terms=None) for _ in bands]
        else:
            blocks = res._eta if res._eta is not None else _k_eta_blocks(_Static(self), jnp.asarray(res.fill), b.packed)
            ims, terms = _k_im_multi(_Static(self), jnp.asarray(res.fill), b.packed, jnp.asarray(lams),
                                     jnp.asarray(res.vectors[:, bands]), ch, cm, blocks)
            nch = int(np.asarray(cm).sum())
            for i, lam in enumerate(lams):
                im = float(ims[i])
                out.append(dict(im_lam=im, im_thz=float(thz(im / (2 * np.sqrt(lam)))), Q=lam / -im if im < 0 else np.inf,
                                channels=np.asarray(ch)[:nch], terms=np.asarray(terms[i]).reshape(2, -1)[:, :nch]))
        return out[0] if band is not None else out

    # --- gradients ---------------------------------------------------------------------------

    def gradient(self, res: GMEResult, band: int, quantity: str = "freq", adjoint_tol: float = 1e-10):
        """
        (value, ∂value/∂fill (n1, n2)) for one band of a solve; quantity "freq" (THz), "Q"
        (Q_rad) or "logQ" (ln Q_rad). Non-degenerate bands only. With sym_y the solver sees the
        σ_y-symmetric part of ε (symmetrised in Fourier space), and the gradient is that of the
        value of the symmetrised structure, w.r.t. the raster as given.
        """
        b, P = res.basis, res.basis.packed
        cfg = _Static(self)
        fill = jnp.asarray(res.fill)
        lam = float(res.lam[band])
        c = res.vectors[:, band]
        others = np.delete(res.lam, band)
        if len(others) and np.min(np.abs(others - lam)) < 1e-7 * lam:
            warnings.warn(f"band {band} is (near-)degenerate; its gradient is not defined")
        if quantity == "freq":
            g_lam = np.asarray(_k_hf_grad(cfg, fill, P, jnp.asarray(c)))
            return float(thz(np.sqrt(lam))), g_lam * C_UM_PER_PS / (4 * np.pi * np.sqrt(lam))
        ch, cm = self._channels(b, lam)
        if ch is None:
            raise ValueError("no radiation channel: the mode is below the light line (Q_rad = ∞)")
        im, d_lam, g_c = _k_im_partials(cfg, fill, P, lam, jnp.asarray(c), ch, cm)
        im, d_lam = float(im), float(d_lam)
        g_c = np.conj(np.asarray(g_c)) / 2  # jax.grad of a real f of complex c is 2 (∂f/∂c*)*
        r = g_c - c * np.vdot(c, g_c)
        w = self._pinv_solve(res, band, r, adjoint_tol)
        # d ln Q = dλ/λ − d(Im λ)/Im λ, one reverse pass:
        # dλ = Re cᴴ dM c;  d Im λ = ∂_fill Im λ + (∂Im λ/∂λ) dλ − 2 Re wᴴ dM c
        dlogQ = np.asarray(_k_q_grad(cfg, fill, P, lam, jnp.asarray(c), jnp.asarray(w), ch, cm, d_lam, 1.0 / lam, 1.0 / im))
        Q = lam / -im
        if quantity == "Q":
            return Q, Q * dlogQ
        if quantity == "logQ":
            return float(np.log(Q)), dlogQ
        raise ValueError(quantity)

    def _pinv_solve(self, res: GMEResult, band: int, r: np.ndarray, tol: float) -> np.ndarray:
        """w ⊥ c with (M − λ) w = r (r ⊥ c): GMRES preconditioned by the shift-invert LU, else a dense solve of M − λ + λ c cᴴ."""
        M = res._M
        lam = float(res.lam[band])
        c = res.vectors[:, band]
        n = M.shape[0]

        def dense():
            return sla.solve(M - lam * np.eye(n) + lam * np.outer(c, c.conj()), r, assume_a="her")

        if res._lu is None:
            w = dense()
        else:
            lu, tr = res._lu, res.basis.tr

            def defl(y):
                return y - c * np.vdot(c, y)

            def lu_apply(y):
                if tr is None:
                    return sla.lu_solve(lu, y, check_finite=False)
                z = apply_vh(y, tr)
                z = sla.lu_solve(lu, z.real, check_finite=False) + 1j * sla.lu_solve(lu, z.imag, check_finite=False)
                return apply_v(z, tr)

            op = spla.LinearOperator((n, n), matvec=lambda y: defl(M @ defl(y) - lam * defl(y)), dtype=complex)
            pre = spla.LinearOperator((n, n), matvec=lambda y: defl(lu_apply(defl(y))), dtype=complex)
            w, info = spla.gmres(op, r, x0=pre.matvec(r), M=pre, rtol=tol, atol=0.0, restart=80, maxiter=10)
            if info != 0:
                warnings.warn(f"GME adjoint GMRES did not converge (info {info}); dense solve instead")
                w = dense()
        return w - c * np.vdot(c, w)

    # --- fields ------------------------------------------------------------------------------

    def _z_parts(self, prof: dict, z: np.ndarray, eps_bar: float, continuous_e: bool):
        """
        Curl factors (N, nz): basis function μ has ∇×H = e^{i kg·ρ} [u_μ f_ip(z) + ẑ f_z(z)]. With
        continuous_e the in-plane part is E_∥ = D_∥/ε_eff(z), continuous across the slab faces.
        """
        d, sz, ec = self.d, self.sigma_z, self.eps_clad
        q, chi = prof["q"][:, None], prof["chi"][:, None]
        zz = z[None, :]
        inside = np.abs(zz) <= d / 2
        sg = np.sign(zz)
        F_ip = np.cos(q * zz) if sz > 0 else np.sin(q * zz)
        F_z = np.sin(q * zz) if sz > 0 else np.cos(q * zz)
        ex = np.exp(-chi * np.maximum(np.abs(zz) - d / 2, 0))
        P_ip = ex * (1.0 if sz > 0 else sg)
        P_z = ex * (sg if sz > 0 else 1.0)
        f_ip = np.where(inside, prof["a_c"][:, None] * F_ip, prof["a_l"][:, None] * P_ip)
        f_z = np.where(inside, prof["b_c"][:, None] * F_z, prof["b_l"][:, None] * P_z)
        if continuous_e:
            f_ip = f_ip / np.where(inside, eps_bar, ec)
        return f_ip, f_z

    def curl_h(self, res: GMEResult, band: int, s1: np.ndarray, s2: np.ndarray, z: np.ndarray, continuous_e: bool = False) -> np.ndarray:
        """
        Periodic part of ∇×H (∝ D, 3 × n1 × n2 × nz) of one band at lattice coordinates s1 (n1,) ×
        s2 (n2,) (centred: r = s1 a1 + s2 a2) and heights z (nz,), normalised as ∫_cell |H|² = 1.
        With continuous_e the in-plane components are E_∥ instead of D_∥.
        """
        b, P = res.basis, res.basis.packed
        c = b.expand(res.vectors[:, band]) * np.asarray(P["mask"])
        prof = {k_: np.asarray(v) for k_, v in _k_profiles(_Static(self), jnp.asarray(res.fill), P).items()}
        f_ip, f_z = self._z_parts(prof, np.asarray(z, float), res.eps_bar, continuous_e)
        mG = b.mG[np.asarray(P["gi"])]
        r1, r2 = int(np.abs(b.mG[:, 0]).max()), int(np.abs(b.mG[:, 1]).max())
        E1 = np.exp(2j * np.pi * np.outer(np.arange(-r1, r1 + 1), s1))
        E2 = np.exp(2j * np.pi * np.outer(np.arange(-r2, r2 + 1), s2))
        out = np.zeros((3, len(s1), len(s2), len(z)), dtype=complex)
        comps = (c[:, None] * prof["u"][:, 0, None] * f_ip, c[:, None] * prof["u"][:, 1, None] * f_ip, c[:, None] * f_z)
        for a, coef in enumerate(comps):
            C = np.zeros((2 * r1 + 1, 2 * r2 + 1, len(z)), dtype=complex)
            np.add.at(C, (mG[:, 0] + r1, mG[:, 1] + r2), coef)
            out[a] = np.einsum("pi,pqz,qj->ijz", E1, C, E2, optimize=True)
        return out / np.sqrt(self.area)

    def _d_coefficients(self, res: GMEResult, band: int, z: np.ndarray, continuous_e: bool = False) -> np.ndarray:
        """Plane-wave coefficients of ∇×H, (3, NG_padded, nz), summed over the guided orders at each G."""
        b, P = res.basis, res.basis.packed
        c = b.expand(res.vectors[:, band]) * np.asarray(P["mask"])
        prof = {k_: np.asarray(v) for k_, v in _k_profiles(_Static(self), jnp.asarray(res.fill), P).items()}
        f_ip, f_z = self._z_parts(prof, np.asarray(z, float), res.eps_bar, continuous_e)
        gi = np.asarray(P["gi"])
        out = np.zeros((3, len(P["gmask"]), len(z)), dtype=complex)
        for a, coef in enumerate((c[:, None] * prof["u"][:, 0, None] * f_ip, c[:, None] * prof["u"][:, 1, None] * f_ip, c[:, None] * f_z)):
            np.add.at(out[a], gi, coef)
        return out

    def _synthesise(self, b: Basis, coef: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """Σ_G coef_G e^{iG·r} on the lattice grid s1 × s2: coef (3, NG_padded, nz) → (3, n1, n2, nz)."""
        ng = b.n_g
        mG = b.mG
        r1, r2 = int(np.abs(mG[:, 0]).max()), int(np.abs(mG[:, 1]).max())
        E1 = np.exp(2j * np.pi * np.outer(np.arange(-r1, r1 + 1), s1))
        E2 = np.exp(2j * np.pi * np.outer(np.arange(-r2, r2 + 1), s2))
        out = np.zeros((coef.shape[0], len(s1), len(s2), coef.shape[-1]), dtype=complex)
        for a in range(coef.shape[0]):
            C = np.zeros((2 * r1 + 1, 2 * r2 + 1, coef.shape[-1]), dtype=complex)
            C[mG[:, 0] + r1, mG[:, 1] + r2] = coef[a, :ng]
            out[a] = np.einsum("pi,pqz,qj->ijz", E1, C, E2, optimize=True)
        return out / np.sqrt(self.area)

    def eta_matrix(self, res: GMEResult) -> np.ndarray:
        """η(G, G') over the padded plane-wave list of a solve (the core's ε⁻¹ in the chosen rule)."""
        P = res.basis.packed
        blocks = res._eta if res._eta is not None else _k_eta_blocks(_Static(self), jnp.asarray(res.fill), P)
        return np.asarray(_k_eta_full(_Static(self), jnp.asarray(res.fill), P, blocks))

    def fields_on_cell(self, res: GMEResult, band: int, cell) -> tuple[np.ndarray, np.ndarray]:
        """
        (e, E) on the voxels of an optics_pwe.SlabCell: e ∝ D as `curl_h_on_cell`, and GME's own
        E = η D in the core (η of the eigenproblem, e.g. the inverse rule), D/ε_c in the claddings;
        in face voxels E_∥ from the core side (continuous) and E_z = g (ηD_z)_core + (1 − g) D_z/ε_c.
        With these, the Kottke-consistent field e_c = n(n·e) + ⟨ε⟩(E − n(n·E)) (cell normals n, as
        for meep fields in research/cavity_gpt07_g0.py) is what omc.coupling's voxel sensitivity
        expects; see research/gme_g0.py.
        """
        n1, n2, nz = cell.eps.shape
        s1 = (np.arange(n1) + 0.5) / n1 - 0.5
        s2 = (np.arange(n2) + 0.5) / n2 - 0.5
        dz = cell.height / nz
        z = np.arange(nz) * dz
        z = np.where(z > cell.height / 2, z - cell.height, z)
        g = cell.z_cover
        e = self.curl_h_on_cell(res, band, cell)
        eta = self.eta_matrix(res)
        core = np.flatnonzero(g >= 1 - 1e-12)
        part = np.flatnonzero((g > 0) & (g < 1))
        clad = np.flatnonzero(g <= 1e-12)
        b = res.basis
        E = np.zeros_like(e)
        if len(core):
            Dc = self._d_coefficients(res, band, z[core])
            E[..., core] = self._synthesise(b, np.einsum("gh,ahz->agz", eta, Dc), s1, s2)
        if len(clad):
            E[..., clad] = self._synthesise(b, self._d_coefficients(res, band, z[clad]) / self.eps_clad, s1, s2)
        if len(part):
            zin = np.sign(z[part]) * (self.d / 2 * (1 - 1e-9))
            Din = self._d_coefficients(res, band, zin)
            Ein = self._synthesise(b, np.einsum("gh,ahz->agz", eta, Din), s1, s2)
            Dout = self._synthesise(b, self._d_coefficients(res, band, z[part]) / self.eps_clad, s1, s2)
            E[:2][..., part] = Ein[:2]
            E[2][..., part] = g[part] * Ein[2] + (1 - g[part]) * Dout[2]
        return e, E

    def curl_h_on_cell(self, res: GMEResult, band: int, cell) -> np.ndarray:
        """
        e ∝ D (3, N1, N2, N3) on the voxels of an optics_pwe.SlabCell on the same lattice (any
        in-plane grid), for omc.coupling.optical_mode_from_field. Full core and cladding voxels
        get D; in voxels cut by a slab face the in-plane part is ⟨ε⟩_z E_∥ (E_∥ continuous;
        ⟨ε⟩_z = g ε̄ + (1 − g) ε_c of the effective slab) and the normal part D_z.
        """
        n1, n2, nz = cell.eps.shape
        s1 = (np.arange(n1) + 0.5) / n1 - 0.5
        s2 = (np.arange(n2) + 0.5) / n2 - 0.5
        dz = cell.height / nz
        z = np.arange(nz) * dz
        z = np.where(z > cell.height / 2, z - cell.height, z)
        g = cell.z_cover
        e = self.curl_h(res, band, s1, s2, z)
        part = np.flatnonzero((g > 0) & (g < 1))
        if len(part):
            eE = self.curl_h(res, band, s1, s2, z[part], continuous_e=True)
            e[:2][..., part] = eE[:2] * (g[part] * res.eps_bar + (1 - g[part]) * self.eps_clad)
        return e

    def y_fraction(self, res: GMEResult, band: int, half_width: float, n: tuple[int, int] = (16, 128)) -> float:
        """Fraction of the mid-plane ∫|D|² with |y| < half_width (localisation of line-defect / cavity modes)."""
        s1 = (np.arange(n[0]) + 0.5) / n[0] - 0.5
        s2 = (np.arange(n[1]) + 0.5) / n[1] - 0.5
        e = self.curl_h(res, band, s1, s2, np.array([0.0]))[..., 0]
        w = (np.abs(e) ** 2).sum(axis=0)
        y = s1[:, None] * self.A[0, 1] + s2[None, :] * self.A[1, 1]
        return float((w * (np.abs(y) < half_width)).sum() / w.sum())


# --- jitted kernels (solver static, hashed by identity; basis data traced) ----------------------


class _Static:
    def __init__(self, obj):
        self.obj = obj

    def __hash__(self):
        return id(self.obj)

    def __eq__(self, other):
        return isinstance(other, _Static) and other.obj is self.obj


@partial(jax.jit, static_argnums=0)
def _k_matrix(cfg, fill, P):
    return cfg.obj._matrix(fill, P)


@partial(jax.jit, static_argnums=0)
def _k_profiles(cfg, fill, P):
    return cfg.obj._profiles(fill, P)


@partial(jax.jit, static_argnums=0)
def _k_im_multi(cfg, fill, P, lams, C, ch, cm, blocks):
    return cfg.obj._im_lam_multi(fill, P, lams, C, ch, cm, blocks)


@partial(jax.jit, static_argnums=0)
def _k_eta_full(cfg, fill, P, blocks):
    idx = jnp.arange(P["mG"].shape[0])
    return cfg.obj._eta_fn(blocks, P)(idx[:, None], idx[None, :])


@partial(jax.jit, static_argnums=0)
def _k_eta_blocks(cfg, fill, P):
    return cfg.obj._eta_blocks(fill, P)


@partial(jax.jit, static_argnums=0)
def _k_matrix_from(cfg, fill, P, blocks):
    return cfg.obj._matrix(fill, P, blocks)


@partial(jax.jit, static_argnums=0)
def _k_hf_grad(cfg, fill, P, c):
    s = cfg.obj
    return jax.grad(lambda fl: jnp.real(jnp.vdot(c, s._matrix(fl, P) @ c)))(fill)


@partial(jax.jit, static_argnums=0)
def _k_im_partials(cfg, fill, P, lam, c, ch, cm):
    s = cfg.obj
    eta, prof = s._eta(fill, P), s._profiles(fill, P)
    im, (d_lam, g_c) = jax.value_and_grad(lambda lm, cc: s._im_lam(fill, P, lm, cc, ch, cm, eta, prof)[0], argnums=(0, 1))(lam, c)
    return im, d_lam, g_c


@partial(jax.jit, static_argnums=0)
def _k_q_grad(cfg, fill, P, lam, c, w, ch, cm, d_lam, inv_lam, inv_im):
    """∇ ln Q_rad = ∇[(1/λ) Re cᴴMc − (1/Im λ)(Im λ(fill) + (∂Im λ/∂λ) Re cᴴMc − 2 Re wᴴMc)] with c, w, λ fixed."""
    s = cfg.obj

    def f(fl):
        blocks = s._eta_blocks(fl, P)
        M = s._matrix(fl, P, blocks)
        Mc = M @ c
        lam_r = jnp.real(jnp.vdot(c, Mc))
        im = s._im_lam_multi(fl, P, lam[None], c[:, None], ch, cm, blocks)[0][0]
        return inv_lam * lam_r - inv_im * (im + d_lam * lam_r - 2 * jnp.real(jnp.vdot(w, Mc)))

    return jax.grad(f)(fill)


__all__ = ["Basis", "GMEResult", "GMESolver", "light_line_thz", "thz", "k0_of_thz", "real_transform"]
