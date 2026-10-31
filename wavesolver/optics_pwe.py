"""
Optical Bloch modes of a patterned slab by plane-wave expansion, the method of MPB.

The supercell is the hexagonal unit cell (a1, a2) times a height Lz in z, with the slab
centred on z = 0 and air above and below. H is expanded in plane waves exp(i(k+G)·r) on an
N1 × N2 × N3 FFT grid in the lattice coordinates, so the oblique lattice needs no special
treatment. Each plane wave carries two transverse amplitudes along unit vectors u, v ⊥ q = k+G,
so ∇·H = 0 holds exactly. The operator

    Θ H = ∇ × (ε⁻¹ ∇ × H) = (ω/c)² H

is applied matrix-free: curl in Fourier space, inverse FFT, multiply by the ε⁻¹ tensor, FFT,
curl again. Θ is Hermitian, so the lowest bands come from a block eigensolver (LOBPCG) with
MPB's approximate-inverse preconditioner, which applies the same chain with ε in place of ε⁻¹
and 1/|q| in place of |q|.

Interfaces that cut a voxel use the anisotropic average of Kottke et al. (PRE 77, 036611):
ε⁻¹ = ⟨ε⁻¹⟩ n nᵀ + ⟨ε⟩⁻¹ (1 − n nᵀ), with n the local interface normal. Fill fractions come
from the in-plane solid fraction (geometry.solid_fraction) times the slab's coverage of each z
layer; normals come from the gradient of that fill.

The slab is mirror symmetric in z. Under the mirror σ_z, H (a pseudovector) maps to
(-H_x, -H_y, H_z)(x, y, -z). parity "even" keeps σ_z H = H, the TE-like modes (in-plane E even
in z); "odd" keeps the TM-like modes. Modes above the light line ω = c|k| are not guided and
depend on Lz.

Units: µm, k in rad/µm, eigenvalue (ω/c)² in (rad/µm)², frequencies returned in THz.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as spla

from .geometry import HexLattice

jax.config.update("jax_enable_x64", True)

C_UM_PER_PS = 299.792458
# ε⁻¹ tensor components, stored in this order.
TENSOR = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def thz(eigenvalue: np.ndarray) -> np.ndarray:
    return C_UM_PER_PS * np.sqrt(np.clip(eigenvalue, 0, None)) / (2 * np.pi)


@dataclass
class SlabCell:
    """
    Dielectric of one supercell on the FFT grid: mean ε and the ε⁻¹ tensor (6, N1, N2, N3).
    The fill F = f(x, y) g(z), interface normals and the two permittivities are kept for the
    sensitivities.
    """

    lattice: HexLattice
    height: float
    eps: np.ndarray
    eps_inv: np.ndarray
    fill: np.ndarray | None = None
    z_cover: np.ndarray | None = None
    normals: np.ndarray | None = None
    grad_norm: np.ndarray | None = None
    eps_slab: float = 1.0
    eps_background: float = 1.0

    @property
    def vectors(self) -> np.ndarray:
        """Rows a1, a2, a3 in 3D."""
        A = np.zeros((3, 3))
        A[:2, :2] = self.lattice.vectors
        A[2, 2] = self.height
        return A


def slab_cell(
    lattice: HexLattice,
    fraction: np.ndarray,
    thickness: float,
    height: float,
    nz: int,
    eps_slab: float,
    eps_background: float = 1.0,
    normal_reg: float = 1e-2,
) -> SlabCell:
    """
    Extrude an in-plane solid fraction (N1, N2) into a slab of the given thickness, centred on
    z = 0 in a supercell of the given height, sampled at nz points z_j = j·height/nz (periodic).
    """
    n1, n2 = fraction.shape
    dz = height / nz
    z = np.arange(nz) * dz
    z = np.where(z > height / 2, z - height, z)
    g = np.clip(np.minimum(z + dz / 2, thickness / 2) - np.maximum(z - dz / 2, -thickness / 2), 0, None) / dz

    # Gradients by periodic central differences, mapped from lattice to Cartesian coordinates.
    b = lattice.reciprocal / (2 * np.pi)
    dfs1 = (np.roll(fraction, -1, 0) - np.roll(fraction, 1, 0)) * n1 / 2
    dfs2 = (np.roll(fraction, -1, 1) - np.roll(fraction, 1, 1)) * n2 / 2
    grad_f = dfs1[..., None] * b[0] + dfs2[..., None] * b[1]  # (n1, n2, 2)
    dg = (np.roll(g, -1) - np.roll(g, 1)) / (2 * dz)

    F = fraction[:, :, None] * g[None, None, :]
    grad = np.empty((3, n1, n2, nz))
    grad[0] = grad_f[..., 0, None] * g
    grad[1] = grad_f[..., 1, None] * g
    grad[2] = fraction[:, :, None] * dg
    # n = ∇F / √(|∇F|² + η²): unit normals at interfaces, fading smoothly to zero where the fill
    # is uniform, so ε⁻¹ stays differentiable there. η is 1 % of a full-voxel step in fill.
    eta = normal_reg * max(n1 / np.linalg.norm(lattice.vectors[0]), n2 / np.linalg.norm(lattice.vectors[1]))
    norm = np.sqrt((grad**2).sum(axis=0) + eta**2)
    nrm = grad / norm

    eps = eps_background + F * (eps_slab - eps_background)
    inv_mean = 1 / eps_background + F * (1 / eps_slab - 1 / eps_background)
    aniso = inv_mean - 1 / eps
    eps_inv = np.empty((6, n1, n2, nz))
    for c, (i, j) in enumerate(TENSOR):
        eps_inv[c] = aniso * nrm[i] * nrm[j] + (1 / eps if i == j else 0.0)
    return SlabCell(lattice, height, eps, eps_inv, F, g, nrm, norm, eps_slab, eps_background)


class PWESolver:
    """Bloch modes of a SlabCell. parity: "even" (TE-like), "odd" (TM-like) or None."""

    def __init__(self, cell: SlabCell, parity: str | None = "even"):
        self.cell = cell
        self.shape = cell.eps.shape
        self.parity = parity
        A = cell.vectors
        B = 2 * np.pi * np.linalg.inv(A).T
        m = [np.fft.fftfreq(n) * n for n in self.shape]
        M = np.meshgrid(*m, indexing="ij")
        self.G = np.einsum("i...,ij->j...", np.stack(M), B)  # (3, N1, N2, N3)
        self.flip_z = (-np.arange(self.shape[2])) % self.shape[2]
        # The z Nyquist plane maps onto itself under the mirror, so it cannot carry a parity.
        n3 = self.shape[2]
        self.keep_z = np.ones(n3, dtype=bool)
        if n3 % 2 == 0:
            self.keep_z[n3 // 2] = False
        self.shift = 1e3  # moves the excluded (wrong-parity, null) subspace above the bands
        self._k = None
        self._prev = None
        self._eps_inv = jnp.asarray(cell.eps_inv)
        self._eps = jnp.asarray(cell.eps)

    # --- basis -------------------------------------------------------------------------------

    def _set_k(self, k: np.ndarray) -> None:
        q = self.G + np.array([k[0], k[1], 0.0])[:, None, None, None]
        absq = np.sqrt((q**2).sum(axis=0))
        self.mask = (absq > 1e-9) & self.keep_z
        qhat = np.divide(q, absq, out=np.zeros_like(q), where=self.mask)
        u = np.stack([-q[1], q[0], np.zeros_like(q[0])])  # ẑ × q
        un = np.sqrt((u**2).sum(axis=0))
        along_z = un < 1e-9 * np.maximum(absq, 1)
        u = np.divide(u, un, out=np.zeros_like(u), where=~along_z)
        u[0][along_z] = 1.0
        v = np.cross(qhat, u, axis=0)
        self.absq, self.u, self.v = absq, u, v
        self.inv_absq = np.divide(1.0, absq, out=np.zeros_like(absq), where=self.mask)
        self._k = np.array(k, dtype=float)
        self._state = self._jax_state()
        self._absq_j, self._inv_absq_j = jnp.asarray(absq), jnp.asarray(self.inv_absq)

    def _blocks(self, X: np.ndarray) -> np.ndarray:
        return X.T.reshape((X.shape[1], 2) + self.shape)

    def _flat(self, h: np.ndarray) -> np.ndarray:
        return h.reshape(h.shape[0], -1).T

    def _vector(self, h: np.ndarray) -> np.ndarray:
        """Transverse amplitudes (nb, 2, ...) → Cartesian H coefficients (nb, 3, ...)."""
        return h[:, 0, None] * self.u + h[:, 1, None] * self.v

    def _amplitudes(self, H: np.ndarray) -> np.ndarray:
        return np.stack([(H * self.u).sum(axis=1), (H * self.v).sum(axis=1)], axis=1) * self.mask

    def _project(self, h: np.ndarray) -> np.ndarray:
        """
        Onto the chosen z-parity and the non-null plane waves. u = ẑ × q depends only on the
        in-plane part of q, so with G' the mirror image of G (G_z → -G_z) the mirror acts on
        the amplitudes as h_u(G) → -h_u(G'), h_v(G) → h_v(G').
        """
        if self.parity is None:
            return h * self.mask
        s = 1.0 if self.parity == "even" else -1.0
        hm = h[..., self.flip_z]
        out = np.empty_like(h)
        out[:, 0] = (h[:, 0] - s * hm[:, 0]) / 2
        out[:, 1] = (h[:, 1] + s * hm[:, 1]) / 2
        return out * self.mask

    # --- operators (jitted JAX, so the elementwise passes fuse around the FFTs) --------------

    def _jax_state(self) -> dict:
        return dict(
            u=jnp.asarray(self.u),
            v=jnp.asarray(self.v),
            mask=jnp.asarray(self.mask),
            flip=jnp.asarray(self.flip_z),
            parity=_PARITY[self.parity],
        )

    def _theta(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(_theta_jit(jnp.asarray(X), self._eps_inv, jnp.asarray(self.absq), self.shift, **self._state))

    def _precondition(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(_precondition_jit(jnp.asarray(X), self._eps, jnp.asarray(self.inv_absq), **self._state))

    # --- solve -------------------------------------------------------------------------------

    def solve(self, k: np.ndarray, n_bands: int = 8, tol: float = 1e-2, maxiter: int = 100, extra: int = 2):
        """
        Lowest n_bands frequencies in THz at in-plane k (rad/µm). Starts from the previous
        k-point's modes when there are any, carried over in Cartesian components.

        tol is LOBPCG's absolute residual norm on unit vectors, for eigenvalues (ω/c)² of
        order 10-100. The eigenvalue error goes as the residual squared: on the slab benchmark
        tol = 0.1 and 1e-3 give the same frequency to 1e-7. The final residuals are kept in
        self.residuals, the iteration count in self.iterations. tol applies to the n_bands
        requested; the `extra` bands above them only speed up convergence and may stop short of
        tol. If maxiter is reached first, the current (unconverged) modes are returned.
        """
        prev = self._prev
        self._set_k(k)
        nb = n_bands + extra
        if prev is not None and prev.shape[0] == nb:
            h0 = self._amplitudes(prev)
        else:
            rng = np.random.default_rng(0)
            h0 = rng.standard_normal((nb, 2) + self.shape) + 1j * rng.standard_normal((nb, 2) + self.shape)
            h0 *= (self.inv_absq**2)[None, None]  # start from smooth fields
        X0 = self._flat(self._project(h0))
        lam, X, res, itn = _lobpcg_jit(
            jnp.asarray(X0), tol, maxiter, n_bands, self._eps_inv, self._eps, self._absq_j, self._inv_absq_j, self.shift, **self._state
        )
        lam, X = np.asarray(lam), np.asarray(X)
        self.residuals, self.iterations = np.asarray(res), int(itn)
        order = np.argsort(lam)
        lam, X = lam[order], X[:, order]
        self._prev = self._vector(self._blocks(X))
        self.last_vectors = X
        self._last_freqs = thz(lam)
        return thz(lam[:n_bands])

    def solve_near(
        self,
        k: np.ndarray,
        f_target: float,
        n_bands: int = 3,
        x0: np.ndarray | None = None,
        tol: float = 1e-8,
        inner: int = 40,
        inner_fine: int = 200,
        max_basis: int = 32,
        min_basis: int = 8,
        maxiter: int = 400,
        switch: float = 1e-2,
        min_iter: int = 1,
        verbose: bool = False,
        callback=None,
    ) -> np.ndarray:
        """
        The n_bands modes nearest f_target (THz) at in-plane k, sorted by frequency (THz). For
        large supercells (a finite cavity), where the mode of interest lies hundreds of bands
        up and LOBPCG on the lowest bands is out of reach.

        Jacobi–Davidson for the Hermitian Θ with harmonic Ritz extraction about τ = (2π f/c)²
        (Ritz values closest to τ, not the extremes), one mode at a time, locking converged
        ones. The correction equation (1 − QQᴴ)(Θ − σ)(1 − QQᴴ) t = −r, t ⊥ Q (Q: the locked
        modes and the current Ritz vector) is solved approximately by `inner` MINRES steps on
        the real form, preconditioned by the deflated MPB preconditioner (as `adjoint_solve`);
        σ = τ until the relative residual is below `switch`, then the Ritz value with
        `inner_fine` steps (in a large supercell the low bands make the shifted system badly
        conditioned, and few steps stall the refinement near 1e-4). x0: start
        vectors (flat, as `last_vectors`' columns), e.g. an envelope times a band-edge mode;
        default smooth random fields. tol: relative residual |Θx − λx| / λ (the eigenvalue
        error goes as its square).

        A two-level preconditioner (the low-|q| plane waves solved densely) was tried and did not
        reduce the MINRES count on the GPT07 cavity supercell: the cost is the dense spectrum next
        to σ (air and waveguide-side supercell modes), not the low bands. Those slow components
        are delocalised, so functionals of the localised mode (g0) converge at a loose tol (~1e-2).
        callback(it, f_THz, rel_residual, u): called every iteration with the current Ritz vector.
        min_iter: no mode is locked before this iteration (a warm start after a small design
        change can be below tol at once, which would leave the eigenvector unchanged). If no
        mode converges, the current Ritz pair is returned with its residual.

        Sets last_vectors, _last_freqs and residuals as `solve` does, so `sensitivities`,
        `curl_h`, `voxel_sensitivity` and `adjoint_solve` apply to the returned bands.
        """
        self._set_k(k)
        n = 2 * np.prod(self.shape)
        tau = (2 * np.pi * f_target / C_UM_PER_PS) ** 2

        def proj(Y):
            return self._flat(self._project(self._blocks(Y)))

        def orth(Y, *bases):
            if Y.shape[1] == 0:
                return Y
            for _ in range(2):  # twice is enough (Kahan)
                for B in bases:
                    if B.shape[1]:
                        Y = Y - B @ (B.conj().T @ Y)
            q, r = np.linalg.qr(Y)
            keep = np.abs(np.diag(r)) > 1e-10 * max(np.abs(np.diag(r)).max(), 1e-300)
            return q[:, keep]

        if x0 is None:
            rng = np.random.default_rng(0)
            h0 = rng.standard_normal((min_basis, 2) + self.shape) + 1j * rng.standard_normal((min_basis, 2) + self.shape)
            h0 *= (self.inv_absq**2)[None, None]
            X0 = self._flat(h0)
        else:
            X0 = np.asarray(x0, dtype=complex).reshape(n, -1)
        Q = np.zeros((n, 0), dtype=complex)  # locked eigenvectors
        lam_q: list[float] = []
        res_q: list[float] = []
        V = orth(proj(X0), Q)
        AV = self._theta(V)
        it, n_inner = 0, 0
        while len(lam_q) < n_bands and it < maxiter:
            it += 1
            # harmonic Ritz about τ, W = (Θ − τ)V: the smallest |μ| = θ − τ is nearest τ
            H = V.conj().T @ AV
            H = (H + H.conj().T) / 2
            W = AV - tau * V
            G = W.conj().T @ W
            G = (G + G.conj().T) / 2
            Hs = H - tau * np.eye(len(H))
            # as G s = μ (H − τ) s, μ = 1/ν: a mode at τ itself has W s = 0 (μ = 0), which the
            # (H − τ) s = ν G s form would lose with G's near-null space (a converged warm start)
            mu, S = _harmonic_pencil(G, Hs)
            order = np.argsort(np.abs(mu))
            S = S[:, order]
            s = S[:, 0]
            u = V @ s
            Au = AV @ s
            nu_ = np.linalg.norm(u)
            u, Au = u / nu_, Au / nu_
            theta = float(np.real(np.vdot(u, Au)))
            r = Au - theta * u
            r -= Q @ (Q.conj().T @ r)
            rel = np.linalg.norm(r) / abs(theta)
            if callback is not None:
                callback(it, float(thz(np.array([theta]))[0]), rel, u)
            if verbose and (it % 5 == 1 or rel < tol):
                print(f"    JD it {it:3d}: basis {V.shape[1]:2d}, locked {len(lam_q)}, f {thz(np.array([theta]))[0]:.5f} THz, rel. residual {rel:.2e}", flush=True)
            if rel < tol and it >= min_iter:
                Q = np.column_stack([Q, u])
                lam_q.append(theta)
                res_q.append(rel)
                if len(lam_q) >= n_bands:  # done (the remaining Ritz space may be empty: a converged one-vector warm start)
                    break
                # continue in the remaining Ritz space, orthogonal to the locked modes
                Vn = orth(V @ S[:, 1:], Q)
                V, AV = Vn, self._theta(Vn)
                continue
            if V.shape[1] >= max_basis:  # restart with the best harmonic Ritz vectors
                Vn = orth(V @ S[:, :min_basis], Q)
                V, AV = Vn, self._theta(Vn)
            sigma, steps = (tau, inner) if rel > switch else (theta, inner_fine)
            t, k_in = self._jd_correction(np.column_stack([Q, u]), r, sigma, steps, proj)
            n_inner += k_in
            t = orth(t[:, None], Q, V)
            if t.shape[1] == 0:  # correction already in the basis: perturb with the residual
                t = orth(proj(r[:, None]), Q, V)
            V = np.column_stack([V, t])
            AV = np.column_stack([AV, self._theta(t)])
        if len(lam_q) < n_bands:
            warnings.warn(f"solve_near: {len(lam_q)} of {n_bands} modes converged in {maxiter} iterations")
            if it and len(lam_q) < n_bands:  # keep the current (unconverged) Ritz pair; its residual is in .residuals
                Q = np.column_stack([Q, u])
                lam_q.append(theta)
                res_q.append(rel)
        order = np.argsort(lam_q)
        lam = np.array(lam_q)[order]
        X = Q[:, order]
        self.residuals = np.array(res_q)[order]
        self.last_vectors = X
        self._last_freqs = thz(lam)
        self._prev = self._vector(self._blocks(X))
        self.jd_info = dict(outer=it, inner=n_inner)
        return thz(lam)

    def _jd_correction(self, Qt: np.ndarray, r: np.ndarray, sigma: float, maxiter: int, proj) -> tuple[np.ndarray, int]:
        """
        Approximate t ⊥ Qt with (1 − QtQtᴴ)(Θ − σ)(1 − QtQtᴴ) t = −r: complex MINRES on the device
        (`_minres_deflated_jit`, deflated MPB preconditioner), at most maxiter steps.
        """

        def defl(y):
            return y - Qt @ (Qt.conj().T @ y)

        b = defl(proj(-r[:, None])[:, 0])
        t, steps = _minres_deflated_jit(jnp.asarray(b), jnp.asarray(Qt), sigma, maxiter, 1e-3, self._eps_inv, self._eps,
                                        self._absq_j, self._inv_absq_j, self.shift, **self._state)
        return defl(proj(np.asarray(t)[:, None])[:, 0]), int(steps)

    def bands(self, ks: np.ndarray, n_bands: int = 8, workers: int = 1, **kwargs) -> np.ndarray:
        """
        Frequencies (len(ks), n_bands) in THz, each k-point warm-started from the one before.
        With workers > 1 the path is cut into `workers` contiguous pieces, each solved in its own
        process pinned to its own share of the CPUs (Linux and Windows; elsewhere only XLA's
        thread flags apply). That gives more throughput than one process, whose threads scale
        poorly. The workers do not update this solver's state (last_vectors, residuals).
        """
        self._prev = None
        if workers <= 1:
            return np.array([self.solve(k, n_bands, **kwargs) for k in ks])
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        pieces = [p for p in np.array_split(np.asarray(ks, dtype=float), workers) if len(p)]
        ctx = multiprocessing.get_context("spawn")
        cpus = ctx.Queue()
        n_cpu = os.cpu_count() or 1
        per = max(1, n_cpu // len(pieces))
        for i in range(len(pieces)):
            cpus.put([(i * per + j) % n_cpu for j in range(per)])
        with ProcessPoolExecutor(
            len(pieces), mp_context=ctx, initializer=_set_worker_solver, initargs=(self.cell, self.parity, cpus)
        ) as pool:
            return np.concatenate(list(pool.map(_worker_bands, pieces, [n_bands] * len(pieces), [kwargs] * len(pieces))))

    def h_field(self, band: int) -> np.ndarray:
        """Periodic part of H for one band of the last solve, on the real-space grid (3, N1, N2, N3)."""
        H = self._vector(self._blocks(self.last_vectors[:, [band]]))[0]
        return np.fft.ifftn(H, axes=(1, 2, 3)) * np.prod(self.shape)

    def curl_h(self, band: int) -> np.ndarray:
        """
        e = F⁻¹ B x for one band of the last solve: the periodic part of ∇ × H on the real-space
        grid (3, N1, N2, N3), in the normalisation where λ = N Σ e^H ε⁻¹ e. ε⁻¹ e is ∝ E.
        """
        h = self._blocks(self.last_vectors[:, [band]])
        C = 1j * self.absq * (h[:, 0, None] * self.v - h[:, 1, None] * self.u)
        return np.fft.ifftn(C[0], axes=(1, 2, 3))

    def sensitivities(self, bands: list[int] | range) -> np.ndarray:
        """
        ∂f/∂fraction (THz per unit fill) of the given bands of the last solve, w.r.t. the
        in-plane solid fraction (N1, N2) the cell was built from.

        λ = x^H Θ x with Θ = B† F ε⁻¹ F⁻¹ B and |x| = 1, so dλ = N Σ_voxels e^H dε⁻¹ e, for the
        Kottke tensor ε⁻¹ = ⟨ε⁻¹⟩ n nᵀ + ⟨ε⟩⁻¹ (1 − n nᵀ) as built by slab_cell. With
        c = ⟨ε⁻¹⟩ − ⟨ε⟩⁻¹,

            dε⁻¹ = (∂⟨ε⟩⁻¹/∂F + ∂c/∂F n nᵀ) dF + c (dn nᵀ + n dnᵀ),
            dn = (1 − n nᵀ) d(∇F) / √(|∇F|² + η²)   (slab_cell's regularised normal).

        F = f(x, y) g(z) and ∇F = (g ∇f, f g'), with ∇f by periodic central differences, so
        both terms map back to the in-plane fill; the central difference is antisymmetric, so
        its adjoint is minus itself. Bands are assumed non-degenerate.
        """
        c = self.cell
        n1, n2, nz = self.shape
        d_eps = c.eps_slab - c.eps_background
        d_inv = 1 / c.eps_slab - 1 / c.eps_background
        a = -d_eps / c.eps**2  # ∂⟨ε⟩⁻¹/∂F
        b = d_inv - a  # ∂c/∂F
        aniso = 1 / c.eps_background + c.fill * d_inv - 1 / c.eps
        inv_g = 1.0 / c.grad_norm
        g = c.z_cover
        dz = c.height / nz
        dg = (np.roll(g, -1) - np.roll(g, 1)) / (2 * dz)
        rb = c.lattice.reciprocal / (2 * np.pi)
        n_tot = np.prod(self.shape)
        lam = (2 * np.pi * self._last_freqs / C_UM_PER_PS) ** 2

        def central_adjoint(s: np.ndarray, axis: int, n: int) -> np.ndarray:
            return -(np.roll(s, -1, axis) - np.roll(s, 1, axis)) * n / 2

        out = []
        for band in bands:
            e = self.curl_h(band)
            ne = (c.normals * e).sum(axis=0)
            q = n_tot * (a * (np.abs(e) ** 2).sum(axis=0) + b * np.abs(ne) ** 2)
            dlam = (q * g).sum(axis=2)
            # Normal term: N · 2c · (1 − n nᵀ) Re(e* (n·e)) / |∇F| contracted with d(∇F).
            r = np.real(np.conj(e) * ne)
            r -= c.normals * (c.normals * r).sum(axis=0)
            s = 2 * n_tot * aniso * inv_g * r
            S1 = ((s[0] * rb[0, 0] + s[1] * rb[0, 1]) * g).sum(axis=2)
            S2 = ((s[0] * rb[1, 0] + s[1] * rb[1, 1]) * g).sum(axis=2)
            dlam += central_adjoint(S1, 0, n1) + central_adjoint(S2, 1, n2) + (s[2] * dg).sum(axis=2)
            out.append(C_UM_PER_PS / (4 * np.pi * np.sqrt(lam[band])) * dlam)
        return np.array(out)

    def voxel_sensitivity(self, band: int) -> np.ndarray:
        """
        ∂λ/∂F (eigenvalue (ω/c)² in (rad/µm)² per unit fill) of one band of the last solve,
        w.r.t. the 3D voxel fill F (N1, N2, N3) of the cell, for a perturbation δF that is not
        of the extruded form f(x, y) g(z) (e.g. δF = −u·∇F of a moving boundary, wavesolver.coupling).

        The same derivation as `sensitivities`, before the sum over z: slab_cell's ∇F equals
        the 3D periodic central differences of F (in-plane ones of f times g, f times the z
        one of g), so the normal term is the adjoint of those differences applied to
        s = 2N c (1 − n nᵀ) Re(e* (n·e)) / √(|∇F|² + η²). Σ_z g(z) ∂λ/∂F, converted to THz,
        reproduces `sensitivities`. δω/ω = δλ / 2λ. Bands are assumed non-degenerate.
        """
        c = self.cell
        n1, n2, nz = self.shape
        d_eps = c.eps_slab - c.eps_background
        d_inv = 1 / c.eps_slab - 1 / c.eps_background
        a = -d_eps / c.eps**2  # ∂⟨ε⟩⁻¹/∂F
        b = d_inv - a  # ∂c/∂F
        aniso = 1 / c.eps_background + c.fill * d_inv - 1 / c.eps
        dz = c.height / nz
        rb = c.lattice.reciprocal / (2 * np.pi)
        n_tot = np.prod(self.shape)
        e = self.curl_h(band)
        ne = (c.normals * e).sum(axis=0)
        dlam = n_tot * (a * (np.abs(e) ** 2).sum(axis=0) + b * np.abs(ne) ** 2)
        r = np.real(np.conj(e) * ne)
        r -= c.normals * (c.normals * r).sum(axis=0)
        s = 2 * n_tot * aniso / c.grad_norm * r  # ∂λ/∂(∇F), Cartesian (3, N1, N2, N3)

        def central_adjoint(v: np.ndarray, axis: int, scale: float) -> np.ndarray:
            return -(np.roll(v, -1, axis) - np.roll(v, 1, axis)) * scale / 2

        dlam += central_adjoint(s[0] * rb[0, 0] + s[1] * rb[0, 1], 0, n1)
        dlam += central_adjoint(s[0] * rb[1, 0] + s[1] * rb[1, 1], 1, n2)
        dlam += central_adjoint(s[2], 2, 1 / dz)
        return dlam

    def adjoint_solve(
        self, band: int, rhs: np.ndarray, tol: float = 1e-10, maxiter: int = 3000, start: bool = True, jax: bool = False
    ) -> tuple[np.ndarray, dict]:
        """
        Linear response of one band of the last solve: w with (Θ − λ) w = r, w ⊥ x (x the
        band's unit eigenvector, λ its Ritz value), where r is rhs (flat, as `last_vectors`'
        columns) projected onto the chosen z-parity and off x. This is the adjoint system of
        any functional of the eigenvector (wavesolver.coupling_adjoint): if dJ = Re(rᴴ dx) for an
        eigenvector change dx, then dJ = −Re(wᴴ dΘ x) for a change dΘ of the operator.

        Θ − λ is Hermitian and indefinite on the deflated subspace, so MINRES is used, on the
        equivalent real symmetric system of twice the size (scipy's MINRES is real only),
        preconditioned by the deflated MPB preconditioner (SPD). The right-hand side is
        normalised for MINRES. With start, the iteration starts from the expansion of r over
        the other computed bands, r_j / (λ_j − λ). Assumes the band is non-degenerate:
        near-degenerate bands make the system ill-conditioned (a warning is issued).

        jax: complex MINRES in one device loop (`_minres_deflated_jit`, as solve_near's
        correction equation), from zero; ~1.6× faster per
        step on the CPU. For a cavity mode in a large supercell (solve_near, one stored band)
        the spectrum near λ is dense, but a functional of the localised mode has a localised
        right-hand side, so a loose tol (~1e-3) suffices.
        """
        X = self.last_vectors
        x = X[:, band]
        lam_all = (2 * np.pi * self._last_freqs / C_UM_PER_PS) ** 2
        lam = lam_all[band]
        gaps = np.append(np.abs(np.delete(lam_all, band) - lam) / lam, np.inf)  # inf: a single stored band (solve_near)
        if gaps.min() < 1e-4:
            warnings.warn(f"band {band} is (near-)degenerate (relative eigenvalue gap {gaps.min():.1e}); its adjoint is ill-conditioned")
        n = X.shape[0]

        def proj(y):
            return self._flat(self._project(self._blocks(y[:, None])))[:, 0]

        def defl(y):
            return y - x * np.vdot(x, y)

        def to_c(y):
            return y[:n] + 1j * y[n:]

        def to_r(y):
            return np.concatenate([y.real, y.imag])

        def a_op(y):
            y = defl(to_c(y))
            return to_r(defl(self._theta(y[:, None])[:, 0] - lam * y))

        def m_op(y):
            return to_r(defl(self._precondition(defl(to_c(y))[:, None])[:, 0]))

        b = defl(proj(np.asarray(rhs, dtype=complex)))
        scale = np.linalg.norm(b)
        if not np.isfinite(scale):
            raise ValueError("adjoint_solve: right-hand side is not finite")
        if scale == 0:
            return np.zeros_like(b), dict(iterations=0, residual=0.0, info=0, min_gap=gaps.min())
        b = b / scale
        if jax:
            w, steps = _minres_deflated_jit(jnp.asarray(b), jnp.asarray(x[:, None]), lam, maxiter, tol, self._eps_inv, self._eps, self._absq_j, self._inv_absq_j,
                                            self.shift, **self._state)
            w = defl(proj(np.asarray(w)))
            res = np.linalg.norm(defl(self._theta(w[:, None])[:, 0] - lam * w) - b)
            return w * scale, dict(iterations=int(steps), residual=res, info=0, min_gap=gaps.min())
        others = np.delete(np.arange(X.shape[1]), band)
        Xo = X[:, others]
        x0 = Xo @ ((Xo.conj().T @ b) / (lam_all[others] - lam)) if start else None
        A = spla.LinearOperator((2 * n, 2 * n), matvec=a_op, dtype=float)
        M = spla.LinearOperator((2 * n, 2 * n), matvec=m_op, dtype=float)
        count = [0]

        def cb(_):
            count[0] += 1

        w, info = spla.minres(A, to_r(b), x0=None if x0 is None else to_r(x0), rtol=tol, maxiter=maxiter, M=M, callback=cb)
        w = defl(to_c(w))
        res = np.linalg.norm(defl(self._theta(w[:, None])[:, 0] - lam * w) - b)
        if info != 0 or res > 1e3 * tol:
            warnings.warn(f"optical adjoint MINRES: info {info}, relative residual {res:.1e} after {count[0]} iterations")
        return w * scale, dict(iterations=count[0], residual=res, info=info, min_gap=gaps.min())


def _harmonic_pencil(G: np.ndarray, Hs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """G s = μ Hs s (G Hermitian PSD, Hs Hermitian indefinite): real parts of μ (inf where Hs s = 0) and S."""
    import scipy.linalg as sla

    mu, S = sla.eig(G, Hs)
    mu = np.where(np.isfinite(mu), mu.real, np.inf)
    return mu, S / np.linalg.norm(S, axis=0)


def light_line_thz(ks: np.ndarray) -> np.ndarray:
    return C_UM_PER_PS * np.linalg.norm(ks, axis=1) / (2 * np.pi)


# --- worker processes for PWESolver.bands ----------------------------------------------------

_WORKER_SOLVER: PWESolver | None = None


def _set_worker_solver(cell: SlabCell, parity: str | None, cpus) -> None:
    # XLA's thread flags confine the FFTs but not its matrix products, which spread over every core,
    # so each worker is also pinned to the CPUs it takes from the queue. Both must happen before
    # the worker's first JAX computation, which is when XLA reads its flags.
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    ).strip()
    mine = cpus.get()
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, mine)
    elif sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask(ctypes.c_void_p(kernel32.GetCurrentProcess()), ctypes.c_size_t(sum(1 << c for c in mine)))
    global _WORKER_SOLVER
    _WORKER_SOLVER = PWESolver(cell, parity)


def _worker_bands(ks: np.ndarray, n_bands: int, kwargs: dict) -> np.ndarray:
    return _WORKER_SOLVER.bands(ks, n_bands, **kwargs)


# --- jitted operator kernels ---------------------------------------------------------------
# Blocks X are (2·N1·N2·N3, nb) as LOBPCG passes them; h is the (nb, 2, N1, N2, N3) view.

_PARITY = {None: 0, "even": 1, "odd": -1}


def _to_blocks(X, shape):
    return X.T.reshape((X.shape[1], 2) + shape)


def _to_flat(h):
    return h.reshape(h.shape[0], -1).T


def _project_j(h, mask, flip, parity):
    if parity == 0:
        return h * mask
    hm = h[..., flip]
    hu = (h[:, 0] - parity * hm[:, 0]) / 2
    hv = (h[:, 1] + parity * hm[:, 1]) / 2
    return jnp.stack([hu, hv], axis=1) * mask


def _curl_chain_j(h, weight, scale, u, v):
    """B† F w F⁻¹ B with B: h ↦ i·scale·(h_u v − h_v u); w is the ε⁻¹ tensor (6, ...) or a scalar field."""
    C = 1j * scale * (h[:, 0, None] * v - h[:, 1, None] * u)
    E = jnp.fft.ifftn(C, axes=(2, 3, 4))
    if weight.ndim == 4:
        w = weight
        D = jnp.stack(
            [
                w[0] * E[:, 0] + w[5] * E[:, 1] + w[4] * E[:, 2],
                w[5] * E[:, 0] + w[1] * E[:, 1] + w[3] * E[:, 2],
                w[4] * E[:, 0] + w[3] * E[:, 1] + w[2] * E[:, 2],
            ],
            axis=1,
        )
    else:
        D = weight * E
    Df = jnp.fft.fftn(D, axes=(2, 3, 4))
    return jnp.stack([-1j * scale * (Df * v).sum(axis=1), 1j * scale * (Df * u).sum(axis=1)], axis=1)


@partial(jax.jit, static_argnames=("parity",))
def _theta_jit(X, eps_inv, absq, shift, u, v, mask, flip, parity):
    h = _to_blocks(X, absq.shape)
    p = _project_j(h, mask, flip, parity)
    y = _project_j(_curl_chain_j(p, eps_inv, absq, u, v), mask, flip, parity)
    return _to_flat(y + shift * (h - p))


@partial(jax.jit, static_argnames=("parity",))
def _precondition_jit(X, eps, inv_absq, u, v, mask, flip, parity):
    h = _project_j(_to_blocks(X, inv_absq.shape), mask, flip, parity)
    return _to_flat(_project_j(_curl_chain_j(h, eps, inv_absq, u, v), mask, flip, parity))


@partial(jax.jit, static_argnames=("parity",))
def _lobpcg_jit(X0, tol, maxiter, n_conv, eps_inv, eps, absq, inv_absq, shift, u, v, mask, flip, parity):
    """
    Lowest eigenpairs of Θ by LOBPCG with the MPB preconditioner, the whole iteration compiled
    (no host round trips between operator applications). Each step does a Rayleigh-Ritz on the
    basis [X, W, P] (modes, preconditioned residuals, previous update), orthonormalised through
    the eigendecomposition of its Gram matrix: near-dependent directions are dropped instead of
    breaking a Cholesky factorisation. Converged columns (residual ≤ tol) add no search direction.
    The iteration stops when the lowest n_conv columns have converged; the columns above them are
    guard vectors, which speed up convergence but need not converge themselves.
    Returns λ, X (orthonormal columns), the residual norms and the number of iterations.
    """
    nb = X0.shape[1]

    def A(Y):
        return _theta_jit(Y, eps_inv, absq, shift, u, v, mask, flip, parity)

    def unit(Y, AY):
        n = jnp.linalg.norm(Y, axis=0)
        s = jnp.where(n > 0, 1 / jnp.where(n > 0, n, 1), 0)
        return Y * s, AY * s

    def rayleigh_ritz(G, H):
        d, V = jnp.linalg.eigh((G + G.conj().T) / 2)
        keep = d > 1e-10 * d[-1]
        T = V * jnp.where(keep, 1 / jnp.sqrt(jnp.where(keep, d, 1)), 0)
        B = T.conj().T @ ((H + H.conj().T) / 2) @ T + jnp.diag(jnp.where(keep, 0.0, 1e30))
        mu, Y = jnp.linalg.eigh(B)
        return mu[:nb], T @ Y[:, :nb]

    X, _ = jnp.linalg.qr(X0)
    AX = A(X)
    lam, C = rayleigh_ritz(jnp.eye(nb, dtype=X.dtype), X.conj().T @ AX)
    X, AX = X @ C, AX @ C
    R = AX - X * lam
    zero = jnp.zeros_like(X)
    state = dict(itn=0, X=X, AX=AX, P=zero, AP=zero, lam=lam, R=R, res=jnp.linalg.norm(R, axis=0))

    def cond(st):
        return (st["itn"] < maxiter) & jnp.any((st["res"] > tol) & (jnp.arange(nb) < n_conv))

    def body(st):
        X, AX = st["X"], st["AX"]
        W = _precondition_jit(st["R"], eps, inv_absq, u, v, mask, flip, parity) * (st["res"] > tol)
        W = W - X @ (X.conj().T @ W)
        W, AW = unit(W, A(W))
        P, AP = unit(st["P"], st["AP"])
        Q, AQ = jnp.concatenate([W, P], axis=1), jnp.concatenate([AW, AP], axis=1)
        # Gram blocks of [X, Q]: X is orthonormal with XᴴAX = diag(λ) from the last step.
        XQ, XAQ = X.conj().T @ Q, X.conj().T @ AQ
        G = jnp.block([[jnp.eye(nb, dtype=X.dtype), XQ], [XQ.conj().T, Q.conj().T @ Q]])
        H = jnp.block([[jnp.diag(st["lam"]).astype(X.dtype), XAQ], [XAQ.conj().T, Q.conj().T @ AQ]])
        lam, C = rayleigh_ritz(G, H)
        P, AP = Q @ C[nb:], AQ @ C[nb:]
        X, AX = X @ C[:nb] + P, AX @ C[:nb] + AP
        R = AX - X * lam
        return dict(itn=st["itn"] + 1, X=X, AX=AX, P=P, AP=AP, lam=lam, R=R, res=jnp.linalg.norm(R, axis=0))

    st = jax.lax.while_loop(cond, body, state)
    return st["lam"], st["X"], st["res"], st["itn"]


@partial(jax.jit, static_argnames=("parity",))
def _minres_deflated_jit(b, Q, sigma, maxiter, rtol, eps_inv, eps, absq, inv_absq, shift, u, v, mask, flip, parity):
    """
    Preconditioned MINRES (Paige–Saunders, as scipy's) for (1 − QQᴴ)(Θ − σ)(1 − QQᴴ) x = b with
    the deflated MPB preconditioner, in complex arithmetic (Θ Hermitian: the Lanczos
    coefficients are real), all on the device in one loop. Stops after maxiter steps or when
    the preconditioned residual estimate falls below rtol of its start. Returns (x, steps).
    """

    def defl(y):
        return y - Q @ (Q.conj().T @ y)

    def a_op(y):
        y = defl(y)
        return defl(_theta_jit(y[:, None], eps_inv, absq, shift, u, v, mask, flip, parity)[:, 0] - sigma * y)

    def m_op(y):
        return defl(_precondition_jit(defl(y)[:, None], eps, inv_absq, u, v, mask, flip, parity)[:, 0])

    zero = jnp.zeros_like(b)
    y = m_op(b)
    beta1 = jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(b, y)), 0.0))
    tiny = jnp.finfo(jnp.float64).tiny
    state = dict(itn=0, x=zero, r1=b, r2=b, y=y, w=zero, w2=zero, beta=beta1, oldb=0.0, dbar=0.0, epsln=0.0,
                 phibar=beta1, cs=-1.0, sn=0.0)

    def cond(st):
        return (st["itn"] < maxiter) & (st["phibar"] > rtol * beta1) & (beta1 > 0)

    def body(st):
        beta, oldb = st["beta"], st["oldb"]
        v_ = st["y"] / beta
        y = a_op(v_)
        y = jnp.where(st["itn"] >= 1, y - (beta / jnp.where(oldb == 0, 1.0, oldb)) * st["r1"], y)
        alfa = jnp.real(jnp.vdot(v_, y))
        y = y - (alfa / beta) * st["r2"]
        r1, r2 = st["r2"], y
        y = m_op(r2)
        oldb, beta = beta, jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(r2, y)), 0.0))
        cs, sn, dbar = st["cs"], st["sn"], st["dbar"]
        oldeps = st["epsln"]
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta
        gamma = jnp.maximum(jnp.hypot(gbar, beta), tiny)
        cs, sn = gbar / gamma, beta / gamma
        phi = cs * st["phibar"]
        phibar = sn * st["phibar"]
        w1, w2 = st["w2"], st["w"]
        w = (v_ - oldeps * w1 - delta * w2) / gamma
        return dict(itn=st["itn"] + 1, x=st["x"] + phi * w, r1=r1, r2=r2, y=y, w=w, w2=w2, beta=beta, oldb=oldb, dbar=dbar,
                    epsln=epsln, phibar=phibar, cs=cs, sn=sn)

    st = jax.lax.while_loop(cond, body, state)
    return st["x"], st["itn"]
