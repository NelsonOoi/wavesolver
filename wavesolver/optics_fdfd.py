"""
Frequency-domain Maxwell on a thin slab by a direct solve: a Yee grid with few cells in z, so the matrix is nearly two-dimensional and a sparse factorisation is affordable.

Why this exists. wavesolver.optics_pwe is spectrally accurate, but a driven solve is iterative and takes many steps on a cavity-sized domain. Here one factorisation gives every right-hand side (forward, adjoint,
shift-invert eigensolves) at a cost that does not depend on Q, and the absorbers make radiation part of the solve. Its second-order grid makes it the wrong tool for a final Q above about 1e5: see validation/fdfd_check.py.

The domain. x ∈ [0, Lx/2], y ∈ [−Ly/2, Ly/2], z ∈ [0, z_top], with an electric wall at x = 0 (E_y = E_z = 0: the sector of a TE0 strip running along y), a magnetic wall at z = 0 (the TE-like, z-even sector) and stretched-coordinate
absorbers on the other sides. Time goes as exp(−iωt). Unknowns are the line integrals e of E along the Yee edges, which makes A = Cᵀ R C − k² W complex symmetric whatever the grid and the stretching:
- C is the integer curl from edges to faces, R = (dual length)/(face area) per face, W = ε (dual area)/(length) per edge, k² = k0²(1 + i/Q_d).
- x: E_x on x-segments, E_y and E_z on interior x-nodes. y the same. z: E_x and E_y at z-cell centres (the "nodes" ζ_k), E_z on the segments between them, which lie at the cell faces. So the slab's faces are cell faces.

The density. `eps_maps` turns a fine raster (m pixels per cell) into one ε per edge: E along an edge sees the harmonic mean along it and the arithmetic mean across it, the diagonal form of subpixel smoothing. Differentiable (JAX).

The port. `strip_mode` solves the discrete cross-section of a uniform waveguide along y, so the mode launched by `port_source` is an exact solution of the discrete A there, propagating towards −y with unit power.
"""

from __future__ import annotations

import ctypes as ct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

C_UM_THZ = 299.792458
EPS_SI = 3.48**2


# --- complex symmetric direct solver ------------------------------------------------------------


class ComplexSymmetricLU:
    """
    MKL PARDISO (mtype 6, complex symmetric) through ctypes, on the upper triangle in CSR: `analyse` once per sparsity pattern, `factor` per set of values, `solve` per right-hand side.
    MKL comes with numpy on conda-forge; set MKL_THREADING_LAYER=TBB for threads (the default layer in this project's pixi environment is SEQUENTIAL).
    """

    def __init__(self, threads: int = 16, single: bool = False):
        name = {"win32": "Library/bin/mkl_rt.3.dll", "darwin": "lib/libmkl_rt.dylib"}.get(sys.platform, "lib/libmkl_rt.so")
        self.rt = ct.CDLL(str(Path(sys.prefix) / name))
        self.rt.MKL_Set_Num_Threads(ct.c_int(threads))
        self.pt = (ct.c_void_p * 64)()
        self.iparm = (ct.c_int * 64)()
        p = self.iparm
        p[0], p[1], p[7], p[9], p[10], p[12], p[17], p[18], p[34] = 1, 2, 2, 8, 1, 1, -1, -1, 1
        p[27] = 1 if single else 0
        self.dtype = np.complex64 if single else np.complex128
        self.active = False

    def _call(self, phase, b=None, x=None):
        err, one, zero = ct.c_int(0), ct.c_int(1), ct.c_int(0)
        cptr = ct.POINTER(ct.c_float if self.dtype == np.complex64 else ct.c_double)
        iptr = ct.POINTER(ct.c_int)
        perm = (ct.c_int * 1)()
        self.rt.pardiso(self.pt, ct.byref(one), ct.byref(one), ct.byref(ct.c_int(6)), ct.byref(ct.c_int(phase)), ct.byref(ct.c_int(self.n)), self.a.ctypes.data_as(cptr), self.ia.ctypes.data_as(iptr), self.ja.ctypes.data_as(iptr),
                        perm, ct.byref(one), self.iparm, ct.byref(zero), b.ctypes.data_as(cptr) if b is not None else None, x.ctypes.data_as(cptr) if x is not None else None, ct.byref(err))
        if err.value:
            raise RuntimeError(f"PARDISO phase {phase}: error {err.value}")

    def analyse(self, A_upper: sp.csr_matrix) -> None:
        self.n = A_upper.shape[0]
        self.ia, self.ja = A_upper.indptr.astype(np.int32), A_upper.indices.astype(np.int32)
        self.a = A_upper.data.astype(self.dtype)
        self._call(11)
        self.active = True

    def factor(self, data: np.ndarray) -> None:
        self.a = np.asarray(data, dtype=self.dtype)
        self._call(22)

    def solve(self, b: np.ndarray) -> np.ndarray:
        b = np.array(b, dtype=self.dtype)
        x = np.zeros(self.n, dtype=self.dtype)
        self._call(33, b, x)
        return x.astype(np.complex128)

    @property
    def stats(self) -> dict:
        p = self.iparm
        return dict(factor_nnz=int(-p[17] if p[17] < 0 else p[17]), mflops=int(p[18]), peak_gb=max(p[14], p[15] + p[16]) / 1e6, perturbed=int(p[13]))

    def release(self) -> None:
        if self.active:
            self._call(-1)
            self.active = False

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


# --- grid ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ZGrid:
    """Cell heights from z = 0 upwards: n_slab cells in the half thickness t/2, then air growing by `growth` up to dz_max, the last n_pml cells the absorber."""

    dz: np.ndarray
    n_slab: int
    n_pml: int

    @classmethod
    def make(cls, t: float = 0.22, n_slab: int = 3, dz_air: float = 0.04, growth: float = 1.5, dz_max: float = 0.3, reach: float = 0.6, n_pml: int = 4) -> "ZGrid":
        """reach: at least this much air (µm) above the slab before the absorber."""
        cells = [t / 2 / n_slab] * n_slab
        d, top = dz_air, 0.0
        while top < reach:
            cells.append(min(d, dz_max))
            top += cells[-1]
            d *= growth
        cells += [dz_max] * n_pml
        return cls(np.array(cells), n_slab, n_pml)


def _stretch(p, n, npml, sigma, k0):
    """s at fractional cell positions p, an absorber over the last npml cells (cubic grading)."""
    d = np.clip((np.asarray(p, float) - (n - npml)) / max(npml, 1), 0, None)
    return 1 + 1j * sigma * d**3 / k0


def _difference(n_seg):
    """G: interior nodes (n_seg − 1) → segments (n_seg), walls at both ends: (G a)_{i+½} = a_{i+1} − a_i."""
    return sp.diags([np.ones(n_seg - 1), -np.ones(n_seg - 1)], [0, -1], shape=(n_seg, n_seg - 1), format="csr")


class SlabFDFD:
    """
    The solver on one grid; see the module docstring. Lx, Ly: the whole domain's size (µm), of which x ≥ 0 is solved; nx, ny: cells of the solved part. npml: absorber cells at the outer x side and at both y ends.
    Set the frequency by set_frequency, the design by set_eps, then factor; solve, or modes.
    """

    def __init__(self, Lx: float, Ly: float, nx: int, ny: int, zgrid: ZGrid | None = None, npml: int = 10, sigma_xy: float = 40.0, sigma_z: float = 12.0, f_ref_thz: float = 195.0, eps_slab: float = EPS_SI, eps_bg: float = 1.0,
                 threads: int = 16, single: bool = False, y_wall: str | None = None):
        """
        y_wall None: y ∈ [−Ly/2, Ly/2] with absorbers at both ends (ny cells). "pec" or "pmc": the mirror y = 0 as a wall too, y ∈ [0, Ly/2] solved (ny cells) with the absorber at the top only, as x.
        "pec" (electric wall): E_x, E_z odd in y, the y = 0 nodes dropped, exactly as the x wall. "pmc" (magnetic wall): E_x, E_z even, E_y odd; the y = 0 nodes kept, with half their dual length (the half cell on this side).
        """
        if y_wall not in (None, "pec", "pmc"):
            raise ValueError(f"y_wall {y_wall!r}")
        self.y_wall = y_wall
        self.Lx, self.Ly, self.nx, self.ny = Lx, Ly, nx, ny
        self.zg = zgrid or ZGrid.make()
        self.nz = nz = len(self.zg.dz)
        self.eps_slab, self.eps_bg = eps_slab, eps_bg
        self.dx, self.dy = Lx / 2 / nx, (Ly / ny if y_wall is None else Ly / 2 / ny)
        self.nyN = ny if y_wall == "pmc" else ny - 1  # y-nodes that carry E_x and E_z
        k_ref = 2 * np.pi * f_ref_thz / C_UM_THZ
        self.npml = npml
        dz = self.zg.dz
        self.dz = dz
        # lengths: S on segments, N (dual) on nodes; x has absorber at the outer end, y at both, z at the top
        sx = _stretch(np.arange(nx) + 0.5, nx, npml, sigma_xy, k_ref)
        sxn = _stretch(np.arange(1, nx), nx, npml, sigma_xy, k_ref)
        self.lxS, self.lxN = self.dx * sx, self.dx * sxn

        def sy_at(p):
            return _stretch(np.maximum(p, ny - p), ny, npml, sigma_xy, k_ref)

        if y_wall is None:
            self.lyS, self.lyN = self.dy * sy_at(np.arange(ny) + 0.5), self.dy * sy_at(np.arange(1, ny).astype(float))
        else:  # one absorber, at the top
            self.lyS = self.dy * _stretch(np.arange(ny) + 0.5, ny, npml, sigma_xy, k_ref)
            self.lyN = self.dy * _stretch(np.arange(ny - self.nyN, ny).astype(float), ny, npml, sigma_xy, k_ref)
            if y_wall == "pmc":
                self.lyN[0] *= 0.5
        np_z = self.zg.n_pml
        self.lzN = dz * _stretch(np.arange(nz) + 0.5, nz, np_z, sigma_z, k_ref)  # nodes ζ_k: the cells themselves
        self.lzS = 0.5 * (dz[:-1] + dz[1:]) * _stretch(np.arange(1, nz).astype(float), nz, np_z, sigma_z, k_ref)  # segments between ζ_k and ζ_{k+1}, at the cell faces
        self.z_edges = np.r_[0.0, np.cumsum(dz)]
        self.zeta = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        # unknowns: e_x (nx, ny−1, nz), e_y (nx−1, ny, nz), e_z (nx−1, ny−1, nz−1)
        nyN = self.nyN
        self.shape_x, self.shape_y, self.shape_z = (nx, nyN, nz), (nx - 1, ny, nz), (nx - 1, nyN, nz - 1)
        self.n_x, self.n_y, self.n_z = int(np.prod(self.shape_x)), int(np.prod(self.shape_y)), int(np.prod(self.shape_z))
        self.n = self.n_x + self.n_y + self.n_z
        Gx = _difference(nx)
        # y-nodes → y-segments: walls at both ends, or with "pmc" the node at y = 0 kept and a wall only at the top
        Gy = _difference(ny) if y_wall != "pmc" else sp.diags([-np.ones(ny), np.ones(ny - 1)], [0, 1], shape=(ny, ny), format="csr")
        Gz = sp.diags([-np.ones(nz - 1), np.ones(nz - 1)], [0, 1], shape=(nz - 1, nz), format="csr")  # z-nodes → segments, no wall ends

        def k3(a, b, c):
            return sp.kron(sp.kron(a, b, format="csr"), c, format="csr")

        I = sp.identity
        Fx_ey, Fx_ez = -k3(I(nx - 1), I(ny), Gz), k3(I(nx - 1), Gy, I(nz - 1))
        Fy_ex, Fy_ez = k3(I(nx), I(nyN), Gz), -k3(Gx, I(nyN), I(nz - 1))
        Fz_ex, Fz_ey = -k3(I(nx), Gy, I(nz)), k3(Gx, I(ny), I(nz))
        self.C = sp.bmat([[None, Fx_ey, Fx_ez], [Fy_ex, None, Fy_ez], [Fz_ex, Fz_ey, None]], format="csr")

        def outer(a, b, c):
            return np.ravel(a[:, None, None] * b[None, :, None] * c[None, None, :])

        rx = outer(self.lxN, 1 / self.lyS, 1 / self.lzS)
        ry = outer(1 / self.lxS, self.lyN, 1 / self.lzS)
        rz = outer(1 / self.lxS, 1 / self.lyS, self.lzN)
        self.R = np.r_[rx, ry, rz]
        # W per edge is ε times this
        self.mass_x = outer(1 / self.lxS, self.lyN, self.lzN).reshape(self.shape_x)
        self.mass_y = outer(self.lxN, 1 / self.lyS, self.lzN).reshape(self.shape_y)
        self.mass_z = outer(self.lxN, self.lyN, 1 / self.lzS).reshape(self.shape_z)
        self.mass = np.r_[self.mass_x.ravel(), self.mass_y.ravel(), self.mass_z.ravel()]
        A0 = (self.C.T @ sp.diags(self.R) @ self.C).tocsr()
        self.A0_diag = A0.diagonal().copy()
        U = sp.triu(A0 + sp.identity(self.n, dtype=complex, format="csr") * 0, format="csr")
        U.sum_duplicates()
        U.sort_indices()
        self.upper = U
        # the positions of the diagonal in U.data
        pos = np.empty(self.n, dtype=np.int64)
        rows = np.repeat(np.arange(self.n), np.diff(U.indptr))
        d = rows == U.indices
        pos[rows[d]] = np.flatnonzero(d)
        if d.sum() != self.n:
            raise RuntimeError("the matrix pattern is missing diagonal entries")
        self.diag_pos = pos
        self.lu = ComplexSymmetricLU(threads, single)
        self._analysed = False
        self.k2 = None
        self.eps_vec = None

    # --- design ---------------------------------------------------------------------------------

    def set_frequency(self, f_thz: float, q_d: float = np.inf) -> None:
        self.f_thz = f_thz
        self.q_d = q_d
        self.k0 = 2 * np.pi * f_thz / C_UM_THZ
        self.k2 = self.k0**2 * (1 + (1j / q_d if np.isfinite(q_d) else 0))

    def eps_maps(self, fine_half, m: int):
        """(ε_x (nx, ny−1), ε_y (nx−1, ny), ε_z (nx−1, ny−1)) in the slab from the fill raster of x ≥ 0, (m·nx, m·ny); m even. JAX."""
        nx, ny, nyN = self.nx, self.ny, self.nyN
        eps = self.eps_bg + fine_half * (self.eps_slab - self.eps_bg)
        h = m // 2
        # the columns around the y-nodes: interior nodes, or with "pmc" from y = 0 on, the cell around it completed by the mirror image
        ycols = (lambda a: a[:, h:-h]) if self.y_wall != "pmc" else (lambda a: jnp.concatenate([a[:, :h][:, ::-1], a], axis=1)[:, : ny * m])
        # E_x: arithmetic across (y) over the cell around the node, harmonic along (x)
        ey = ycols(eps).reshape(nx * m, nyN, m).mean(axis=2)
        ex = 1 / (1 / ey).reshape(nx, m, nyN).mean(axis=1)
        # E_y: arithmetic across (x), harmonic along (y)
        ax = eps[h : -h, :].reshape(nx - 1, m, ny * m).mean(axis=1)
        ey_ = 1 / (1 / ax).reshape(nx - 1, ny, m).mean(axis=2)
        # E_z: along z, so arithmetic in the plane
        ez = ycols(eps[h:-h]).reshape(nx - 1, m, nyN, m).mean(axis=(1, 3))
        return ex, ey_, ez

    def _face_weights(self):
        """w_in, w_out of the z segment that crosses the slab's face: 1/ε_z = w_in/ε + w_out/ε_bg."""
        k = self.zg.n_slab - 1
        dz = self.dz
        tot = 0.5 * (dz[k] + dz[k + 1])
        return k, 0.5 * dz[k] / tot, 0.5 * dz[k + 1] / tot

    def set_eps(self, ex: np.ndarray, ey: np.ndarray, ez: np.ndarray) -> None:
        """ε per edge from the slab's maps: uniform in z in the slab, background above, the harmonic mean on the face segment."""
        ns = self.zg.n_slab
        ex3, ey3, ez3 = np.full(self.shape_x, self.eps_bg, float), np.full(self.shape_y, self.eps_bg, float), np.full(self.shape_z, self.eps_bg, float)
        ex3[:, :, :ns] = np.asarray(ex)[:, :, None]
        ey3[:, :, :ns] = np.asarray(ey)[:, :, None]
        ez3[:, :, : ns - 1] = np.asarray(ez)[:, :, None]
        k, wi, wo = self._face_weights()
        ez3[:, :, k] = 1 / (wi / np.asarray(ez) + wo / self.eps_bg)
        self.eps_x, self.eps_y, self.eps_z = ex3, ey3, ez3
        self.eps_vec = np.r_[ex3.ravel(), ey3.ravel(), ez3.ravel()]

    def factor(self) -> dict:
        t0 = time.perf_counter()
        data = self.upper.data.copy()
        data[self.diag_pos] = self.A0_diag - self.k2 * self.mass * self.eps_vec
        if not self._analysed:
            self.upper.data = data
            self.lu.analyse(self.upper)
            self._analysed = True
        t1 = time.perf_counter()
        self.lu.factor(data)
        self._data = data
        return dict(t_analyse=t1 - t0, t_factor=time.perf_counter() - t1, **self.lu.stats)

    def solve(self, b: np.ndarray) -> np.ndarray:
        return self.lu.solve(b)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """A x, without the factorisation."""
        return self.C.T @ (self.R * (self.C @ x)) - self.k2 * self.mass * self.eps_vec * x

    def residual(self, x: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(self.apply(x) - b) / np.linalg.norm(b))

    def split(self, x: np.ndarray):
        return x[: self.n_x].reshape(self.shape_x), x[self.n_x : self.n_x + self.n_y].reshape(self.shape_y), x[self.n_x + self.n_y :].reshape(self.shape_z)

    def modes(self, f_thz: float, n: int = 4, tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """n modes of the absorbing structure nearest the factorised k²: (f THz, Q, vectors), by shift-invert on the current factorisation; f_thz is unused beyond labelling, the shift is the factorised k²."""
        W = self.mass * self.eps_vec
        op = spla.LinearOperator((self.n, self.n), matvec=lambda v: self.lu.solve(W * v), dtype=complex)
        mu, X = spla.eigs(op, k=n, which="LM", tol=tol)
        k = np.sqrt(self.k2 + 1 / mu)
        k = np.where(k.real < 0, -k, k)
        f = k.real * C_UM_THZ / (2 * np.pi)
        Q = k.real / (-2 * k.imag)
        order = np.argsort(np.abs(f - f_thz))
        return f[order], Q[order], X[:, order]

    def eig_residual(self, lam: complex, x: np.ndarray) -> float:
        """‖(K − λW) x‖ / ‖K x‖, K = CᵀRC, W = mass ε: how far (λ, x) is from an eigenpair of the factorised structure."""
        Kx = self.C.T @ (self.R * (self.C @ x))
        return float(np.linalg.norm(Kx - lam * self.mass * self.eps_vec * x) / np.linalg.norm(Kx))

    def eig_adjoint(self, lam: complex, x: np.ndarray, c: np.ndarray, rtol: float = 1e-10, maxiter: int = 300) -> tuple[np.ndarray, dict]:
        """
        The adjoint of a functional of an eigenvector: y with (K − λW) y = c and yᵀW x = 0, for K x = λ W x (K = CᵀRC, W = mass ε, both complex symmetric) and c bilinearly orthogonal to x (cᵀx = 0, which holds for the
        cotangent of any g(x) invariant under x → αx, α complex). Then for a design change dg = Re(yᵀ(dλ W + λ dW) x) = Re(λ yᵀ dW x), the dλ term dropping out by the pinning; see research/cavity_inverse.py.
        B = K − λW is singular along x, and the held factorisation M = K − σW is at σ = k² ≠ λ. GMRES on P M⁻¹ B with P = I − x xᵀW / xᵀWx: M⁻¹B leaves range P = {v : xᵀW v = 0} invariant (xᵀW M⁻¹ = μ xᵀ with
        M⁻¹W x = μ x, and xᵀB = 0), and there its eigenvalues are (λ_j − λ)/(λ_j − σ) over the other modes j, so 1 but for the few modes near σ and λ, and none is 0. The right-hand side P M⁻¹ c lies in range P because
        cᵀx = 0. One LU solve and one curl product per iteration. Returns y (projected once more) and dict(iters, time, res = ‖B y − c‖/‖c‖, orth = |cᵀx|/(‖c‖‖x‖)).
        """
        t0 = time.perf_counter()
        Wx = self.mass * self.eps_vec * x
        n = x @ Wx

        def proj(v):
            return v - x * ((Wx @ v) / n)

        def B(v):
            return self.C.T @ (self.R * (self.C @ v)) - lam * self.mass * self.eps_vec * v

        op = spla.LinearOperator((self.n, self.n), matvec=lambda v: proj(self.lu.solve(B(v))), dtype=complex)
        rhs = proj(self.lu.solve(c))
        its = [0]

        def count(_):
            its[0] += 1

        y, info = spla.gmres(op, rhs, rtol=rtol, atol=0.0, restart=maxiter, maxiter=1, callback=count, callback_type="pr_norm")
        y = proj(y)
        cn = np.linalg.norm(c)
        return y, dict(iters=its[0], info=int(info), time=time.perf_counter() - t0, res=float(np.linalg.norm(B(y) - c) / cn), orth=float(abs(c @ x) / (cn * np.linalg.norm(x))))

    # --- port -----------------------------------------------------------------------------------

    def strip_mode(self, j0: int, n_guess: float = 3.4) -> dict:
        """
        The fundamental mode in this sector of a guide uniform along y around node j0, travelling towards −y with unit power (half cross-section, Z0 = 1): the discrete cross-section problem with ∂_y → iτ, with τ related to the Bloch phase by
        τ = 2 sin(θ/2)/Δy, so that it is an exact solution of the discrete A wherever ε does not change along y. Returns E_x, E_z on the y-nodes, E_y, H_x, H_z on the y-segments, θ, n_eff.
        """
        nx, nz, k0 = self.nx, self.nz, self.k0
        sc = self.k2 / k0**2  # the drive's width, as a loss on ε
        ex = self.eps_x[:, j0 - 1, :] * sc
        ez = self.eps_z[:, j0 - 1, :] * sc
        ey = self.eps_y[:, j0 - 1, :] * sc
        Gx = sp.diags(1 / self.lxS) @ _difference(nx)  # x-nodes → x-segments
        Dx = -sp.diags(1 / self.lxN) @ _difference(nx).T  # x-segments → x-nodes
        Gz0 = sp.diags([-np.ones(nz - 1), np.ones(nz - 1)], [0, 1], shape=(nz - 1, nz), format="csr")
        Gz = sp.diags(1 / self.lzS) @ Gz0  # z-nodes → z-segments
        Dz = -sp.diags(1 / self.lzN) @ Gz0.T  # z-segments → z-nodes
        I = sp.identity
        nnode, nseg = nx - 1, nx
        dz_ey = sp.kron(I(nnode), Gz, format="csr")
        dx_ey = sp.kron(Gx, I(nz), format="csr")
        dz_ex = sp.kron(I(nseg), Gz, format="csr")
        dx_ez = sp.kron(Gx, I(nz - 1), format="csr")
        dz_hx = sp.kron(I(nnode), Dz, format="csr")
        dx_hz = sp.kron(Dx, I(nz), format="csr")
        dz_hy = sp.kron(I(nseg), Dz, format="csr")
        dx_hy = sp.kron(Dx, I(nz - 1), format="csr")
        Ex, Ez, Ey = sp.diags(ex.ravel()), sp.diags(ez.ravel()), sp.diags(1 / ey.ravel())
        hy_ex, hy_ez = dz_ex / (1j * k0), -dx_ez / (1j * k0)
        # iτ (H_x, H_z) = B (E_x, E_z)
        B = sp.bmat([[dx_hy @ hy_ex, 1j * k0 * Ez + dx_hy @ hy_ez], [-1j * k0 * Ex + dz_hy @ hy_ex, dz_hy @ hy_ez]], format="csc")
        # E_y from (H_x, H_z)
        ey_hx, ey_hz = (1j / k0) * Ey @ dz_hx, -(1j / k0) * Ey @ dx_hz
        # iτ (E_x, E_z) = D (H_x, H_z)
        D = sp.bmat([[dx_ey @ ey_hx, -1j * k0 * I(nseg * nz) + dx_ey @ ey_hz], [1j * k0 * I(nnode * (nz - 1)) + dz_ey @ ey_hx, dz_ey @ ey_hz]], format="csc")
        M = (D @ B).tocsc()
        mu, V = spla.eigs(M, k=1, sigma=-((n_guess * k0) ** 2), which="LM", tol=1e-10)
        tau = np.sqrt(-mu[0] + 0j)
        tau = -tau if tau.real > 0 else tau  # towards −y; with Q_d this decays along −y
        T = V[:, 0]
        n_ex = nseg * nz
        W = B @ T / (1j * tau)
        n_hx = nnode * (nz - 1)
        Hx, Hz = W[:n_hx], W[n_hx:]
        Ey_ = ey_hx @ Hx + ey_hz @ Hz
        Ex_, Ez_ = T[:n_ex], T[n_ex:]
        # power through the half cross-section, along −y
        a_x = (np.real(self.lxS)[:, None] * np.real(self.lzN)[None, :]).ravel()
        a_z = (np.real(self.lxN)[:, None] * np.real(self.lzS)[None, :]).ravel()
        theta = 2 * np.arcsin(tau * self.dy / 2)
        # E_t on the nodes and H_t on the segments are half a cell apart, so the discrete power that is conserved carries cos(θ/2) (`flux_y`)
        P = -np.real(np.sum(a_z * Ez_ * np.conj(Hx)) - np.sum(a_x * Ex_ * np.conj(Hz))) * np.cos(np.real(theta) / 2)
        s = 1 / np.sqrt(abs(P))
        return dict(Ex=(Ex_ * s).reshape(nseg, nz), Ez=(Ez_ * s).reshape(nnode, nz - 1), Ey=(Ey_ * s).reshape(nnode, nz), Hx=(Hx * s).reshape(nnode, nz - 1), Hz=(Hz * s).reshape(nseg, nz), theta=theta, tau=tau, n_eff=abs(theta.real) / self.dy / np.real(k0), P_sign=np.sign(P))

    def port_source(self, j0: int, mode: dict | None = None, reach: int = 3) -> tuple[np.ndarray, dict]:
        """
        b for a mode launched at y-node j0 towards −y: total field for y ≤ y_j0, scattered field above. b = (Q A − A Q) e_inc with Q the total-field mask and e_inc the mode's line integrals near the plane.
        Needs set_frequency and set_eps. ε must not change along y within `reach` nodes of j0.
        """
        mode = mode or self.strip_mode(j0)
        th = mode["theta"]
        ny = self.ny
        ex = np.zeros(self.shape_x, complex)
        ey = np.zeros(self.shape_y, complex)
        ez = np.zeros(self.shape_z, complex)
        for j in range(j0 - reach, j0 + reach + 1):  # node j; e_x, e_z index j − 1 (nodes 1 … ny − 1)
            ex[:, j - 1, :] = mode["Ex"] * self.lxS[:, None] * np.exp(1j * th * (j - j0))
            ez[:, j - 1, :] = mode["Ez"] * self.lzS[None, :] * np.exp(1j * th * (j - j0))
        for j in range(j0 - reach, j0 + reach):  # segment j + ½
            ey[:, j, :] = mode["Ey"] * self.lyS[j] * np.exp(1j * th * (j + 0.5 - j0))
        e_inc = np.r_[ex.ravel(), ey.ravel(), ez.ravel()]
        qx = np.zeros(self.shape_x, bool)
        qy = np.zeros(self.shape_y, bool)
        qz = np.zeros(self.shape_z, bool)
        qx[:, : j0, :] = True  # nodes 1 … j0
        qz[:, : j0, :] = True
        qy[:, : j0, :] = True  # segments below node j0
        Q = np.r_[qx.ravel(), qy.ravel(), qz.ravel()]
        b = Q * self.apply(e_inc) - self.apply(Q * e_inc)
        # only the rows at the plane survive; the truncation of e_inc at ±reach cancels because Q is 1 or 0 throughout there
        return b, mode

    # --- a current source in the structure ------------------------------------------------------------

    def current_source(self, y0: float = 0.0, width: float = 0.0) -> np.ndarray:
        """
        b for an x-directed current in the slab on the axis x = 0, where this sector's E_x is even: unit peak density with a Gaussian profile of rms `width` (µm) in the plane around (0, y0), or the one edge at the y-node nearest y0 when width is 0,
        uniform across the slab's thickness. b = i k0 I per edge (Z0 = 1), I the current through the edge's dual face, so the source delivers P = Im(bᴴ e)/k0 (`source_power`) in flux_y's units. Needs set_frequency; keep it clear of the absorbers.
        """
        nx, ny, ns = self.nx, self.ny, self.zg.n_slab
        xs = (np.arange(nx) + 0.5) * self.dx
        yn = (-self.Ly / 2 + np.arange(1, ny) * self.dy) if self.y_wall is None else np.arange(ny - self.nyN, ny) * self.dy
        if width > 0:
            prof = np.exp(-(xs[:, None] ** 2 + (yn[None, :] - y0) ** 2) / (2 * width**2))
            prof[prof < 1e-8] = 0.0
        else:
            prof = np.zeros((nx, self.nyN))
            prof[0, np.argmin(np.abs(yn - y0))] = 1.0
        bx = np.zeros(self.shape_x, complex)
        bx[:, :, :ns] = 1j * self.k0 * prof[:, :, None] * (self.lyN[None, :, None] * self.lzN[None, None, :ns])
        return np.r_[bx.ravel(), np.zeros(self.n_y + self.n_z, complex)]

    def source_power(self, b: np.ndarray, x: np.ndarray) -> float:
        """The power a current source b delivers to its solution x: −Re ∫ J*·E = Im(bᴴ x)/k0 (Z0 = 1, the half domain). With U = Σ ε|E|²dV, k0 U/P is the Q of what it drives: exactly Q_d in a lossless closed box."""
        return float(np.imag(np.vdot(b, x)) / np.real(self.k0))

    def energy(self, x: np.ndarray, mask_cells: np.ndarray | None = None) -> float:
        """U = Σ ε|E|²dV over the half domain, or over the (nx, ny) cells of a mask (`energy_masks`)."""
        wx, wy, wz = self.energy_masks(np.ones((self.nx, self.ny)) if mask_cells is None else mask_cells)
        ex, ey, ez = self.split(x)
        return float(np.sum(self.eps_x * wx * np.abs(ex) ** 2) + np.sum(self.eps_y * wy * np.abs(ey) ** 2) + np.sum(self.eps_z * wz * np.abs(ez) ** 2))

    def fields(self, x: np.ndarray):
        """E_x, E_y, E_z on their edges (physical lengths; inside the absorbers they are the stretched coordinate's)."""
        ex, ey, ez = self.split(x)
        return ex / self.lxS[:, None, None], ey / self.lyS[None, :, None], ez / self.lzS[None, None, :]

    def flux_y(self, x: np.ndarray, j: int) -> float:
        """Re ∫ (E × H*)·ŷ over the half cross-section at segment j + ½ (Z0 = 1), positive towards +y: E_t averaged from the nodes to the segment, H_t from the curl there."""
        k0 = np.real(self.k0)
        F = self.C @ x
        n1 = (self.nx - 1) * self.ny * (self.nz - 1)
        n2 = self.nx * (self.ny - 1) * (self.nz - 1)
        fx = F[:n1].reshape(self.nx - 1, self.ny, self.nz - 1)
        fz = F[n1 + n2 :].reshape(self.nx, self.ny, self.nz)
        area_hx = self.lyS[j] * self.lzS[None, :]  # face of H_x at (x-node, y-seg, z-seg)
        area_hz = self.lxS[:, None] * self.lyS[j]
        hx = fx[:, j, :] / area_hx / (1j * k0)
        hz = fz[:, j, :] / area_hz / (1j * k0)
        Ex, _, Ez = self.fields(x)
        ez = 0.5 * (Ez[:, j - 1, :] + Ez[:, j, :]) if 0 < j < self.ny - 1 else Ez[:, min(j, self.ny - 2), :]
        ex = 0.5 * (Ex[:, j - 1, :] + Ex[:, j, :]) if 0 < j < self.ny - 1 else Ex[:, min(j, self.ny - 2), :]
        wz = np.real(self.lxN)[:, None] * np.real(self.lzS)[None, :]
        wx = np.real(self.lxS)[:, None] * np.real(self.lzN)[None, :]
        return float(np.real(np.sum(wz * ez * np.conj(hx)) - np.sum(wx * ex * np.conj(hz))))

    # --- the field on another solver's voxels --------------------------------------------------------

    def voxel_tables(self, nz_v: int, height: float, z_cover: np.ndarray) -> dict:
        """Interpolation in z to the voxels of optics_pwe.slab_cell (heights K·dz, wrapped): E_x, E_y between the ζ_k (even in z), D_z between the faces (odd, zero at z = 0). z_cover: the slab's share of each voxel."""
        nz = self.nz
        dz_v = height / nz_v
        z = np.arange(nz_v) * dz_v
        z = np.where(z > height / 2, z - height, z)
        az = np.abs(z)
        k1 = np.clip(np.searchsorted(self.zeta, az), 0, nz - 1)
        k0 = np.clip(k1 - 1, 0, nz - 1)
        dk = self.zeta[k1] - self.zeta[k0]
        t = np.where(dk > 0, (az - self.zeta[k0]) / np.where(dk > 0, dk, 1), 0.0)
        t = np.clip(t, 0, 1)
        faces = self.z_edges[: nz]  # the first is z = 0, where D_z vanishes
        m1 = np.clip(np.searchsorted(faces, az), 1, nz - 1)
        m0 = m1 - 1
        u = np.clip((az - faces[m0]) / (faces[m1] - faces[m0]), 0, 1)
        return {k: jnp.asarray(v) for k, v in dict(k0=k0, k1=k1, t=t, m0=m0, m1=m1, u=u, sign=np.where(z < 0, -1.0, 1.0), g=np.asarray(z_cover)).items()}

    def lift(self, x, ex, ey, ez, tab: dict):
        """
        e ∝ D (3, 2 nx, ny, nz_v) on the voxel columns of the whole domain, x ≥ 0 mirrored by the electric wall: D_x even, D_y and D_z odd in x. x: the solution; ex, ey, ez: the ε maps (`eps_maps`). JAX, linear in x.
        Between z cells E_x and E_y are interpolated, being continuous at the slab's face, then D = (g ε + (1 − g) ε_bg) E per voxel; D_z, continuous there, is interpolated itself. Then the edges are averaged to the cell centres.
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        Ex = x[: self.n_x].reshape(self.shape_x) / jnp.asarray(self.lxS)[:, None, None]
        Ey = x[self.n_x : self.n_x + self.n_y].reshape(self.shape_y) / jnp.asarray(self.lyS)[None, :, None]
        Ez = x[self.n_x + self.n_y :].reshape(self.shape_z) / jnp.asarray(self.lzS)[None, None, :]
        k0, k1, t, g = tab["k0"], tab["k1"], tab["t"], tab["g"]
        eb = self.eps_bg
        Dx = ((1 - t) * Ex[:, :, k0] + t * Ex[:, :, k1]) * (g * ex[:, :, None] + (1 - g) * eb)
        Dy = ((1 - t) * Ey[:, :, k0] + t * Ey[:, :, k1]) * (g * ey[:, :, None] + (1 - g) * eb)
        ns = self.zg.n_slab
        kf, wi, wo = self._face_weights()
        layers = [ez] * (ns - 1) + [1 / (wi / ez + wo / eb)] + [jnp.full_like(ez, eb)] * (nz - 1 - ns)
        ez3 = jnp.stack(layers, axis=2)
        D = ez3 * Ez
        D = jnp.concatenate([jnp.zeros_like(D[:, :, :1]), D], axis=2)  # faces: z = 0, then the z_{k+1}
        m0, m1, u = tab["m0"], tab["m1"], tab["u"]
        Dz = ((1 - u) * D[:, :, m0] + u * D[:, :, m1]) * tab["sign"]
        if self.y_wall == "pmc":  # nodes from y = 0: only the top wall is padded
            pad_y = lambda a: jnp.concatenate([a, jnp.zeros_like(a[:, :1])], axis=1)  # noqa: E731
        else:
            pad_y = lambda a: jnp.concatenate([jnp.zeros_like(a[:, :1]), a, jnp.zeros_like(a[:, :1])], axis=1)  # noqa: E731
        pad_x = lambda a: jnp.concatenate([jnp.zeros_like(a[:1]), a, jnp.zeros_like(a[:1])], axis=0)  # noqa: E731
        Dx = pad_y(Dx)
        Dx = 0.5 * (Dx[:, :-1] + Dx[:, 1:])
        Dy = pad_x(Dy)
        Dy = 0.5 * (Dy[:-1] + Dy[1:])
        Dz = pad_y(pad_x(Dz))
        Dz = 0.25 * (Dz[:-1, :-1] + Dz[1:, :-1] + Dz[:-1, 1:] + Dz[1:, 1:])
        Dx, Dy, Dz = jnp.concatenate([Dx[::-1], Dx], axis=0), jnp.concatenate([-Dy[::-1], Dy], axis=0), jnp.concatenate([-Dz[::-1], Dz], axis=0)
        if self.y_wall is not None:  # y ≥ 0 mirrored: "pmc" D_x, D_z even and D_y odd in y, "pec" the opposite
            p = 1.0 if self.y_wall == "pmc" else -1.0
            Dx, Dy, Dz = (jnp.concatenate([s * D[:, ::-1], D], axis=1) for s, D in ((p, Dx), (-p, Dy), (p, Dz)))
        return jnp.stack([Dx, Dy, Dz])

    # --- energy and derivatives with respect to ε -------------------------------------------------

    def energy_masks(self, mask_cells: np.ndarray):
        """Real edge weights (ε|e|²·weight = ε|E|²dV, half domain) restricted to a mask on the (nx, ny) cells: an edge counts if the cells either side of it do."""
        m = np.asarray(mask_cells, float)
        if self.y_wall == "pmc":  # the y = 0 nodes: the cell below is the mirror image of the one above
            m = np.concatenate([m[:, :1], m], axis=1)
        mx = 0.5 * (m[:, :-1] + m[:, 1:])
        my = 0.5 * (m[:-1, :] + m[1:, :])[:, -self.ny :]
        mz = 0.25 * (m[:-1, :-1] + m[1:, :-1] + m[:-1, 1:] + m[1:, 1:])
        wx = np.real(self.mass_x) * mx[:, :, None]
        wy = np.real(self.mass_y) * my[:, :, None]
        wz = np.real(self.mass_z) * mz[:, :, None]
        return wx, wy, wz

    def eps2d_bilinear(self, y: np.ndarray, x: np.ndarray):
        """∂/∂(ε maps) of Re(yᵀ (−dA) x) = Re(k² Σ mass yᵀ x dε): (g_x, g_y, g_z), summed over the slab's cells, for the set_eps above."""
        yx, yy, yz = self.split(y)
        xx, xy, xz = self.split(x)
        ns = self.zg.n_slab
        gx = np.real(self.k2 * np.sum((self.mass_x * yx * xx)[:, :, :ns], axis=2))
        gy = np.real(self.k2 * np.sum((self.mass_y * yy * xy)[:, :, :ns], axis=2))
        gz = np.real(self.k2 * np.sum((self.mass_z * yz * xz)[:, :, : ns - 1], axis=2))
        k, wi, _ = self._face_weights()
        ez2d = self.eps_z[:, :, 0] if ns > 1 else None
        d = (self.eps_z[:, :, k] ** 2) * wi / ez2d**2
        gz = gz + np.real(self.k2 * (self.mass_z[:, :, k] * yz[:, :, k] * xz[:, :, k])) * d
        return gx, gy, gz


__all__ = ["ComplexSymmetricLU", "SlabFDFD", "ZGrid"]
