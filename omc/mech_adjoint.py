"""
Sensitivities of mechanical Bloch eigenfrequencies to the element solid fractions.

`BlochSolver` assembles K(k) = Σ_e f_e P_e(k)ᴴ Ke P_e(k) and the same for M, where P_e(k)
gathers element e's DOFs from the global vector with the Bloch phases of the nodes that
wrap around the cell. Both are linear in f_e, so for an eigenpair K x = λ M x

    ∂λ/∂f_e = u_eᴴ (Ke - λ Me) u_e / (xᴴ M x),    u_e = P_e(k) x,

a quadratic form of the eigenvector restricted to one element: no matrix is assembled per
element, and the whole gradient costs one gather and two small dense products over all
elements. With several parts (the layers of a "stack"), each part has its own fraction map
and gets its own gradient.

Degenerate eigenvalues (at Γ, K and M of a C3v cell, broken only slightly by the skewed
mesh) are not differentiable one by one. For a cluster of m (near-)degenerate bands the
directional derivatives along a design change δf are the eigenvalues of the m × m matrix
Σ_e δf_e U_eᴴ (Ke - λ̄ Me) U_e (`cluster_derivatives`); `cluster_matrix` keeps the split
λ_i, so diag(λ) + h·A also follows a cluster that the mesh has split slightly. The
cluster's mean is smooth, and its gradient is the mean of the bands' individual formulas. `sensitivities`
returns that mean for every band of a cluster (degenerate="mean"), or the raw per-band
formula (degenerate="raw").

Chain rule to shape parameters or a density field: dλ/dθ = Σ_e ∂λ/∂f_e ∂f_e/∂θ, with the
fraction map's vector-Jacobian product from `omc.parametrize`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .mech_bloch import OMEGA_UNIT, BlochSolver, ElementPart


@dataclass
class BandSensitivity:
    """
    Bands at one k and their gradients with respect to each part's fraction map.

    freqs: (nb,) Hz. dfreq: (nparts, nb, n, n) ∂f_band/∂fraction in Hz. clusters: index
    arrays of the (near-)degenerate groups. lam: eigenvalues in solver units. X: eigenvectors
    (ndof, nb), kept only when asked.
    """

    k: np.ndarray
    freqs: np.ndarray
    lam: np.ndarray
    dfreq: np.ndarray
    clusters: list[np.ndarray]
    X: np.ndarray | None = None


def element_vectors(part: ElementPart, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """U[e, a, v] = (P_e(k) X)[a, v]: eigenvectors gathered onto each element's DOFs, phased."""
    phase = np.exp(1j * (part.W1 * theta[0] + part.W2 * theta[1]))
    U = X[np.maximum(part.edof, 0)] * phase[..., None]
    U[part.edof < 0] = 0
    return U


def element_forms(part: ElementPart, U: np.ndarray, full: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Element quadratic forms of Ke and Me: (ne, nv) diagonal u_eᴴ Ke u_e (real), or with full
    the (ne, nv, nv) matrices U_eᴴ Ke U_e.
    """
    KU, MU = part.Ke @ U, part.Me @ U
    Uc = U.conj()
    if full:
        Uh = Uc.transpose(0, 2, 1)
        return Uh @ KU, Uh @ MU
    return np.einsum("eav,eav->ev", Uc, KU).real, np.einsum("eav,eav->ev", Uc, MU).real


def find_clusters(lam: np.ndarray, rtol: float) -> list[np.ndarray]:
    """Group sorted eigenvalues whose frequencies differ by less than rtol (relative)."""
    f = np.sqrt(np.clip(lam, 0, None))
    groups, start = [], 0
    for i in range(1, len(f) + 1):
        if i == len(f) or f[i] - f[i - 1] > rtol * max(f[i], 1e-12):
            groups.append(np.arange(start, i))
            start = i
    return groups


def sensitivities(
    solver: BlochSolver,
    k: np.ndarray,
    n_bands: int = 10,
    shift_hz: float = -0.5e9,
    degenerate: str = "mean",
    rtol: float = 1e-4,
    keep_vectors: bool = False,
) -> BandSensitivity:
    """
    Lowest bands at k and ∂f/∂fraction for every part. Elements dropped by the solver
    (fraction ≤ min_fraction) get zero gradient. rtol: relative frequency splitting below
    which bands count as one degenerate cluster.
    """
    freqs, X = solver.solve(k, n_bands, shift_hz=shift_hz, vectors=True)
    lam = (2 * np.pi * freqs / OMEGA_UNIT) ** 2
    theta = solver.lattice.vectors @ np.asarray(k, dtype=float)
    n = solver.n
    nb = len(freqs)
    dlam = np.zeros((len(solver.parts), nb, n * n))
    for p, part in enumerate(solver.parts):
        U = element_vectors(part, theta, X)
        eK, eM = element_forms(part, U)
        g = eK - lam[None] * eM  # (ne, nb)
        for b in range(nb):
            dlam[p, b] = np.bincount(part.cell, weights=g[:, b], minlength=n * n)
    _, M = solver.matrices(k)
    norm = np.einsum("iv,iv->v", X.conj(), M @ X).real
    dlam /= norm[None, :, None]
    clusters = find_clusters(lam, rtol)
    # f = Ω √λ / 2π, so df/dλ = f / 2λ; the zero-frequency acoustic modes at Γ get zero.
    scale = np.where(lam > 0, freqs / (2 * np.where(lam > 0, lam, 1)), 0.0)
    dfreq = dlam * scale[None, :, None]
    if degenerate == "mean":  # gradient of the cluster's mean frequency
        for c in clusters:
            if len(c) > 1:
                dfreq[:, c] = dfreq[:, c].mean(axis=1, keepdims=True)
    dfreq = dfreq.reshape(len(solver.parts), nb, n, n)
    return BandSensitivity(np.asarray(k, float), freqs, lam, dfreq, clusters, X if keep_vectors else None)


def cluster_matrix(solver: BlochSolver, sens: BandSensitivity, cluster: np.ndarray, dfrac: list[np.ndarray]) -> np.ndarray:
    """
    First-order change of the eigenvalues λ of a cluster along the fraction change dfrac
    (one (n, n) array per part), as an m × m Hermitian matrix in the M-orthonormal cluster
    basis: A_ij = x_iᴴ δK x_j - ½(λ_i + λ_j) x_iᴴ δM x_j. The cluster's eigenvalues at step
    h are those of diag(λ) + h A to first order (exactly the degenerate subspace formula
    when the λ_i coincide; its diagonal is the per-band formula). Needs keep_vectors.
    """
    theta = solver.lattice.vectors @ sens.k
    X = sens.X[:, cluster]
    _, M = solver.matrices(sens.k)
    Li = np.linalg.inv(np.linalg.cholesky(X.conj().T @ (M @ X)))
    X = X @ Li.conj().T  # M-orthonormal (ARPACK's vectors already are, up to round-off)
    lam = sens.lam[cluster]
    dK = np.zeros((len(cluster), len(cluster)), dtype=complex)
    dM = np.zeros_like(dK)
    for part, df in zip(solver.parts, dfrac):
        fK, fM = element_forms(part, element_vectors(part, theta, X), full=True)
        w = df.ravel()[part.cell]
        dK += np.einsum("e,evw->vw", w, fK)
        dM += np.einsum("e,evw->vw", w, fM)
    return dK - 0.5 * (lam[:, None] + lam[None, :]) * dM


def cluster_derivatives(solver: BlochSolver, sens: BandSensitivity, cluster: np.ndarray, dfrac: list[np.ndarray]) -> np.ndarray:
    """
    Directional derivatives (Hz per unit step) of the bands of an exactly degenerate cluster
    along dfrac: eigenvalues of `cluster_matrix`, converted at the cluster's mean λ.
    """
    lam_bar = sens.lam[cluster].mean()
    f_bar = OMEGA_UNIT * np.sqrt(lam_bar) / (2 * np.pi)
    return np.linalg.eigvalsh(cluster_matrix(solver, sens, cluster, dfrac)) * f_bar / (2 * lam_bar)


# -- batched over k, in worker processes ------------------------------------------------

_WORKER: dict = {}


def _worker_task(task):
    """task: (call id, task id, solver kwargs, ks, per-k rtol, n_bands, sensitivity kwargs)."""
    call, tid, kwargs, ks, rtols, n_bands, skw = task
    if _WORKER.get("call") != call:
        _WORKER.clear()
        _WORKER["call"] = call
    if tid not in _WORKER:
        _WORKER[tid] = BlochSolver(**kwargs)
    out = [sensitivities(_WORKER[tid], k, n_bands, rtol=r, **skw) for k, r in zip(ks, rtols)]
    return np.array([s.freqs for s in out]), np.array([s.dfreq for s in out])


class BandPool:
    """
    A persistent process pool evaluating bands and their fraction gradients over many k for
    several solvers at once (e.g. the two parities of a slab). Each worker builds a solver at
    most once per call and the k-points are split between workers. Use as a context manager;
    workers=1 runs in-process.
    """

    def __init__(self, workers: int = 2):
        self.workers = workers
        self.pool = ProcessPoolExecutor(workers) if workers > 1 else None
        self._calls = 0

    def __enter__(self) -> BandPool:
        return self

    def __exit__(self, *exc) -> None:
        if self.pool is not None:
            self.pool.shutdown()

    def submit(
        self, solvers: list[dict], ks: np.ndarray, n_bands: int, rtol: float | np.ndarray = 1e-4, **sens_kwargs
    ):
        """
        Start `run` without waiting: returns a callable that blocks for, and returns, the
        result. The caller can meanwhile work in the main process (e.g. the optical solve).
        Without a pool the work is done here, at submission.
        """
        self._calls += 1
        ks = np.asarray(ks, dtype=float)
        rtols = np.broadcast_to(np.asarray(rtol, dtype=float), len(ks))
        nchunk = max(1, self.workers // len(solvers)) if self.pool is not None else 1
        tasks = [
            (self._calls, t, kw, c, r, n_bands, sens_kwargs)
            for t, kw in enumerate(solvers)
            for c, r in zip(np.array_split(ks, nchunk), np.array_split(rtols, nchunk))
            if len(c)
        ]
        if self.pool is None:
            done = [_worker_task(t) for t in tasks]
            futures = None
        else:
            futures = [self.pool.submit(_worker_task, t) for t in tasks]

        def collect() -> list[tuple[np.ndarray, np.ndarray]]:
            results = done if futures is None else [f.result() for f in futures]
            out = []
            for t in range(len(solvers)):
                rs = [r for task, r in zip(tasks, results) if task[1] == t]
                out.append((np.concatenate([r[0] for r in rs]), np.concatenate([r[1] for r in rs])))
            return out

        return collect

    def run(
        self, solvers: list[dict], ks: np.ndarray, n_bands: int, rtol: float | np.ndarray = 1e-4, **sens_kwargs
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        For each BlochSolver(**kwargs) in solvers: freqs (nk, nb) in Hz and dfreq
        (nk, nparts, nb, n, n) from `sensitivities`. rtol: cluster tolerance, scalar or per k
        (e.g. only at Γ and K, where the degeneracies are symmetry-protected).
        """
        return self.submit(solvers, ks, n_bands, rtol, **sens_kwargs)()


# -- eigenvector adjoint (linear response) -------------------------------------------------


def eigenvector_adjoint_solver(K, M, lam: float, x: np.ndarray, pivot_thresh: float = 0.1):
    """
    For a simple eigenpair K x = λ M x (sparse Hermitian K, M), return solve(r) → z with

        (K − λ M) z = r − M x (xᴴ r)/(xᴴ M x),    xᴴ M z = 0,

    i.e. the M-orthogonal solution of the singular system restricted to the range (r ⊥ x is
    the case of interest: the gradient of a functional invariant under the scale of x). If
    dJ = Re(rᴴ dx) for an eigenvector change dx, then dJ = −Re(zᴴ (dK − λ dM) x): the
    adjoint of eigenvector-dependent functionals (omc.coupling_adjoint), one factorisation
    for any number of right-hand sides.

    The bordered system [[K − λM, Mx], [(Mx)ᴴ, 0]] [z; μ] = [r; 0] is nonsingular for a simple
    eigenvalue. It is factorised by SuperLU on a METIS ordering of K − λM with the border
    last; threshold pivoting swaps the border row in where K − λM is (near) singular. Real
    matrices (Γ point) are factorised in real arithmetic. One step of iterative refinement.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    A = (K - lam * M).tocsc()
    b = M @ x
    real = not (np.iscomplexobj(A.data) and np.any(A.data.imag)) and not np.any(np.imag(b))
    if real:
        A = sp.csc_matrix((np.real(A.data), A.indices, A.indptr), shape=A.shape)
        b = np.real(b)
    n = A.shape[0]
    B = sp.bmat([[A, sp.csc_matrix(b[:, None])], [sp.csc_matrix(b.conj()[None, :]), None]], format="csc")
    try:
        from sksparse.cholmod import metis

        pattern = sp.csc_matrix((np.ones(A.nnz), A.indices, A.indptr), shape=A.shape)
        p = np.concatenate([np.asarray(metis((pattern + pattern.T).tocsc())), [n]])
        spec = "NATURAL"
    except ImportError:
        p = np.arange(n + 1)
        spec = "MMD_AT_PLUS_A"
    inv = np.empty_like(p)
    inv[p] = np.arange(n + 1)
    lu = spla.splu(B[p][:, p].tocsc(), permc_spec=spec, diag_pivot_thresh=pivot_thresh, options=dict(SymmetricMode=True))
    xMx = np.vdot(x, b).real

    def raw(rr):
        if real and np.iscomplexobj(rr):
            return raw(rr.real) + 1j * raw(rr.imag)
        return lu.solve(np.ascontiguousarray(rr[p]))[inv]

    def solve(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r)
        r = r - b * (np.vdot(x, r) / xMx)
        rhs = np.concatenate([r, np.zeros(1, dtype=r.dtype)])
        s = raw(rhs)
        res = rhs - B @ s
        s = s + raw(res)
        z = s[:n]
        return z - x * (np.vdot(b, z) / xMx)

    return solve
