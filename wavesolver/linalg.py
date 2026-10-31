"""
Sparse factorisation backends behind one interface, so the solvers do not depend on which
is installed. CPU: CHOLMOD (scikit-sparse) for Hermitian positive definite systems, SuperLU
otherwise. A GPU backend (cuDSS) slots in here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:  # scikit-sparse >= 0.5
    from sksparse.cholmod import cho_factor as _cho_factor

    def _cholesky(A):
        # Supernodal is several times faster here than the simplicial factorisation CHOLMOD
        # picks by default. A factor object reused for a new matrix falls back to simplicial,
        # so each matrix gets a fresh factorisation.
        return _cho_factor(A, supernodal_mode="supernodal").solve

except ImportError:
    try:  # scikit-sparse 0.4
        from sksparse.cholmod import cholesky as _cholesky
    except ImportError:
        _cholesky = None


def factorize(A: sp.spmatrix, hpd: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    """Return x ↦ A⁻¹ x. hpd: A is Hermitian positive definite, so Cholesky applies."""
    A = A.tocsc()
    if hpd and _cholesky is not None:
        return _cholesky(A)
    options = dict(permc_spec="MMD_AT_PLUS_A", options=dict(SymmetricMode=True))
    if hpd:
        options["diag_pivot_thresh"] = 0.0
    return spla.splu(A, **options).solve


def factorize_indefinite(A: sp.spmatrix, pivot_thresh: float = 0.1) -> Callable[..., np.ndarray]:
    """
    Return solve(b, trans="N") ↦ A⁻¹ b (trans "T": A⁻ᵀ b, "H": A⁻ᴴ b) for a sparse symmetric
    (or Hermitian) indefinite matrix with a non-zero diagonal, such as the piezoelectric
    saddle-point system [[K - σM, C], [Cᴴ, -E]]. SuperLU in symmetric mode on a METIS
    nested-dissection ordering with threshold pivoting on the diagonal: `factorize`'s partial
    pivoting more than doubles the fill on these systems. A real A is factorised in real
    arithmetic (several times faster) and a complex right-hand side is split. Falls back to
    `factorize` without METIS (then trans is not supported). A GPU / MUMPS backend should
    take an LDLᵀ with Bunch-Kaufman pivoting here.
    """
    import gc

    gc.collect()  # release earlier factorisations still held in reference cycles
    A = A.tocsc()
    if np.iscomplexobj(A.data) and not np.any(A.data.imag):
        A = A.copy()
        A.data = A.data.real
    try:
        from sksparse.cholmod import metis
    except ImportError:
        f = factorize(A)
        return lambda b, trans="N": f(b)
    pattern = sp.csc_matrix((np.ones(A.nnz), A.indices, A.indptr), shape=A.shape)
    p = metis((pattern + pattern.T).tocsc())
    inv = np.empty_like(p)
    inv[p] = np.arange(len(p))
    lu = spla.splu(A[p][:, p].tocsc(), permc_spec="NATURAL", diag_pivot_thresh=pivot_thresh, options=dict(SymmetricMode=True))
    real = np.isrealobj(A.data)

    def solve(b: np.ndarray, trans: str = "N") -> np.ndarray:
        b = b[p]
        if real:
            trans = "T" if trans == "H" else trans
            if np.iscomplexobj(b):
                return (lu.solve(np.ascontiguousarray(b.real), trans) + 1j * lu.solve(np.ascontiguousarray(b.imag), trans))[inv]
        return lu.solve(b, trans)[inv]

    return solve
