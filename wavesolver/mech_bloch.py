"""
Mechanical Bloch modes of a periodic membrane by finite elements on a lattice-aligned grid.

The unit cell is meshed with n1 × n2 parallelogram elements along a1, a2 (n1 = n2 for a
hexagonal cell; a line-defect supercell takes n2 ≫ n1), so every element has the same shape
and one reference element matrix serves the whole mesh. Each element carries a
solid fraction f (from `geometry.solid_fraction`): its stiffness and mass are both scaled by f,
so partly filled elements keep the local wave speed, and empty elements are dropped.

Models:
- "plane_stress", "plane_strain": 2D, in-plane displacement, bilinear (Q1) elements.
- "slab_p": slab of thickness t extruded from the 2D pattern. In-plane Q1 elements; through
  the thickness each displacement component is a sum of Legendre polynomials P_m(2z/t) of one
  parity, so the mesh stays 2D. parity "even" (u_x, u_y even in z, u_z odd) holds the
  symmetric modes, including the breathing modes; "odd" holds the flexural modes. The
  integration through the thickness is exact, as the material does not vary with z.
- "slab": the same slab meshed with nz layers of trilinear elements over half the thickness,
  with the mirror plane at z = 0. Slower; kept as a cross-check of "slab_p".
- "stack": layers of different materials and in-plane patterns (e.g. LN on Si), without the
  z-mirror. Through each layer a hierarchical p-element (`lobatto`): linear functions shared
  with the neighbouring layers keep u continuous at the interfaces, where the strain has a
  kink that one Legendre series over the whole stack would resolve poorly. All three
  components use every function, so modes of both z-parities come out of one solve.

Bloch periodicity u(x + R) = exp(i k·R) u(x) enters through the element nodes that wrap
around the cell. Each stiffness and mass entry couples two nodes whose wrap counts differ by
(Δ1, Δ2) ∈ {-1, 0, 1}², and picks up exp(i (Δ1 k·a1 + Δ2 k·a2)). The entries are grouped by
(Δ1, Δ2) once, so K(k) and M(k) cost one weighted sum of nine arrays per k.

Units inside: µm, stiffness in 1e11 Pa, density in 1e3 kg/m³, so ω = 1e10 √λ rad/s.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial import legendre

from .geometry import HexLattice, Lattice2D
from .linalg import factorize
from .materials import Elastic

C_UNIT = 1e11
RHO_UNIT = 1e3
OMEGA_UNIT = 1e10  # sqrt(C_UNIT / (RHO_UNIT * 1e-12 m²))
GAUSS2 = (1 + np.array([-1, 1]) / np.sqrt(3)) / 2  # 2-point Gauss on [0, 1]


def plane_stress(C: np.ndarray) -> np.ndarray:
    """In-plane (xx, yy, xy) stiffness with σ_zz = σ_yz = σ_xz = 0."""
    a, b = [0, 1, 5], [2, 3, 4]
    return C[np.ix_(a, a)] - C[np.ix_(a, b)] @ np.linalg.solve(C[np.ix_(b, b)], C[np.ix_(b, a)])


def plane_strain(C: np.ndarray) -> np.ndarray:
    a = [0, 1, 5]
    return C[np.ix_(a, a)]


def _q1_shape(corners: np.ndarray, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Q1 shape functions and their reference gradients at xi ∈ [0, 1]^dim."""
    factors = np.where(corners == 1, xi, 1 - xi)
    N = factors.prod(axis=1)
    dN = np.empty(corners.shape)
    for d in range(corners.shape[1]):
        dN[:, d] = np.delete(factors, d, axis=1).prod(axis=1) * np.where(corners[:, d] == 1, 1.0, -1.0)
    return N, dN


def _strain_matrix(grads: np.ndarray, comps: np.ndarray, voigt: int) -> np.ndarray:
    """
    Strain-displacement matrix for DOFs whose displacement is along comps[j] with gradient
    grads[j] (3-vector; z-derivative zero in 2D). Voigt 3: (xx, yy, xy); 6: full.
    """
    B = np.zeros((voigt, len(comps)))
    gx, gy, gz = grads.T
    ux, uy, uz = comps == 0, comps == 1, comps == 2
    if voigt == 3:
        B[0] = gx * ux
        B[1] = gy * uy
        B[2] = gy * ux + gx * uy
    else:
        B[0] = gx * ux
        B[1] = gy * uy
        B[2] = gz * uz
        B[3] = gz * uy + gy * uz
        B[4] = gz * ux + gx * uz
        B[5] = gy * ux + gx * uy
    return B


def q1_element(J: np.ndarray, D: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Stiffness and consistent mass of a Q1 element with constant Jacobian J (columns are the
    edge vectors). Nodes follow itertools.product((0, 1), repeat=dim); DOFs are node-major.
    """
    dim = J.shape[0]
    corners = np.array(list(itertools.product((0, 1), repeat=dim)))
    detJ = abs(np.linalg.det(J))
    Jinv_T = np.linalg.inv(J).T
    comps = np.tile(np.arange(dim), len(corners))
    nd = len(comps)
    Ke, Me = np.zeros((nd, nd)), np.zeros((nd, nd))
    for xi in itertools.product(GAUSS2, repeat=dim):
        N, dN = _q1_shape(corners, np.array(xi))
        g = np.zeros((len(corners), 3))
        g[:, :dim] = dN @ Jinv_T.T
        B = _strain_matrix(np.repeat(g, dim, axis=0), comps, D.shape[0])
        Nd = np.repeat(N, dim)
        w = detJ / 2**dim
        Ke += w * B.T @ D @ B
        Me += w * rho * np.outer(Nd, Nd) * (comps[:, None] == comps[None, :])
    return Ke, Me


def thickness_basis(order: int, parity: str) -> list[tuple[int, int]]:
    """(component, Legendre degree) pairs: u_x, u_y take degrees of the slab's parity, u_z the other."""
    even = parity == "even"
    basis = []
    for comp in range(3):
        start = 0 if (comp < 2) == even else 1
        basis += [(comp, m) for m in range(start, order + 1, 2)]
    return basis


def slab_p_element(J: np.ndarray, C: np.ndarray, rho: float, thickness: float, basis: list[tuple[int, int]]):
    """
    Stiffness and mass of an in-plane Q1 element (Jacobian J, 2 × 2) carrying the thickness
    basis on every node, over the full thickness. DOFs are node-major, then basis order.
    """
    h = thickness / 2
    degs = [m for _, m in basis]
    zq, zw = legendre.leggauss(max(degs) + 2)
    P = np.array([legendre.legval(zq, np.eye(m + 1)[m]) for m in degs])  # (nb, nq)
    dP = np.array([legendre.legval(zq, legendre.legder(np.eye(m + 1)[m])) for m in degs]) / h
    return _extruded_element(J, C, rho, np.array([c for c, _ in basis]), P, dP, h * zw)


def lobatto(order: int, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    1D hierarchical p-element on s ∈ [-1, 1]: values and s-derivatives (order + 1, len(s)) of
    (1 - s)/2, the bubbles (P_m - P_{m-2}) / √(2(2m - 1)) for m = 2..order, then (1 + s)/2.
    """
    V = [(1 - s) / 2]
    dV = [-np.ones_like(s) / 2]
    for m in range(2, order + 1):
        c = (np.eye(m + 1)[m] - np.eye(m + 1)[m - 2]) / np.sqrt(2 * (2 * m - 1))
        V.append(legendre.legval(s, c))
        dV.append(legendre.legval(s, legendre.legder(c)))
    V.append((1 + s) / 2)
    dV.append(np.ones_like(s) / 2)
    return np.array(V), np.array(dV)


def stack_layer_element(J: np.ndarray, C: np.ndarray, rho: float, thickness: float, order: int):
    """
    Stiffness and mass of one layer of a stack: in-plane Q1 element (Jacobian J) times the
    `lobatto` functions through the layer, all three components on each. DOFs are node-major,
    then thickness function, then component.
    """
    h = thickness / 2
    zq, zw = legendre.leggauss(order + 2)
    V, dV = lobatto(order, zq)
    return _extruded_element(J, C, rho, np.tile(np.arange(3), order + 1), np.repeat(V, 3, axis=0), np.repeat(dV, 3, axis=0) / h, h * zw)


def _extruded_element(J, C, rho, comps_b, Z, dZ, zw):
    """
    Q1 element (Jacobian J, 2 × 2) whose nodes carry thickness functions: DOF b of a node is
    displacement along comps_b[b] with z-profile Z[b] and z-derivative dZ[b], tabulated at
    quadrature points of weights zw (in µm).
    """
    corners = np.array(list(itertools.product((0, 1), repeat=2)))
    detJ = abs(np.linalg.det(J))
    Jinv_T = np.linalg.inv(J).T
    comps = np.tile(comps_b, len(corners))
    nd = len(comps)
    Ke, Me = np.zeros((nd, nd)), np.zeros((nd, nd))
    for xi in itertools.product(GAUSS2, repeat=2):
        N, dN = _q1_shape(corners, np.array(xi))
        gxy = dN @ Jinv_T.T  # (4, 2)
        for q in range(len(zw)):
            phi = np.outer(N, Z[:, q]).ravel()  # value of each DOF's shape function
            g = np.zeros((nd, 3))
            g[:, 0] = np.outer(gxy[:, 0], Z[:, q]).ravel()
            g[:, 1] = np.outer(gxy[:, 1], Z[:, q]).ravel()
            g[:, 2] = np.outer(N, dZ[:, q]).ravel()
            B = _strain_matrix(g, comps, 6)
            w = detJ / 4 * zw[q]
            Ke += w * B.T @ C @ B
            Me += w * rho * np.outer(phi, phi) * (comps[:, None] == comps[None, :])
    return Ke, Me


@dataclass
class Layer:
    """One layer of a "stack" model, listed bottom to top. fraction: (n1, n2) as in BlochSolver."""

    material: Elastic
    thickness: float
    fraction: np.ndarray


@dataclass
class ElementPart:
    """
    The kept elements of one part (the slab, or one layer of a stack), for sensitivities:
    the element matrices Ke, Me (scaled by the fraction in K, M), the fraction-map cell
    i·n2 + j of each element, its local DOFs (-1 where fixed by a mirror plane), and the wrap
    counts W1, W2 of each DOF's node. ∂K/∂f[cell] = Σ over the part's elements in that cell
    of Ke placed on edof with phases exp(i(W1 θ1 + W2 θ2)).
    """

    Ke: np.ndarray
    Me: np.ndarray
    cell: np.ndarray
    edof: np.ndarray
    W1: np.ndarray
    W2: np.ndarray


@dataclass
class _PhaseClass:
    d1: int
    d2: int
    index: np.ndarray
    K: np.ndarray
    M: np.ndarray


class BlochSolver:
    """
    Bloch eigenmodes of one unit cell.

    lattice: anything with `.vectors` (rows a1, a2), e.g. HexLattice or the Lattice2D of
    `geometry.line_defect_supercell`. fraction: (n1, n2) solid fraction from
    `geometry.solid_fraction`. self.shape = (n1, n2); self.n = n1 for a square grid (as used by
    wavesolver.mech_adjoint) and None otherwise.
    model: "plane_stress", "plane_strain", "slab_p", "slab" or "stack".
    thickness: slab thickness in µm. order: highest Legendre degree through the thickness
    ("slab_p"), or through each layer ("stack"). nz: element layers over the half thickness
    ("slab"). layers: the `Layer`s of a "stack", bottom to top, replacing fraction and
    material. `parts` keeps each part's kept elements (`ElementPart`) for the sensitivities
    in wavesolver.mech_adjoint.
    loads ("slab_p" only): extra layers (material, layer thickness µm, fraction map) added as
    parts 1, 2, … on the same σ_z-even thickness basis, with C and ρ scaled by t_layer/thickness:
    a membrane model of thin films on the slab (their in-plane stiffness and mass per area are
    exact; their offset from the mid-plane and the flexural coupling it causes are not). Part 0
    stays the slab material on `fraction`.
    """

    def __init__(
        self,
        lattice: HexLattice | Lattice2D,
        fraction: np.ndarray | None = None,
        material: Elastic | None = None,
        model: str = "plane_stress",
        thickness: float | None = None,
        parity: str = "even",
        order: int = 4,
        nz: int = 4,
        min_fraction: float = 1e-3,
        layers: list[Layer] | None = None,
        loads: list | None = None,
    ):
        self.lattice = lattice
        n1, n2 = (layers[0].fraction if model == "stack" else fraction).shape
        if model != "stack":
            C = material.C / C_UNIT
            rho = material.rho / RHO_UNIT
        A = lattice.vectors
        J2 = np.column_stack([A[0] / n1, A[1] / n2])
        nzl = 0
        # Each part: element matrices, the fraction scaling them, and the element's DOFs
        # within a node's dpn.
        if model == "slab":
            J = np.zeros((3, 3))
            J[:2, :2] = J2
            J[2, 2] = thickness / 2 / nz
            dpn, nzl = 3, nz
            parts = [(*q1_element(J, C, rho), fraction, np.arange(dpn))]
        elif model == "slab_p":
            basis = thickness_basis(order, parity)
            dpn = len(basis)
            parts = [(*slab_p_element(J2, C, rho, thickness, basis), fraction, np.arange(dpn))]
            for mat, t_l, frac_l in loads or []:
                s = t_l / thickness
                parts.append((*slab_p_element(J2, s * mat.C / C_UNIT, s * mat.rho / RHO_UNIT, thickness, basis), np.asarray(frac_l, float), np.arange(dpn)))
        elif model == "stack":
            # Layer l carries the thickness functions l·order .. (l+1)·order; the end ones
            # are shared with the neighbouring layers.
            dpn = 3 * (len(layers) * order + 1)
            parts = [
                (
                    *stack_layer_element(J2, L.material.C / C_UNIT, L.material.rho / RHO_UNIT, L.thickness, order),
                    L.fraction,
                    3 * l * order + np.arange(3 * (order + 1)),
                )
                for l, L in enumerate(layers)
            ]
        else:
            D = plane_stress(C) if model == "plane_stress" else plane_strain(C)
            dpn = 2
            parts = [(*q1_element(J2, D, rho), fraction, np.arange(dpn))]
        mesh_dim = 3 if nzl else 2
        # Kept for field evaluation (wavesolver.coupling): the thickness basis of each node's DOFs.
        self.model, self.thickness, self.parity = model, thickness, parity
        self.basis = basis if model == "slab_p" else None

        corners = np.array(list(itertools.product((0, 1), repeat=mesh_dim)))
        axes = [np.arange(n1), np.arange(n2)] + ([np.arange(nzl)] if nzl else [])
        el = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)
        gn = el[:, None, :] + corners[None]  # grid coordinates of element nodes
        w1, w2 = gn[..., 0] // n1, gn[..., 1] // n2
        node = (gn[..., 0] % n1) * n2 + gn[..., 1] % n2
        if nzl:
            node = node * (nzl + 1) + gn[..., 2]

        edof, W1, W2, kv, mv = [], [], [], [], []
        self.shape = (n1, n2)
        self.n = n1 if n1 == n2 else None
        self.parts: list[ElementPart] = []
        self.part_fractions = [np.asarray(frac, float) for _, _, frac, _ in parts]  # (n1, n2) per part (loads: fixed)
        for Ke, Me, frac, offset in parts:
            f = frac[el[:, 0], el[:, 1]]
            keep = f > min_fraction
            edof.append((node[keep][..., None] * dpn + offset).reshape(keep.sum(), -1))
            W1.append(np.repeat(w1[keep], len(offset), axis=1))
            W2.append(np.repeat(w2[keep], len(offset), axis=1))
            self.parts.append(ElementPart(Ke, Me, el[keep, 0] * n2 + el[keep, 1], edof[-1], W1[-1], W2[-1]))
            kv.append(f[keep, None, None] * Ke)
            mv.append(f[keep, None, None] * Me)
        edof, W1, W2, kv, mv = (np.concatenate(x) for x in (edof, W1, W2, kv, mv))

        # Global DOFs of the kept elements, less those fixed by the mirror plane.
        used = np.unique(edof)
        if nzl:
            comp = used % dpn
            on_mirror = (used // dpn) % (nzl + 1) == 0
            used = used[~(on_mirror & ((comp == 2) if parity == "even" else (comp < 2)))]
        local = np.full(edof.max() + 1, -1)
        local[used] = np.arange(len(used))
        edof = local[edof]
        for part in self.parts:
            part.edof = local[part.edof]
        self.ndof = len(used)
        self.dofs = used

        r, c = edof[:, :, None], edof[:, None, :]
        valid = (r >= 0) & (c >= 0)
        rows = np.broadcast_to(r, valid.shape)[valid]
        cols = np.broadcast_to(c, valid.shape)[valid]
        d1 = (W1[:, None, :] - W1[:, :, None])[valid]
        d2 = (W2[:, None, :] - W2[:, :, None])[valid]
        kv, mv = kv[valid], mv[valid]

        keys, inv = np.unique(rows.astype(np.int64) * self.ndof + cols, return_inverse=True)
        self.nnz = len(keys)
        self.indices = (keys % self.ndof).astype(np.int32)
        self.indptr = np.searchsorted(keys // self.ndof, np.arange(self.ndof + 1)).astype(np.int32)
        cls = (d1 + 1) * 3 + (d2 + 1)
        self.classes = []
        for k in np.unique(cls):
            sel = cls == k
            idx, sub = np.unique(inv[sel], return_inverse=True)
            self.classes.append(
                _PhaseClass(
                    int(k // 3 - 1),
                    int(k % 3 - 1),
                    idx,
                    np.bincount(sub, weights=kv[sel]),
                    np.bincount(sub, weights=mv[sel]),
                )
            )

    def matrices(self, k: np.ndarray) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """K(k), M(k) for a Bloch wavevector k in 1/µm."""
        theta = self.lattice.vectors @ np.asarray(k, dtype=float)
        Kd = np.zeros(self.nnz, dtype=complex)
        Md = np.zeros(self.nnz, dtype=complex)
        for pc in self.classes:
            phase = np.exp(1j * (pc.d1 * theta[0] + pc.d2 * theta[1]))
            Kd[pc.index] += phase * pc.K
            Md[pc.index] += phase * pc.M
        shape = (self.ndof, self.ndof)
        return (
            sp.csr_matrix((Kd, self.indices, self.indptr), shape=shape),
            sp.csr_matrix((Md, self.indices, self.indptr), shape=shape),
        )

    def solve(self, k: np.ndarray, n_bands: int = 10, shift_hz: float = -0.5e9, vectors: bool = False):
        """
        n_bands frequencies in Hz nearest shift_hz (signed, so a negative shift gives the lowest
        bands), by shift-invert. With a negative shift K - σM is Hermitian positive definite, so
        it is factorised by Cholesky.
        """
        K, M = self.matrices(k)
        sigma = np.sign(shift_hz) * (2 * np.pi * shift_hz / OMEGA_UNIT) ** 2
        solve = factorize(K - sigma * M, hpd=sigma < 0)
        OPinv = spla.LinearOperator(K.shape, matvec=solve, dtype=complex)
        lam, vec = spla.eigsh(K, k=n_bands, M=M, sigma=sigma, which="LM", OPinv=OPinv, tol=1e-8)
        order = np.argsort(lam.real)
        freqs = OMEGA_UNIT * np.sqrt(np.clip(lam.real[order], 0, None)) / (2 * np.pi)
        return (freqs, vec[:, order]) if vectors else freqs

    def bands(self, ks: np.ndarray, n_bands: int = 10, workers: int = 1, **kwargs) -> np.ndarray:
        """
        Frequencies (len(ks), n_bands) in Hz. With workers > 1 the k-points are split across
        processes. Limit BLAS to one thread per process (OMP_NUM_THREADS=1,
        VECLIB_MAXIMUM_THREADS=1 before numpy is imported): the eigensolver's small dense
        operations run several times slower when BLAS threads compete.
        """
        if workers <= 1:
            return np.array([self.solve(k, n_bands, **kwargs) for k in ks])
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers, initializer=_set_worker_solver, initargs=(self,)) as pool:
            return np.array(list(pool.map(_worker_solve, ks, [n_bands] * len(ks), [kwargs] * len(ks))))


_WORKER_SOLVER: BlochSolver | None = None


def _set_worker_solver(solver: BlochSolver) -> None:
    global _WORKER_SOLVER
    _WORKER_SOLVER = solver


def _worker_solve(k: np.ndarray, n_bands: int, kwargs: dict) -> np.ndarray:
    return _WORKER_SOLVER.solve(k, n_bands, **kwargs)
