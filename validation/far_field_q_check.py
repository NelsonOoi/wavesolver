"""
Checks of the far-field sum behind wavesolver.cavity_adjoint.far_field_q, no eigensolve:

- Larmor: a point current J in the plane (Ĵ independent of q) must radiate P = k0² |J|² / (12π) (units ε0 = μ0 = c = 1,
  ω = k0), through the cone weights, the shell projection and the q cells, for any oversampling.
- Cone edge: the weights' sum must stay smooth as k0 moves across the q grid.
- Slab: T(q) of a z-even current in a slab of ε̄ against the 1D wave equation with outgoing boundaries, TE and TM.

    pixi run python validation/far_field_q_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wavesolver.cavity_adjoint import _cone_weights, _far_terms  # noqa: E402


def cells(k0: float, dqx: float, dqy: float) -> tuple[np.ndarray, np.ndarray]:
    mx, my = int(np.ceil(k0 / dqx)) + 2, int(np.ceil(k0 / dqy)) + 2
    qx, qy = np.meshgrid(np.arange(-mx, mx + 1) * dqx, np.arange(-my, my + 1) * dqy, indexing="ij")
    return qx.ravel(), qy.ravel()


def radiated(k0: float, J: np.ndarray, dqx: float, dqy: float) -> float:
    """P with the q integral done as far_field_q does, for a point current at the origin (dV² |Ĵ|² = |J|²)."""
    qx, qy = cells(k0, dqx, dqy)
    W = _cone_weights(k0, qx, qy, dqx, dqy, n_y=16)
    Jq = np.broadcast_to(J[:, None, None], (3, len(qx), 1))
    T = np.asarray(_far_terms(jnp.asarray(Jq), jnp.asarray(qx), jnp.asarray(qy), jnp.zeros(1), k0))
    # Σ_q(2π/L grid) k0 / (8 k_z A) → k0 / 8 · ∫ d²q / (2π)² / k_z
    return float(k0 / 8 * dqx * dqy / (2 * np.pi) ** 2 * np.sum(W * T))


def wave_1d(k0: float, q: float, eps_bar: float, t: float, jq, jz, js, half: float = 4.0, n: int = 16001):
    """
    The field radiated by currents in a slab, q along x: |E|² + |E|² at the two far ends. TE: E_y'' + (εk0² − q²)E_y = −ik0 J_y. TM: (H_y'/ε)' + (k0² − q²/ε)H_y = −(J_x/ε)' + iq J_z/ε.
    Second-order differences on a uniform grid (ε, 1/ε at half nodes), one-sided outgoing ends.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    z = np.linspace(-half, half, n)
    dz = z[1] - z[0]
    kz = np.sqrt(k0**2 - q**2 + 0j)
    eps = np.where(np.abs(z) < t / 2, eps_bar, 1.0)
    zh = (z[1:] + z[:-1]) / 2
    ieh = np.where(np.abs(zh) < t / 2, 1 / eps_bar, 1.0)

    def ends(A, b):
        A[0, 0], A[0, 1] = -1 / dz + 1j * kz / 2, 1 / dz + 1j * kz / 2
        A[-1, -1], A[-1, -2] = 1 / dz - 1j * kz / 2, -1 / dz - 1j * kz / 2
        b[0] = b[-1] = 0
        return spla.spsolve(A.tocsc(), b)

    A = sp.diags([np.full(n - 1, 1 / dz**2), -2 / dz**2 + eps * k0**2 - q**2, np.full(n - 1, 1 / dz**2)], [-1, 0, 1], format="lil", dtype=complex)
    E = ends(A, -1j * k0 * js(z) + 0j)
    P_te = abs(E[0]) ** 2 + abs(E[-1]) ** 2
    A = sp.diags([ieh / dz**2, -(np.r_[ieh, 1.0] + np.r_[1.0, ieh]) / dz**2 + k0**2 - q**2 / eps, ieh / dz**2], [-1, 0, 1], format="lil", dtype=complex)
    jq_e = jq(zh) * ieh
    b = np.zeros(n, complex)
    b[1:-1] = -(jq_e[1:] - jq_e[:-1]) / dz
    b += 1j * q * jz(z) / eps
    H = ends(A, b)
    P_tm = abs(H[0]) ** 2 + abs(H[-1]) ** 2
    return P_te, P_tm


def main() -> None:
    k0 = 2 * np.pi * 194.0 / 299.792458
    print(f"k0 = {k0:.4f} /µm")
    for name, J in (("J along x", np.array([1.0, 0, 0])), ("J along y", np.array([0, 1.0, 0]))):
        exact = k0**2 / (12 * np.pi) * float(J @ J)
        for dqx, dqy in ((0.30, 0.50), (0.15, 0.50), (0.04, 0.50), (0.04, 0.12), (0.02, 0.06)):
            p = radiated(k0, J, dqx, dqy)
            print(f"  {name}  dq ({dqx:.2f}, {dqy:.2f}): P / Larmor {p / exact:.4f}")
    # smoothness at the cone edge: a spectrum steep across the edge, as a mode's tail around X
    dqx, dqy = 0.0375, 0.49
    print("cone edge: ln Σ W exp(s |qx|) as k0 moves by a fifth of a q cell (s = 30 /µm); differences should be even")
    prev = None
    for t in np.arange(0, 2.01, 0.2):
        kk = k0 + t * dqx
        qx, qy = cells(kk, dqx, dqy)
        v = np.log(np.sum(_cone_weights(kk, qx, qy, dqx, dqy, n_y=16) * np.exp(30 * (np.abs(qx) - k0))))
        print(f"  k0 + {t:.1f} dq: {v:+.4f}" + (f"  diff {v - prev:+.4f}" if prev is not None else ""))
        prev = v
    print("slab: T(q) from _far_terms / (2 k_z / k0)² (|field|² at both ends) of the 1D wave equation; should be 1")
    t = 0.22
    nz = 2200
    z = (np.arange(nz) + 0.5) / nz * t - t / 2
    dz = t / nz
    js = lambda u: np.where(np.abs(u) < t / 2, 1 + 2 * (u / t) ** 2, 0.0)  # noqa: E731
    jq = lambda u: np.where(np.abs(u) < t / 2, np.cos(np.pi * u / t), 0.0)  # noqa: E731
    jz = lambda u: np.where(np.abs(u) < t / 2, 3 * u / t, 0.0)  # noqa: E731
    for eps_bar in (1.0, 6.6, 12.1):
        for frac in (0.1, 0.5, 0.9, 0.98):
            q = frac * k0
            p_te, p_tm = wave_1d(k0, q, eps_bar, t, jq, jz, js)
            kz = np.sqrt(k0**2 - q**2)
            qx, qy = jnp.asarray([q]), jnp.asarray([0.0])
            T_te = _far_terms(jnp.asarray(np.stack([0 * z, js(z) * dz, 0 * z])[:, None]), qx, qy, jnp.asarray(z), k0, eps_bar, t)
            T_tm = _far_terms(jnp.asarray(np.stack([jq(z) * dz + 0j, 0 * z + 0j, jz(z) * dz + 0j])[:, None]), qx, qy, jnp.asarray(z), k0, eps_bar, t)
            print(f"  ε̄ {eps_bar:4.1f}, q/k0 {frac:.2f}: TE {float(T_te[0]) / ((2 * kz / k0) ** 2 * p_te):.4f}, TM {float(T_tm[0]) / ((2 * kz / k0) ** 2 * p_tm):.4f}")


if __name__ == "__main__":
    main()
