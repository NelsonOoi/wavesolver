"""
Electromechanical coupling of the S0 and SH0 plate waves of a thin X-cut LN plate against the
thin-plate (long-wavelength) limit, as a function of the in-plane propagation angle.

Thin-plate limit (fields uniform through the thickness, propagation along lab x, no y
dependence): S_yy = 0, E_y = 0 and T_zz = T_yz = T_xz = 0. Short circuit (both faces
grounded): E = 0. Open circuit (bare faces, no vacuum): D_z = 0 at the faces, hence D_z = 0
and, from ∇·D = 0, D_x = 0. Eliminating the free strains (and E_x, E_z when open) leaves a
2 × 2 stiffness on (S_xx, S_xy) whose eigenvalues are ρ v² of the S0 / SH0 pair, and
k² = (v_oc² - v_sc²) / v_oc².

FEM: one Bloch cell of length λ/4 along x (phase π/2), one periodic element along y, the
plate meshed by one layer of order 4; short circuit by grounded electrodes covering both
faces, open circuit without electrodes. The lab frame is the crystal frame rotated by
x_cut(angle): angle is the angle from the propagation direction to the c-axis, so angle 90°
is propagation along crystal ∓Y. A last column adds 2 µm of vacuum on both faces of the open
plate (with D·n = 0 on its outer faces), the physically open surface.

    pixi run python validation/piezo_plate_k2.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wavesolver.materials import EPS0, LN_PIEZO, x_cut  # noqa: E402
from wavesolver.piezo import Electrode, PiezoLayer, PiezoSolver  # noqa: E402

OUT = ROOT / "validation/out"
T = 0.25  # plate thickness, µm
LAM = 10.0  # wavelength, µm
N1 = 16
ANGLES = np.arange(0, 181, 15)


def thin_plate(mat, short: bool):
    """ρ v² (Pa) of the two in-plane plate waves along lab x, thin-plate limit."""
    C, e, eps = mat.C, mat.e, mat.eps_r * EPS0
    G = np.block([[C, -e.T], [e, eps]])  # (S, E) -> (T, D)
    known = [0, 5]  # S_xx, S_xy
    unknown = [2, 3, 4] + ([] if short else [6, 8])  # S_zz, S_yz, S_xz (+ E_x, E_z)
    zero_out = [2, 3, 4] + ([] if short else [6, 8])  # T_zz, T_yz, T_xz (+ D_x, D_z)
    A = G[np.ix_(zero_out, unknown)]
    Bq = G[np.ix_(zero_out, known)]
    Ceff = G[np.ix_([0, 5], known)] - G[np.ix_([0, 5], unknown)] @ np.linalg.solve(A, Bq)
    return np.sort(np.linalg.eigvalsh((Ceff + Ceff.T) / 2))


def plate_solver(mat, short: bool, air: float = 0.0) -> PiezoSolver:
    cell = np.diag([LAM / 4, 0.05])
    layers = [PiezoLayer(T, 1.0, mat, order=4)]
    el = []
    if short:
        full = np.ones((N1, 1), bool)
        el = [Electrode("bottom", full, 0, "ground"), Electrode("top", full, 1, "ground")]
    if air:
        layers = [PiezoLayer(air, 1.0, eps_r=1.0, order=4)] + layers + [PiezoLayer(air, 1.0, eps_r=1.0, order=4)]
    return PiezoSolver(cell, layers, bc=("periodic", "periodic"), electrodes=el, shape=(N1, 1))


def in_plane_freqs(solver: PiezoSolver) -> np.ndarray:
    """The two lowest modes with in-plane polarisation (S0, SH0 pair), Hz."""
    k = np.array([2 * np.pi / LAM, 0.0])
    f, u, _ = solver.eigenmodes(6, -1e6, k=k, vectors=True)
    pol = np.sum(np.abs(u) ** 2, axis=(1, 2, 3))  # (modes, 3)
    inplane = pol[:, 2] < 0.5 * pol.sum(axis=1)
    return np.sort(f.real[inplane])[:2]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for ang in ANGLES:
        mat = LN_PIEZO.rotated(x_cut(float(ang)))
        v_sc = np.sqrt(thin_plate(mat, True) / mat.rho)
        v_oc = np.sqrt(thin_plate(mat, False) / mat.rho)
        f_sc = in_plane_freqs(plate_solver(mat, True))
        f_oc = in_plane_freqs(plate_solver(mat, False))
        f_air = in_plane_freqs(plate_solver(mat, False, air=2.0))
        rows.append((ang, *v_sc, *v_oc, *(f_sc * LAM * 1e-6), *(f_oc * LAM * 1e-6), *(f_air * LAM * 1e-6)))
    rows = np.array(rows)
    dt = time.perf_counter() - t0
    ang = rows[:, 0]
    k2 = lambda sc, oc: (oc**2 - sc**2) / oc**2  # noqa: E731
    print(f"{len(ANGLES)} angles x 3 solves in {dt:.1f} s")
    print(" angle | v_sc analytic / FEM (m/s)  | v_oc analytic / FEM (m/s)   | k² analytic / FEM / FEM+vacuum (%)")
    worst_v, worst_k = 0.0, 0.0
    for r in rows:
        for m, name in ((0, "slow"), (1, "fast")):
            vsa, voa = r[1 + m], r[3 + m]
            vsf, vof, vaf = r[5 + m], r[7 + m], r[9 + m]
            worst_v = max(worst_v, abs(vsf / vsa - 1), abs(vof / voa - 1))
            worst_k = max(worst_k, abs(k2(vsf, vof) - k2(vsa, voa)))
            print(
                f" {r[0]:5.0f} {name} | {vsa:7.1f} / {vsf:7.1f} | {voa:7.1f} / {vof:7.1f} | "
                f"{100 * k2(vsa, voa):5.2f} / {100 * k2(vsf, vof):5.2f} / {100 * k2(vsf, vaf):5.2f}"
            )
    print(f"(b) worst velocity error {100 * worst_v:.3f} %, worst k² error {100 * worst_k:.3f} %-points")
    print(f"    (kt = {2 * np.pi * T / LAM:.3f}; the FEM includes the finite-thickness dispersion the limit drops)")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m, (c, name) in enumerate((("C0", "slow (SH0-like)"), ("C3", "fast (S0-like)"))):
        ax[0].plot(ang, rows[:, 1 + m], c + "-", lw=1, label=f"{name}, short, thin-plate limit")
        ax[0].plot(ang, rows[:, 5 + m], c + "o", ms=4, mfc="none")
        ax[0].plot(ang, rows[:, 3 + m], c + "--", lw=1, label=f"{name}, open")
        ax[0].plot(ang, rows[:, 7 + m], c + "s", ms=4, mfc="none")
        ax[1].plot(ang, 100 * k2(rows[:, 1 + m], rows[:, 3 + m]), c + "-", lw=1, label=f"{name}, thin-plate limit")
        ax[1].plot(ang, 100 * k2(rows[:, 5 + m], rows[:, 7 + m]), c + "o", ms=4, mfc="none", label=f"{name}, FEM")
        ax[1].plot(ang, 100 * k2(rows[:, 5 + m], rows[:, 9 + m]), c + "x", ms=4, label=f"{name}, FEM with vacuum")
    ax[0].set_ylabel("phase velocity (m/s)")
    ax[1].set_ylabel("k² (%)")
    for a in ax:
        a.set_xlabel("angle from propagation direction to c-axis (deg)")
        a.grid(alpha=0.3)
        a.legend(fontsize=6)
    fig.suptitle(f"X-cut LN plate, t = {T * 1e3:.0f} nm, λ = {LAM} µm (markers: FEM)")
    fig.tight_layout()
    fig.savefig(OUT / "piezo_plate_k2.png", dpi=150)
    np.savez(OUT / "piezo_plate_k2.npz", rows=rows)


if __name__ == "__main__":
    main()
