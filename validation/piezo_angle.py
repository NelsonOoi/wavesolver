"""
k_eff² of a periodic IDT section against the in-plane LN crystal angle θ, with analytic
dk²/dθ, and a 1-parameter gradient optimisation of θ.

Cell: one electrical period (2a = 520 nm, one hot and one grounded 130 nm × 100 nm Al finger
across the ridge top) of a 600 nm wide nanobeam along x: 220 nm Si under 250 nm X-cut LN
(12° sidewalls as two stepped sub-layers; Si cubic, [100] along x), LN = LN_PIEZO.rotated(x_cut(θ)), Bloch-periodic
along x at k = 0, free along y. θ is the angle from the beam axis (x, the propagation
direction) to the c-axis; the user's default orientation is x_cut(45). For each θ: the modes
between 3 and 7 GHz, short and open circuit, labelled S0-like (u_x dominant) or SH0-like (u_y
dominant) from the short-circuit displacement; the strongest of each class is followed.
FD of k_eff² at a few angles (±0.05°) checks the analytic gradient on the curve; then a
safeguarded Newton-secant iteration on dk²/dθ = 0 from the best sampled angle of each class.

Note: at this pitch the stack (≈ 570 nm with Al) is about one wavelength thick, so the band
holds dispersive, x- or y-polarised beam modes rather than thin-plate S0/SH0; "S0-like" and
"SH0-like" mean u_x- and u_y-dominant. The x-polarised modes couple weakly (≤ 0.5 %); the
thin-plate S0 coupling (validation/piezo_plate_k2.py) needs a pitch with t/λ ≪ 1. Away from
θ = 0°, 90° the y → -y mirror is broken: the polarisations mix (up to ~15 % u_x in the
SH0-like mode) and the strongest mode of a class can change identity (jumps in the curve).

    pixi run python validation/piezo_angle.py [--quick]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omc.materials import AL_COMSOL, LN_PIEZO, SI, SI_EPS_R  # noqa: E402
from omc.piezo import Electrode, PiezoLayer, PiezoSolver  # noqa: E402

OUT = ROOT / "validation/out"
A = 0.26
W = 0.6
T_SI, T_LN, T_AL = 0.22, 0.25, 0.1
SIDEWALL = 12.0
W_TOP = W - 2 * T_LN * np.tan(np.radians(SIDEWALL))
BAND = (3.0e9, 7.0e9)
N_MODES = 24
MESH = (8, 16, 3, 2, 2, 2)  # cells per a, elements across w, Si order, LN sub-layers, LN order, Al order


def overlap_1d(lo, hi, a, b):
    return np.clip((np.minimum(hi, b) - np.maximum(lo, a)) / (hi - lo), 0, 1)


def idt_cell(angle: float, m=8, n2=16, p_si=3, n_ln=2, p_ln=2, p_al=2, mass_scale=None, si=SI) -> PiezoSolver:
    """
    Periodic IDT period with LN at crystal angle `angle` (degrees, a design variable). Si is
    cubic with [100] along the beam (`materials.SI`, unrotated).
    """
    dx = A / m
    n1 = 2 * m
    xe = -A / 2 + dx * np.arange(n1 + 1)
    ye = -W / 2 + W / n2 * np.arange(n2 + 1)
    fy = lambda w: overlap_1d(ye[:-1], ye[1:], -w / 2, w / 2)  # noqa: E731
    layers = [PiezoLayer(T_SI, np.ones((n1, n2)), si, eps_r=SI_EPS_R, order=p_si, mass_scale=mass_scale)]
    h = T_LN / n_ln
    for s in range(n_ln):
        w = W - 2 * (s + 0.5) * h * np.tan(np.radians(SIDEWALL))
        layers.append(PiezoLayer(h, np.outer(np.ones(n1), fy(w)), LN_PIEZO, order=p_ln, angle=angle))
    hot = overlap_1d(xe[:-1], xe[1:], -A / 4, A / 4)
    gnd = overlap_1d(xe[:-1], xe[1:], A - A / 4, A + A / 4)
    layers.append(PiezoLayer(T_AL, np.outer(hot + gnd, fy(W_TOP)), AL_COMSOL, eps_r=None, order=p_al))
    on = fy(W_TOP) > 0.5
    top = len(layers) - 1
    el = [Electrode("hot", np.outer(hot > 0.5, on), top), Electrode("gnd", np.outer(gnd > 0.5, on), top, "ground")]
    return PiezoSolver(np.diag([2 * A, W]), layers, bc=("periodic", "free"), electrodes=el, origin=(-A / 2, -W / 2))


def polarisation(u: np.ndarray) -> np.ndarray:
    """Fractions of |u|² along x, y, z per mode (u: (nm, n1+1, n2+1, F, 3))."""
    p = np.sum(np.abs(u) ** 2, axis=(1, 2, 3))
    return p / p.sum(axis=1, keepdims=True)


def analyse(sol: PiezoSolver):
    """Coupling sensitivities and per-mode class (0: S0-like, 1: SH0-like, -1: other) in the band."""
    cs = sol.coupling_sensitivities(N_MODES, 2.5e9)
    pol = polarisation(cs.short.u)
    cls = np.where(pol[:, 0] > 0.5, 0, np.where(pol[:, 1] > 0.5, 1, -1))
    inband = (cs.f_sc > BAND[0]) & (cs.f_sc < BAND[1]) & (cs.overlap > 0.5)
    return cs, np.where(inband, cls, -1), pol


def strongest(cs, cls, kind: int, near: float | None = None):
    """Index of the class-`kind` mode with the largest k² (or, given near, the class mode closest in f_sc)."""
    idx = np.flatnonzero(cls == kind)
    if not len(idx):
        return None
    if near is not None:
        return idx[np.argmin(np.abs(cs.f_sc[idx] - near))]
    return idx[np.argmax(cs.k2[idx])]


def main() -> None:
    quick = "--quick" in sys.argv
    OUT.mkdir(exist_ok=True)
    angles = np.arange(0, 180, 30.0 if quick else 7.5)
    sol = idt_cell(float(angles[0]), *MESH)
    print(f"{sol.ndof} dof (full grid), mesh {MESH}", flush=True)
    rows = {0: [], 1: []}
    t_sens = []
    for th in angles:
        sol.set_angle(float(th))
        t0 = time.perf_counter()
        cs, cls, pol = analyse(sol)
        t_sens.append(time.perf_counter() - t0)
        for kind in (0, 1):
            i = strongest(cs, cls, kind)
            if i is None:
                continue
            rows[kind].append((th, cs.f_sc[i], cs.f_oc[i], cs.k2[i], cs.dk2_angle[i], pol[i, 0], pol[i, 1], pol[i, 2]))
        print(
            f"θ = {th:5.1f}°  ({t_sens[-1]:.1f} s)  "
            + "  ".join(
                f"{name}: f_sc {r[-1][1] / 1e9:.4f} GHz k² {100 * r[-1][3]:5.2f} % dk²/dθ {100 * r[-1][4]:+.4f} %/° pol {r[-1][5]:.2f}/{r[-1][6]:.2f}/{r[-1][7]:.2f}"
                for name, r in (("S0", rows[0]), ("SH0", rows[1]))
                if r and r[-1][0] == th
            ),
            flush=True,
        )
    rows = {k: np.array(v) for k, v in rows.items()}

    # FD of the followed mode's k² at a few angles.
    fd = {0: [], 1: []}
    h = 0.05
    for kind in (0, 1):
        r = rows[kind]
        for j in np.linspace(0, len(r) - 1, 3 if quick else 4).astype(int):
            th, fsc = r[j, 0], r[j, 1]
            vals = []
            for s in (+1, -1):
                sol.set_angle(th + s * h)
                cs, cls, _ = analyse(sol)
                i = strongest(cs, cls, kind, near=fsc)
                vals.append(cs.k2[i])
            fd[kind].append((th, (vals[0] - vals[1]) / (2 * h), r[j, 4]))
            print(f"FD check {'S0 SH0'.split()[kind]} θ = {th:5.1f}°: FD {100 * fd[kind][-1][1]:+.5f} %/°, analytic {100 * r[j, 4]:+.5f} %/°", flush=True)
    fd = {k: np.array(v) for k, v in fd.items()}

    # Gradient optimisation of θ: safeguarded Newton-secant on g(θ) = dk²/dθ = 0.
    optima = {}
    for kind in (0, 1):
        r = rows[kind]
        j = int(np.argmax(r[:, 3]))
        th, fsc = r[j, 0], r[j, 1]
        g, hist = r[j, 4], []
        step = 5.0 * np.sign(g) if g != 0 else 0.0
        th_prev, g_prev = None, None
        for it in range(12):
            if th_prev is not None and g != g_prev:
                step = -g * (th - th_prev) / (g - g_prev)  # secant on the gradient
            step = float(np.clip(step, -10, 10))
            th_prev, g_prev = th, g
            th = th + step
            sol.set_angle(th)
            cs, cls, _ = analyse(sol)
            i = strongest(cs, cls, kind, near=fsc)
            fsc, k2, g = cs.f_sc[i], cs.k2[i], cs.dk2_angle[i]
            hist.append((th, k2, g))
            print(f"opt {'S0 SH0'.split()[kind]} it {it}: θ = {th:8.4f}°  k² {100 * k2:.4f} %  dk²/dθ {100 * g:+.2e} %/°  f_sc {fsc / 1e9:.4f} GHz", flush=True)
            if abs(step) < 1e-3:
                break
        optima[kind] = (th, k2, fsc, np.array(hist))
    print(f"mean sensitivity time per θ (2 × {N_MODES} modes, short + open, all gradients): {np.mean(t_sens):.1f} s")
    for kind, name in ((0, "S0-like"), (1, "SH0-like")):
        th, k2, fsc, _ = optima[kind]
        print(f"optimum {name}: θ = {th % 180:.2f}° (mod 180), k_eff² = {100 * k2:.3f} %, f_sc = {fsc / 1e9:.4f} GHz")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for kind, (c, name) in enumerate((("C3", "S0-like"), ("C0", "SH0-like"))):
        r = rows[kind]
        ax[0].plot(r[:, 0], 100 * r[:, 3], c + "o-", ms=4, label=name)
        for th, k2, g in zip(r[:, 0], r[:, 3], r[:, 4]):
            ax[0].plot([th - 4, th + 4], [100 * (k2 - 4 * g), 100 * (k2 + 4 * g)], c + "-", lw=2, alpha=0.5)
        th, k2, *_ = optima[kind]
        ax[0].plot([th % 180], [100 * k2], c + "*", ms=14, label=f"{name} optimum {th % 180:.1f}°")
        ax[1].plot(r[:, 0], 100 * r[:, 4], c + "-", lw=1, label=f"{name} analytic")
        ax[1].plot(fd[kind][:, 0], 100 * fd[kind][:, 1], c + "s", mfc="none", ms=7, label=f"{name} FD")
    ax[0].set_ylabel("k_eff² (%)")
    ax[0].set_title("strongest mode per class (bars: analytic slope)")
    ax[1].set_ylabel("dk_eff²/dθ (% per degree)")
    for a in ax:
        a.set_xlabel("θ: angle from beam axis to c-axis, x_cut(θ) (deg)")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle("Periodic IDT section, 250 nm X-cut LN on 220 nm Si, 2a = 520 nm")
    fig.tight_layout()
    fig.savefig(OUT / "piezo_angle.png", dpi=150)
    np.savez(OUT / "piezo_angle.npz", s0=rows[0], sh0=rows[1], fd_s0=fd[0], fd_sh0=fd[1], opt_s0=optima[0][3], opt_sh0=optima[1][3])


if __name__ == "__main__":
    main()
