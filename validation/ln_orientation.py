"""
LN crystal axes against the global frame for COMSOL Euler angles (z-x-z), and the thin-plate
coupling k² of SH0 and S0 against propagation direction in the global frame.

The OMC spine is along global x. The k² data are the thin X-cut LN plate sweep from
validation/piezo_plate_k2.py (FEM, no vacuum), tabulated against the angle θ from the
propagation direction to the c-axis; with the c-axis at angle φ_c in the global frame, a wave
along global direction φ has θ = φ_c − φ, and k² repeats every 180°.

    pixi run python validation/ln_orientation.py [alpha beta gamma, in units of π]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omc.materials import x_cut  # noqa: E402

OUT = ROOT / "validation/out"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def euler_zxz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """COMSOL rotated system: columns are the crystal (local) axes in global coordinates."""

    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    return rz(alpha) @ rx(beta) @ rz(gamma)


def main() -> None:
    args = [float(a) for a in sys.argv[1:4]] if len(sys.argv) >= 4 else [0.25, -0.5, -0.5]
    R = euler_zxz(*(np.pi * np.array(args)))
    phi_c = np.degrees(np.arctan2(R[1, 2], R[0, 2]))
    match = [a for a in range(-180, 181, 15) if np.allclose(x_cut(a), R)]
    label = f"Euler ({args[0]:g}π, {args[1]:g}π, {args[2]:g}π)"
    print(label)
    for name, col in zip("XYZ", R.T):
        print(f"  crystal {name} → global {np.round(col, 3)}")
    print(f"  c-axis at {phi_c:.1f}° from the spine; omc.materials.x_cut({match[0] if match else '?'})")

    rows = np.load(OUT / "piezo_plate_k2.npz")["rows"]
    theta = rows[:, 0]
    k2 = lambda vs, vo: 100 * (vo**2 - vs**2) / vo**2  # noqa: E731
    sh0, s0 = k2(rows[:, 5], rows[:, 7]), k2(rows[:, 6], rows[:, 8])  # lower-velocity mode is SH0

    fig = plt.figure(figsize=(11, 5.2))
    fig.patch.set_facecolor("#fcfcfb")

    # Top view: global x-y plane, z out of the page.
    ax = fig.add_subplot(1, 2, 1)
    ax.set_facecolor("#fcfcfb")
    ax.add_patch(plt.Rectangle((-1.25, -0.08), 2.5, 0.16, color=GRID, zorder=0))
    ax.text(1.27, 0.0, "OMC spine\n(global x)", va="center", ha="left", color=INK2, fontsize=9)
    for name, col, color in zip(("Y", "Z"), R.T[1:], (BLUE, ORANGE)):
        v = col[:2]
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2, color=color))
        extra = " (c-axis)" if name == "Z" else ""
        ax.text(*(1.12 * v), f"crystal +{name}{extra}", color=INK, fontsize=10, ha="center", va="center")
    xz = R[2, 0]
    ax.plot(0, 0, "o", ms=16, mfc="none", mec=AQUA, mew=2)
    ax.plot(0, 0, "o", ms=4, color=AQUA)
    ax.text(0.05, -0.2, f"crystal +X {'out of' if xz > 0 else 'into'} page", color=INK, fontsize=10)
    ax.annotate("", xy=(-1.05, -0.75), xytext=(-1.35, -0.75), arrowprops=dict(arrowstyle="-|>", lw=1, color=INK2))
    ax.annotate("", xy=(-1.35, -0.45), xytext=(-1.35, -0.75), arrowprops=dict(arrowstyle="-|>", lw=1, color=INK2))
    ax.text(-1.02, -0.75, "x", color=INK2, va="center")
    ax.text(-1.35, -0.4, "y", color=INK2, ha="center")
    ax.text(-1.4, -0.92, "global (z out of page)", color=INK2, fontsize=8)
    ax.set_xlim(-1.5, 1.8)
    ax.set_ylim(-1.05, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    spine = R.T @ [1, 0, 0]
    ax.set_title(
        f"{label}: spine ∥ ({spine[1]:+.2f} Y {spine[2]:+.2f} Z)",
        color=INK,
        fontsize=11,
    )

    # k² vs propagation direction in the global frame (180° periodic).
    axp = fig.add_subplot(1, 2, 2, projection="polar")
    axp.set_facecolor("#fcfcfb")
    phis = np.concatenate([phi_c - theta, phi_c - theta + 180])
    order = np.argsort(np.mod(phis, 360))
    phi_sorted = np.radians(np.mod(phis, 360)[order])
    for data, color, name in ((sh0, BLUE, "SH0"), (s0, ORANGE, "S0")):
        vals = np.concatenate([data, data])[order]
        axp.plot(np.append(phi_sorted, phi_sorted[0]), np.append(vals, vals[0]), "-", color=color, lw=2, label=name)
        axp.plot(phi_sorted, vals, "o", color=color, ms=4)
    for d in (0, 180):
        axp.plot([np.radians(d)] * 2, [0, 40], color=INK2, lw=1, ls="--")
    axp.text(np.radians(8), 41, "spine", color=INK2, fontsize=9)
    s_sh0 = np.interp(np.mod(phi_c, 180), theta, sh0)
    s_s0 = np.interp(np.mod(phi_c, 180), theta, s0)
    axp.set_title(
        f"thin X-cut LN plate k² (%) vs propagation direction\nalong the spine: SH0 {s_sh0:.1f} %, S0 {s_s0:.1f} %",
        color=INK,
        fontsize=11,
    )
    axp.set_rlim(0, 40)
    axp.tick_params(colors=INK2, labelsize=8)
    axp.grid(color=GRID)
    axp.legend(loc="lower left", bbox_to_anchor=(-0.12, -0.08), frameon=False, fontsize=9)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    name = "ln_orientation_" + "_".join(f"{a:g}" for a in args).replace("-", "m").replace(".", "p") + ".png"
    fig.savefig(OUT / name, dpi=150, facecolor=fig.get_facecolor())
    print("  saved", OUT / name)


if __name__ == "__main__":
    main()
