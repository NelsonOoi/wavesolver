"""
Three-mode transducer (microwave resonator – mechanics – optics): checks and design window
for the electromechanical coupling g_em.

Nominal: Q_i = 2.4e5 (κ_i/2π ≈ 0.8 GHz), critical optics κ_ex = κ_i, γ_i/2π = 30 kHz,
g0/2π = 0.5 MHz, f_m = 4 GHz, red sideband; single-mode overcoupled resonator κ_r/2π = 1 MHz
(κ_r,i/2π = 0.1 MHz assumed, η_r = 0.9), n_r = 0; n_th from `Heating()` (placeholder law).

Checks: the RWA three-mode response against the non-RWA 6 × 6 Langevin solution; the
resonant closed form η_oη_r 4C_oC_e/(1 + C_o + C_e)²; and the reduction to the two-port model
(`two_port_equivalent`) for g_em ≪ κ_r.
Design window: η_max, bandwidth (contiguous width of η ≥ η_max/2 around the peak; one peak of a split pair) and N_add vs g_em for n_c = 1, 10,
100 and κ_r/2π = 0.3, 1, 3 MHz, including the hybridised regime g_em ≳ κ_r/2.

    pixi run python validation/system_three_mode.py
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

from omc.system import (  # noqa: E402
    TWO_PI,
    Heating,
    Resonator,
    Transducer,
    added_noise,
    bandwidth,
    conversion_peak,
    efficiency,
    exact_three_mode_response,
    matched_g_em,
    three_mode_efficiency_resonant,
    two_port_equivalent,
)

OUT = ROOT / "validation/out"
MHZ = TWO_PI * 1e6
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
P = Transducer(Q_i=2.4e5, gamma_i=TWO_PI * 30e3, g0=0.5 * MHZ)
R = Resonator(kappa_ex=0.9 * MHZ, kappa_i=0.1 * MHZ)
HEAT = Heating()
NC = (1.0, 10.0, 100.0)
KAPPA_R = (0.3, 1.0, 3.0)  # MHz, with κ_r,i fixed at 0.1 MHz
G_EM = np.logspace(np.log10(0.01), np.log10(10.0), 61)  # MHz


def checks() -> None:
    print("=== Checks (n_th = 0.3) ===")
    print("  g_em/2π  n_c | RWA η_max   exact η_max  |rel|   | N_add RWA  exact | resonant formula | 2-port η   BW RWA/2-port (kHz)")
    worst_x = 0.0
    for G in (0.02, 0.1, 0.5, 2.0):
        for nc in (1.0, 100.0):
            r = R.replace(g_em=G * MHZ)
            e, nu, bw, N = conversion_peak(P, r, nc, n_th=0.3)
            w = P.omega_m + nu + np.linspace(-1, 1, 4001) * 0.02 * max(bw, 1.0)
            ex, Nx = exact_three_mode_response(P, r, nc, w, n_th=0.3)
            i = ex.argmax()
            worst_x = max(worst_x, abs(ex[i] / e - 1))
            q = two_port_equivalent(P, r)
            print(
                f"  {G:5.2f} MHz {nc:4.0f} | {e:.6f}  {ex[i]:.6f}  {abs(ex[i] / e - 1):.1e} | {N:.4f}  {Nx[i]:.4f} | "
                f"{float(three_mode_efficiency_resonant(P, r, nc)):.6f} | {float(efficiency(q, nc)):.6f}  "
                f"{bw / TWO_PI / 1e3:7.1f} / {float(bandwidth(q, nc)) / TWO_PI / 1e3:7.1f}"
            )
    print(f"  worst RWA vs exact η_max: {worst_x:.1e}")
    # reduction: g_em ≪ κ_r, the resonator bath at n_th too (as the two-port lumps it with γ_i)
    r = R.replace(g_em=0.01 * MHZ, n_th=0.3)
    e, _, bw, N = conversion_peak(P, r, 10.0, n_th=0.3)
    q = two_port_equivalent(P, r)
    print(
        f"  reduction g_em/κ_r = 0.01, n_c = 10: η {e:.6e} vs two-port {float(efficiency(q, 10.0)):.6e}; "
        f"BW {bw / TWO_PI:.0f} vs {float(bandwidth(q, 10.0)) / TWO_PI:.0f} Hz; N_add {N:.4f} vs {float(added_noise(q, 10.0, 0.3)):.4f}"
    )


def window():
    res = {}
    for kr in KAPPA_R:
        r0 = Resonator(kappa_ex=(kr - 0.1) * MHZ, kappa_i=0.1 * MHZ)
        for nc in NC:
            nth = float(HEAT.n_th(P, nc))
            rows = [conversion_peak(P, r0.replace(g_em=G * MHZ), nc, n_th=nth) for G in G_EM]
            res[kr, nc] = np.array(rows)
    print("\n=== Design window (κ_r,i/2π = 0.1 MHz; n_th from Heating(): n_th(1, 10, 100) = "
          + ", ".join(f"{float(HEAT.n_th(P, n)):.2f}" for n in NC) + ") ===")
    print("  κ_r   n_c  C_o  | g_em* (MHz)  η(g*)   BW(g*) kHz  N_add | η ≥ ½η(g*) for g_em in (MHz) | at g_em = κ_r/2: η, BW | max η·BW at g_em: η, BW (kHz)")
    for kr in KAPPA_R:
        r0 = Resonator(kappa_ex=(kr - 0.1) * MHZ, kappa_i=0.1 * MHZ)
        for nc in NC:
            a = res[kr, nc]
            Co = 4 * P.g0**2 * nc / (P.kappa * P.gamma_i)
            gs = float(matched_g_em(P, r0, nc)) / MHZ
            es, _, bs, Ns = conversion_peak(P, r0.replace(g_em=gs * MHZ), nc, n_th=float(HEAT.n_th(P, nc)))
            ok = a[:, 0] >= es / 2
            lo, hi = G_EM[ok].min(), G_EM[ok].max()
            eh, _, bh, _ = conversion_peak(P, r0.replace(g_em=kr / 2 * MHZ), nc, n_th=float(HEAT.n_th(P, nc)))
            j = int(np.argmax(a[:, 0] * a[:, 2]))
            print(
                f"  {kr:3.1f} {nc:5.0f} {float(Co):5.2f} | {gs:9.3f}  {es:.4f}  {bs / TWO_PI / 1e3:8.1f}  {Ns:.3f} | {lo:6.3f} – {hi:6.3f} | "
                f"{eh:.4f}, {bh / TWO_PI / 1e3:7.0f} | {G_EM[j]:.2f}: {a[j, 0]:.4f}, {a[j, 2] / TWO_PI / 1e3:.0f}"
            )
    return res


def plot(res) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for c, nc in zip(COLORS, NC):
        for kr, ls in zip(KAPPA_R, (":", "-", "--")):
            a = res[kr, nc]
            lab = f"n_c = {nc:g}" if kr == 1.0 else None
            ax[0].plot(G_EM, a[:, 0], ls, color=c, lw=1.5 if kr == 1.0 else 1, label=lab)
            ax[1].plot(G_EM, a[:, 2] / TWO_PI / 1e6, ls, color=c, lw=1.5 if kr == 1.0 else 1)
            ax[2].plot(G_EM, a[:, 3], ls, color=c, lw=1.5 if kr == 1.0 else 1)
        r0 = Resonator(kappa_ex=0.9 * MHZ, kappa_i=0.1 * MHZ)
        gs = float(matched_g_em(P, r0, nc)) / MHZ
        ax[0].axvline(gs, color=c, lw=0.8, alpha=0.5)
    for kr, ls in zip(KAPPA_R, (":", "-", "--")):
        ax[0].plot([], [], ls, color="k", label=f"κ_r/2π = {kr} MHz")
    for a in ax:
        a.set_xscale("log")
        a.axvline(0.5, color="k", lw=0.6, ls="-.", alpha=0.6)
        a.set_xlabel("g_em/2π (MHz)")
        a.grid(alpha=0.3, which="both")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("peak η (microwave line → optical line)")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("bandwidth: η ≥ η_max/2 around the peak (MHz)")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("N_add at the peak (quanta)")
    ax[0].legend(fontsize=7)
    fig.suptitle("Three-mode transducer: g0/2π = 0.5 MHz, γ_i/2π = 30 kHz, Q_i = 2.4e5 (critical), κ_r,i/2π = 0.1 MHz; "
                 "thin lines: matched g_em*, dash-dot: κ_r/2 (1 MHz)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "system_three_mode.png", dpi=150)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    checks()
    res = window()
    plot(res)
    np.savez(OUT / "system_three_mode.npz", g_em=G_EM, nc=NC, kappa_r=KAPPA_R, **{f"kr{kr}_nc{nc:g}": v for (kr, nc), v in res.items()})


if __name__ == "__main__":
    main()
