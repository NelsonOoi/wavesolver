"""
Piezoelectric FEM of one period of an interdigital transducer (IDT): electromechanical coupling
to a microwave resonator and its gradient with respect to the LN crystal angle.

250 nm X-cut LN ridge (12° sidewalls, two sub-layers) on a 600 nm wide, 220 nm thick cubic Si beam,
100 nm Al fingers, period 2a = 520 nm, periodic along the beam and free at its sides. For the
short-circuit modes in 3-7 GHz that couple to the fingers: the BVD motional branch (C_m, and g_em
to a 700 Ω resonator) from PiezoSolver.motional. Its exact gradient in the angle θ is checked
against a central difference.

    pixi run python examples/04_piezo_idt.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omc.materials import AL_COMSOL, LN_PIEZO, SI, SI_EPS_R  # noqa: E402
from omc.piezo import Electrode, PiezoLayer, PiezoSolver  # noqa: E402

A, W = 0.26, 0.6  # half period and beam width (µm)
T_SI, T_LN, T_AL = 0.22, 0.25, 0.1
SIDEWALL = 12.0  # degrees
W_TOP = W - 2 * T_LN * np.tan(np.radians(SIDEWALL))


def overlap_1d(lo, hi, a, b):
    """Share of each interval [lo, hi] inside [a, b]."""
    return np.clip((np.minimum(hi, b) - np.maximum(lo, a)) / (hi - lo), 0, 1)


def idt_period(angle: float, m: int = 6, n2: int = 12) -> PiezoSolver:
    """One IDT period with the LN c-axis at `angle` degrees from the beam axis (materials.x_cut)."""
    n1 = 2 * m
    xe = -A / 2 + A / m * np.arange(n1 + 1)
    ye = -W / 2 + W / n2 * np.arange(n2 + 1)
    fy = lambda w: overlap_1d(ye[:-1], ye[1:], -w / 2, w / 2)  # noqa: E731
    layers = [PiezoLayer(T_SI, np.ones((n1, n2)), SI, eps_r=SI_EPS_R, order=3)]
    for s in range(2):  # the sloped LN ridge, staircased in two sub-layers
        w = W - 2 * (s + 0.5) * (T_LN / 2) * np.tan(np.radians(SIDEWALL))
        layers.append(PiezoLayer(T_LN / 2, np.outer(np.ones(n1), fy(w)), LN_PIEZO, order=2, angle=angle))
    hot = overlap_1d(xe[:-1], xe[1:], -A / 4, A / 4)
    gnd = overlap_1d(xe[:-1], xe[1:], 3 * A / 4, 5 * A / 4)
    layers.append(PiezoLayer(T_AL, np.outer(hot + gnd, fy(W_TOP)), AL_COMSOL, eps_r=None, order=2))
    on, top = fy(W_TOP) > 0.5, len(layers) - 1
    electrodes = [Electrode("hot", np.outer(hot > 0.5, on), top), Electrode("gnd", np.outer(gnd > 0.5, on), top, "ground")]
    return PiezoSolver(np.diag([2 * A, W]), layers, bc=("periodic", "free"), electrodes=electrodes, origin=(-A / 2, -W / 2))


def main() -> None:
    angle = 45.0
    sol = idt_period(angle)
    t0 = time.perf_counter()
    mb = sol.motional("hot", n_modes=16, shift_hz=2.5e9, Z_r=700.0)
    print(f"{len(mb.freqs)} short-circuit modes in {time.perf_counter() - t0:.1f} s; C0 = {mb.C0 * 1e18:.1f} aF per period")
    band = (mb.freqs > 3e9) & (mb.freqs < 7e9)
    coupled = band & (mb.C_m > 1e-3 * mb.C_m[band].max())
    for f, c, g in zip(mb.freqs[coupled], mb.C_m[coupled], mb.g_em[coupled]):
        print(f"  {f / 1e9:6.3f} GHz  C_m {c * 1e18:7.3f} aF  g_em/2π {g / 2 / np.pi / 1e6:7.2f} MHz")

    # Exact gradient of the strongest mode's g_em in θ, against a central difference.
    j = int(np.flatnonzero(band)[np.argmax(mb.g_em[band])])
    mb = sol.motional("hot", n_modes=16, shift_hz=2.5e9, Z_r=700.0, gradient=[j])
    h = 1e-3
    g = []
    for s in (1, -1):
        sol.set_angle(angle + s * h)
        r = sol.motional("hot", n_modes=16, shift_hz=2.5e9, Z_r=700.0)
        g.append(r.g_em[np.argmin(abs(r.freqs - mb.freqs[j]))])
    fd = (g[0] - g[1]) / (2 * h)
    print(f"mode at {mb.freqs[j] / 1e9:.3f} GHz: dg_em/dθ adjoint {mb.dg_angle[j]:.6e}, FD {fd:.6e} rad/s per degree, "
          f"rel diff {abs(mb.dg_angle[j] - fd) / abs(fd):.1e}")


if __name__ == "__main__":
    main()
