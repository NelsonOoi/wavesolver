"""
Smoke test of the absorbing options: reflection of an SH0 wave from the end of a plate strip.

An isotropic Si plate (220 nm, infinite along y: one periodic element), 12 µm long, is driven
by a line force along y at x = 2 µm; the left end is free. The right 4 µm (x > 8 µm) is
either (i) nothing special (free end), (ii) a damping layer with density ρ (1 - i α ((x - 8)/4)²)
(COMSOL-style, lossy sign for e^{iωt}) or (iii) the coordinate-stretch PML
s(x) = 1 - i β ((x - 8)/4)². In 3 µm < x < 8 µm the field u_y is fitted to a e^{-ikx} + b e^{ikx}
(k scanned around ω/v_SH0); |R| = |b/a|.

    pixi run python validation/piezo_absorber.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wavesolver.materials import SI_ISOTROPIC  # noqa: E402
from wavesolver.piezo import PiezoLayer, PiezoSolver  # noqa: E402

L, X_SRC, X_ABS, DX = 12.0, 2.0, 8.0, 0.025
FREQS = (2e9, 4e9, 6e9)


def reflection(kind: str, strength: float, freq: float) -> float:
    n1 = int(round(L / DX))
    xc = (np.arange(n1) + 0.5) * DX
    ramp = np.clip((xc - X_ABS) / (L - X_ABS), 0, None) ** 2
    mass, stretch = None, (None, None)
    if kind == "damping":
        mass = (1 - 1j * strength * ramp)[:, None]
    elif kind == "pml":
        stretch = (1 - 1j * strength * ramp, None)
    layer = PiezoLayer(0.22, np.ones((n1, 1)), SI_ISOTROPIC, eps_r=11.7, order=3, mass_scale=mass)
    sol = PiezoSolver(np.diag([L, 0.05]), [layer], bc=("free", "periodic"), stretch=stretch, shape=(n1, 1))
    force = np.zeros((n1 + 1, 2, sol.n_func, 3))
    i = int(round(X_SRC / DX))
    force[i, :, 0, 1] = force[i, :, -1, 1] = 1e-6  # top and bottom faces, along y
    u = sol.harmonic(freq, {}, force=force).u[:, 0, -1, 1]  # top face, u_y
    x = np.arange(n1 + 1) * DX
    sel = (x > 3.0) & (x < X_ABS)
    G = SI_ISOTROPIC.C[5, 5]
    k0 = 2 * np.pi * freq / np.sqrt(G / SI_ISOTROPIC.rho) * 1e-6
    best = (np.inf, 0.0)
    for k in k0 * np.linspace(0.97, 1.03, 601):
        Bm = np.column_stack([np.exp(-1j * k * x[sel]), np.exp(1j * k * x[sel])])
        c, res, *_ = np.linalg.lstsq(Bm, u[sel], rcond=None)
        r = np.linalg.norm(Bm @ c - u[sel])
        if r < best[0]:
            best = (r, abs(c[1] / c[0]))
    return best[1]


def main() -> None:
    cases = [("free end", "none", 0.0), ("damping α=1", "damping", 1.0), ("damping α=4", "damping", 4.0)]
    cases += [("PML β=2", "pml", 2.0), ("PML β=5", "pml", 5.0)]
    print("|R| of SH0 at " + ", ".join(f"{f / 1e9:.0f} GHz" for f in FREQS))
    for name, kind, s in cases:
        print(f"  {name:12s}: " + "  ".join(f"{reflection(kind, s, f):.3e}" for f in FREQS), flush=True)


if __name__ == "__main__":
    main()
