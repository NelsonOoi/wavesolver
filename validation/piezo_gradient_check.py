"""
Finite-difference checks of the piezo sensitivities (wavesolver.piezo).

Problem: the periodic IDT section of validation/piezo_angle.py (250 nm LN at θ = 30°, which
breaks every mirror symmetry) on a coarse mesh. Design variables: θ, and the fraction maps of
the Si layer and of the two LN sub-layers (random directions on their solid elements, kept
inside the elements' solid set so the DOF set does not change). Checked: material-tensor
θ-derivatives; short- and open-circuit eigenfrequencies and k_eff² (k = 0, real path); the
same at a Bloch k ≠ 0 (Hermitian, complex path); the admittance Y(ω) and a linear field
functional (complex weights on the top-surface displacement) by the adjoint; and a lossy
case (complex Si density, complex-symmetric path) for eigenfrequencies and Y. Central
differences: θ ± 1e-3°, fraction ± 1e-4 along the direction. Modes closer than 1e-3 to a
neighbour are skipped (non-degenerate quantities only).

    pixi run python validation/piezo_gradient_check.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

from piezo_angle import idt_cell  # noqa: E402

from wavesolver.materials import LN_PIEZO, rotation_derivatives, x_cut, x_cut_derivative  # noqa: E402

MESH = (4, 8, 2, 2, 2, 1)
THETA = 30.0
H_ANG, H_FRAC = 1e-3, 1e-4
rng = np.random.default_rng(1)
worst = {}
SECTION = [""]


def report(name, an, fd):
    an, fd = np.atleast_1d(an), np.atleast_1d(fd)
    err = np.abs(an - fd) / np.maximum(np.abs(fd), 1e-3 * np.abs(fd).max() + 1e-30)
    key = f"{SECTION[0]}: {name.split(' (')[0]}"
    worst[key] = max(worst.get(key, 0.0), err.max())
    print(f"  {name:38s} max rel err {err.max():.1e}   (analytic {np.array2string(an[:3], precision=6)}, FD {np.array2string(fd[:3], precision=6)})", flush=True)


def directions(sol):
    """Random fraction directions on the solid elements of layers 0 (Si), 1, 2 (LN)."""
    dirs = {}
    for l in (0, 1, 2):
        d = rng.uniform(-1, 1, sol.shape) * sol.parts[l].solid.reshape(sol.shape)
        dirs[l] = d
    return dirs


def perturbed(make, sol, l, d, h):
    base = sol.parts[l].f.reshape(sol.shape).copy()
    out = []
    for s in (+1, -1):
        sol.set_fraction(l, base + s * h * d)
        out.append(make())
    sol.set_fraction(l, base)
    return out


def align(f_ref, f):
    """f reordered to the nearest frequency of each reference mode (ARPACK may miss a mode of a cluster)."""
    f_ref, f = np.asarray(f_ref), np.asarray(f)
    return f[np.argmin(np.abs(f_ref.real[:, None] - f.real[None, :]), axis=1)]


def separated(f, rtol=1e-3):
    f = np.maximum(np.asarray(f).real, 1.0)
    gap = np.full(len(f), np.inf)
    gap[1:] = np.minimum(gap[1:], np.diff(f) / f[1:])
    gap[:-1] = np.minimum(gap[:-1], np.diff(f) / f[1:])
    return gap > rtol


def check_eigen(sol, label, k=(0.0, 0.0), n=10, shift=3.0e9):
    print(f"{label}:")
    SECTION[0] = label.split(",")[0] + (" lossy" if sol.lossy else "") + (" Bloch" if any(k) else "")
    dirs = directions(sol)
    for elec in ("short", "open"):
        t0 = time.perf_counter()
        es = sol.eigen_sensitivities(n, shift, k, elec)
        dt = time.perf_counter() - t0
        ok = separated(es.freqs) & (es.freqs.real > 1e8)
        th = sol.angle
        sol.set_angle(th + H_ANG)
        fp = align(es.freqs, sol.eigenmodes(n, shift, k, elec))
        sol.set_angle(th - H_ANG)
        fm = align(es.freqs, sol.eigenmodes(n, shift, k, elec))
        sol.set_angle(th)
        report(f"{elec} df/dθ ({ok.sum()} modes, {dt:.1f} s)", es.dfreq_angle[ok], ((fp - fm) / (2 * H_ANG))[ok])
        for l, d in dirs.items():
            fp, fm = perturbed(lambda: align(es.freqs, sol.eigenmodes(n, shift, k, elec)), sol, l, d, H_FRAC)
            an = np.einsum("mij,ij->m", es.dfreq_frac[l], d)
            report(f"{elec} df/dfrac layer {l}", an[ok], ((fp - fm) / (2 * H_FRAC))[ok])


def check_k2(sol):
    print(f"k_eff² (k = 0, θ = {sol.angle}°):")
    SECTION[0] = f"k² θ={sol.angle:g}"
    t0 = time.perf_counter()
    cs = sol.coupling_sensitivities(10, 3.0e9)
    dt = time.perf_counter() - t0
    ok = (cs.overlap > 0.9) & separated(cs.f_sc) & (cs.f_sc > 1e8) & (cs.k2 > 1e-5)
    th = sol.angle
    vals = []
    for s in (+1, -1):
        sol.set_angle(th + s * H_ANG)
        fs, _, k2, _ = sol.coupling(10, 3.0e9)
        vals.append(k2[np.argmin(np.abs(cs.f_sc[:, None] - fs[None, :]), axis=1)])
    sol.set_angle(th)
    report(f"dk²/dθ ({ok.sum()} modes, {dt:.1f} s for both states)", cs.dk2_angle[ok], ((vals[0] - vals[1]) / (2 * H_ANG))[ok])
    for l, d in directions(sol).items():
        def k2f():
            fs, _, k2, _ = sol.coupling(10, 3.0e9)
            return k2[np.argmin(np.abs(cs.f_sc[:, None] - fs[None, :]), axis=1)]

        kp, km = perturbed(k2f, sol, l, d, H_FRAC)
        report(f"dk²/dfrac layer {l}", np.einsum("mij,ij->m", cs.dk2_frac[l], d)[ok], ((kp - km) / (2 * H_FRAC))[ok])


def check_driven(sol, label, freq):
    print(f"{label}, f = {freq / 1e9} GHz:")
    SECTION[0] = "driven" + (" lossy" if sol.lossy else "")
    t0 = time.perf_counter()
    Y, dYa, dYf = sol.admittance_gradient(freq, "hot")
    dt = time.perf_counter() - t0
    Yf = lambda: sol.admittance([freq], ["hot"])[0, 0, 0]  # noqa: E731
    th = sol.angle
    sol.set_angle(th + H_ANG)
    yp = Yf()
    sol.set_angle(th - H_ANG)
    ym = Yf()
    sol.set_angle(th)
    report(f"dY/dθ (Y = {Y:.3e} S; {dt:.1f} s incl. adjoint)", dYa, (yp - ym) / (2 * H_ANG))
    dirs = directions(sol)
    for l, d in dirs.items():
        yp, ym = perturbed(Yf, sol, l, d, H_FRAC)
        report(f"dY/dfrac layer {l}", np.sum(dYf[l] * d), (yp - ym) / (2 * H_FRAC))
    # field functional: complex weights on u_x, u_y of the top surface of the LN
    wts = np.zeros((sol.shape[0] + 1, sol.shape[1] + 1, sol.n_func, 3), complex)
    top = sol.interface_function(3)
    wts[:, :, top, :2] = rng.normal(size=wts[:, :, top, :2].shape) + 1j * rng.normal(size=wts[:, :, top, :2].shape)
    J, dJa, dJf = sol.field_gradient(freq, {"hot": 1.0}, wts)
    Jf = lambda: np.sum(wts * sol.harmonic(freq, {"hot": 1.0}).u)  # noqa: E731
    sol.set_angle(th + H_ANG)
    jp = Jf()
    sol.set_angle(th - H_ANG)
    jm = Jf()
    sol.set_angle(th)
    report(f"dJ/dθ (field functional, J = {J:.3e})", dJa, (jp - jm) / (2 * H_ANG))
    for l, d in dirs.items():
        jp, jm = perturbed(Jf, sol, l, d, H_FRAC)
        report(f"dJ/dfrac layer {l}", np.sum(dJf[l] * d), (jp - jm) / (2 * H_FRAC))


def main() -> None:
    print("material tensors:")
    for th in (0.0, 30.0, 45.0, 101.0):
        R, dR = x_cut(th), x_cut_derivative(th)
        an = rotation_derivatives(LN_PIEZO, R, dR)
        p, m = LN_PIEZO.rotated(x_cut(th + H_ANG)), LN_PIEZO.rotated(x_cut(th - H_ANG))
        for name, a, fp, fm in zip(("C", "e", "eps"), an, (p.C, p.e, p.eps_r), (m.C, m.e, m.eps_r)):
            fd = (fp - fm) / (2 * H_ANG)
            err = np.abs(a - fd).max() / np.abs(a).max()
            worst[f"material: d{name}/dθ"] = max(worst.get(f"material: d{name}/dθ", 0), err)
    print("  " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))

    sol = idt_cell(THETA, *MESH)
    print(f"IDT cell: {sol.ndof} dof (full grid)")
    check_eigen(sol, "eigenfrequencies, k = 0")
    check_k2(sol)
    sol.set_angle(80.0)
    check_k2(sol)
    sol.set_angle(THETA)
    check_eigen(sol, "eigenfrequencies, Bloch k = 3/µm (Hermitian)", k=(3.0, 0.0))
    check_driven(sol, "admittance and field functional", 4.6e9)
    n1, n2 = sol.shape
    lossy = idt_cell(THETA, *MESH, mass_scale=np.full((n1, n2), 1 - 0.01j))
    check_eigen(lossy, "lossy (complex Si density), k = 0, complex symmetric", n=6)
    check_driven(lossy, "lossy admittance", 4.6e9)
    print("worst relative errors:")
    for k, v in worst.items():
        print(f"  {k:36s} {v:.1e}")


if __name__ == "__main__":
    main()
