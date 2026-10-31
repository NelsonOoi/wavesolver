"""
End-to-end finite-difference check of the device-level adjoint (omc.codesign).

Test structure (small on purpose): one period a = 0.48 µm along x (Bloch k = 0 for the
mechanics, X point for the optics), and across y from −1.36 to 2.0 µm:
  - a 0.5 µm Si nanobeam with one r = 100 nm hole per period, which holds the optical mode;
  - 80 nm Si tethers to Si membranes on both sides;
  - on the top membrane, 250 nm X-cut LN (x_cut(45)) with one hot and one grounded electrode
    finger per period: the transducer;
  - a bottom membrane, ending in a PML (coordinate stretch) in the lossy copy: mechanical
    leakage.
PiezoSolver 12 × 84 elements (40 nm; Si and LN order 2; about 12k DOFs). Optics in the window
|y| < 0.96 µm on 6 × 24 × 32 (Lz 1.28 µm), moving-boundary raster ×8 of the optics, Si fill a
4 × 4 raster per mechanical element (one Si raster for both physics), cubic Si [100] ∥ x.

Checks (central FD; every FD point re-solves optics, mechanics and the PML copy; modes followed
by overlap):
  0. the JAX FOM against system.three_mode_response, and its partials against FD;
     g0 of the adapted piezo mode against coupling.g0; g_em against PiezoSolver.motional;
  1. g0, g_em, f_m and γ_leak separately, and η (and ∫η dν) through the combined adjoint, along
     a smooth and an edge-localised Si raster direction, an LN-fraction direction and the LN
     angle;
  2. the light-cone Q proxy gradient (LightConeQ) on the blade cell at Γ.

    pixi run python validation/codesign_gradient_check.py
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omc.codesign import (  # noqa: E402
    Design,
    DeviceModel,
    LightConeQ,
    OpticalWindow,
    SystemParams,
    eta_partials,
    piezo_mech_mode,
    PiezoSiMap,
    system_check,
)
from omc.coupling import block_mean, g0 as g0_value, optical_mode  # noqa: E402
from omc.materials import LN_PIEZO, SI, SI_EPS_R  # noqa: E402
from omc.piezo import Electrode, PiezoLayer, PiezoSolver  # noqa: E402

LX, YLO, YHI = 0.48, -1.36, 2.0
N1, N2, SUB = 12, 84, 4
OUT = ROOT / "validation/out"


def si_raster(r_hole=0.1, w_beam=0.5, w_teth=0.08):
    m1, m2, aa = N1 * SUB, N2 * SUB, 4
    x = -LX / 2 + (np.arange(m1 * aa) + 0.5) * LX / (m1 * aa)
    y = YLO + (np.arange(m2 * aa) + 0.5) * (YHI - YLO) / (m2 * aa)
    X, Y = np.meshgrid(x, y, indexing="ij")
    beam = (np.abs(Y) < w_beam / 2) & (X**2 + Y**2 > r_hole**2)
    teth = (np.abs(X) > LX / 2 - w_teth / 2) & (np.abs(Y) <= 0.96)
    memb = np.abs(Y) > 0.96
    return (beam | teth | memb).astype(float).reshape(m1, aa, m2, aa).mean(axis=(1, 3))


def build(pml: bool, fine, ln, angle=45.0):
    yc = YLO + (np.arange(N2) + 0.5) * (YHI - YLO) / N2
    xc = -LX / 2 + (np.arange(N1) + 0.5) * LX / N1
    rows = (yc > 1.04) & (yc < 1.96)
    hot = np.outer(xc < -0.08, rows)
    gnd = np.outer((xc > 0.0) & (xc < 0.16), rows)
    layers = [
        PiezoLayer(0.22, block_mean(fine, N1, N2), SI, eps_r=SI_EPS_R, order=2),
        PiezoLayer(0.25, ln, LN_PIEZO, order=2, angle=angle),
    ]
    st = 1 - 3j * np.clip((-1.04 - yc) / 0.32, 0, None) ** 2 if pml else None
    return PiezoSolver(
        np.diag([LX, YHI - YLO]), layers, bc=("periodic", "free"),
        electrodes=[Electrode("hot", hot, 2), Electrode("gnd", gnd, 2, "ground")],
        origin=(-LX / 2, YLO), stretch=(None, st), min_fraction=1e-9,
    )


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def smooth(shape, rng, width):
    k = np.add.outer((np.fft.fftfreq(shape[0]) * shape[0] / width) ** 2, (np.fft.fftfreq(shape[1]) * shape[1] / width) ** 2)
    s = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape)) * np.exp(-0.5 * k)))
    return s / np.abs(s).max()


def check_fom(P: SystemParams, q: dict):
    print("0. FOM model")
    sc = system_check(q, P)
    val, part = eta_partials(q, replace(P, fom="eta"))
    if P.fom != "eta":
        print("  (value check on the resonant η; the partials below are of the chosen FOM)")
        part = eta_partials(q, P)[1]
    print(f"  η(ν=0) JAX {val:.10e}; system.three_mode_response {sc['eta_response']:.10e} (rel {rel(val, sc['eta_response']):.1e}); "
          f"closed form (no Stokes term) {sc['eta_closed']:.10e}")
    for k in ("g0", "g_em", "omega_m", "omega_o", "gamma_leak"):
        h = (1e-3 if k == "omega_m" else 1e-6) * max(abs(q[k]), 1.0)  # η depends on ω_m only through the tiny Stokes term
        qp, qm = dict(q), dict(q)
        qp[k] += h
        qm[k] -= h
        fd = (eta_partials(qp, P)[0] - eta_partials(qm, P)[0]) / (2 * h)
        print(f"    ∂η/∂{k:10s} JAX {part[k]:+.6e}  FD {fd:+.6e}  rel {rel(part[k], fd):.1e}")
    Pb = replace(P, fom="eta_bw")
    vb, _ = eta_partials(q, Pb)
    Pb2 = replace(Pb, n_nu=16001, nu_span=24.0)
    print(f"  ∫η dν/2π = {vb:.6e} Hz (quadrature check with 4× points, 2× span: {eta_partials(q, Pb2)[0]:.6e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=1e-3)
    ap.add_argument("--angle_step", type=float, default=1e-2)
    ap.add_argument("--fom", default="eta", choices=("eta", "eta_bw"))
    ap.add_argument("--skip_q", action="store_true")
    args = ap.parse_args()
    warnings.simplefilter("ignore", UserWarning)
    OUT.mkdir(exist_ok=True)
    fine = si_raster()
    yc = YLO + (np.arange(N2) + 0.5) * (YHI - YLO) / N2
    ln = np.outer(np.ones(N1), ((yc > 1.04) & (yc < 1.96)).astype(float))
    P = SystemParams(fom=args.fom)
    win = OpticalWindow(j0=10, n2w=48, n_opt=(6, 24), nz=32, height=1.28, eps_si=3.48**2, k_opt=np.array([np.pi / LX, 0.0]))
    pml_rows = np.outer(np.ones(N1, bool), yc < -1.04)
    model = DeviceModel(build(False, fine, ln), win, "hot", P, sol_pml=build(True, fine, ln), leak_regions={"bottom PML": pml_rows})
    base = Design(fine, [ln], 45.0)
    model.set_design(base)
    ev = model.evaluate(4.0e9, 1, shift_hz=3.8e9, n_modes=8)
    model.opt_ref = None
    t = time.perf_counter()
    model.set_design(base)
    ev = model.evaluate(4.0e9, 1, shift_hz=3.8e9, n_modes=8)
    t_eval = time.perf_counter() - t
    mode_ref, band_ref = ev.mode_u, ev.opt_x
    q = ev.q
    inf = ev.info
    print(f"structure: PiezoSolver {model.sol.shape} ({model.sol.ndof} dof), optics {win.n_opt + (win.nz,)}; "
          f"mode {q['f_m'] / 1e9:.4f} GHz, optical {q['f_o'] / 1e12:.2f} THz ({299792.458 / (q['f_o'] / 1e12):.0f} nm)")
    print(f"  g0/2π {q['g0'] / 2 / np.pi / 1e3:+.2f} kHz, g_em/2π {q['g_em'] / 2 / np.pi / 1e3:.2f} kHz, γ_leak/2π {q['gamma_leak'] / 2 / np.pi / 1e6:.2f} MHz "
          f"(PML-copy match overlap {inf['leak_match'][0]:.3f}); FOM ({args.fom}) {ev.fom:.4e}")
    print(f"  leakage split: {inf.get('leak_split')}")
    print(f"  optical window: {inf['window']}")
    print(f"  timing (η + ∇η + constraints): total {t_eval:.1f} s = optics {inf['t_optics']:.1f} + mechanics eigen {inf['t_mech_eigen']:.1f} + "
          f"g0 core (incl. MINRES {inf['g0_opt_iterations']} it, {inf['g0_t_opt_adjoint']:.1f} s) {inf['t_g0_core']:.1f} + PML copy {inf['t_leak']:.1f} + "
          f"bordered solve and forms {inf['t_gradient']:.1f}; mechanical adjoint residual {inf['mech_adjoint_residual']:.1e}")
    # adapter check against coupling.g0
    so = model.optics()
    sol = model.sol
    fr, U, PHI, ctx = model.mechanics(8, 3.8e9)
    m = inf["mode"]
    nur = ctx["red"][0].shape[1]
    y = np.real(ctx["ys"][m])
    smap = PiezoSiMap(sol, ctx["red"], win)
    fsi_w = block_mean(fine, N1, N2)[:, win.j0 : win.j0 + win.n2w]
    mode, _ = piezo_mech_mode(smap, y[:nur], fr[m].real, float(y[:nur] @ (ctx["Mr"] @ y[:nur])), win.lattice(sol), fsi_w)
    c = g0_value(optical_mode(so, inf["band"]), mode, win.fine_window(fine))
    print(f"  adapter: coupling.g0 on the adapted mode {c.g0 / 2 / np.pi / 1e3:+.4f} kHz vs pipeline {q['g0'] / 2 / np.pi / 1e3:+.4f} kHz "
          f"(rel {rel(abs(c.g0), abs(q['g0'])):.1e}; ARPACK vs Rayleigh frequency)")
    mb = sol.motional("hot", 8, 3.8e9)
    j = int(np.argmin(abs(mb.freqs - q["f_m"])))
    print(f"  g_em vs PiezoSolver.motional: rel {rel(q['g_em'], mb.g_em[j]):.1e}")
    check_fom(P, q)

    # gradients of the separate quantities (weights) and of the FOM, one evaluation each
    grads = {}
    for name, wts in (("g0", {"g0": 1.0}), ("g_em", {"g_em": 1.0})):
        model.set_design(base)
        e2 = model.evaluate(mode_ref, band_ref, shift_hz=3.8e9, n_modes=8, weights=wts)
        grads[name] = e2.grad["fom"]
        if name == "g0":
            sign0 = np.sign(e2.q["g0"])  # the eigenvector sign of that evaluation
    grads["fom"] = ev.grad["fom"]
    grads["f_m"] = ev.grad["constraints"]["f_m"]
    grads["leak"] = ev.grad["constraints"]["leak"]
    grads["wavelength"] = ev.grad["constraints"]["wavelength"]

    def values(e):
        return dict(g0=abs(e.q["g0"]), g_em=e.q["g_em"], fom=e.fom, f_m=e.constraints["f_m"], leak=e.constraints["leak"],
                    wavelength=e.constraints["wavelength"])

    rng = np.random.default_rng(0)
    solid = (fine > 0).astype(float)
    dirs = {
        "Si raster smooth": ("si", smooth(fine.shape, rng, 8) * solid),
        "Si raster edge": ("si", ((fine > 0.02) & (fine < 0.98)) * rng.standard_normal(fine.shape)),
        "LN fraction": ("ln", (ln > 0) * rng.uniform(-1, 1, ln.shape)),
        "LN angle": ("angle", None),
    }
    print(f"1. FD along design directions (step {args.step:g} on max|δ| = 1; angle step {args.angle_step:g}°)")
    results = {}
    for dname, (kind, d) in dirs.items():
        if d is not None:
            d = d / np.abs(d).max()
        h = args.angle_step if kind == "angle" else args.step
        vals = []
        for s in (h, -h):
            if kind == "si":
                dz = Design(fine + s * d, [ln], 45.0)
            elif kind == "ln":
                dz = Design(fine, [ln + s * d], 45.0)
            else:
                dz = Design(fine, [ln], 45.0 + s)
            model.set_design(dz)
            model.opt_ref = so
            vals.append(values(model.evaluate(mode_ref, band_ref, shift_hz=3.8e9, n_modes=8, gradient=False)))
        model.set_design(base)
        for qn in ("g0", "g_em", "f_m", "leak", "wavelength", "fom"):
            g = grads[qn]
            if kind == "si":
                an = (g["si_fine"] * d).sum()
            elif kind == "ln":
                an = (g["layers"][0] * d).sum()
            else:
                an = g["angle"]
            if qn == "g0":
                an = sign0 * an
            fd = (vals[0][qn] - vals[1][qn]) / (2 * h)
            if qn == "wavelength" and kind != "si":
                print(f"  {dname:17s} {qn:10s} adjoint {an:+.6e}  FD {fd:+.6e}  (no dependence)")
                continue
            e = rel(an, fd)
            results[f"{dname}|{qn}"] = (an, fd, e)
            print(f"  {dname:17s} {qn:10s} adjoint {an:+.6e}  FD {fd:+.6e}  rel err {e:.1e}", flush=True)

    if not args.skip_q:
        print("2. light-cone Q proxy (blade cell, Γ, band 1)")
        from omc.geometry import HexLattice, blade_hole, solid_fraction
        from omc.optics_pwe import PWESolver, slab_cell

        lat = HexLattice(0.448)
        fr0 = solid_fraction(lat, [blade_hole(0.172, 0.093, 0.025)], 16)
        qm = LightConeQ(z0=0.3)

        def qval(fr, prev=None):
            s = PWESolver(slab_cell(lat, fr, 0.22, 1.5, 48, 3.48**2), parity="even")
            if prev is not None:
                s._prev = prev._prev
            s.solve(np.array([0.3, 0.1]), 4, tol=1e-9, maxiter=600)
            return s

        s0 = qval(fr0)
        bq = 2
        Q0, dQ = qm(s0, bq)
        print(f"  band {bq} at k = (0.3, 0.1)/µm: {s0._last_freqs[bq]:.2f} THz, Q_proxy {Q0:.4e}")
        for dname, d in (("smooth", smooth(fr0.shape, rng, 3)), ("edge", ((fr0 > 0.02) & (fr0 < 0.98)) * rng.standard_normal(fr0.shape))):
            d = d / np.abs(d).max()
            hq = 2.5e-4
            qp = qm(qval(fr0 + hq * d, s0), bq)[0]
            qn_ = qm(qval(fr0 - hq * d, s0), bq)[0]
            fd = (qp - qn_) / (2 * hq)
            an = (dQ * d).sum()
            print(f"  Q {dname:7s} adjoint {an:+.6e}  FD {fd:+.6e}  rel err {rel(an, fd):.1e}")
            results[f"Q|{dname}"] = (an, fd, rel(an, fd))
    np.savez(OUT / "codesign_gradient_check.npz", **{k: np.array(v) for k, v in results.items()}, t_eval=t_eval)


if __name__ == "__main__":
    main()
