"""
Finite-difference check of the adjoint g0 gradient (omc.coupling_adjoint.g0_gradient).

Case (default): blade cell (a = 448 nm, r = 172 nm, w = 93 nm, 25 nm fillet), cubic Si with
[100] ∥ x (materials.SI, SI_P unrotated). Optics N × N × NZ at M (TE-like); mechanics slab_p
even at Γ on a 2N grid; moving-boundary raster ×8. The optical band and mechanical mode are
picked as a well-separated, strongly coupled, non-degenerate pair.

g0 is a function of three rasters, g0(f_fine, f_o, f_m) (coupling.g0). Checks, each a central
difference of coupling.g0 after re-solving both eigenproblems from scratch:
  - f_o only (optics re-solved; f_m, f_fine fixed), smooth and edge-localised directions;
  - f_m only (mechanics re-solved), smooth and edge (also m_eff at the fixed peak point, and f_m);
  - f_fine only (explicit moving-boundary raster);
  - all three together through the block means (f_o, f_m = block means of f_fine);
  - shape parameters r, w of omc.parametrize.ParametricCell(TripodHole) through the raster.
Smooth directions are restricted to where the mechanics keeps elements, so no element crosses
the solver's min_fraction (dropped elements have zero gradient by construction). The sign of
the mechanical mode is fixed by its overlap with the unperturbed one.

Notes from the runs (validation/out/g0_gradient_check_blade.log):
- min_fraction. The SDF raster (smoothed coverage) gives element fractions of any small size,
  so with BlochSolver's default min_fraction = 1e-3 a θ step can add or drop an element (the
  DOF set changes), a jump that FD sees and the gradient cannot. The θ check therefore uses
  min_fraction = 1e-9 (--min_fraction). The fraction-level checks are unaffected.
- Step sizes. The remaining FD error is truncation and falls as h²: the regularised interface
  normals (η ≈ 0.5/µm here) make ε⁻¹ curve strongly for smooth fill changes in uniform regions
  (f_o smooth: 1.7e-3, 4.2e-4, 1.1e-4 at h = 1e-3, 5e-4, 2.5e-4), and the smoothed-coverage
  raster curves on its smoothing width a/(n·supersample) ≈ 1.2 nm (θ: 5e-3 at 0.2 nm, 4e-4 at
  0.05 nm, 4e-5 at 0.015 nm).

    pixi run python validation/g0_gradient_check.py              # blade cell
    pixi run python validation/g0_gradient_check.py --quick      # fewer checks
    pixi run python validation/g0_gradient_check.py --case line  # coarse GPT line defect: timing + 2 directions
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omc.coupling import block_mean, g0 as g0_value, mech_mode, optical_mode  # noqa: E402
from omc.coupling_adjoint import fine_total_gradient, g0_gradient  # noqa: E402
from omc.geometry import HexLattice, blade_hole, line_defect_supercell, solid_fraction  # noqa: E402
from omc.materials import SI, SI_P  # noqa: E402
from omc.mech_adjoint import find_clusters  # noqa: E402
from omc.mech_bloch import BlochSolver  # noqa: E402
from omc.optics_pwe import PWESolver, slab_cell  # noqa: E402
from omc.parametrize import ParametricCell, TripodHole  # noqa: E402

T, N_SI = 0.22, 3.48
OUT = ROOT / "validation/out"


class Problem:
    """Forward model g0(f_fine, f_o, f_m) with fixed band / mode tracking."""

    def __init__(self, lattice, k_opt, nz, height, nb_opt, nb_mech, tol, maxiter, min_fraction=1e-3):
        self.min_fraction = min_fraction
        self.lattice, self.k_opt, self.nz, self.height = lattice, k_opt, nz, height
        self.nb_opt, self.nb_mech, self.tol, self.maxiter = nb_opt, nb_mech, tol, maxiter
        self.ref_opt = None
        self.ref_y = None
        self.band = self.mode = None
        self.t_opt = self.t_mech = 0.0
        self.ref_ndof, self.dof_changes = None, 0

    def optics(self, fo):
        t = time.perf_counter()
        s = PWESolver(slab_cell(self.lattice, fo, T, self.height, self.nz, N_SI**2), parity="even")
        if self.ref_opt is not None:
            s._prev = self.ref_opt._prev  # warm start from the unperturbed modes
        s.solve(self.k_opt, self.nb_opt, tol=self.tol, maxiter=self.maxiter)
        self.t_opt = time.perf_counter() - t
        self.last_residual = float(np.abs(s.residuals).max())
        return s

    def mechanics(self, fm):
        t = time.perf_counter()
        ms = BlochSolver(self.lattice, fm, SI, model="slab_p", thickness=T, parity="even", min_fraction=self.min_fraction)
        f, X = ms.solve(np.zeros(2), self.nb_mech, vectors=True)
        self.t_mech = time.perf_counter() - t
        return ms, f, X

    def real_mode(self, ms, X, j):
        """Real Γ mode j, tracked by overlap with the reference and sign-aligned, on the full node DOF array."""

        def full(jj):
            y = (X[:, jj] * np.exp(-0.5j * np.angle(np.sum(X[:, jj] ** 2)))).real
            out = np.zeros(ms.shape[0] * ms.shape[1] * len(ms.basis))
            out[ms.dofs] = y
            return y, out

        y, yf = full(j)
        if self.ref_y is not None:
            ov = [abs(full(jj)[1] @ self.ref_y) / np.linalg.norm(full(jj)[1]) for jj in range(X.shape[1])]
            jj = int(np.argmax(ov))
            if jj != j:
                j = jj
                y, yf = full(j)
            if yf @ self.ref_y < 0:
                y = -y
        self.n_dof = len(y)
        return j, y, yf

    def value(self, fine, fo, fm, opt=None, mech=None):
        so = opt if opt is not None else self.optics(fo)
        ms, f, X = mech if mech is not None else self.mechanics(fm)
        j, y, _ = self.real_mode(ms, X, self.mode)
        if self.n_dof != self.ref_ndof:
            self.dof_changes += 1
        mode = mech_mode(ms, np.zeros(2), f[j], y.astype(complex), fraction=fm)
        c = g0_value(optical_mode(so, self.band), mode, fine, p=SI_P)
        return c.g0, mode.mass, f[j]


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def smooth_field(shape, rng, width):
    n1, n2 = shape
    k = np.add.outer((np.fft.fftfreq(n1) * n1 / width) ** 2, (np.fft.fftfreq(n2) * n2 / width) ** 2)
    s = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape)) * np.exp(-0.5 * k)))
    return s / np.abs(s).max()


def edge_field(frac, rng):
    d = ((frac > 0.02) & (frac < 0.98)) * rng.standard_normal(frac.shape)
    return d / np.abs(d).max()


def blade_case(args):
    lattice = HexLattice(0.448)
    N, m, nm = args.n, args.m, 2 * args.n
    pc = ParametricCell(lattice, TripodHole(), free=("r", "w"))  # the blade, r = 172, w = 93 nm, 25 nm rounding
    theta0 = pc.theta0()
    if args.raster == "sdf":
        fine = pc.fraction(theta0, N * m, supersample=2)
    else:
        fine = solid_fraction(lattice, [blade_hole(0.172, 0.093, 0.025)], N * m, supersample=2)
    k = np.pi / 0.448 * np.array([1.0, 1 / np.sqrt(3)])  # M
    prob = Problem(lattice, k, args.nz, 1.5, 4, args.nbm, args.tol, args.maxiter, args.min_fraction)
    return prob, fine, (N, N), (nm, nm), pc, theta0


def line_case(args):
    lattice, holes = line_defect_supercell(cell="defect", shield_rows=args.rows)
    a = lattice.vectors[0, 0]
    n1o, n2o = args.n1o, args.n2o
    fine = solid_fraction(lattice, holes, (n1o * args.m, n2o * args.m), supersample=2)
    k = np.array([np.pi / a, 0.0])
    prob = Problem(lattice, k, args.nz, 1.5, args.nbo, args.nbm, args.tol, args.maxiter, args.min_fraction)
    return prob, fine, (n1o, n2o), (2 * n1o, 2 * n2o), None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="blade", choices=("blade", "line"))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--nz", type=int, default=64)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--nbm", type=int, default=20)
    ap.add_argument("--raster", default="sdf", choices=("sdf", "polygon"))
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--n1o", type=int, default=8)
    ap.add_argument("--n2o", type=int, default=64)
    ap.add_argument("--nbo", type=int, default=12)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=800)
    ap.add_argument("--step", type=float, default=1e-3)
    ap.add_argument("--theta_step", type=float, default=1.5e-5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--checks", default="mech,fine,opt,all,theta")
    ap.add_argument("--min_fraction", type=float, default=1e-9, help="mechanical element threshold (BlochSolver default 1e-3)")
    ap.add_argument("--opt_steps", default="1e-3,2.5e-4", help="steps for the smooth f_o check (truncation scan)")
    args = ap.parse_args()
    warnings.simplefilter("always")
    OUT.mkdir(exist_ok=True)
    prob, fine, so_shape, sm_shape, pc, theta0 = (blade_case if args.case == "blade" else line_case)(args)
    fo, fm = block_mean(fine, *so_shape), block_mean(fine, *sm_shape)
    rng = np.random.default_rng(0)

    # Base solves and mode choice.
    so = prob.optics(fo)
    t_opt0 = prob.t_opt
    ms, fmech, X = prob.mechanics(fm)
    t_mech0 = prob.t_mech
    fo_thz = so._last_freqs[: prob.nb_opt]
    print(f"case {args.case}: optics {so.shape} (ndof {2 * np.prod(so.shape)}), {t_opt0:.1f} s, max residual {so.residuals.max():.1e}; "
          f"mechanics {ms.shape} ndof {ms.ndof}, {t_mech0:.1f} s", flush=True)
    print(f"  optical bands (THz) {np.round(fo_thz, 2)}; mechanical Γ (GHz) {np.round(fmech / 1e9, 3)}")
    lam_o = fo_thz**2
    singles_o = [b for b in range(prob.nb_opt) if np.abs(np.delete(lam_o, b) - lam_o[b]).min() > 2e-2 * lam_o[b]]
    clusters = find_clusters((fmech / 1e9) ** 2, 3e-3)
    singles_m = [int(c[0]) for c in clusters if len(c) == 1 and fmech[c[0]] > 0.3e9]
    best = (0, None, None)
    for b in singles_o:
        opt = optical_mode(so, b)
        for j in singles_m:
            mode = mech_mode(ms, np.zeros(2), fmech[j], X[:, j], fraction=fm)
            g = abs(g0_value(opt, mode, fine).g0)
            if g > best[0]:
                best = (g, b, j)
    prob.band, prob.mode = best[1], best[2]
    j, y0, y0f = prob.real_mode(ms, X, prob.mode)
    prob.ref_y, prob.ref_opt, prob.ref_ndof = y0f / np.linalg.norm(y0f), so, len(y0)
    print(f"  pair: optical band {prob.band} ({fo_thz[prob.band]:.2f} THz), mechanical mode {prob.mode} ({fmech[prob.mode] / 1e9:.3f} GHz)")

    # Adjoint gradient (twice: the first call includes JAX compilation).
    r = g0_gradient(so, prob.band, ms, fmech[prob.mode], y0.astype(complex), fine, fm, p=SI_P, mech_freqs=fmech)
    t = time.perf_counter()
    r = g0_gradient(so, prob.band, ms, fmech[prob.mode], y0.astype(complex), fine, fm, p=SI_P, mech_freqs=fmech)
    t_grad = time.perf_counter() - t
    g_ref = prob.value(fine, fo, fm, opt=so, mech=(ms, fmech, X))[0]
    inf = r.info
    print(f"  g0/2π = {r.g0 / 2 / np.pi / 1e3:+.2f} kHz (MB {r.g_mb / 2 / np.pi / 1e3:+.2f}, PE {r.g_pe / 2 / np.pi / 1e3:+.2f}); "
          f"coupling.g0 {g_ref / 2 / np.pi / 1e3:+.2f} kHz, rel diff {rel(r.g0, g_ref):.1e}; m_eff {r.m_eff:.3e} kg")
    print(f"  timing: forward optics {t_opt0:.2f} s + mechanics {t_mech0:.2f} s; gradient {t_grad:.2f} s "
          f"(fields+JAX {inf['t_fields']:.2f}, optical MINRES {inf['opt_iterations']} it {inf['t_opt_adjoint']:.2f} s, "
          f"mech factor {inf['t_mech_factor']:.2f} s, mech total {inf['t_mech_adjoint']:.2f} s)")
    print(f"  residuals: optical adjoint {inf['opt_residual']:.1e}, mechanical {inf['mech_residual']:.1e}; ε⁻¹ rebuild {inf['eps_rebuild_err']:.1e}; "
          f"rhs ⊥ eigvec: optics {inf['opt_rhs_overlap']:.1e}, mechanics {inf['mech_rhs_overlap']:.1e}", flush=True)

    h = args.step
    results = {}

    def report(name, an, fp, fm_, step, extra=""):
        fd = (fp - fm_) / (2 * step)
        e = rel(an, fd)
        results[name] = (an, fd, e)
        print(f"  {name:28s} adjoint {an:+.6e}  FD {fd:+.6e}  rel err {e:.1e}{extra}", flush=True)

    def check_opt(name, d, steps=(h,)):
        an = (r.d_opt * d).sum()
        for st in steps:
            gp = prob.value(fine, fo + st * d, fm, mech=(ms, fmech, X))[0]
            gm = prob.value(fine, fo - st * d, fm, mech=(ms, fmech, X))[0]
            report(name + ("" if len(steps) == 1 else f" h={st:.1e}"), an, gp, gm, st, f"   (LOBPCG residual {prob.last_residual:.0e})")

    def check_mech(name, d):
        vp = prob.value(fine, fo, fm + h * d, opt=so)
        vm = prob.value(fine, fo, fm - h * d, opt=so)
        report(name, (r.d_mech * d).sum(), vp[0], vm[0], h)
        report(name + " [m_eff]", (r.d_meff * d).sum(), vp[1], vm[1], h)
        report(name + " [f_m]", (r.d_mech_freq * d).sum(), vp[2], vm[2], h)

    def check_fine(name, d):
        gp = prob.value(fine + h * d, fo, fm, opt=so, mech=(ms, fmech, X))[0]
        gm = prob.value(fine - h * d, fo, fm, opt=so, mech=(ms, fmech, X))[0]
        report(name, (r.d_fine * d).sum(), gp, gm, h)

    def check_all(name, d, step=h):
        an = (fine_total_gradient(r) * d).sum()
        vals = []
        for s in (step, -step):
            f2 = fine + s * d
            vals.append(prob.value(f2, block_mean(f2, *so_shape), block_mean(f2, *sm_shape))[0])
        report(name, an, *vals, step)

    keep_m = (fm > 1e-3).astype(float)
    print("checks (central FD, step {:.0e} on max|δf| = 1):".format(h), flush=True)
    checks = args.checks.split(",")
    dirs = dict(
        m_s=smooth_field(fm.shape, rng, 3) * keep_m, m_e=edge_field(fm, rng),
        f_s=smooth_field(fine.shape, rng, 6), f_e=edge_field(fine, rng),
        o_s=smooth_field(fo.shape, rng, 3), o_e=edge_field(fo, rng),
        a_s=smooth_field(fine.shape, rng, 6) * (fine > 0), a_e=edge_field(fine, rng),
    )
    if "mech" in checks:
        check_mech("f_m smooth", dirs["m_s"])
        check_mech("f_m edge", dirs["m_e"])
    if "fine" in checks:
        check_fine("f_fine smooth", dirs["f_s"])
        check_fine("f_fine edge", dirs["f_e"])
    if "opt" in checks:
        check_opt("f_o smooth", dirs["o_s"], tuple(float(x) for x in args.opt_steps.split(",")))
        check_opt("f_o edge", dirs["o_e"])
    if "all" in checks and not args.quick:
        check_all("all (block means) smooth", dirs["a_s"])
        check_all("all (block means) edge", dirs["a_e"])
    if pc is not None and "theta" in checks and not args.quick:
        # Shape parameters: the raster is pc.fraction(θ) on the fine grid, f_o and f_m its block means.
        N = fine.shape[0]
        J = pc.fraction_jacobian(theta0, N, supersample=2)  # (N, N, 2)
        G = fine_total_gradient(r)
        dth = np.einsum("ij,ijp->p", G, J)
        for p_i, name in enumerate(pc.free):
            vals = []
            for s in (args.theta_step, -args.theta_step):
                th = theta0.copy()
                th[p_i] += s
                f2 = pc.fraction(th, N, supersample=2)
                vals.append(prob.value(f2, block_mean(f2, *so_shape), block_mean(f2, *sm_shape))[0])
            report(f"θ: {name}", dth[p_i], *vals, args.theta_step,
                   f"   (per µm; {(abs(J[..., p_i]) > 0).sum()} raster cells move; mech DOF set changed in {prob.dof_changes} of the solves)")
            prob.dof_changes = 0
    np.savez(OUT / f"g0_gradient_check_{args.case}.npz", **{k: np.array(v) for k, v in results.items()},
             g0=r.g0, t_opt=t_opt0, t_mech=t_mech0, t_grad=t_grad)


if __name__ == "__main__":
    main()
