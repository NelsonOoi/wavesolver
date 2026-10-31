# omc: fast, differentiable mode solvers for optomechanical and phoxonic crystals

`omc` computes the mechanical, optical and piezoelectric modes of patterned membranes (Si slabs,
LN-on-Si stacks), the optomechanical coupling g₀ between them, and the exact gradient of each of
these quantities with respect to every pixel of the pattern. It is built for inverse design: the
geometry enters every solver as a smooth solid-fraction map on a fixed grid, so nothing is remeshed
and every result is differentiable.

On a unit cell, the mechanical solver matches COMSOL to about 0.25 % and needs 40–50× fewer
core-seconds per k-point. Its gradients cost less than one extra solve. Details are under
[Performance against COMSOL](#performance-against-comsol) and [Accuracy](#accuracy).

The code is plain Python (NumPy, SciPy, JAX), runs on a CPU, and is MIT-licensed.

## Contents

- [Quick start](#quick-start)
- [Guide for agents](#guide-for-agents)
- [Performance against COMSOL](#performance-against-comsol)
- [Accuracy](#accuracy)
- [Reproducing the validation](#reproducing-the-validation)
- [Repository layout](#repository-layout)

## Quick start

The recommended environment is [pixi](https://pixi.sh), because it installs CHOLMOD (scikit-sparse)
from conda-forge. CHOLMOD has no PyPI wheels and makes the mechanical solves about 3× faster.

```bash
pixi install                                        # linux-64, osx-arm64 or win-64
pixi run python examples/01_mechanical_bands.py     # about 10 s
```

[uv](https://docs.astral.sh/uv/) also works. Without CHOLMOD, factorisations fall back to SuperLU:

```bash
uv run python examples/01_mechanical_bands.py
```

The pixi environment sets `OMP_NUM_THREADS=1` and `MKL_THREADING_LAYER=SEQUENTIAL`. Set these
yourself if you run Python another way (see [Pitfalls](#pitfalls)).

## Guide for agents

This section is for coding agents (and people) that call the solvers from their own scripts.

### The model in one paragraph

A design is a **solid-fraction map** `f[i, j] ∈ [0, 1]`. Each entry is the share of one grid
element that is solid, on an `n1 × n2` grid aligned with the lattice vectors `a1, a2`. Every solver
takes that map:

- The mechanics FEM scales each element's stiffness and mass by `f`.
- The optics PWE builds a subpixel-smoothed permittivity from `f`.
- The piezo FEM takes one map per layer.

You build `f` in one of three ways:

- from polygons with `geometry.solid_fraction`;
- from shape parameters with `parametrize.ParametricCell`;
- from free-form pixels with `parametrize.DensityCell`, which applies a filter and a projection
  for a minimum feature size.

The solvers return frequencies, plus **∂(quantity)/∂f for every element**. You chain that to your
design variables with `cell.fraction_vjp(theta, dq_df, ...)`. One adjoint gives the whole gradient,
whatever the number of variables.

### Units and conventions

| Quantity | Unit |
|---|---|
| Lengths, lattice constants, thicknesses | µm |
| Mechanical frequencies | Hz |
| Optical frequencies | THz |
| k | rad/µm |
| g₀, g_em | rad/s (divide by 2π for Hz) |

- **Lattice:** `HexLattice(a)` is triangular, with `a1 = (a, 0)` and `a2 = a(1/2, √3/2)`.
  `gamma_m_k_path(n)` returns the Γ–M–K–Γ path with `n` points per leg, plus the corner indices.
  For rectangular cells and supercells, use `Lattice2D(vectors)`.
- **Mechanical parity:** `"even"` means symmetric about the slab mid-plane (in-plane and breathing
  modes); `"odd"` means flexural.
- **Optical parity:** `"even"` means TE-like (σ_z H = H).
- **Materials** (`omc.materials`):

  | Name | Material |
  |---|---|
  | `SI` | Cubic Si, [100] along x |
  | `SI_ISOTROPIC` | Isotropic Si |
  | `SI_P` | Si photoelastic tensor |
  | `LN_PIEZO` | LiNbO₃ stiffness, piezo and permittivity tensors |
  | `AL_COMSOL` | Al |
  | `x_cut(θ)` | X-cut LN rotation, with the c-axis at θ from x |
  | `rot_z`, `rotate_stiffness` | Rotations for other orientations |

### Which module to use

| Task | Module and entry point |
|---|---|
| Lattices, polygon → fraction raster, tripod hole | `omc.geometry`: `HexLattice`, `Lattice2D`, `solid_fraction`, `blade_hole`, `tripod` |
| Differentiable geometry: shape parameters or free-form pixels with a minimum feature size | `omc.parametrize`: `ParametricCell` + `TripodHole`, `DensityCell` (`fraction`, `fraction_vjp`, `feature_check`, `length_scale`) |
| Mechanical Bloch bands (2D, 3D slab, LN/Si stack) | `omc.mech_bloch.BlochSolver` (`model="slab_p"` for membranes), `.solve(k)`, `.bands(ks, workers=…)` |
| Mechanical band gradients, including degenerate clusters | `omc.mech_adjoint.sensitivities`; `BandPool` for parallel k-points |
| Fast band sweeps and finite chains of cells (reduced-order models) | `omc.mech_rom.BlochROM` (`reproject` for new designs on a frozen basis), `ChainROM` (`set_cell`) |
| Optical bands of a patterned slab (MPB's method) | `omc.optics_pwe`: `slab_cell`, `PWESolver` (`.solve`, `.bands`, `.solve_near` targeted, `.sensitivities`) |
| Fast approximate optics with radiation loss Q_rad | `omc.optics_gme.GMESolver` (`.solve(target_thz=…)`, `.radiation`, `.gradient(quantity="freq" or "lnQ")`) |
| Optomechanical g₀ (moving boundary + photoelastic) | `omc.coupling`: `optical_mode`, `mech_mode`, `g0` |
| Adjoint gradient of g₀ | `omc.coupling_adjoint`: `g0_gradient`, `fine_total_gradient` |
| Finite cavities (mechanics with absorbers, tight-binding envelopes) | `omc.cavity`: `chain_domain`, `mech_cavity`, `cavity_modes`; `omc.cavity_optics` |
| Piezoelectric FEM (u, φ), admittance, k_eff², g_em to a resonator, gradients in angle and fractions | `omc.piezo.PiezoSolver` (`.eigenmodes`, `.admittance`, `.motional`, `.eigen_sensitivities`) |
| Transducer system model (efficiency, noise, three-mode response) | `omc.system` |
| One device-level figure of merit η with one combined adjoint | `omc.codesign.DeviceModel` |
| Sparse factorisation backend (CHOLMOD or SuperLU) | `omc.linalg.factorize` |

`omc.linedefect`, `omc.layout`, `omc.channel` and `omc.transducer` hold the line-defect, device
layout, phononic channel and IDT builders used in the original project. `omc.layout` and
`geometry.line_defect_supercell` need an external fab generator, `gpt_omc.py`, which is not
included (see [Repository layout](#repository-layout)). Everything in the table above works without
it.

### Recipes

The four scripts in `examples/` are short, run in 3–70 s, and finish with a gradient check against
finite differences. Run them first to confirm that the install works.

| Example | Shows |
|---|---|
| `01_mechanical_bands.py` | Bands, both parities, full phononic gap, ∂f/∂fraction |
| `02_optical_bands.py` | TE-like PWE bands, the light line, gap below it, ∂f/∂fill |
| `03_g0_and_shape_gradient.py` | g₀ on one shared geometry, the adjoint gradient chained to shape parameters (r, w) |
| `04_piezo_idt.py` | IDT period: motional branch C_m, g_em to a 700 Ω resonator, ∂g_em/∂θ |

The core pattern:

```python
import numpy as np
from omc.geometry import HexLattice, blade_hole, solid_fraction
from omc.materials import SI
from omc.mech_bloch import BlochSolver
from omc.mech_adjoint import sensitivities
from omc.optics_pwe import PWESolver, slab_cell

lattice = HexLattice(0.448)                                    # a in µm
f = solid_fraction(lattice, [blade_hole(0.172, 0.093, 0.025)], 32)
ks, corners = lattice.gamma_m_k_path(15)

# Mechanics: Hz, (n_k, n_bands). workers > 1 splits the k-points across processes.
mech = BlochSolver(lattice, f, SI, model="slab_p", thickness=0.22, parity="even")
f_mech = mech.bands(ks, n_bands=10, workers=8)
s = sensitivities(mech, ks[corners[1]], n_bands=10)            # s.dfreq[layer, band] is (n, n) Hz per unit fraction

# Optics: THz. The supercell is 1.5 µm tall with 64 z samples, and the Si index is 3.48.
opt = PWESolver(slab_cell(lattice, f, 0.22, 1.5, 64, 3.48**2), parity="even")
f_opt = opt.bands(ks, n_bands=6)
opt.solve(ks[corners[1]], 6, tol=1e-8)
df_dfill = opt.sensitivities([0])[0]                           # THz per unit fill, (n, n)
```

To optimise shape parameters or pixels, build `f` from a `ParametricCell` or a `DensityCell`. Then
pass each band gradient `G` (shape `(n, n)`) through `cell.fraction_vjp(theta, G, n)` to get
dq/dθ. Example 03 does this end to end for g₀.

### Pitfalls

- **Threads.** Set `OMP_NUM_THREADS=1` (and `OPENBLAS_`/`VECLIB_MAXIMUM_THREADS`) before NumPy is
  imported. Get parallelism from `workers=` or from separate processes per k-point: ARPACK runs
  several times slower when BLAS threads compete.
- **Windows.** The MKL BLAS shipped in the win-64 env crashes silently (exit code 127) unless
  `MKL_THREADING_LAYER=SEQUENTIAL` is set. Set `PYTHONUTF8=1` too, because logs contain Greek
  letters. `pixi run` sets both.
- **Mesh bias.** At `n = 32` (about 20 nm elements) the mechanical gap edges sit 1.4–6 % high.
  Optimise at `n = 32`, then re-verify the final design at `n = 96`.
- **Shape gradients.** Pass `min_fraction=1e-9` to `BlochSolver`. With the default `1e-3`, nearly
  empty edge elements are dropped and about 17 % of the edge gradient is lost.
- **Degenerate bands.** Two modes at the same frequency (common at Γ and K) are not individually
  differentiable. `sensitivities` differentiates the cluster mean. Check `s.clusters` before using
  a single band's gradient.
- **Finite-difference checks** of PWE quantities need `tol ≤ 1e-8` and small steps, about 1e-5 µm
  for shape parameters. At 2e-4 µm the truncation error in example 03 is 5e-3 to 9e-3, which can
  look like a wrong gradient.
- **Optical modes near the light line** depend on the PWE supercell height (the cell is periodic
  in z). Only trust bands below the light line, or increase `height` and `nz`.
- **GME for g₀.** GME is a coarse model with an absolute error of about ±1 %. Its geometric
  sensitivities are about 1.6× too large for slot-like features. Take g₀ from PWE fields, and use
  GME as the fast inner model of a multi-fidelity loop.
- **First call.** The first PWE or GME call compiles with JAX and takes several seconds longer.
  Time the second call.
- **Not modelled:** sloped sidewalls in `slab_p` (the pattern is the same through the thickness);
  the `stack` model and `PiezoSolver` approximate them with stepped sub-layers. There is no GPU
  backend yet.

## Performance against COMSOL

These are unit-cell timings for one free-form phoxonic cell (a = 621.5 nm, 220 nm cubic Si slab),
per k-point and per parity. They were measured on an AMD Ryzen 9 7950X while two COMSOL jobs shared
the CPU, so all numbers are somewhat high. The COMSOL model uses the same cell, from the smooth
exported outline, with a Floquet unit cell and the mid-plane symmetry.

| Solver | Unknowns | Time per k-point | Cores |
|---|---|---|---|
| `BlochSolver`, `slab_p`, n = 32 (optimisation mesh) | 4,928 / 4,312 | **0.15 / 0.11 s** | 1 |
| `BlochSolver`, `slab_p`, n = 96 (verification mesh) | 37,576 / 32,879 | **2.3 / 1.6 s** | 1 |
| COMSOL solid mechanics, quadratic tetrahedra, hmax = t/7.5 | 65,565 | 10–15 s | 8 |
| `PWESolver`, 32 × 32 × 64, 1.5 µm supercell | 131,072 | 6 s | 1 process |
| `PWESolver`, 32 × 32 × 128, 2.0 µm supercell | 262,144 | 13 s | 1 process |
| COMSOL electromagnetic waves, hmax = t/6 (Si), t_air/10 (air) | 89,100 | 10 s | 8 |

**Mechanics.**

- At verification resolution (both codes within 0.25 % of each other), the FEM here is 5–7×
  faster in wall time on 1 core than COMSOL on 8 cores. That is **40–50× fewer core-seconds**.
- At the optimisation mesh it is about **100× faster in wall time**.

**Optics.** The PWE takes about as long per k-point as COMSOL, though on 1 process rather than 8.
It is used because:

- its gradients are nearly free;
- its frequencies depend smoothly on the fill;
- it reproduces MPB to 0.1 %.

Optics timings in more detail. Hardware differs between rows; the MPB comparison was a single run
on a heavily loaded Apple M1, and the rest are on the 7950X.

| Solver and case | Time | Reference |
|---|---|---|
| PWE, tripod cell, 24 × 24 × 64 (`examples/02`) | 2.5 s per k-point | — |
| PWE, tripod cell, resolution matched to MPB | 33 s per k-point | MPB 1.35: 58 s |
| PWE, line-defect supercell, band edge | 112–265 s | — |
| GME, tripod cell, g = 5, 128² raster (`gme_bands.py`) | 0.07–0.11 s per k-point | PWE: 6–13 s per k-point |
| GME, line-defect supercell, band edge | 1.7 s + 3.2 s for the gradient | PWE: 112–265 s |
| GME, finite cavity, 13 columns × 2 rows | 40 s | 3D FDTD (meep): 21 min |

GME is about 100× faster than PWE but accurate only to about ±1 % (see [Limits](#limits)). Use it to
screen designs, then verify with PWE.

**Gradients are where the gap becomes decisive.** One adjoint solve gives the derivative of a band
with respect to every pixel. Take one robust free-form optimisation step:

- 3 eroded/nominal/dilated designs × 2 parities × 76 k-points = 456 mechanical eigenproblems,
  plus their gradients over 342 pixel variables;
- here: about **1 core-minute**;
- with the COMSOL model above: about 1.5 h for the frequencies alone, and about **3 weeks** with
  finite-difference gradients.

Other measured costs:

| Quantity | Forward | Gradient | Reference |
|---|---|---|---|
| g₀ on a unit cell, all pixels (`examples/03`) | 21 s (both eigenproblems, pair search) | 0.9 s | one central difference = 2 forward solves per variable |
| g₀ on a 57k-DOF line-defect supercell | 116 s | 13.6 s, 2.5 GB | — |
| Piezo η + ∇η + constraints, 12k DOF | about 5 s in total | (included) | — |
| `BlochROM` band sweep (free-form cell, n = 96, 240 k) | 2.5 min including the reduction | — | full FEM 32 min (12–15×) |
| `ChainROM` single-cell update in a finite cavity | about 0.1 s | — | 3–6× faster than the full FEM, ⅓ of the memory |

**Why the FEM is faster:**

1. **2D mesh, polynomial thickness.** Each displacement component is a Legendre series of one
   z-parity through the 220 nm slab: 7–8 unknowns per node, and a 2D sparsity pattern. A sparse
   factorisation of a 2D pattern costs about N^1.5 against N² for 3D, with less fill.
2. **No per-k assembly or constraints.** The Bloch phase is carried by how the element nodes wrap
   around the cell. A new k is a weighted sum of nine stored arrays, where COMSOL re-assembles and
   eliminates Floquet constraints in every study run.
3. **Cholesky instead of LU.** A shift slightly below zero keeps K − σM positive definite, so
   supernodal CHOLMOD can be used.
4. **Parallel across k-points.** k-points are independent, so 16 cores solve 16 k-points with
   almost perfect scaling. A threaded direct solve of 65k unknowns gains far less from 8 threads.
5. **No remeshing.** The fixed grid makes each band a smooth function of the design, with no mesh
   noise in the objective.

## Accuracy

These figures are validated agreement with independent references, under the stated settings.
They are not formal error bounds, but they hold for the configurations listed, and the
[Limits](#limits) say where they stop holding. Rows marked "internal" were run against reference
data that is not distributed (licensed COMSOL models); the other rows reproduce from
`validation/`.

### Mode frequencies

**Mechanics against COMSOL, tripod cell.** a = 448 nm, r = 172 nm, w = 93 nm, 25 nm fillet,
isotropic Si:

| Case | Mesh | Mean error | Max error | Reproduce |
|---|---|---|---|---|
| 2D plane strain, 43 k | n = 96 | 0.08 % | 0.24 % | `blade_mech_2d.py` + COMSOL export |
| 3D slab 220 nm, symmetric, 46 k | n = 96, p = 4 | 0.08 % | 0.26 % | `blade_mech_3d.py` + COMSOL export |
| 3D slab, antisymmetric | n = 96, p = 4 | 0.12 % | 0.30 % | same |
| 3D slab, both parities | n = 64, p = 4 | 0.23–0.26 % | 0.55–0.83 % | same |

**Mechanics against COMSOL, two optimised free-form cells.** These are 4 GHz phononic and
1550 nm photonic cells in cubic Si, compared over the full Γ–M–K–Γ path (480 solves per cell),
internal:

| Quantity | Cell A | Cell B |
|---|---|---|
| Even parity, mean / max | 0.26 % / 0.53 % | 0.24 % / 0.50 % |
| Odd parity, mean / max | 0.17 % / 0.24 % | 0.19 % / 0.28 % |
| Full phononic gap, COMSOL | 3.539–4.501 GHz | 3.232–4.177 GHz |
| Full phononic gap, this solver | 3.546–4.511 GHz | 3.243–4.188 GHz |

COMSOL is about 0.2 % lower throughout, so the gap edges agree to within 0.35 %.

**Optics against COMSOL, the same two cells.** TE-like bands for modes confined to the slab, whose
field falls to e⁻³ before the edge of the air box, internal:

| Quantity | Cell A | Cell B |
|---|---|---|
| Mean / max | 0.29 % / 1.0 % | 0.26 % / 0.51 % |
| TE gap, COMSOL | 167.0–224.2 THz | 165.9–217.9 THz |
| TE gap, PWE | 167.6–224.4 THz | 166.7–217.9 THz |

The gap edges agree to within 0.5 %. Modes that reach the box edge differ by up to 15 %, because
they depend on the box (COMSOL uses PEC at 1.5 µm, PWE a 2 µm periodic supercell). No gap edge is
such a mode.

**Other references:**

| Solver and case | Reference | Agreement | Reproduce |
|---|---|---|---|
| PWE, unpatterned 220 nm slab, TE₀ | Exact dispersion | 0.06–0.08 % at 15.6 nm z-resolution | `optics_slab_analytic.py` |
| PWE, tripod cell, TE-like bands below the light line | MPB 1.35, similar resolution | 0.11 % mean, 0.31 % max over 151 band points | `optics_blade.py` |
| GME, tripod cell | MPB | 0.27–0.38 % mean (max ≤ 1.35 %) at g = 5 with TE₀(+TM₀); not monotonic in g or in the number of orders | `gme_bands.py blade` |
| GME, tripod cell, equal settings | legume | Bands 0.11 % at 128 px/a; Q_rad of leaky bands 1–2 % | `gme_legume.py` |
| GME, finite cavities | legume, FDTD | f ±1 % absolute; Q_rad 0.1–11 % for Q ≲ 4×10⁶ | internal |
| Piezo FEM, 3D LN-on-Si IDT, 6 fingers, open circuit | COMSOL | All 10 modes with Q > 100 found, at matching Q; f 0 to +1.2 % (mean 0.7 %), of which 0.3–0.4 % is this mesh | internal |
| Piezo FEM, thin X-cut LN plate | Analytic | Velocities 0.08 %; k² to 0.05 points | `piezo_plate_k2.py` |
| Piezo FEM, coupling switched off | `BlochSolver` | 1e-12 | `piezo_elastic_limit.py` |
| Piezo FEM, reciprocity | Y₁₂ = Y₂₁ | 1e-14 | `piezo_elastic_limit.py` |
| Piezo FEM, PML | SH0 reflection | \|R\| ≈ 1e-5 | `piezo_absorber.py` |
| Piezo FEM, modal C_m | BVD fit of the driven Y(ω) | 1.2e-4 | `piezo_gem.py` |
| `BlochROM` | Full FEM, tripod n = 64 and free-form n = 96 | ≤ 5e-4 relative (gap edges ≤ 3e-4) | internal |
| `ChainROM`, 30 modes of a finite cavity | Full FEM | f 7e-5; Q ratio 0.98–1.005; mode overlap ≥ 0.999 | internal |
| Three-mode model (RWA) | Non-RWA Langevin solution | ≤ 4.7e-3 | `system_three_mode.py` |

**Optomechanical coupling g₀:**

| Check | Agreement | Reproduce |
|---|---|---|
| Dilation term against a finite-difference rebuild of the geometry | 0.01–0.6 % | `g0_validate.py` |
| Photoelastic term against an analytic strained slab | 3e-4 | `g0_validate.py` |
| Rigid z-translation (must give 0) | 1e-16 | `g0_validate.py` |
| Line-defect supercell against fields from 3D FDTD | 4 % (2.68 vs 2.57 MHz) | internal |
| Convergence in mesh and supersampling | about 1 % | — |

The moving-boundary term is computed on a supersampled raster. Central differences on the voxel
grid are 20–40 % off.

### Gradients

Every gradient is checked against central finite differences of a full re-solve. Where the residual
falls as h² with the step, it is finite-difference truncation, not gradient error.

| Gradient | Agreement with FD | Reproduce |
|---|---|---|
| Mechanical bands: fills (2D, `slab_p`, `stack`), shape parameters, degenerate clusters, free-form pixels | 1e-7 to 4e-5 | `mech_gradient_check.py` |
| Optical bands, hole-edge perturbations | 1e-6 to 3e-4 (smooth interior perturbations about 1e-3) | `optics_gradient_check.py` |
| g₀: optical fill, mechanical fractions, raster, shape parameters | 1e-13 to 2e-4 (shape parameters 3–5e-5) | `g0_gradient_check.py`, `examples/03` |
| Piezo: eigenfrequencies, k_eff², Y(ω), driven fields; in θ and fractions | ≤ 7.3e-7 | `piezo_gradient_check.py` |
| g_em (BVD motional branch) in θ and fractions | ≤ 1.5e-7 | `piezo_gem.py`, `examples/04` |
| GME: f and ln Q_rad | 1e-9 to 3e-7; 8e-9 to 2e-5 | internal |
| Device-level η with constraints (f_m, λ_o, leakage, Q_o), end to end | 4e-8 to 3.5e-5 | `codesign_gradient_check.py` |
| Reduced-order models, projected sensitivities | 3e-8 to 3.5e-4 (within about 1 % of the full adjoint) | internal |

### Limits

- **Pixel geometry.** Edges are partly filled elements, not a boundary-fitted mesh. Re-verify
  final designs at n = 96, or in a boundary-fitted solver.
- **`slab_p`** assumes straight sidewalls, a pattern that does not vary with z, and mid-plane
  symmetry. Layered or sloped structures need `stack` or `PiezoSolver`, which staircase slopes in
  sub-layers.
- **PWE optics** covers the TE-like (σ_z-even) family only. Above the light line, results depend on
  the supercell height.
- **GME** is a coarse model (±1 % in f, slot sensitivities 1.6× too large). Use it for screening
  and multi-fidelity loops, not for final numbers.
- **Mechanical g₀** needs the mechanical mode at Γ, and both modes non-degenerate. g₀ does not
  depend on the mode's scale, but the reported m_eff does: it is evaluated at a fixed peak point.
  With `min_fraction=1e-9` that point can land on a nearly empty element, and then the m_eff
  gradient is unreliable (`g0_gradient_check.py` shows 0.9 relative error for one smooth
  direction), while g₀ and its gradient are unaffected.
- **`codesign`** uses the lossless mode for η; the leakage rate comes from a separate PML solve.
  k = 0, non-degenerate modes only.

## Reproducing the validation

Every script in `validation/` writes to `validation/out/` (ignored by git) and prints its
comparison. Most run in a few minutes; the gradient checks take 1–20 min.

```bash
pixi run python validation/optics_slab_analytic.py
pixi run python validation/mech_gradient_check.py
pixi run python validation/g0_gradient_check.py --quick
```

- **MPB reference bands** for the tripod cell are included as
  `validation/reference/optics_mpb_blade.npz`. To regenerate them, run
  `validation/optics_mpb_reference.py` in an environment with MPB (Linux or macOS).
- **legume** (`gme_legume.py`) runs in its own environment. The steps are in the script's
  docstring.
- **COMSOL band exports** for `blade_mech_2d.py` and `blade_mech_3d.py` are not distributed. Pass
  the path to your own export as the first argument.
- A few scripts have an extra case, a line-defect supercell or a finite cavity, that needs the
  external `gpt_omc.py`. Their default case does not.

## Repository layout

```
omc/          the solver package
examples/     four short, runnable recipes with gradient checks
validation/   scripts behind the accuracy tables; reference/ holds small reference data
docs/         piezoelectricity.md: a primer on LN piezoelectricity and its simulation pitfalls
research/     (git-ignored) the original design studies, their results and notes
```

`research/` stays on the original machine. Some docstrings in `omc/` cite scripts there as the
source of a number. Builders that need the fab generator `gpt_omc.py` import it on first use from
the Python path, from `$GPT_OMC_PATH`, or from `research/`. Without it, those builders raise an
`ImportError` that says so; nothing else depends on it.

## License

MIT; see [LICENSE](LICENSE).
