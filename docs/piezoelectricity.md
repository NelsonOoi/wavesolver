# Piezoelectricity for the LN-on-SOI transducer

A working primer: the physics, the conventions that silently break simulations, and what
matters specifically for a 250 nm X-cut LN on 220 nm Si transducer feeding a 4 GHz
optomechanical crystal. Numbers quoted are from this repo's material data (`wavesolver.materials.LN_PIEZO`)
unless stated otherwise.

Contents

1. [What piezoelectricity is](#1-what-piezoelectricity-is)
2. [Constitutive relations: the four forms](#2-constitutive-relations-the-four-forms)
3. [Tensor bookkeeping: Voigt, Bond, rotations](#3-tensor-bookkeeping-voigt-bond-rotations)
4. [Symmetry: which couplings exist in LN](#4-symmetry-which-couplings-exist-in-ln)
5. [Field equations and boundary conditions](#5-field-equations-and-boundary-conditions)
6. [Coupling coefficients: k², K², k_eff²](#6-coupling-coefficients-k-k-k_eff)
7. [Waves in piezoelectric media and plates](#7-waves-in-piezoelectric-media-and-plates)
8. [k_eff² as an overlap integral](#8-k_eff-as-an-overlap-integral)
9. [Circuit models: BVD, admittance, matching](#9-circuit-models-bvd-admittance-matching)
10. [Transducer geometries](#10-transducer-geometries)
11. [Coupling to a qubit or microwave resonator](#11-coupling-to-a-qubit-or-microwave-resonator)
12. [Lithium niobate specifics](#12-lithium-niobate-specifics)
13. [Losses and cryogenic effects](#13-losses-and-cryogenic-effects)
14. [Simulation pitfalls checklist](#14-simulation-pitfalls-checklist)
15. [Where this lives in the repo](#15-where-this-lives-in-the-repo)
16. [Reading list](#16-reading-list)

---

## 1. What piezoelectricity is

Strain in a crystal without an inversion centre displaces the positive and negative
sublattices unequally, creating a polarisation **P** linear in strain (direct effect).
Conversely an applied field produces strain (converse effect). Both are described by the
same tensor, by thermodynamics.

- Requires a **non-centrosymmetric** point group. Of the 21 such classes, 20 are
piezoelectric (432 is the exception: its cubic symmetry forces the tensor to zero).
- A subset (10 classes) is **pyroelectric**: they have a spontaneous polarisation that
changes with temperature. A subset of those is **ferroelectric**: the polarisation can be
reversed by a field. LN (class 3m) is all three. Its spontaneous polarisation lies along the
c-axis (crystal Z), and commercial wafers are poled single-domain. Domains can be inverted
locally by high-voltage poling, which is how periodically poled LN is made.
- The effect is linear: reversing the strain reverses **P**. That is different from
**electrostriction** (strain ∝ E²), which every dielectric has and which is negligible here.
- Piezoelectricity, the linear electro-optic (Pockels) effect and second-harmonic
generation all come from the same broken inversion symmetry. That is why LN is good at
all three.

## 2. Constitutive relations: the four forms

The independent variables can be chosen four ways. Each pair of material tensors is
measured under a different condition, and **mixing them is the single most common error**.

| Form | Equations | Tensors | Used by |
|---|---|---|---|
| stress-charge | T = C^E S − eᵀE,  D = eS + ε^S E | C^E, e, ε^S | FEM (COMSOL, `wavesolver.piezo`) |
| strain-charge | S = s^E T + dᵀE,  D = dT + ε^T E | s^E, d, ε^T | data sheets, actuators |
| stress-voltage | T = C^D S − hᵀD,  E = −hS + β^S D | C^D, h, β^S | rarely |
| strain-voltage | S = s^D T + gᵀD,  E = −gT + β^T D | s^D, g, β^T | sensors |

Superscripts give the condition held fixed: ^E at constant (zero) field (short circuit),
^D at constant displacement (open circuit), ^S at constant strain (clamped), ^T at constant
stress (free).

Conversions:

$$
e = d\,C^E,\qquad d = e\,s^E,\qquad
\varepsilon^T = \varepsilon^S + d\,C^E d^{\mathsf T} = \varepsilon^S + e\,s^E e^{\mathsf T},
$$

$$
C^D = C^E + e^{\mathsf T}(\varepsilon^S)^{-1}e,\qquad h = (\varepsilon^S)^{-1}e .
$$

**The free and clamped permittivities of LN are very different.** From the repo's data,
ε^S = (43.6, 43.6, 29.2)ε₀ and ε^T = (84, 84, 30)ε₀. The stress-charge form needs ε^S. If
you put in ε^T (the "dielectric constant" on most data sheets), C₀ comes out almost 2× too
big along X and Y, and k² comes out wrong.

The thermodynamic potential behind the stress-charge form is the electric enthalpy

$$
H(S,E) = \tfrac12 S\cdot C^E S - E\cdot e S - \tfrac12 E\cdot\varepsilon^S E,
\qquad T = \partial H/\partial S,\quad D = -\partial H/\partial E .
$$

Its stationarity gives the symmetric but indefinite (u, φ) FEM system used in `wavesolver.piezo`.
The −½E·εE term is why the φ-block has a negative sign. As a consequence, the system is not
positive definite, and an LDLᵀ or LU solver is needed instead of Cholesky.

Sign conventions: some texts write T = CS − eE without the transpose, or use −e. The
physics is fixed by requiring that e = dC^E and that the energy above is the work done.
When importing constants, check them against a known number (Section 12).

## 3. Tensor bookkeeping: Voigt, Bond, rotations

**Voigt order in this repo:** 1 = xx, 2 = yy, 3 = zz, 4 = yz, 5 = xz, 6 = xy, with
**engineering shear strains** S₄ = 2S_yz etc. and tensor (not doubled) shear stresses. Then
C_IJ = C_ijkl, e_iJ = e_ijk, d_iJ = d_ijk for J ≤ 3 but d_iJ = 2d_ijk for J ≥ 4 (d
multiplies stress). Photoelastic p follows the same convention as stiffness here.

**Rotations.** A rotation matrix R whose columns are the crystal axes expressed in lab
coordinates transforms

$$
C_{\rm lab} = M C M^{\mathsf T},\qquad e_{\rm lab} = R\,e\,M^{\mathsf T},\qquad
\varepsilon_{\rm lab} = R\,\varepsilon R^{\mathsf T},
$$

with M = bond(R), the 6 × 6 Bond matrix (`materials.bond`, `rotate_*`). Since everything is
polynomial in R, the θ-derivatives used by the adjoint follow from ∂R/∂θ and ∂M/∂θ
(`x_cut_derivative`, `bond_derivative`).

**Cut names.** "X-cut" means the plate normal is the crystal X axis, so the plate plane is
the crystal Y–Z plane. "128° Y-cut" means the normal is Y rotated by 128° about X. The
IEEE notation (e.g. "YXl/128°") encodes the same thing more precisely. The in-plane propagation
direction is a separate angle. In this repo, `x_cut(θ)` puts crystal X along lab z and the
c-axis at angle θ from lab x (the spine). The design default is θ = 45°, i.e. COMSOL Euler
(−π/4, −π/2, −π/2), with the spine along (Y + Z)/√2.

**COMSOL Euler angles** are z-x-z, and the columns of R_z(α)R_x(β)R_z(γ) are the crystal axes
in global coordinates (`validation/ln_orientation.py`). Other tools use other orders and
active vs passive conventions. To be sure, always print where crystal X, Y, Z land in lab
coordinates.

**Lab-frame vs crystal-frame components.** "S_xx" in a plate model means strain along lab x,
the propagation direction. For X-cut at angle θ, lab x̂ = (0, sin θ, cos θ) in crystal
coordinates. So lab S_xx mixes crystal S_YY, S_ZZ and S_YZ, and lab e_x,xx is

$$
e_{x,xx}(\theta) = \sin^3\theta\; e_{22} + \sin^2\theta\cos\theta\,(e_{31}+2e_{15}) + \cos^3\theta\; e_{33}.
$$

The large S0 coupling comes mostly from **e15** (a crystal shear constant). It appears in an
extensional coupling only because lab x is tilted between crystal Y and Z.

## 4. Symmetry: which couplings exist in LN

Neumann's principle: every material tensor is invariant under the crystal's point-group
operations. For 3m (IEEE axes: Z = c-axis, mirror plane ⊥ X, X along an a-axis):

```
        xx      yy      zz      yz      xz      xy
e = [   0       0       0       0       e15    −e22  ]   (D_X)
    [ −e22     e22      0      e15      0       0    ]   (D_Y)
    [  e31     e31     e33      0       0       0    ]   (D_Z)

e15 = 3.70, e22 = 2.54, e31 = 0.19, e33 = 1.31 C/m²
```

Stiffness has six independent constants including **C14** (it couples yz shear to xx/yy
normal strain; 9.0 GPa). Permittivity is uniaxial: ε_XX = ε_YY ≠ ε_ZZ.

Useful consequences:

- **Mirror ⊥ X.** Every component of e with an odd number of X indices vanishes. Because
X-cut puts X along the plate normal, **a field across the thickness (E_z in the lab) cannot
drive any in-plane strain**: e_z,xx = e_z,yy = e_z,xy = 0 at every in-plane angle. It drives
only S_xz and S_yz, i.e. thickness-shear modes. S0 and SH0 in X-cut LN need **lateral**
fields from coplanar electrodes.
- **Y and Z are polar** (e22 has three Y indices, e33 three Z indices). Reversing +Y or +Z
matters. Reversing X does not: ±X faces are related by the mirror. **Flipping an X-cut chip
face-down changes nothing for in-plane waves.** Mirroring the c-axis across the spine
changes a lot (Section 12).
- **The c-axis is a 3-fold axis, not 6-fold.** Properties in the Y–Z plane have period 180°
(θ and θ + 180° are related by inversion of the wave direction), not 90°.

## 5. Field equations and boundary conditions

**Quasi-static approximation.** Acoustic waves are about 10⁵ times slower than light. Over
an acoustic wavelength the magnetic induction is negligible, so ∇×E ≈ 0 and E = −∇φ. The
equations are then

$$
\rho\,\ddot u_i = \partial_j T_{ij},\qquad \partial_i D_i = \rho_{\rm free}\;(=0\ \text{in insulators}),
$$

with four unknowns (u, φ) per point. At 4 GHz the EM wavelength is about 7.5 cm, so the
approximation is excellent at chip scale. Its errors appear only in long bus lines, which are
the circuit's job.

**Mechanical boundary conditions.** Free surfaces have T·n = 0. Electrodes add mass, and
also stiffness if they are thick.

**Electrical boundary conditions** are where most of the physics lives:

- **Electroded surface** (short circuit / driven): φ is prescribed and equal on the whole
electrode (an equipotential).
- **Bare surface**: D·n and φ are continuous into the surrounding medium (vacuum, air, or Si
underneath). The field fringes outside the LN.
- The approximation D·n = 0 on a bare face ignores the fringing field. It overestimates k²
slightly. In `validation/piezo_plate_k2.py`, adding vacuum lowers the thin-plate k² by about 4 points. Mesh enough
vacuum, a few wavelengths' worth, for C₀ and k² you trust.
- **Floating electrode**: unknown V, total Q = 0. Used for reflectors and for "open"
terminals.

**Si is a dielectric here** (ε ≈ 11.7). It is not piezoelectric but carries field. At room
temperature, doped or lossy Si can screen fields and add loss. At millikelvin, carriers
freeze out. Interface charge at the LN/Si or oxide interfaces is a known, hard-to-model
nuisance.

## 6. Coupling coefficients: k², K², k_eff²

"Coupling" means at least five different numbers in the literature. Always check which.

**Material (1D) coupling.** For a single strain–field pair with stiffness c^E, piezo
constant e and permittivity ε^S:

$$
K^2 = \frac{e^2}{c^E\varepsilon^S}\ \ (\text{stiffening factor}),\qquad
k^2 = \frac{e^2}{c^D\varepsilon^S} = \frac{K^2}{1+K^2},\qquad
c^D = c^E(1+K^2).
$$

k² is the fraction of the stored energy that is converted between electrical and mechanical
form. K² is how much the electrical boundary condition stiffens the material.

**Wave coupling.** For a guided wave, measure the velocity with the surface shorted and open:

$$
k^2 = \frac{v_{\rm oc}^2 - v_{\rm sc}^2}{v_{\rm oc}^2}\;\approx\;\frac{2\Delta v}{v}.
$$

SAW literature often quotes K² = 2(v_free − v_metallised)/v_free, which equals this to first
order. This is what `validation/piezo_plate_k2.py` computes: SH0 36% along Y, S0 31% at 30°
from Y, and S0 26.5% along the spine.

**Resonator coupling.** For a single mode, take its frequencies with the terminals shorted
(series resonance f_s) and open (parallel resonance f_p):

$$
k_{\rm eff}^2 = \frac{f_p^2 - f_s^2}{f_p^2}.
$$

This is what `PiezoSolver` reports. Other definitions exist: IEEE's thickness-mode
k_t² = (π/2)(f_s/f_p)cot((π/2)(f_s/f_p)), and the cruder 2(f_p − f_s)/f_p. They agree at small
k² and differ by several percent of themselves at 20–30%.

**The hierarchy.** The material or wave k² is an upper bound for any transducer's
k_eff². The transducer loses because the electrode field does not match the mode strain
everywhere: strain energy sits in the Si, field fringes into air, and only part of the mode
is under electrodes. Example from this repo's runs: plate S0 along the spine is 26.5%, while a
6-finger IDT mode is 7% and the 520 nm pitch IDT period of `validation/piezo_angle.py` is 1%.

**Figure of merit.** For resonators, k_eff²·Q bounds achievable bandwidth and insertion
loss. For transduction, what matters is the coupling *rate* (Section 11), which also depends
on how C₀ compares with the rest of the circuit.

## 7. Waves in piezoelectric media and plates

**Bulk waves: piezoelectric stiffening.** In a plane wave along n̂ the quasi-static field is
longitudinal (E ∥ n̂). Eliminating it modifies the Christoffel tensor:

$$
\rho v^2 u_i = \Big[C^E_{ijkl}n_jn_l + \frac{(e_{pij}n_pn_j)(e_{qkl}n_qn_l)}{\varepsilon^S_{pq}n_pn_q}\Big]u_k .
$$

There are three bulk modes per direction: quasi-longitudinal and two quasi-shear. In
anisotropic crystals, polarisation is generally not exactly along or perpendicular to n̂,
and energy (group velocity) is not parallel to the wavevector. That **beam steering**
matters for long channels at off-axis angles.

**Surface waves (SAW).** Rayleigh-type waves are bound to a thick substrate's surface,
roughly 0.9 × the shear velocity. Leaky SAWs radiate into the bulk. They are not your case,
but most IDT design lore was developed for them.

**Plate (Lamb) waves**, for thickness h ≪ λ:

| Mode | Polarisation (wave along x) | Strain | Notes |
|---|---|---|---|
| A0 | mostly u_z, flexural | bending | strongly dispersive, slowest at low f |
| S0 | mostly u_x, small antisymmetric u_z | S_xx | ≈ 6000–6800 m/s in free LN plate |
| SH0 | u_y, uniform through thickness | S_xy | ≈ 3500–4000 m/s; non-dispersive if isotropic |

In anisotropic crystals off symmetry directions, S0 and SH0 hybridise, giving "S0-like" and
"SH0-like" modes. In a patterned membrane or phononic crystal, every band is a hybrid. The
labels then mean only which polarisation dominates.

At 4 GHz, the free-plate S0 wavelength along the spine is about 1.5 µm and SH0 about 1 µm.
That gives h/λ ≈ 0.16–0.25 for 250 nm LN, so dispersion is not negligible. The LN/Si
bilayer, with 220 nm of non-piezo Si, changes both velocity and coupling.

**Why the S0 coupling is strong along the spine.** E_x along the propagation direction is
longitudinal, so for a free plate D_x = 0 automatically (∇·D = ∂_xD_x = 0). That is the
"open" case. Metallising the faces kills E_x, which is the "short" case. A 1D estimate
k² ≈ e²/(cε + e²), with e_x,xx = 4.04 C/m², c ≈ ρv² and ε_xx = 36.4ε₀, gives about 22%. The
exact 2 × 2 thin-plate model, which includes the S_xy coupling and C14, gives 26.5%.

## 8. k_eff² as an overlap integral

This is the most useful intuition for transducer design. Take a mechanical mode (computed
short-circuited) with strain S_m, frequency ω_m and modal mass m = ∫ρ|u_m|². Take the static
potential φ₀ with 1 V on the drive electrode and the others grounded, so E₀ = −∇φ₀ and
C₀ = ∫E₀·ε^S E₀ dV over all space. The force the voltage exerts on the mode is

$$
\Gamma = \int E_0\cdot e\,S_m\,dV \quad(\text{units: C/m per V, i.e. N/V}),
$$

and the mode appears as a series motional branch with

$$
C_m = \frac{\Gamma^2}{m\,\omega_m^2} = \frac{\Gamma^2}{\int S_m\cdot C\,S_m\,dV},\qquad
k_{\rm eff}^2 \approx \frac{C_m}{C_0 + C_m}\approx
\frac{\big(\int E_0\cdot e\,S_m\big)^2}{\int S_m\cdot C S_m\;\int E_0\cdot\varepsilon E_0}.
$$

Reading it:

- It is a normalised **projection** of the electrode field onto the mode's "piezo-charge
density" e S_m. It has the form of Cauchy–Schwarz, so it is maximised when E₀ ∝ (e S_m)
pattern-wise.
- **Sign cancellation** is the main loss mechanism. Wherever e S_m changes sign, the
electrode polarity must change sign too. That is the whole logic of interdigitated
fingers at the wavelength, and of electrode inverse design: polarity = sign(e S_m) at each
location.
- **Energy outside the LN** (Si, tethers) enlarges the denominator without adding to the
numerator. So does field in air or Si.
- Valid for an isolated mode at small k². For strong coupling, or near-degenerate modes,
use the full admittance.

## 9. Circuit models: BVD, admittance, matching

**Butterworth–Van Dyke (BVD).** Near one resonance, the terminal sees a static capacitance C₀
in parallel with a series R_m L_m C_m branch:

$$
Y(\omega) = i\omega C_0 + \frac{1}{R_m + i\omega L_m + 1/(i\omega C_m)},\qquad
\omega_s = \frac{1}{\sqrt{L_mC_m}},\quad \omega_p = \omega_s\sqrt{1 + C_m/C_0}.
$$

So C_m/C₀ = k_eff²/(1 − k_eff²), Q_m = ω_sL_m/R_m, and L_m = m/Γ², C_m = Γ²/(mω_m²) from
Section 8. The repo uses the e^{iωt} convention (like COMSOL), so Y = iωQ/V. Several modes
form several parallel branches ("modified BVD"). Electrode resistance adds a series R_s,
and dielectric loss gives C₀ a loss tangent.

**Impedance is the practical problem with tiny transducers.** A 6-finger IDT on this platform has
C₀ ≈ 0.58 fF, so 1/(ωC₀) ≈ 69 kΩ at 4 GHz. It cannot be matched to 50 Ω with a simple
network. Nanoscale transducers are therefore coupled to:

- a high-impedance resonator (high-kinetic-inductance or compact LC),
- the qubit's own capacitance (Section 11),
- or a matching network with a large transformation ratio, at the cost of bandwidth.

**Radiation conductance (travelling-wave IDT).** When the transducer launches waves into a
channel rather than forming a resonator, the BVD branch is replaced by a frequency-dependent
radiation conductance G_a(ω) plus a susceptance B_a(ω), in parallel with C₀. Classic
Smith-model scalings for an N-period uniform IDT:

- G_a(f₀) ∝ K² f₀ C₀ N, and the fractional bandwidth ≈ 1/N.
- The electrical Q is ≈ ωC₀/G_a, and the "optimum" N for match-limited bandwidth is
N_opt ≈ √(π/(4K²)).

With K² ~ 0.2–0.3 you only need a handful of periods. The transducer is then short compared
with the wavelength-scale features of the channel.

**Two-port quantities.** For transducer → channel → mode, the useful outputs are:

- the microwave-to-phonon conversion efficiency, |S₂₁|² into the desired channel mode,
- reflection |S₁₁|²,
- leakage into other modes or directions,
- for a resonator readout, the external coupling rate of the mechanical mode to the
microwave port.

`PiezoSolver` gives admittance and driven-field functionals with adjoint gradients, so all of
these are differentiable objectives.

## 10. Transducer geometries

| Geometry | Field | Couples to (X-cut LN) | Notes |
|---|---|---|---|
| IDT, single-electrode (λ/4 fingers, alternating polarity) | lateral, periodic | S0, SH0, SAW-like at v = f·λ | bidirectional; fingers Bragg-reflect at f₀ (stopband) |
| IDT, split-finger (λ/8 fingers, ++−−) | lateral | same | reflections cancel at f₀; needs finer features |
| SPUDT / reflector-backed IDT | lateral | same | unidirectional; a phononic mirror behind the IDT does the same |
| Single-gap or comb capacitor on a resonator | lateral | the resonator mode | common for phononic-crystal resonators coupled to qubits |
| Top/bottom electrodes (FBAR/HBAR style) | thickness | thickness shear only | weak or useless for S0/SH0 in X-cut |
| Lateral field on a plate, λ ≫ gap (XBAR-like) | lateral | thickness shear via e15 | a different mode family |

Points that are easy to miss:

- **Bidirectionality costs 3 dB.** A symmetric IDT sends half the power each way. Into a
single channel, the efficiency is ≤ 50% unless something reflects the backward wave in
phase: a phononic mirror, a reflector grating, or an asymmetric (SPUDT-like) design.
Freeform electrode/LN design can find this by itself, if the objective sees the backward
port.
- **Electrode reflections** come from mass loading and from electrical shorting, which
changes the local velocity by Δv/v ~ k²/2. With k² ≈ 25%, a shorted region is a
significant impedance step. At the Bragg condition (pitch = λ/2), single-electrode IDTs
reflect strongly. That can help (a resonator) or hurt (a broadband launcher).
- **Mode selectivity.** The overlap integral selects polarisation. At the spine angle, S0
couples about 13× more strongly than SH0 (26.5% vs 2%), which helps keep a clean single
mode. Other angles (e.g. θ ≈ 94° in the IDT period of `validation/piezo_angle.py`) favour SH0. Off 0° or 90°, the
mirror symmetry about the spine breaks and polarisations mix.
- **Electrode material.** Al or Nb are superconducting at mK. Their thickness adds mass
loading and stiffness, and sets the electrode resistance at room-temperature testing.
Electrodes on LN form a non-piezo layer that also loads the mode.

## 11. Coupling to a qubit or microwave resonator

Model the mechanical mode as its BVD branch (L_m, C_m) connected across a microwave circuit
whose total capacitance at that node is C_Σ = C_q + C₀ + parasitics. This is two resonators
coupled through C_m, and for C_m ≪ C_Σ

$$
g \approx \frac{\omega_m}{2}\sqrt{\frac{C_m}{C_\Sigma}}
 = \frac{\omega_m}{2}\sqrt{\frac{k_{\rm eff}^2}{1-k_{\rm eff}^2}}\sqrt{\frac{C_0}{C_\Sigma}} .
$$

- Coupling depends on **k_eff and on the participation ratio C₀/C_Σ**. A transducer with
high k² but sub-fF C₀ on a ~80 fF transmon is diluted by √(C₀/C_Σ) ≈ 0.08. Reducing
parasitic capacitance (wiring, pads) is as valuable as raising k².
- Larger C₀ means more fingers, more area or narrower gaps. Each pushes the transducer
towards multimode behaviour and more TLS-hosting surface, so it is a trade-off.
- For transduction efficiency through a channel, the analogous statement is that the
external coupling rate of the transducer port, κ_e = G_a/C_Σ for a resonant circuit, must
be matched to the circuit's other rates. It is the same impedance-matching problem as
Section 9.

Arithmetic illustration only, with an assumed C_Σ = 80 fF, k_eff² = 7% and C₀ = 0.58 fF at
4 GHz: g/2π ≈ 2 GHz × 0.27 × 0.085 ≈ 46 MHz. Real devices land lower once losses, parasitics
and multimode splitting are included.

## 12. Lithium niobate specifics

**Constants in the repo** (`LN_PIEZO`, COMSOL library data, stress-charge, crystal frame):

| | Value |
|---|---|
| C11, C12, C13, C14, C33, C44, C66 (GPa) | 202.9, 52.9, 74.9, 9.0, 243.1, 59.9, 74.9 |
| e15, e22, e31, e33 (C/m²) | 3.70, 2.54, 0.19, 1.31 |
| ε^S / ε₀ (X, Y, Z) | 43.6, 43.6, 29.2 |
| ε^T / ε₀ | 84, 84, 30 |
| ρ (kg/m³) | 4700 (congruent LN is usually quoted ≈ 4640) |

Published datasets (Warner–Onoe–Coquin 1967, Kovacs et al. 1990, Smith & Welsh 1971,
Jazbinšek & Zgonik 2002) differ by a few percent in C and e. Thin-film LN made by ion
slicing may differ again: residual strain, implantation damage, Li out-diffusion. Expect
percent-level frequency offsets from material data alone. That is larger than the
piezo FEM's agreement with COMSOL (0.7 % mean, see the README).

**Sign conventions: the trap that costs a factor of 6.** C14 and e22 change sign together
under Y → −Y, which is a 180° rotation about the c-axis. Different references and standards
(IEEE 176-1949 vs later) have used both signs. The repo uses C14 > 0 with e15, e22, e31,
e33 > 0.

The effect on your design is large. With the spine along ±(Y + Z)/√2, i.e. `x_cut(45)`, S0
has k² = 26.5%. Along ±(Y − Z)/√2, i.e. `x_cut(135)`, S0 has only **4.3%**. Both are "the
c-axis at 45° to the spine". They differ only in which side of the c-axis the +Y direction
is on, so they differ by exactly this sign.

On a real chip, which diagonal is which depends on:

- how the vendor marks +Y and +Z (flats, and the convention used to define + by the
piezo/pyro charge sign),
- whether that convention matches the constants.

Flipping the chip over does not matter, because the ±X faces are mirror-related. So:

- Get the +Z and +Y directions of your film from the vendor in a stated convention.
- Better, measure k² on test IDTs at both diagonals early. The difference is unmistakable.

**Other LN facts that matter:**

- **Ferroelectric with a high Curie temperature** (≈ 1140 °C congruent). Domains are stable
in processing, but domain inversion can occur under high fields at elevated temperature.
- **Pyroelectric.** Temperature changes, especially the 300 K → mK cooldown, produce surface
charge along Z. Z lies in-plane for X-cut, so the charge appears across electrode gaps.
This causes drifting offsets, possible discharges across narrow gaps, and trapped charge.
Keep electrodes shorted or grounded during cooldown and wire-bonding.
- **Photorefractive / photogalvanic.** Light, mostly visible but also any absorbed IR,
creates space-charge fields that drift. It is less severe at 1550 nm, but relevant if the
optical mode overlaps LN.
- **Electro-optic** (r33 ≈ 31 pm/V) and photoelastic. Where the optical mode overlaps LN, it
modulates the optical mode. This is usually negligible in a Si-confined cavity, but it is
another coupling path.
- **Anisotropic etching.** Sidewalls are typically about 65°, and the angle and
redeposition depend on crystal direction. The repo models slopes as staircased sub-layers.
- **Temperature dependence.** LN has a large negative TCF at room temperature, roughly
−70 to −90 ppm/K. Its constants stiffen on cooling, so expect GHz modes to move up by of
order a percent between 300 K and mK. Calibrate on a test structure rather than trusting
room-temperature constants.

## 13. Losses and cryogenic effects

| Loss | Mechanism | Notes |
|---|---|---|
| Two-level systems (TLS) | defects in amorphous surfaces, oxides, damaged LN, interfaces | often dominant for LN phononic resonators at mK; piezoelectric materials couple TLS to *both* strain and field, so LN can be worse than Si |
| Piezoelectric radiation | the mode's field couples to electrodes, bus lines or free charges and radiates or dissipates | wanted at the transducer, unwanted at the OMC; keep LN and electrodes away from the optomechanical mode unless intended |
| Clamping / anchor | phonons leaking into the substrate or frame | the phononic shield's job |
| Electrode resistance | normal-metal loss at RT, quasiparticles at mK | optical absorption breaks Cooper pairs in nearby superconducting electrodes |
| Dielectric loss | tan δ of LN, Si, oxides | sets the electrical Q of C₀ |
| Thermoelastic, Akhiezer | phonon–phonon | small at mK, relevant at RT |

Practical consequences for the design:

- Minimise LN area and electrode–LN interface where the mechanical energy lives, except
where you need coupling.
- If the OMC mode itself has LN under it, the piezo field turns any nearby electrode or lossy
dielectric into a damping channel. In the repo's parametrisation LN can be removed per
element (`LN ⊆ Si`), so the optimiser can learn to keep LN out of the OMC region.
- Losses depend on power and temperature (TLS saturate). Room-temperature Q tells you
little about mK Q.

## 14. Simulation pitfalls checklist

- [ ] **ε^S, not ε^T**, in the stress-charge form.
- [ ] **Voigt with engineering shear**, consistent between C, e, d and p. A factor of 2 in shear terms is invisible in normal-strain benchmarks.
- [ ] **Rotation convention**: print where crystal X, Y, Z land. Check one known number, e.g. SH0 k² ≈ 36% along Y in a thin X-cut plate.
- [ ] **Sign of C14/e22 vs the physical wafer** (Section 12). Factor ~6 in S0 k² at your angle.
- [ ] **Open-circuit surfaces need vacuum**: D·n = 0 overestimates k² by a few points. C₀ needs a large enough air box.
- [ ] **Mesh**: at least order 2 through thin LN, since order 1 overstiffens by several percent. About 30 Q1 elements per wavelength in-plane for 0.3%. Refine at electrode edges, where the field is singular (∝ r^(−1/2)) and C₀ converges slowly.
- [ ] **Compare sc and oc on the same mesh** for k_eff². Mesh errors largely cancel in the difference.
- [ ] **Electrode mass and thickness**: massless equipotentials are fine for coupling, but shift frequencies. Add a metal layer for the final check.
- [ ] **Time convention**: e^{iωt} (COMSOL, this repo) needs Im < 0 for loss. Absorbers written for e^{−iωt} become gain, which is easy to miss when porting an absorber from another code.
- [ ] **Reciprocity and symmetry checks**: Y₁₂ = Y₂₁. A Hermitian system without loss. e = 0 reproduces the purely mechanical solver. Cheap and catches most assembly bugs.
- [ ] **Multimode**: a high k_eff² of the target mode means little if neighbouring modes also couple. Look at the admittance spectrum, not one eigenvalue.
- [ ] **Degeneracies**: sensitivities of degenerate clusters need subspace treatment. `wavesolver.piezo` averages them.
- [ ] **Material data**: a few percent between datasets, plus thin-film and cryogenic shifts. Do not over-optimise below that tolerance; use robustness (eroded/dilated) instead.

## 15. Where this lives in the repo

| Concept | Code |
|---|---|
| Constitutive data, rotations, Bond matrix | `wavesolver/materials.py` (`LN_PIEZO`, `x_cut`, `rotate_piezo`, `bond`) |
| (u, φ) FEM, electrodes, k_eff², admittance, PML, sensitivities | `wavesolver/piezo.py` (`PiezoSolver`, `PiezoLayer`, `Electrode`) |
| Thin-plate S0/SH0 k² vs angle (analytic + FEM) | `validation/piezo_plate_k2.py` |
| Orientation picture, spine vs crystal axes | `validation/ln_orientation.py` |
| Angle optimisation of an IDT cell | `validation/piezo_angle.py` |
| Gradient checks | `validation/piezo_gradient_check.py` |
| Absorber reflection | `validation/piezo_absorber.py` |

## 16. Reading list

Foundations

- B. A. Auld, *Acoustic Fields and Waves in Solids*, Vols. I–II. The reference for
quasi-static piezoelectric waves, the perturbation and overlap formulas, and plate modes.
- D. Royer and E. Dieulesaint, *Elastic Waves in Solids*, Vols. I–II. Clear on anisotropy
and on Christoffel with piezoelectric stiffening.
- J. F. Nye, *Physical Properties of Crystals*. Tensor symmetry and Neumann's principle.
- IEEE Std 176-1987, *IEEE Standard on Piezoelectricity*. Definitions of k, cut notation,
sign conventions.

Devices

- D. Morgan, *Surface Acoustic Wave Devices for Signal Processing*. IDT design, the
Smith/crossed-field models, reflections, SPUDTs.
- K.-y. Hashimoto, *Surface Acoustic Wave Devices in Telecommunications*. COM and P-matrix
models.

Lithium niobate

- R. S. Weis and T. K. Gaylord, "Lithium niobate: summary of physical properties and crystal
structure", Appl. Phys. A (1985).
- A. W. Warner, M. Onoe, G. A. Coquin, J. Acoust. Soc. Am. (1967). Original elastic,
piezoelectric and dielectric constants.
- G. Kovacs et al., IEEE Ultrasonics Symposium (1990). Revised LN/LT constants.

Piezo-optomechanics and quantum acoustics (LN and related)

- W. Jiang et al., "Lithium niobate piezo-optomechanical crystals", Optica (2019).
- P. Arrangoiz-Arriola et al., "Resolving the energy levels of a nanomechanical oscillator",
Nature (2019). LN phononic-crystal resonators coupled to a qubit.
- W. Jiang et al., "Efficient bidirectional piezo-optomechanical transduction between
microwave and optical frequency", Nat. Commun. (2020).
- M. Mirhosseini et al., "Superconducting qubit to optical photon transduction", Nature
(2020). AlN-on-Si, same architecture class.
- E. A. Wollack et al., "Loss channels affecting lithium niobate phononic crystal resonators
at cryogenic temperature", Appl. Phys. Lett. (2021).
- W. Weaver et al., "An integrated microwave-to-optics interface for scalable quantum
computing", Nat. Nanotechnol. (2024).
