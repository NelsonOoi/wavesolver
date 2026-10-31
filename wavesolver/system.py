"""
System model of a piezo-optomechanical transducer: microwave ↔ mechanics ↔ optics.

One optical mode a (ω_c, λ ≈ 1550 nm) with a single end-fire port: intrinsic loss κ_i,
external κ_ex, κ = κ_i + κ_ex, η_o = κ_ex/κ. One mechanical mode b (ω_m = 2π·4 GHz) with
intrinsic loss γ_i and the piezo (microwave) port γ_ex, γ = γ_i + γ_ex, η_m = γ_ex/γ. A pump
at detuning Δ = ω_L − ω_c (Δ = −ω_m is red, beam splitter) linearises the interaction to
H_int = ħ g (a + a†)(b + b†), g = g0 √n_c. All rates are angular (rad/s); `two_pi(f)` and
`/ (2π)` convert from and to Hz.

Formulas (derivations in the docstrings):
- n_c = κ_ex/(Δ² + κ²/4) · P/(ħω_L)                         single-port pump
- Γ_∓ = g² κ/(κ²/4 + (Δ ± ω_m)²)                            anti-Stokes / Stokes rates
- η = η_o η_m 4C₋/(1 + C₋ − C₊)², C_∓ = Γ_∓/γ               conversion at the dressed resonance
  At Δ = −ω_m: C₋ = C = 4g²/(κγ), C₊ = rC, r = κ²/(κ² + 16ω_m²) ≈ (κ/4ω_m)², so the only
  non-resolved-sideband correction is C → C(1 − r) in the damping term.
- N_add = (γ_i n_th + Γ₊)/γ_ex                               input-referred added noise (quanta)

The model keeps the full optical susceptibility (no optical RWA) and takes the mechanical
RWA (g, γ ≪ ω_m) and a flat optical response across the dressed mechanical line
(γ(1 + C) ≪ κ). `exact_response` drops both approximations by solving the 4 × 4 linearised
Langevin equations, as a check.

Coupling optimum. η = 4G κ_ex γ_ex/(κγ + G)² with G = 4g². At fixed n_c (i.e. fixed
absorption heating) ∂η/∂κ_ex = 0 gives κ_ex = κ_i + G/γ = κ_i (1 + C_i), C_i = 4g²/(κ_i γ):
critical coupling for C_i ≪ 1, overcoupled as C_i → 1. Optimising both ports gives
κ_ex/κ_i = γ_ex/γ_i = √(1 + C₀), C₀ = 4g²/(κ_i γ_i), with η_max = (√(1+C₀) − 1)/(√(1+C₀) + 1).
At fixed pump power instead (heating by stray pump light outside the cavity), n_c/P peaks at
κ_ex = √(κ_i² + 4Δ²), heavily overcoupled at 4 GHz. `Heating.xi` interpolates between the two.

Placeholders until the solvers deliver them: g0 (default 2π·0.5 MHz; 2D Si OMCs report
~0.3–1 MHz), γ_i (default 2π·100 kHz) and the heating law n_th = n_0 + A h^α with
A = 0.1, α = 1/3 (sub-linear, as reported for Si OMCs at mK, α ~ 0.3–0.5).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

HBAR = 1.054571817e-34  # J s
C_LIGHT = 299792458.0  # m/s
TWO_PI = 2 * np.pi


def two_pi(f):
    """Frequency in Hz → angular rate in rad/s."""
    return TWO_PI * np.asarray(f, dtype=float)


def kappa_from_q(Q, wavelength: float = 1550e-9):
    """Energy decay rate κ = ω/Q in rad/s."""
    return TWO_PI * C_LIGHT / wavelength / np.asarray(Q, dtype=float)


def intrinsic_q(Q_loaded, eta_o):
    """
    Q_i from the loaded Q and η_o = κ_ex/κ: κ_i = (1 − η_o) κ, so Q_i = Q_L/(1 − η_o).
    For a single-port reflection dip R_min = (1 − 2η_o)², the depth fixes η_o only up to
    η_o ↔ 1 − η_o; this assumes the undercoupled branch η_o < 1/2.
    """
    return np.asarray(Q_loaded, dtype=float) / (1 - np.asarray(eta_o, dtype=float))


@dataclass(frozen=True)
class Transducer:
    """Device parameters, SI units, rates in rad/s. Fields may be numpy arrays (broadcast)."""

    Q_i: float = 3e5  # optical intrinsic Q
    kappa_ex: float | None = None  # rad/s; None → critical (κ_ex = κ_i)
    gamma_i: float = TWO_PI * 100e3  # mechanical intrinsic loss, rad/s (placeholder)
    gamma_ex: float | None = None  # piezo port, rad/s; None → critical (γ_ex = γ_i)
    g0: float = TWO_PI * 0.5e6  # vacuum optomechanical coupling, rad/s (placeholder)
    f_m: float = 4e9  # Hz
    wavelength: float = 1550e-9  # m
    detuning: float | None = None  # Δ = ω_L − ω_c, rad/s; None → −ω_m (red sideband)

    def replace(self, **kw) -> Transducer:
        return dataclasses.replace(self, **kw)

    @property
    def omega_m(self):
        return two_pi(self.f_m)

    @property
    def omega_L(self):
        return TWO_PI * C_LIGHT / self.wavelength

    @property
    def kappa_i(self):
        return kappa_from_q(self.Q_i, self.wavelength)

    @property
    def k_ex(self):
        return self.kappa_i if self.kappa_ex is None else np.asarray(self.kappa_ex, dtype=float)

    @property
    def kappa(self):
        return self.kappa_i + self.k_ex

    @property
    def eta_o(self):
        return self.k_ex / self.kappa

    @property
    def g_ex(self):
        return np.asarray(self.gamma_i if self.gamma_ex is None else self.gamma_ex, dtype=float)

    @property
    def gamma(self):
        return self.gamma_i + self.g_ex

    @property
    def eta_m(self):
        return self.g_ex / self.gamma

    @property
    def delta(self):
        return -self.omega_m if self.detuning is None else np.asarray(self.detuning, dtype=float)


# --- Pump ---------------------------------------------------------------------------------


def photons_per_watt(p: Transducer):
    """
    n_c/P for a single-port cavity. Steady state of ȧ = (iΔ − κ/2) a + √κ_ex a_in with
    |a_in|² = P/(ħω_L): |a|² = κ_ex/(Δ² + κ²/4) · P/(ħω_L). Maximal over κ_ex at
    κ_ex = √(κ_i² + 4Δ²) (set d/dκ_ex = 0).
    """
    return p.k_ex / (p.delta**2 + p.kappa**2 / 4) / (HBAR * p.omega_L)


def pump_power(p: Transducer, n_c):
    """Input pump power (W) for n_c intracavity photons."""
    return np.asarray(n_c, dtype=float) / photons_per_watt(p)


def kappa_ex_max_photons(p: Transducer):
    """κ_ex maximising n_c/P at the pump detuning: √(κ_i² + 4Δ²)."""
    return np.sqrt(p.kappa_i**2 + 4 * p.delta**2)


# --- Rates and figures of merit --------------------------------------------------------------


def scattering_rates(p: Transducer, n_c):
    """
    (Γ₋, Γ₊): anti-Stokes (phonon → photon at ω_L + ω_m) and Stokes (photon + phonon pair
    at ω_L − ω_m) rates, Fermi golden rule with the cavity density of states:
    Γ_∓ = g² κ/(κ²/4 + (Δ ± ω_m)²), g² = g0² n_c. Net optical damping Γ₋ − Γ₊.
    """
    g2 = p.g0**2 * np.asarray(n_c, dtype=float)
    den = p.kappa**2 / 4
    return g2 * p.kappa / (den + (p.delta + p.omega_m) ** 2), g2 * p.kappa / (den + (p.delta - p.omega_m) ** 2)


def cooperativity(p: Transducer, n_c):
    """C = 4 g0² n_c/(κγ), the resolved-sideband cooperativity (= Γ₋/γ at Δ = −ω_m)."""
    return 4 * p.g0**2 * np.asarray(n_c, dtype=float) / (p.kappa * p.gamma)


def intrinsic_cooperativity(p: Transducer, n_c):
    """C₀ = 4 g0² n_c/(κ_i γ_i): independent of both couplers; sets the coupling optimum."""
    return 4 * p.g0**2 * np.asarray(n_c, dtype=float) / (p.kappa_i * p.gamma_i)


def electromechanical_g(gamma_e, f_m, Z_r: float = 700.0, R_0: float = 50.0):
    """
    g_em (rad/s) between the mechanical mode and a microwave resonator of impedance Z_r, from
    γ_e (rad/s), the mode's external loss rate into a load R_0. A BVD motional branch C_m
    radiates γ_e = R_0 ω_m² C_m into R_0, and couples to a resonator of C_r = 1/(ω_m Z_r) at
    g_em = (ω_m/2)√(C_m/C_r), so g_em² = γ_e ω_m Z_r/(4 R_0).
    """
    return np.sqrt(np.asarray(gamma_e, dtype=float) * two_pi(f_m) * Z_r / (4 * R_0))


def resonator_gamma_ex(g_em, kappa_r, detuning=0.0):
    """
    Mechanical external rate (rad/s) through a microwave resonator of linewidth κ_r with
    g_em ≪ κ_r (the resonator adiabatically eliminated): γ_ex = 4g²κ_r/(κ_r² + 4δ²), δ the
    mechanics-resonator detuning. Only the fraction κ_r,ex/κ_r of it reaches the line.
    """
    g, k = np.asarray(g_em, dtype=float), np.asarray(kappa_r, dtype=float)
    return 4 * g**2 * k / (k**2 + 4 * np.asarray(detuning, dtype=float) ** 2)


def sideband_ratio(p: Transducer):
    """
    Γ₊/Γ₋ = (κ²/4 + (Δ + ω_m)²)/(κ²/4 + (Δ − ω_m)²); at Δ = −ω_m this is
    r = κ²/(κ² + 16 ω_m²) ≈ (κ/4ω_m)². Also the quantum back-action floor r/(1 − r) of the
    phonon occupation under sideband cooling.
    """
    den = p.kappa**2 / 4
    return (den + (p.delta + p.omega_m) ** 2) / (den + (p.delta - p.omega_m) ** 2)


def efficiency(p: Transducer, n_c):
    """
    Microwave → optical photon conversion efficiency at the dressed mechanical resonance,
    any pump detuning (red-sideband operation at Δ = −ω_m).

    In the mechanical RWA, b[ω] = χ_eff(ω)(√γ_ex b_in − i g a_x + ...), with a_x = a[ω] +
    a†[ω] carrying χ_c(ω) and χ_c*(−ω), χ_c(ω) = 1/(κ/2 − i(ω + Δ)). The optical
    self-energy has real part (Γ₋ − Γ₊)/2, so 1/χ_eff = (γ + Γ₋ − Γ₊)/2 − i(ω − ω_m').
    The output a_out = √κ_ex a[ω] then gives at ω = ω_m'
        |S|² = κ_ex γ_ex g² |χ_c(ω_m)|² · 4/(γ + Γ₋ − Γ₊)²,   g²|χ_c(ω_m)|² = Γ₋/κ,
    i.e. η = η_o η_m 4C₋/(1 + C₋ − C₊)². Resolved sideband: η = η_o η_m 4C/(1 + C)².
    """
    gm, gp = scattering_rates(p, n_c)
    cm, cp = gm / p.gamma, gp / p.gamma
    return p.eta_o * p.eta_m * 4 * cm / (1 + cm - cp) ** 2


def bandwidth(p: Transducer, n_c):
    """Transduction bandwidth (FWHM of the dressed mechanical line), rad/s: γ + Γ₋ − Γ₊."""
    gm, gp = scattering_rates(p, n_c)
    return p.gamma + gm - gp


def added_noise(p: Transducer, n_c, n_th):
    """
    Input-referred added noise (quanta per mode, normally ordered), red-detuned operation.
    Output photon flux density at the anti-Stokes frequency gets, besides the signal
    |S|² n_in: the intrinsic mechanical bath |S|² (γ_i/γ_ex) n_th, and optical vacuum at the
    Stokes frequency (the a† inputs of both optical ports, through χ_c*(−ω)), which drives
    b at rate Γ₊: |S|² Γ₊/γ_ex. Hence N_add = (γ_i n_th + Γ₊)/γ_ex, independent of C. The
    microwave input is taken as vacuum (mK). n_th is the occupation of the intrinsic bath.
    """
    _, gp = scattering_rates(p, n_c)
    return (p.gamma_i * np.asarray(n_th, dtype=float) + gp) / p.g_ex


def blue_detuned(p: Transducer, n_c, tau: float | None = None):
    """
    Blue-sideband (Δ = +ω_m) two-mode squeezing, for heralding. Returns
    (Γ₊, C_net, herald_prob): pair rate Γ₊ = 4g²/κ on resonance, net anti-damping
    C_net = (Γ₊ − Γ₋)/γ (CW operation needs C_net < 1, the parametric instability), and,
    if tau (pulse length, s) is given, the probability that a Stokes photon leaves through
    the port, η_o (1 − exp(−Γ₊ τ)) (short pulse, Γ₊τ ≪ 1 regime; detection losses excluded).
    """
    q = p.replace(detuning=p.omega_m)
    gm, gp = scattering_rates(q, n_c)
    herald = None if tau is None else q.eta_o * (1 - np.exp(-gp * tau))
    return gp, (gp - gm) / q.gamma, herald


# --- Heating -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Heating:
    """
    Optically induced heating of the intrinsic mechanical bath, n_th = n_0 + A h^α. The
    heating drive h is in units of intracavity photons:
        h = n_c + ξ n_ref(P),   n_ref(P) = P κ_i/(ħω_L (Δ² + κ_i²)),
    where n_ref is the n_c that P gives at the same detuning in a critically coupled cavity
    (`photons_per_watt` with κ_ex = κ_i). ξ = 0: absorption inside the cavity only (heating
    ∝ n_c^α); ξ = 1: at critical coupling, stray pump light (taper, mirror, substrate) heats
    as much as intracavity light; ξ ≫ 1: heating ∝ P^α.
    A = 0.1, α = 1/3 are placeholders (sub-linear, as reported for Si OMCs at mK); the
    coupling optimum at fixed heating does not depend on A, α or n_0, only on ξ.
    """

    n0: float = 0.01
    A: float = 0.1
    alpha: float = 1 / 3
    xi: float = 0.0

    @staticmethod
    def _ref_photons_per_watt(p: Transducer):
        return p.kappa_i / (p.delta**2 + p.kappa_i**2) / (HBAR * p.omega_L)

    def drive(self, p: Transducer, n_c):
        n_c = np.asarray(n_c, dtype=float)
        return n_c + self.xi * pump_power(p, n_c) * self._ref_photons_per_watt(p)

    def n_th(self, p: Transducer, n_c):
        return self.n0 + self.A * self.drive(p, n_c) ** self.alpha

    def n_c_at_drive(self, p: Transducer, h):
        """Intracavity photons for heating drive h: n_c = h/(1 + ξ L), L = n_ref/n_c."""
        L = self._ref_photons_per_watt(p) / photons_per_watt(p)
        return np.asarray(h, dtype=float) / (1 + self.xi * L)


# --- Coupling optimum ---------------------------------------------------------------------


def optimal_kappa_ex(p: Transducer, h: float, heating: Heating = Heating()) -> float:
    """
    κ_ex (rad/s) maximising η at fixed heating drive h (fixed n_th, hence fixed thermal noise),
    with the full non-resolved-sideband efficiency. ξ = 0 is the fixed-n_c problem.
    """

    def neg_eta(log_x):
        q = p.replace(kappa_ex=p.kappa_i * np.exp(log_x))
        return -efficiency(q, heating.n_c_at_drive(q, h))

    res = minimize_scalar(neg_eta, bounds=(np.log(1e-2), np.log(1e3)), method="bounded", options={"xatol": 1e-6})
    return float(p.kappa_i * np.exp(res.x))


def optimal_couplings_analytic(C0):
    """
    Resolved-sideband optimum at fixed n_c, both ports free: κ_ex/κ_i = γ_ex/γ_i = s,
    s = √(1 + C₀), η_max = (s − 1)/(s + 1). From η = 4G κ_ex γ_ex/(κγ + G)²: the stationarity
    conditions κ_ex = κ_i + G/γ, γ_ex = γ_i + G/κ have the symmetric solution s² = 1 + C₀.
    Returns (s, η_max).
    """
    s = np.sqrt(1 + np.asarray(C0, dtype=float))
    return s, (s - 1) / (s + 1)


# --- Exact linearised Langevin check ------------------------------------------------------------


def exact_response(p: Transducer, n_c, omega, n_th: float = 0.0):
    """
    Solve the linearised Langevin equations for (a, a†, b, b†) without RWA,
        ȧ = (iΔ − κ/2) a − i g (b + b†) + √κ_ex a_ex + √κ_i a_i,
        ḃ = (−iω_m − γ/2) b − i g (a + a†) + √γ_ex b_ex + √γ_i b_i,
    at signal frequencies omega (rad/s, lab frame for b, pump frame for a), a_out = √κ_ex a −
    a_ex. Returns (η(ω), N_add(ω)): |S_{a_out ← b_ex}|² and the normally ordered output noise
    referred to the input, counting thermal (b_i), and vacuum from every creation-operator input.
    Scalar parameters only.
    """
    g = p.g0 * np.sqrt(n_c)
    k, kex, ki = float(p.kappa), float(p.k_ex), float(p.kappa_i)
    gam, gex, gi = float(p.gamma), float(p.g_ex), float(p.gamma_i)
    D, wm = float(p.delta), float(p.omega_m)
    M = np.array(
        [
            [1j * D - k / 2, 0, -1j * g, -1j * g],
            [0, -1j * D - k / 2, 1j * g, 1j * g],
            [-1j * g, -1j * g, -1j * wm - gam / 2, 0],
            [1j * g, 1j * g, 0, 1j * wm - gam / 2],
        ]
    )
    # Inputs: a_ex, a_ex†, a_i, a_i†, b_ex, b_ex†, b_i, b_i†
    K = np.zeros((4, 8))
    K[0, 0] = K[1, 1] = np.sqrt(kex)
    K[0, 2] = K[1, 3] = np.sqrt(ki)
    K[2, 4] = K[3, 5] = np.sqrt(gex)
    K[2, 6] = K[3, 7] = np.sqrt(gi)
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    A = -1j * omega[:, None, None] * np.eye(4) - M
    T = np.linalg.solve(A, np.broadcast_to(K, (len(omega), 4, 8)))
    S = np.sqrt(kex) * T[:, 0, :]
    S[:, 0] -= 1
    P2 = np.abs(S) ** 2
    eta = P2[:, 4]
    noise = P2[:, 6] * n_th + P2[:, 7] * (n_th + 1) + P2[:, 1] + P2[:, 3] + P2[:, 5]
    return eta, noise / eta


# --- Microwave resonator: three-mode model -------------------------------------------------
#
# Microwave line → single-mode resonator c (ω_r = ω_m + δ_r, κ_r = κ_r,ex + κ_r,i, overcoupled)
# → mechanics b (γ_i) → optics a → optical line. H_int = ħ g_em (b†c + bc†) + ħ g (a†b + ab†)
# (beam splitters, red sideband), g = g0 √n_c. In a frame at ω_m, signal offset ν:
#
#     [κ/2 − i(ν + δ_o)   i g          0              ] [a]
#     [i g                γ'/2 − iν    i g_em         ] [b] = L x_in,
#     [0                  i g_em       κ_r/2 − i(ν−δ_r)] [c]
#
# δ_o = Δ + ω_m (0 on the red sideband), γ' = γ_i − Γ₊: the Stokes process (photon pairs at
# ω_L − ω_m, non-resolved sideband) enters as anti-damping Γ₊ and a vacuum input of rate Γ₊.
# On resonance (ν = δ_o = δ_r = 0, Γ₊ → 0) the chain gives exactly
#     η = η_o η_r · 4 C_o C_e/(1 + C_o + C_e)²,  C_o = 4g²/(κγ_i), C_e = 4g_em²/(κ_r γ_i),
# η_r = κ_r,ex/κ_r: the two-port result with γ_ex = 4g_em²/κ_r, for any g_em. It peaks at
# C_e = 1 + C_o, g_em* = ½√((1 + C_o) κ_r γ_i), at η_o η_r C_o/(1 + C_o), with a dressed
# linewidth ≈ γ_i (1 + C_o + C_e) = 2γ_i(1 + C_o) independent of κ_r. For g_em ≳ κ_r/2 the
# resonator and the mechanics hybridise (splitting 2g_em) and the response leaves the
# two-port (adiabatic) form; `three_mode_response` keeps the resonator as a mode throughout.


@dataclass(frozen=True)
class Resonator:
    """Single-mode microwave resonator and its piezo coupling to the mechanical mode (rad/s)."""

    kappa_ex: float = TWO_PI * 0.9e6  # into the microwave line (overcoupled)
    kappa_i: float = TWO_PI * 0.1e6  # intrinsic (assumed; Q_i ≈ 4e4 at 4 GHz)
    g_em: float = TWO_PI * 0.1e6  # electromechanical coupling to the mechanical mode
    detuning: float = 0.0  # δ_r = ω_r − ω_m
    n_th: float = 0.0  # resonator bath occupation (mK: ~0)

    def replace(self, **kw) -> Resonator:
        return dataclasses.replace(self, **kw)

    @property
    def kappa(self):
        return np.asarray(self.kappa_ex, dtype=float) + np.asarray(self.kappa_i, dtype=float)

    @property
    def eta_r(self):
        return self.kappa_ex / self.kappa


def two_port_equivalent(p: Transducer, r: Resonator) -> Transducer:
    """
    The resonator adiabatically eliminated (g_em ≪ κ_r): the mechanics sees Γ_em =
    4g_em²κ_r/(κ_r² + 4δ_r²) (`resonator_gamma_ex`), of which the fraction η_r reaches the line:
    γ_ex → η_r Γ_em, γ_i → γ_i + (1 − η_r) Γ_em.
    """
    G = resonator_gamma_ex(r.g_em, r.kappa, r.detuning)
    return p.replace(gamma_ex=r.eta_r * G, gamma_i=p.gamma_i + (1 - r.eta_r) * G)


def _three_mode_T(p: Transducer, r: Resonator, n_c, nu):
    """(a, b, c) response T(ν) = M(ν)⁻¹, shape (len(ν), 3, 3), and Γ₊."""
    g = p.g0 * np.sqrt(float(n_c))
    _, gp = scattering_rates(p, n_c)
    d_o = float(p.delta + p.omega_m)
    nu = np.atleast_1d(np.asarray(nu, dtype=float))
    M = np.zeros((len(nu), 3, 3), complex)
    M[:, 0, 0] = float(p.kappa) / 2 - 1j * (nu + d_o)
    M[:, 1, 1] = (float(p.gamma_i) - float(gp)) / 2 - 1j * nu
    M[:, 2, 2] = float(r.kappa) / 2 - 1j * (nu - float(r.detuning))
    M[:, 0, 1] = M[:, 1, 0] = 1j * g
    M[:, 1, 2] = M[:, 2, 1] = 1j * float(r.g_em)
    return np.linalg.inv(M), float(gp)


def three_mode_response(p: Transducer, r: Resonator, n_c, nu, n_th: float = 0.0):
    """
    Microwave line → optical line conversion η(ν) = κ_ex κ_r,ex |T_ac(ν)|² and the added noise
    referred to the microwave input (quanta, normally ordered): the intrinsic mechanical bath
    (γ_i, n_th), the Stokes vacuum (Γ₊), and the resonator's intrinsic bath (κ_r,i, r.n_th):
        N_add = [|T_ab|² (γ_i n_th + Γ₊) + |T_ac|² κ_r,i n_r] / (κ_r,ex |T_ac|²).
    ν: signal offset from ω_m (rad/s). p.gamma_ex is ignored (the resonator is the port).
    """
    T, gp = _three_mode_T(p, r, n_c, nu)
    Tab, Tac = np.abs(T[:, 0, 1]) ** 2, np.abs(T[:, 0, 2]) ** 2
    eta = float(p.k_ex) * float(r.kappa_ex) * Tac
    noise = Tab * (float(p.gamma_i) * n_th + gp) + Tac * float(r.kappa_i) * r.n_th
    return eta, noise / (float(r.kappa_ex) * Tac)


def three_mode_efficiency_resonant(p: Transducer, r: Resonator, n_c):
    """η at ν = 0 for δ_o = δ_r = 0, resolved sideband: η_o η_r 4C_oC_e/(1 + C_o + C_e)²."""
    Co = 4 * p.g0**2 * np.asarray(n_c, dtype=float) / (p.kappa * p.gamma_i)
    Ce = 4 * np.asarray(r.g_em, dtype=float) ** 2 / (r.kappa * p.gamma_i)
    return p.eta_o * r.eta_r * 4 * Co * Ce / (1 + Co + Ce) ** 2


def matched_g_em(p: Transducer, r: Resonator, n_c):
    """g_em (rad/s) maximising the resonant η: C_e = 1 + C_o, g_em* = ½√((1 + C_o) κ_r γ_i)."""
    Co = 4 * p.g0**2 * np.asarray(n_c, dtype=float) / (p.kappa * p.gamma_i)
    return 0.5 * np.sqrt((1 + Co) * r.kappa * p.gamma_i)


def conversion_peak(p: Transducer, r: Resonator, n_c, n_th: float = 0.0, n: int = 40001):
    """
    Peak of η(ν) and its bandwidth from `three_mode_response` on an adaptive grid. Returns
    (η_max, ν_peak, bandwidth, N_add at the peak); bandwidth = width (rad/s) of the contiguous
    region η ≥ η_max/2 around the peak. A hybridised response (g_em ≳ κ_r/2) splits into two
    equal peaks at ν ≈ ±g_em; the bandwidth is that of one peak unless they merge.
    """
    Co = 4 * float(p.g0) ** 2 * float(n_c) / (float(p.kappa) * float(p.gamma_i))
    scale = max(float(p.gamma_i) * (1 + Co), float(r.kappa), 2 * float(r.g_em), 4 * float(r.g_em) ** 2 / float(r.kappa))
    span = 8 * scale + 2 * abs(float(r.detuning))
    nu = np.linspace(-span, span, n)
    eta, N = three_mode_response(p, r, n_c, nu, n_th)
    i = int(eta.argmax())
    # refine the peak once
    nu2 = nu[i] + np.linspace(-2, 2, 2001) * (nu[1] - nu[0])
    eta2, N2 = three_mode_response(p, r, n_c, nu2, n_th)
    j = int(eta2.argmax())
    emax = eta2[j]
    above = eta >= emax / 2
    lo, hi = i, i
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < len(nu) - 1 and above[hi + 1]:
        hi += 1
    bw = (hi - lo + 1) * (nu[1] - nu[0])
    return float(emax), float(nu2[j]), float(bw), float(N2[j])


def exact_three_mode_response(p: Transducer, r: Resonator, n_c, omega, n_th: float = 0.0):
    """
    `exact_response` with the resonator: linearised Langevin equations for (a, a†, b, b†, c, c†)
    without any RWA, piezo coupling ħ g_em (b + b†)(c + c†), ω_r = ω_m + δ_r. Returns
    (η(ω), N_add(ω)) for the microwave line input c_ex, counting thermal inputs b_i (n_th),
    c_i (r.n_th) and vacuum from every creation-operator input. Scalar parameters only.
    """
    g, G = p.g0 * np.sqrt(n_c), float(r.g_em)
    k, kex, ki = float(p.kappa), float(p.k_ex), float(p.kappa_i)
    gi = float(p.gamma_i)
    D, wm = float(p.delta), float(p.omega_m)
    wr, kr, krex, kri = wm + float(r.detuning), float(r.kappa), float(r.kappa_ex), float(r.kappa_i)
    M = np.zeros((6, 6), complex)
    M[0, 0], M[1, 1] = 1j * D - k / 2, -1j * D - k / 2
    M[2, 2], M[3, 3] = -1j * wm - gi / 2, 1j * wm - gi / 2
    M[4, 4], M[5, 5] = -1j * wr - kr / 2, 1j * wr - kr / 2
    M[0, 2:4] = -1j * g
    M[1, 2:4] = 1j * g
    M[2, 0:2] = -1j * g
    M[3, 0:2] = 1j * g
    M[2, 4:6] = -1j * G
    M[3, 4:6] = 1j * G
    M[4, 2:4] = -1j * G
    M[5, 2:4] = 1j * G
    # inputs: a_ex, a_ex†, a_i, a_i†, b_i, b_i†, c_ex, c_ex†, c_i, c_i†
    K = np.zeros((6, 10))
    K[0, 0] = K[1, 1] = np.sqrt(kex)
    K[0, 2] = K[1, 3] = np.sqrt(ki)
    K[2, 4] = K[3, 5] = np.sqrt(gi)
    K[4, 6] = K[5, 7] = np.sqrt(krex)
    K[4, 8] = K[5, 9] = np.sqrt(kri)
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    A = -1j * omega[:, None, None] * np.eye(6) - M
    T = np.linalg.solve(A, np.broadcast_to(K, (len(omega), 6, 10)))
    S = np.sqrt(kex) * T[:, 0, :]
    S[:, 0] -= 1
    P2 = np.abs(S) ** 2
    eta = P2[:, 6]
    n_r = r.n_th
    noise = P2[:, 4] * n_th + P2[:, 5] * (n_th + 1) + P2[:, 8] * n_r + P2[:, 9] * (n_r + 1) + P2[:, 1] + P2[:, 3] + P2[:, 7]
    return eta, noise / eta
