"""
LN-patch transducer on a Si strip, for the mechanical port: the IDT period cell (g_em, C₀, k_eff² per
finger pair, and their θ gradients from wavesolver.piezo), the guided modes of the Si output strip at f_m,
and the launch problem (finite IDT → bare Si strip → PML) with the radiated power split over the strip
modes (conversion efficiency into the channel mode versus parasitic modes).

Geometry (local frame, propagation along +x, aperture along y, z up): a Si strip of width w_si
(220 nm, cubic [100] ∥ x — also [010] ∥ x, so the frame may be rotated by 90° with the device, as for
option A), under an LN ridge (250 nm, X-cut, in-plane angle θ of wavesolver.materials.x_cut, top width
w_ln, sidewall angle `sidewall` from the substrate plane: the bottom face is wider by t/tan(sidewall)
per side; stepped in n_ln sub-layers), with Al fingers (t_al) across the LN top: one hot and one
grounded finger per period p, width fw·p/2 (metallisation ratio fw).

Modal power. A guided strip mode m at frequency f has wavenumber k_m, transverse profile φ_m (nodal
values across y and through the thickness functions, from a one-element Bloch cell of the strip) and
group velocity v_m (finite difference of f(k)). In the driven field on the bare strip, each node column
x_i is fitted (least squares over the forward and backward modes) to Σ_m (a_m e^{−ik_m x} + b_m
e^{+ik_m x}) φ_m. The time-averaged power of mode m is P_m = |a_m|² v_m (ω²/2) ∫ρ|φ_m|² dA (energy
velocity × energy per length, lossless guide). The electrical input P_in = ½ Re(Y)|V|² closes the
balance (everything radiated leaves through the strip into the PML).

Units: µm, Hz, SI at the interface (as wavesolver.piezo).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .materials import AL_COMSOL, LN_PIEZO, SI, SI_EPS_R
from .piezo import Electrode, PiezoLayer, PiezoSolver


def overlap_1d(lo, hi, a, b):
    """Fraction of each interval [lo, hi] inside [a, b]."""
    return np.clip((np.minimum(hi, b) - np.maximum(lo, a)) / (hi - lo), 0, 1)


@dataclass
class StripIDT:
    """
    Cross-section and fingers of the transducer (see the module docstring). Lengths in µm, angles in
    degrees. dy: element size across the strip; m: elements per period along x.
    """

    p: float = 0.63
    w_si: float = 0.80
    w_ln: float = 0.50
    t_si: float = 0.22
    t_ln: float = 0.25
    t_al: float = 0.06
    sidewall: float = 65.0
    fw: float = 0.5
    theta: float = 45.0
    n_ln: int = 2
    p_si: int = 3
    p_ln: int = 2
    p_al: int = 1
    m: int = 16
    dy: float = 0.05

    @property
    def out(self) -> float:
        return self.t_ln / math.tan(math.radians(self.sidewall))

    def n2(self) -> int:
        return max(2, int(round(self.w_si / self.dy)))

    def ye(self) -> np.ndarray:
        n2 = self.n2()
        return -self.w_si / 2 + self.w_si / n2 * np.arange(n2 + 1)

    def ln_width(self, s: int) -> float:
        """Width of LN sub-layer s (0 = bottom) at its mid-height."""
        z = (s + 0.5) / self.n_ln
        return self.w_ln + 2 * (1 - z) * self.out

    def finger_masks(self, xe: np.ndarray, x_fingers: list[tuple[float, str]]):
        """Per element column: fraction covered by hot and by ground fingers."""
        hot = np.zeros(len(xe) - 1)
        gnd = np.zeros(len(xe) - 1)
        wf = self.fw * self.p / 2
        for x, net in x_fingers:
            o = overlap_1d(xe[:-1], xe[1:], x - wf / 2, x + wf / 2)
            (hot if net == "hot" else gnd)[:] += o
        return np.clip(hot, 0, 1), np.clip(gnd, 0, 1)

    def layers(self, si_x: np.ndarray, ln_x: np.ndarray, hot_x: np.ndarray, gnd_x: np.ndarray, mass_scale=None, si=SI):
        """
        Layers on an (n1, n2) grid from per-column profiles along x: si_x (Si present), ln_x (LN present),
        hot_x / gnd_x (finger coverage); across y the strip, LN sub-layer and LN-top widths.
        """
        ye = self.ye()
        fy = lambda w: overlap_1d(ye[:-1], ye[1:], -w / 2, w / 2)  # noqa: E731
        L = [PiezoLayer(self.t_si, np.outer(si_x, fy(self.w_si)), si, eps_r=SI_EPS_R, order=self.p_si, mass_scale=mass_scale)]
        h = self.t_ln / self.n_ln
        for s in range(self.n_ln):
            L.append(PiezoLayer(h, np.outer(ln_x, fy(self.ln_width(s))), LN_PIEZO, order=self.p_ln, angle=self.theta))
        cov = np.clip(hot_x + gnd_x, 0, 1)
        L.append(PiezoLayer(self.t_al, np.outer(cov, fy(self.w_ln)), AL_COMSOL, eps_r=None, order=self.p_al))
        on = fy(self.w_ln) > 0.5
        top = len(L) - 1
        el = [Electrode("hot", np.outer(hot_x > 0.5, on), top), Electrode("gnd", np.outer(gnd_x > 0.5, on), top, "ground")]
        return L, el

    def period_cell(self) -> PiezoSolver:
        """One IDT period, Bloch-periodic along x (k = 0: the IDT's synchronous standing wave), free along y."""
        n1 = self.m
        xe = -self.p / 2 + self.p / n1 * np.arange(n1 + 1)
        hot, gnd = self.finger_masks(xe, [(-self.p / 4, "hot"), (self.p / 4, "gnd")])
        L, el = self.layers(np.ones(n1), np.ones(n1), hot, gnd)
        return PiezoSolver(np.diag([self.p, self.w_si]), L, bc=("periodic", "free"), electrodes=el, origin=(-self.p / 2, -self.w_si / 2))

    def strip_cell(self, length: float = 0.02, with_ln: bool = False) -> PiezoSolver:
        """A one-element Bloch cell of the bare (or LN-loaded) strip, for its guided modes f(k_x)."""
        n1 = 1
        L, _ = self.layers(np.ones(n1), np.ones(n1) * (1.0 if with_ln else 0.0), np.zeros(n1), np.zeros(n1))
        if not with_ln:
            L = [L[0]]
        return PiezoSolver(np.diag([length, self.w_si]), L, bc=("periodic", "free"), origin=(0.0, -self.w_si / 2))


# --- strip modes -------------------------------------------------------------------------------------


@dataclass
class StripModes:
    """Guided modes of the bare strip at frequency f: k (1/µm), v_g (m/s), node-column profiles (n2+1, F, 3), labels."""

    f: float
    k: np.ndarray
    vg: np.ndarray
    phi: np.ndarray
    label: list[str]
    mass: np.ndarray  # ∫ρ|φ|² dA per unit length, in kg·m (φ in m) per (unit profile)
    pol: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


def _parities(phi: np.ndarray, sol: PiezoSolver) -> tuple[float, float]:
    """(σ_y parity, σ_z parity) of a profile (n2+1, F, 3): ±1-ish (Re⟨φ, Sφ⟩/⟨φ,φ⟩)."""
    # σ_y: y → −y (node reversal), u_y flips sign
    sy = phi[::-1].copy()
    sy[..., 1] *= -1
    py = float(np.real(np.vdot(phi, sy)) / np.real(np.vdot(phi, phi)))
    # σ_z about the Si mid-plane only meaningful for the bare strip: Lobatto end functions swap (0 ↔ 1), bubbles of odd degree flip
    return py, np.nan


def strip_modes(idt: StripIDT, f: float, k_max: float = 16.0, nk: int = 81, n_modes: int = 12, with_ln: bool = False) -> StripModes:
    """
    Guided modes of the strip at f: bands f_n(k) on a k grid (one-element Bloch cell, free sides), the
    crossings of f by linear interpolation refined by secant, their profiles, group velocities and
    ∫ρ|φ|² per unit length.
    """
    L0 = 0.02
    sol = idt.strip_cell(L0, with_ln)
    ks = np.linspace(1e-3, k_max, nk)
    F = np.array([np.sort(sol.eigenmodes(n_modes, -0.5e9, k=(k, 0.0)).real) for k in ks])

    def f_of(k, b):
        return np.sort(sol.eigenmodes(n_modes, -0.5e9, k=(k, 0.0)).real)[b]

    found = []
    for b in range(n_modes):
        for i in range(nk - 1):
            fa, fb = F[i, b], F[i + 1, b]
            if (fa - f) * (fb - f) <= 0 and fa != fb:
                k0, k1, f0, f1 = ks[i], ks[i + 1], fa, fb
                for _ in range(4):  # secant on f_b(k) = f
                    kn = k0 + (f - f0) * (k1 - k0) / (f1 - f0)
                    fn = f_of(kn, b)
                    k0, f0, k1, f1 = k1, f1, kn, fn
                    if abs(fn - f) < 1e-6 * f:
                        break
                found.append((b, k1))
    ks_f, vgs, phis, masses, labels, pols = [], [], [], [], [], []
    for b, k in found:
        dk = 1e-3 * max(k, 0.1)
        vg = 2 * np.pi * (f_of(k + dk, b) - f_of(k - dk, b)) / (2 * dk * 1e6)
        fr, U, _ = sol.eigenmodes(n_modes, -0.5e9, k=(k, 0.0), vectors=True)
        o = np.argsort(fr.real)
        u = U[o[b]]  # (n1+1, n2+1, F, 3); node column 0 (column 1 is its Bloch image)
        phi = u[0]
        # ∫ρ|u|² over the cell / length → per unit length (internal units: ρ 1e3 kg/m³, µm)
        uv = u.reshape(-1)[sol.udofs]
        m_int = float(np.real(np.vdot(uv, sol.M @ uv)))
        mass = m_int * 1e3 * 1e-18 * 1e-12 / (L0 * 1e-6)  # kg·m²/m for φ in µm → per (µm²)
        p = np.sum(np.abs(phi) ** 2, axis=(0, 1))
        p = p / p.sum()
        py, _ = _parities(phi, sol)
        lab = "xyz"[int(np.argmax(p))] + ("+" if py > 0 else "-")
        ks_f.append(k)
        vgs.append(vg)
        phis.append(phi)
        masses.append(mass)
        labels.append(lab)
        pols.append(p)
    o = np.argsort(ks_f)
    return StripModes(f, np.array(ks_f)[o], np.array(vgs)[o], np.array(phis)[o] if phis else np.zeros((0,)), [labels[i] for i in o],
                      np.array(masses)[o], np.array(pols)[o] if pols else np.zeros((0, 3)))


# --- launch problem ------------------------------------------------------------------------------------


@dataclass
class Launch:
    """A finite IDT on the strip with bare-strip outputs and PMLs (see build_launch)."""

    sol: PiezoSolver
    xe: np.ndarray
    out_windows: list[tuple[float, float]]  # x-ranges of bare strip used for the modal fit (right, left)
    fingers: list[tuple[float, str]]
    idt: StripIDT


def build_launch(idt: StripIDT, n_pairs: int, back: str = "pml", d_back: float = 0.5, L_out: float = 2.0, L_pml: float = 2.5,
                 beta: float = 3.0, ln_margin: float | None = None, dx: float | None = None) -> Launch:
    """
    x-layout: [PML | bare strip L_out]? | LN margin | n_pairs IDT periods | LN margin | bare strip L_out | PML.
    back = "pml": symmetric (bidirectional) launch; "free": the strip ends d_back past the LN (free-end
    reflector, as the plate end in the layout). Mesh: dx (default p/m).
    """
    p = idt.p
    dx = p / idt.m if dx is None else dx
    lm = p / 2 if ln_margin is None else ln_margin
    L_idt = n_pairs * p
    segs = []  # (length, kind)
    if back == "pml":
        segs += [(L_pml, "pml_l"), (L_out, "strip")]
    else:
        segs += [(d_back, "strip")]
    segs += [(lm, "ln"), (L_idt, "idt"), (lm, "ln"), (L_out, "strip"), (L_pml, "pml_r")]
    Lx = sum(s for s, _ in segs)
    n1 = int(round(Lx / dx))
    xe = np.linspace(0, Lx, n1 + 1)
    xc = (xe[:-1] + xe[1:]) / 2
    edges = np.cumsum([0] + [s for s, _ in segs])
    kind = np.empty(n1, object)
    for (a, b), (_, kd) in zip(zip(edges[:-1], edges[1:]), segs):
        kind[(xc >= a) & (xc < b)] = kd
    ln_x = np.isin(kind, ["ln", "idt"]).astype(float)
    x_idt0 = edges[[k for _, k in segs].index("idt")]
    fingers = []
    for q in range(n_pairs):
        fingers += [(x_idt0 + q * p + p / 4, "hot"), (x_idt0 + q * p + 3 * p / 4, "gnd")]
    hot, gnd = idt.finger_masks(xe, fingers)
    L, el = idt.layers(np.ones(n1), ln_x, hot, gnd)
    s = np.ones(n1, complex)
    for kd, sign in (("pml_l", -1), ("pml_r", 1)):
        m = kind == kd
        if m.any():
            a, b = xc[m].min(), xc[m].max()
            t = (xc[m] - a) / (b - a) if sign > 0 else (b - xc[m]) / (b - a)
            s[m] = 1 - 1j * beta * t**2
    sol = PiezoSolver(np.diag([Lx, idt.w_si]), L, bc=("free", "free"), electrodes=el, origin=(0.0, -idt.w_si / 2), stretch=(s, None))
    wins = []
    i_r = [k for _, k in segs].index("pml_r")
    wins.append((edges[i_r - 1] + 0.1 * L_out, edges[i_r] - 0.1 * L_out))
    if back == "pml":
        wins.append((edges[1] + 0.1 * L_out, edges[2] - 0.1 * L_out))
    return Launch(sol, xe, wins, fingers, idt)


def modal_fit(u: np.ndarray, xn: np.ndarray, window: tuple[float, float], modes: StripModes, origin: float):
    """
    Forward/backward amplitudes (a_m, b_m) of the strip modes in the node columns of `window`
    (u: (n1+1, n2+1, F, 3) driven field in µm), the phase reference at x = origin; and the relative
    residual of the fit.
    """
    sel = np.flatnonzero((xn >= window[0]) & (xn <= window[1]))
    nm = len(modes.k)
    Fm = modes.phi.shape[2]  # the bare strip's thickness functions: the Si layer's, first in the stack
    u = u[:, :, :Fm, :]
    rows, rhs = [], []
    B = np.stack([ph.ravel() for ph in modes.phi], axis=1)  # (n2+1)·F·3 × nm
    for i in sel:
        x = xn[i] - origin
        # Bloch profiles φ(k) belong to e^{+ikx}; the e^{−ikx} wave of a lossless guide has the profile φ(−k) = conj φ(k)
        E = np.concatenate([B.conj() * np.exp(-1j * modes.k * x)[None, :], B * np.exp(1j * modes.k * x)[None, :]], axis=1)
        rows.append(E)
        rhs.append(u[i].ravel())
    A = np.concatenate(rows, axis=0)
    y = np.concatenate(rhs)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = np.linalg.norm(A @ c - y) / np.linalg.norm(y)
    return c[:nm], c[nm:], float(res)


def modal_powers(amp: np.ndarray, modes: StripModes, f: float) -> np.ndarray:
    """P_m = |a_m|² v_m (ω²/2) ∫ρ|φ_m|² (W), amplitudes multiplying φ in µm."""
    w = 2 * np.pi * f
    return np.abs(amp) ** 2 * np.abs(modes.vg) * 0.5 * w**2 * modes.mass
