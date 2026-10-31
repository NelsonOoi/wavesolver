"""
Elastic materials as 6×6 Voigt stiffness matrices and densities.

Voigt order is (xx, yy, zz, yz, xz, xy) with engineering shear strains. Stiffness is in
Pa and density in kg/m³. Rotations act on the crystal frame: a material rotated by R has
its crystal axis i along the lab direction R[:, i].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Elastic:
    name: str
    C: np.ndarray
    rho: float

    def rotated(self, R: np.ndarray) -> Elastic:
        return Elastic(self.name, rotate_stiffness(self.C, R), self.rho)


def isotropic(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] += 2 * mu
    C[np.arange(3, 6), np.arange(3, 6)] = mu
    return C


def cubic(C11: float, C12: float, C44: float) -> np.ndarray:
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    C[np.arange(3), np.arange(3)] = C11
    C[np.arange(3, 6), np.arange(3, 6)] = C44
    return C


def rot_z(angle_deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def bond(R: np.ndarray) -> np.ndarray:
    """Bond matrix M with C_lab = M C M^T for a stress-like Voigt stiffness."""
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    M = np.zeros((6, 6))
    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            M[I, J] = R[i, k] * R[j, l] + (R[i, l] * R[j, k] if k != l else 0.0)
    return M


def rotate_stiffness(C: np.ndarray, R: np.ndarray) -> np.ndarray:
    M = bond(R)
    return M @ C @ M.T


# COMSOL's isotropic "Silicon" material, used by the 2D blade benchmark.
SI_ISOTROPIC = Elastic("Si (isotropic)", isotropic(170e9, 0.28), 2329.0)
# Single-crystal Si, crystal axes along lab x, y, z ([100] along x on a (001) wafer).
SI = Elastic("Si", cubic(165.7e9, 63.9e9, 79.6e9), 2329.0)


def trigonal_3m(C11: float, C12: float, C13: float, C14: float, C33: float, C44: float) -> np.ndarray:
    """Class 3m stiffness, crystal x along an a-axis (IEEE 1949 axes)."""
    C = np.zeros((6, 6))
    C[:2, :2] = [[C11, C12], [C12, C11]]
    C[:2, 2] = C[2, :2] = C13
    C[2, 2] = C33
    C[3, 3] = C[4, 4] = C44
    C[5, 5] = (C11 - C12) / 2
    C[0, 3] = C[3, 0] = C14
    C[1, 3] = C[3, 1] = -C14
    C[4, 5] = C[5, 4] = C14
    return C


def x_cut(angle_deg: float) -> np.ndarray:
    """
    Rotation for an X-cut wafer: crystal +x along lab +z (surface normal), crystal c-axis in
    the plane at angle_deg from lab x. Flipping the wafer (-X cut) changes the sign of C14.
    """
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    return np.column_stack([[0.0, 0.0, 1.0], [s, -c, 0.0], [c, s, 0.0]])


# COMSOL's LiNbO3 (stiffness at constant electric field, crystal frame). Purely elastic: the
# piezoelectric stiffening is not included.
LN = Elastic("LiNbO3", trigonal_3m(2.03e11, 0.53e11, 0.75e11, 0.09e11, 2.45e11, 0.60e11), 4700.0)


# --- Piezoelectric materials (appended for wavesolver.piezo) -------------------------------------
#
# Stress-charge form: T = C^E S - e^T E, D = e S + ε^S E, with e the 3 × 6 piezoelectric
# stress constants (C/m²) in the Voigt order above and ε^S the clamped permittivity. Under a
# rotation R (same convention as `rotate_stiffness`) e_lab = R e M^T with M = bond(R), and
# ε_lab = R ε R^T.

EPS0 = 8.8541878128e-12  # F/m


def piezo_3m(e15: float, e22: float, e31: float, e33: float) -> np.ndarray:
    """
    Class 3m piezoelectric stress constants in the axes of `trigonal_3m`: e21 = e16 = -e22.
    Rotating the crystal by 180° about its c-axis (x, y → -x, -y), which is what separates
    some published axis conventions and what flipping an X-cut wafer does, changes the signs
    of C14 and e22 together and leaves e15, e31, e33 unchanged.
    """
    e = np.zeros((3, 6))
    e[0, 4] = e[1, 3] = e15
    e[1, 1] = e22
    e[1, 0] = e[0, 5] = -e22
    e[2, 0] = e[2, 1] = e31
    e[2, 2] = e33
    return e


def rotate_piezo(e: np.ndarray, R: np.ndarray) -> np.ndarray:
    return R @ e @ bond(R).T


def rotate_permittivity(eps: np.ndarray, R: np.ndarray) -> np.ndarray:
    return R @ eps @ R.T


@dataclass(frozen=True)
class Piezoelectric:
    """Piezoelectric solid: stiffness C^E (Pa), density, e (C/m², 3 × 6), relative ε^S (3 × 3)."""

    name: str
    C: np.ndarray
    rho: float
    e: np.ndarray
    eps_r: np.ndarray

    @property
    def elastic(self) -> Elastic:
        return Elastic(self.name, self.C, self.rho)

    def rotated(self, R: np.ndarray) -> Piezoelectric:
        return Piezoelectric(
            self.name, rotate_stiffness(self.C, R), self.rho, rotate_piezo(self.e, R), rotate_permittivity(self.eps_r, R)
        )


# LiNbO3 exactly as stored in the user's COMSOL models (model54_3D_piezo_LNSOI_IDT_full_20201029
# and the other model54 piezo models; COMSOL material library, stress-charge form, crystal
# frame). COMSOL derives c^E and e from its s^E, d data, hence the small departures from exact
# 3m symmetry (C44 ≠ C55, C56 ≠ C14, C66 ≠ (C11 - C12)/2, e15 and e22 entries differing in the
# 4th digit); they are kept so the benchmarks are like-for-like. Signs: C14 > 0 with e15, e22,
# e31, e33 > 0 (Warner et al. 1967 / IEEE 176-1949 axes), as in `trigonal_3m`.
# The `LN` above is close but not identical (C11 203, C12 53, C13 75, C14 9, C33 245, C44 60 GPa
# vs COMSOL 202.9, 52.9, 74.9, 9.0, 243.1, 59.9 GPa): use LN_COMSOL / LN_PIEZO for COMSOL
# comparisons.
LN_C_COMSOL = np.array(
    [
        [2.02897e11, 5.29177e10, 7.49098e10, 8.99874e9, 0.0, 0.0],
        [5.29177e10, 2.02897e11, 7.49098e10, -8.99874e9, 0.0, 0.0],
        [7.49098e10, 7.49098e10, 2.43075e11, 0.0, 0.0, 0.0],
        [8.99874e9, -8.99874e9, 0.0, 5.99034e10, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 5.99018e10, 8.98526e9],
        [0.0, 0.0, 0.0, 0.0, 8.98526e9, 7.48772e10],
    ]
)
LN_E_COMSOL = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 3.69594, -2.53384],
        [-2.53764, 2.53764, 0.0, 3.69548, 0.0, 0.0],
        [0.193644, 0.193644, 1.30863, 0.0, 0.0, 0.0],
    ]
)
LN_EPS_S_COMSOL = np.diag([43.6, 43.6, 29.16])  # relative; ε^T = (84, 84, 30) in the same data
LN_PIEZO = Piezoelectric("LiNbO3 (COMSOL)", LN_C_COMSOL, 4700.0, LN_E_COMSOL, LN_EPS_S_COMSOL)
LN_COMSOL = LN_PIEZO.elastic  # corrected c^E for purely elastic use

# Other materials of model54_3D_piezo_LNSOI_IDT_full_20201029: isotropic Si as set in its
# "Linear Elastic Material 2 Si" (E = 165 GPa, ν = 0.22, ρ = 2330), COMSOL Aluminum for the
# 100 nm electrodes, and the relative permittivity of Si, which COMSOL solves as a dielectric.
SI_IDT_COMSOL = Elastic("Si (isotropic, model54 IDT)", isotropic(165e9, 0.22), 2330.0)
AL_COMSOL = Elastic("Al (COMSOL)", isotropic(70e9, 0.33), 2700.0)
SI_EPS_R = 11.7


# --- Photoelastic constants (appended for wavesolver.coupling) -----------------------------------
#
# Δ(ε⁻¹)_I = p_IJ S_J in the Voigt order above, with engineering shear strains (S_4 = 2 S_yz)
# and no factors of 2 on the impermeability, so p_IJ = p_ijkl and p rotates exactly as a
# stiffness: p_lab = M p Mᵀ (`rotate_stiffness`), crystal axis i along lab R[:, i].
# Si at 1550 nm (Biegelsen 1974 values, as used in the user's COMSOL g0 models). On a (001)
# wafer, rot_z(45) puts [110] along lab x; for a cubic crystal rot_z(-45) is the same tensor.
SI_P = cubic(-0.094, 0.017, -0.051)
SI_N_OPT = 3.48  # refractive index at 1550 nm, as in wavesolver.optics_pwe benchmarks


# --- Orientation derivatives (appended for wavesolver.piezo sensitivities) -----------------------


def x_cut_derivative(angle_deg: float) -> np.ndarray:
    """d x_cut(angle) / d angle, per degree."""
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    return np.column_stack([[0.0, 0.0, 0.0], [c, s, 0.0], [-s, c, 0.0]]) * (np.pi / 180)


def bond_derivative(R: np.ndarray, dR: np.ndarray) -> np.ndarray:
    """Derivative of `bond(R)` along dR (product rule on its bilinear entries)."""
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    dM = np.zeros((6, 6))
    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            dM[I, J] = dR[i, k] * R[j, l] + R[i, k] * dR[j, l]
            if k != l:
                dM[I, J] += dR[i, l] * R[j, k] + R[i, l] * dR[j, k]
    return dM


def rotation_derivatives(mat: Piezoelectric, R: np.ndarray, dR: np.ndarray):
    """
    Derivatives (dC, de, deps_r) of mat.rotated(R) along dR, for mat in the crystal frame:
    C' = M C Mᵀ, e' = R e Mᵀ, ε' = R ε Rᵀ with M = bond(R).
    """
    M, dM = bond(R), bond_derivative(R, dR)
    dC = dM @ mat.C @ M.T + M @ mat.C @ dM.T
    de = dR @ mat.e @ M.T + R @ mat.e @ dM.T
    deps = dR @ mat.eps_r @ R.T + R @ mat.eps_r @ dR.T
    return dC, de, deps
