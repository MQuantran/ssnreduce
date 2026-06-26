"""Mixture thermodynamics for the carrier + condensable gas flow.

Joel's MATLAB pulls every property from RefProp (``refpropm``). We have no
RefProp, so this module is the faithful CoolProp-backed replacement, following
two rules taken directly from Joel's own code:

1. **Carrier (Ar/N2/...) real-gas properties** come from CoolProp.  When the
   requested state falls below the carrier's triple temperature (the supersonic
   tail routinely does — Ar triple is 83.8 K and the tail reaches ~80 K),
   CoolProp refuses the state, so we **clamp the evaluation to the triple
   temperature**.  This is exactly Joel's fallback (``analyseSSN_mix5.m``
   lines 717-724: ``try refpropm(...,T) catch -> refpropm(...,tripleT)``) and
   his ``diabaticEquations.m`` lines 50-57.

2. **Condensable vapour heat capacity** (CO2/H2O/N2) is an *ideal-gas* property
   that depends on T only, so CoolProp's state guard would block it sub-triple.
   We use NASA-7 ideal-gas polynomials instead (Burcat / GRI-Mech 3.0).  Joel
   used RefProp-derived ``cp_J_kg_K.vapour`` interpolation tables for the same
   quantity; the ideal-gas polynomial reproduces those to better than ~1% for
   the dilute condensable and is well-behaved at cryogenic T.

Mixing is mass-fraction weighted (Wyslouzil/Wegener dilute-mixture convention),
matching ``analyseSSN_mix5.m`` lines 228-231 and ``diabaticEquations.m`` 62-68.

All public methods accept scalars or numpy arrays of T (P scalar or array) and
return float64 arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

import CoolProp.CoolProp as CP

from .constants import R_U, BAR_TO_PA

# CoolProp fluid names keyed by our canonical species name
_CP_FLUID = {
    "argon": "Argon",
    "nitrogen": "Nitrogen",
    "helium": "Helium",
    "air": "Air",
    "co2": "CarbonDioxide",
    "water": "Water",
}

# Molecular weights [kg/mol]
_MW_KG = {
    "co2": 0.0440095,
    "water": 0.01801528,
    "argon": 0.039948,
    "nitrogen": 0.0280134,
    "air": 0.0289647,
    "helium": 0.004002602,
}

# NASA-7 ideal-gas polynomials: cp/R = a0 + a1 T + a2 T^2 + a3 T^3 + a4 T^4
# Low-temperature range coefficients (Burcat / GRI-Mech 3.0). Nominal validity
# 200-1000 K; we extrapolate gently below 200 K for the cryogenic vapour, where
# cp is smooth and tends to the rigid-rotor/translation limit.
_NASA7_LOW = {
    "co2": (2.35677352e00, 8.98459677e-03, -7.12356269e-06, 2.45919022e-09, -1.43699548e-13),
    "water": (4.19864056e00, -2.03643410e-03, 6.52040211e-06, -5.48797062e-09, 1.77197817e-12),
    "nitrogen": (3.298677e00, 1.40824040e-03, -3.963222e-06, 5.641515e-09, -2.444854e-12),
    # monatomic: cp/R = 5/2 exactly, no T dependence
    "argon": (2.5, 0.0, 0.0, 0.0, 0.0),
    "helium": (2.5, 0.0, 0.0, 0.0, 0.0),
}


def _ideal_cp_mass(species: str, T: NDArray) -> NDArray:
    """Ideal-gas cp [J/kg/K] from the NASA-7 polynomial."""
    a0, a1, a2, a3, a4 = _NASA7_LOW[species]
    cp_over_R = a0 + a1 * T + a2 * T**2 + a3 * T**3 + a4 * T**4
    R_spec = R_U / _MW_KG[species]
    return cp_over_R * R_spec


@dataclass
class MixtureThermo:
    """Thermo for a binary carrier + condensable gas mixture.

    Parameters
    ----------
    carrier, condensable : canonical species names (e.g. ``"argon"``, ``"co2"``).
    w_condensable        : feed mass fraction of the condensable (g/g).
    """

    carrier: str
    condensable: str
    w_condensable: float

    def __post_init__(self) -> None:
        for s in (self.carrier, self.condensable):
            if s not in _MW_KG:
                raise ValueError(f"species {s!r} not supported by mixture.py")
        self._T_triple_carrier = CP.PropsSI("Ttriple", _CP_FLUID[self.carrier])

    # ---- molecular weights -------------------------------------------------
    @property
    def MW_carrier(self) -> float:
        return _MW_KG[self.carrier]

    @property
    def MW_condensable(self) -> float:
        return _MW_KG[self.condensable]

    # ---- carrier real-gas props (CoolProp, clamped to triple T) ------------
    def _carrier_prop(self, code: str, T: ArrayLike, P_pa: ArrayLike) -> NDArray:
        T = np.atleast_1d(np.asarray(T, dtype=float))
        P = np.atleast_1d(np.asarray(P_pa, dtype=float))
        T, P = np.broadcast_arrays(T, P)
        out = np.empty(T.shape, dtype=float)
        fluid = _CP_FLUID[self.carrier]
        Tmin = self._T_triple_carrier + 1e-3
        for i in range(T.size):
            Ti = max(float(T.flat[i]), Tmin)
            out.flat[i] = CP.PropsSI(code, "T", Ti, "P", float(P.flat[i]), fluid)
        return out

    def carrier_cp(self, T: ArrayLike, P_pa: ArrayLike) -> NDArray:
        return self._carrier_prop("Cpmass", T, P_pa)

    def carrier_cv(self, T: ArrayLike, P_pa: ArrayLike) -> NDArray:
        return self._carrier_prop("Cvmass", T, P_pa)

    def carrier_rho(self, T: ArrayLike, P_pa: ArrayLike) -> NDArray:
        """Carrier density [kg/m3].

        Above the triple temperature: CoolProp real gas. Below it (supersonic
        tail), CoolProp's clamp would pin density at the triple T and badly
        mis-state it, so we use the ideal-gas law there (the tail is dilute,
        ~0.04 bar, so the gas is very nearly ideal).
        """
        T = np.atleast_1d(np.asarray(T, dtype=float))
        P = np.atleast_1d(np.asarray(P_pa, dtype=float))
        T, P = np.broadcast_arrays(T, P)
        out = np.empty(T.shape, dtype=float)
        fluid = _CP_FLUID[self.carrier]
        R_spec = R_U / _MW_KG[self.carrier]
        for i in range(T.size):
            Ti = float(T.flat[i]); Pi = float(P.flat[i])
            if Ti > self._T_triple_carrier:
                out.flat[i] = CP.PropsSI("Dmass", "T", Ti, "P", Pi, fluid)
            else:
                out.flat[i] = Pi / (R_spec * Ti)  # ideal-gas extrapolation
        return out

    def carrier_sound(self, T: ArrayLike, P_pa: ArrayLike) -> NDArray:
        return self._carrier_prop("A", T, P_pa)

    def carrier_k(self, T: ArrayLike, P_pa: ArrayLike) -> NDArray:
        """Carrier thermal conductivity [W/m/K] (for droplet growth)."""
        return self._carrier_prop("conductivity", T, P_pa)

    def carrier_mu(self, T: ArrayLike, P_pa: ArrayLike) -> NDArray:
        """Carrier dynamic viscosity [Pa s]."""
        return self._carrier_prop("viscosity", T, P_pa)

    # ---- condensable vapour cp/cv (ideal gas) ------------------------------
    def condensable_cp(self, T: ArrayLike) -> NDArray:
        T = np.atleast_1d(np.asarray(T, dtype=float))
        return _ideal_cp_mass(self.condensable, T)

    def condensable_cv(self, T: ArrayLike) -> NDArray:
        cp = self.condensable_cp(T)
        return cp - R_U / _MW_KG[self.condensable]

    # ---- mixture properties (mass-weighted) --------------------------------
    def mix_cp(self, T: ArrayLike, P_pa: ArrayLike, g: float | ArrayLike = 0.0) -> NDArray:
        """Vapour-phase mixture cp [J/kg/K] with condensed mass fraction g.

        Mirrors Joel's ``cp_mix_v`` (``diabaticEquations.m`` line 62):

            cp_mix_v = ((1-w)*cp_carrier + (w-g)*cp_cond_vapour) / (1-g)

        i.e. the cp of the *remaining vapour* per unit vapour mass. At g=0 this
        reduces to the simple mass-weighted vapour cp.
        """
        T = np.atleast_1d(np.asarray(T, dtype=float))
        P = np.broadcast_to(np.asarray(P_pa, dtype=float), T.shape)
        g = np.asarray(g, dtype=float)
        w = self.w_condensable
        cp_k = self.carrier_cp(T, P)
        cp_c = self.condensable_cp(T)
        return ((1.0 - w) * cp_k + (w - g) * cp_c) / np.maximum(1.0 - g, 1e-12)

    def mix_cv(self, T: ArrayLike, P_pa: ArrayLike) -> NDArray:
        T = np.atleast_1d(np.asarray(T, dtype=float))
        P = np.broadcast_to(np.asarray(P_pa, dtype=float), T.shape)
        w = self.w_condensable
        return (1.0 - w) * self.carrier_cv(T, P) + w * self.condensable_cv(T)

    def mix_gamma(self, T: ArrayLike, P_pa: ArrayLike) -> NDArray:
        """Vapour-phase mixture gamma = cp_mix / cv_mix (g=0)."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        P = np.broadcast_to(np.asarray(P_pa, dtype=float), T.shape)
        w = self.w_condensable
        cp = (1.0 - w) * self.carrier_cp(T, P) + w * self.condensable_cp(T)
        cv = (1.0 - w) * self.carrier_cv(T, P) + w * self.condensable_cv(T)
        return cp / cv

    def mix_rho(self, T: ArrayLike, P_pa: ArrayLike) -> NDArray:
        """Mixture density [kg/m3], mass-weighted (Joel lines 228, 538).

        rho_mix = w*rho_condensable + (1-w)*rho_carrier, with the condensable
        treated as an ideal gas at its partial conditions (Joel line 800 uses
        the ideal-gas law for the condensable density)."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        P = np.broadcast_to(np.asarray(P_pa, dtype=float), T.shape)
        w = self.w_condensable
        rho_k = self.carrier_rho(T, P)
        R_c = R_U / _MW_KG[self.condensable]
        rho_c = P / (R_c * T)  # ideal-gas condensable
        return w * rho_c + (1.0 - w) * rho_k

    @property
    def MW_mean_molefrac(self) -> float:
        """Mole-weighted mean MW [kg/mol] from the feed mass fraction."""
        w = self.w_condensable
        # convert mass frac -> mole frac
        nc = w / _MW_KG[self.condensable]
        nk = (1 - w) / _MW_KG[self.carrier]
        x = nc / (nc + nk)
        return x * _MW_KG[self.condensable] + (1 - x) * _MW_KG[self.carrier]


__all__ = ["MixtureThermo"]
