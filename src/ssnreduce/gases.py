"""Property database for the condensable species (Joel's ``getGases``).

This is the property layer the nucleation models need. It ports the
**validated, CoolProp-anchored** correlations from ``D:/PhD/MATLAB code/
substance_props.m`` (whose constants were generated from CoolProp 7.2.0 and
pass ``test_mknt`` 10/10) and fills in the solid/metastable-liquid branches
from ``co2ssn/eos.py`` (Span-Wagner sublimation line, dry-ice density, Halonen
2021 solid surface tension).

Phase convention (the central CO2 physics question, see knowledge/09)
---------------------------------------------------------------------
Cryogenic CO2 in the SSN sits **below its triple point (216.59 K)**, so the
thermodynamically stable condensate is **solid** (desublimation), not liquid.
Joel's MATLAB used the *liquid* (metastable supercooled) branch. To stay
faithful to his pipeline the default here is ``phase="liquid"``, but every
property also has a ``"solid"`` branch so the solid-nucleation pathway can be
run and compared. **Switching phase changes p_sat, sigma, density and latent
heat together** -- never mix branches.

All methods accept scalar or array T and return float64 arrays (Pa, N/m,
kg/m3, m3/molecule, J/kg as documented).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

import CoolProp.CoolProp as CP

from .constants import K_B, N_A, R_U

Phase = Literal["liquid", "solid"]

_CP_NAME = {"co2": "CarbonDioxide", "water": "Water", "nitrogen": "Nitrogen", "argon": "Argon"}


@dataclass(frozen=True)
class SpeciesConstants:
    """Per-species constants (ported from substance_props.m where available)."""
    name: str
    MW: float            # molar mass [kg/mol]
    Tc: float            # critical T [K]
    Pc: float            # critical p [Pa]
    omega: float         # acentric factor
    Tt: float            # triple-point T [K]
    sigma_LJ: float      # Lennard-Jones diameter [m]
    epsilon_k: float     # LJ energy / k_B [K]
    N1: int              # bulk coordination number (MKNT)
    # metastable-liquid Clausius-Clapeyron p_sat: ln p[Pa] = Acc - Bcc/T
    Acc: float
    Bcc: float
    # Guggenheim liquid surface tension: sigma = sig0 (1 - T/Tc)^mu  [N/m]
    sig0: float
    mu: float
    # Pitzer-Curl B2 scale to match CoolProp
    kB2: float

    @property
    def m_molec(self) -> float:
        """Mass of one molecule [kg]."""
        return self.MW / N_A


# Constants ported verbatim from substance_props.m (CoolProp 7.2.0 fits)
_CONST = {
    "co2": SpeciesConstants(
        name="co2", MW=44.01e-3, Tc=304.13, Pc=7.3773e6, omega=0.225, Tt=216.592,
        sigma_LJ=3.941e-10, epsilon_k=195.2, N1=12,
        Acc=22.35148, Bcc=1989.9075, sig0=0.078634, mu=1.25406, kB2=0.97432,
    ),
    "argon": SpeciesConstants(
        name="argon", MW=39.948e-3, Tc=150.69, Pc=4.863e6, omega=-0.002, Tt=83.806,
        sigma_LJ=3.405e-10, epsilon_k=119.8, N1=12,
        Acc=20.65469, Bcc=797.0253, sig0=0.037004, mu=1.25014, kB2=0.97534,
    ),
}


class Species:
    """Phase-aware thermophysical properties for one condensable species."""

    def __init__(self, name: str):
        name = name.strip().lower()
        if name not in _CONST:
            raise ValueError(f"Species {name!r} not in gases database (have {list(_CONST)})")
        self.c = _CONST[name]
        self.name = name
        self._cp_fluid = _CP_NAME.get(name)
        self._rho_liq_anchor: tuple[float, float] | None = None  # (rho@Tt, dRho/dT)

    # ---- molecular constants passthrough ----------------------------------
    @property
    def MW(self) -> float: return self.c.MW
    @property
    def m_molec(self) -> float: return self.c.m_molec
    @property
    def Tt(self) -> float: return self.c.Tt
    @property
    def Tc(self) -> float: return self.c.Tc

    # ---- equilibrium (saturation) pressure --------------------------------
    def p_eq(self, T: ArrayLike, phase: Phase = "liquid") -> NDArray:
        """Equilibrium vapour pressure over the condensed phase [Pa]."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        if phase == "solid":
            return self._p_sub(T)
        # liquid / metastable-liquid
        out = np.empty_like(T)
        for i, t in enumerate(T):
            if self._cp_fluid and self.c.Tt <= t <= self.c.Tc - 1e-2:
                out[i] = CP.PropsSI("P", "T", float(t), "Q", 0, self._cp_fluid)
            else:  # metastable extrapolation (substance_props C-C fit)
                out[i] = np.exp(self.c.Acc - self.c.Bcc / t)
        return out

    def _p_sub(self, T: NDArray) -> NDArray:
        """Span-Wagner sublimation line (CO2). Generic C-C fallback otherwise."""
        if self.name == "co2":
            Tt, p_t = 216.592, 5.1795e5
            a1, a2, a3 = -14.740846, 2.4327015, -5.3061778
            tt = 1.0 - T / Tt
            ln_pr = (Tt / T) * (a1 * tt + a2 * tt**1.9 + a3 * tt**2.9)
            return p_t * np.exp(ln_pr)
        # fallback: same C-C as liquid (no dedicated solid line)
        return np.exp(self.c.Acc - self.c.Bcc / T)

    # ---- surface tension ---------------------------------------------------
    def sigma(self, T: ArrayLike, phase: Phase = "liquid") -> NDArray:
        """Surface tension [N/m]."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        if phase == "solid":
            if self.name == "co2":
                return np.full_like(T, 0.084)  # Halonen 2021 MD, weak T-dep
            return self.c.sig0 * np.maximum(1.0 - T / self.c.Tc, 0.0) ** self.c.mu
        # liquid Guggenheim (substance_props)
        return self.c.sig0 * np.maximum(1.0 - T / self.c.Tc, 0.0) ** self.c.mu

    # ---- condensed-phase density and molecular volume ---------------------
    def rho_cond(self, T: ArrayLike, phase: Phase = "liquid") -> NDArray:
        """Condensed-phase density [kg/m3]."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        if phase == "solid":
            if self.name == "co2":
                return 1562.0 + 0.6 * (self.c.Tt - T)   # dry ice (co2ssn)
            # generic: hold at triple liquid density
            phase = "liquid"
        return self._rho_liq(T)

    def _rho_liq(self, T: NDArray) -> NDArray:
        """Saturated-liquid density [kg/m3], CoolProp above Tt, anchored
        linear metastable extrapolation below Tt."""
        out = np.empty_like(T)
        if self._rho_liq_anchor is None and self._cp_fluid:
            rho_tt = CP.PropsSI("D", "T", self.c.Tt + 1e-3, "Q", 0, self._cp_fluid)
            rho_hi = CP.PropsSI("D", "T", self.c.Tt + 5.0, "Q", 0, self._cp_fluid)
            slope = (rho_hi - rho_tt) / 5.0     # dRho/dT (negative)
            self._rho_liq_anchor = (rho_tt, slope)
        for i, t in enumerate(T):
            if self._cp_fluid and self.c.Tt <= t <= self.c.Tc - 1e-2:
                out[i] = CP.PropsSI("D", "T", float(t), "Q", 0, self._cp_fluid)
            elif self._rho_liq_anchor is not None:
                rho_tt, slope = self._rho_liq_anchor
                out[i] = rho_tt + slope * (t - self.c.Tt)  # metastable extrap
            else:
                out[i] = 1100.0
        return out

    def v_m(self, T: ArrayLike, phase: Phase = "liquid") -> NDArray:
        """Molecular volume of the condensed phase [m3/molecule]."""
        return self.c.m_molec / self.rho_cond(T, phase)

    # ---- second virial coefficient ----------------------------------------
    def B2(self, T: ArrayLike) -> NDArray:
        """Second virial coefficient [m3/molecule].

        CoolProp BVIRIAL above the triple point; Pitzer-Curl correlation
        (scaled to CoolProp via kB2) below it (substance_props strategy).
        """
        T = np.atleast_1d(np.asarray(T, dtype=float))
        out = np.empty_like(T)
        for i, t in enumerate(T):
            done = False
            if self._cp_fluid and t >= self.c.Tt:
                try:
                    ps = CP.PropsSI("P", "T", float(t), "Q", 1, self._cp_fluid)
                    out[i] = CP.PropsSI("BVIRIAL", "T", float(t), "P", 0.5 * ps, self._cp_fluid) / N_A
                    done = True
                except Exception:
                    done = False
            if not done:
                Tr = t / self.c.Tc
                B0 = 0.083 - 0.422 / Tr**1.6
                B1 = 0.139 - 0.172 / Tr**4.2
                B2molar = self.c.kB2 * (R_U * self.c.Tc / self.c.Pc) * (B0 + self.c.omega * B1)
                out[i] = B2molar / N_A
        return out

    # ---- saturated-vapour number density ----------------------------------
    def n_sat(self, T: ArrayLike, phase: Phase = "liquid") -> NDArray:
        """Number density of the saturated vapour [1/m3].

        From p_eq via the virial EOS p = n k T (1 + B2 n), solved for n
        (leading term is the ideal-gas n = p_eq/kT). Joel uses the virial-
        corrected saturated-vapour density.
        """
        T = np.atleast_1d(np.asarray(T, dtype=float))
        pe = self.p_eq(T, phase)
        B2 = self.B2(T)
        n_ideal = pe / (K_B * T)
        # solve B2 n^2 + n - n_ideal = 0 for the physical (small) root
        disc = 1.0 + 4.0 * B2 * n_ideal
        disc = np.clip(disc, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            n = np.where(np.abs(B2) > 0, (-1.0 + np.sqrt(disc)) / (2.0 * B2), n_ideal)
        return np.where(np.isfinite(n) & (n > 0), n, n_ideal)

    # ---- latent heat -------------------------------------------------------
    def L(self, T: ArrayLike, phase: Phase = "liquid") -> NDArray:
        """Latent heat [J/kg]: sublimation (solid) or vaporisation (liquid)."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        if phase == "solid":
            if self.name == "co2":
                return 5.9e5 + (self.c.Tt - T) * 200.0   # co2ssn L_sub
            return np.full_like(T, 5.9e5)
        # liquid: CoolProp h_vap above Tt, else hold triple value
        out = np.empty_like(T)
        L_tt = None
        for i, t in enumerate(T):
            if self._cp_fluid and self.c.Tt <= t < self.c.Tc - 1e-2:
                hv = CP.PropsSI("H", "T", float(t), "Q", 1, self._cp_fluid)
                hl = CP.PropsSI("H", "T", float(t), "Q", 0, self._cp_fluid)
                out[i] = hv - hl
            else:
                if L_tt is None and self._cp_fluid:
                    hv = CP.PropsSI("H", "T", self.c.Tt + 1e-2, "Q", 1, self._cp_fluid)
                    hl = CP.PropsSI("H", "T", self.c.Tt + 1e-2, "Q", 0, self._cp_fluid)
                    L_tt = hv - hl
                out[i] = L_tt if L_tt is not None else 3.5e5
        return out


def species(name: str) -> Species:
    """Factory: ``species("co2")``."""
    return Species(name)


__all__ = ["Species", "SpeciesConstants", "species", "Phase"]
