"""Stage 4 -- droplet growth (Joel's ``analyseDropletGrowth``, Gyarmathy 1963).

Faithful port of the single-droplet growth law in ``compareSSN_4.m`` (the
``dropletGrowthResidual_Gyarmathy1963`` residual, lines 1360-1376, and the
mean-free-path / binary-diffusivity setup, lines 1060-1098).

Model (Gyarmathy 1963)
----------------------
The droplet surface sits at temperature ``Td`` > gas ``T`` (latent heat of
condensation warms it). ``Td`` is found by balancing the mass-transfer and
heat-transfer growth rates:

  dr/dt|mass = (p_v - p_e^surf) / (r + 1.59 lambda) * D / (rho_d R_d T) * p/(p - p_v)
  dr/dt|heat = (Td - T)        / (r + 1.59 lambda) * k_carrier / (rho_d L)
  solve  dr/dt|mass = dr/dt|heat  for Td,  then  dr/dt = dr/dt|heat(Td)

with
  lambda  = k_B T / (sqrt(2) dm^2 pi p)                Peters & Paikert 1989
  D       = Chapman-Enskog binary diffusivity (condensable in carrier)
  p_e^surf= p_e(Td) exp(2 sigma v_m /(r k_B Td))       Ostwald-Freundlich (Kelvin)
  R_d     = R_u / MW_condensable                       specific gas constant
  Kn      = lambda / (2 r)

Properties at the droplet surface (rho_d, v_m, sigma, p_e, L) are taken at Td on
the condensable's ``phase`` branch (default "liquid", matching Joel).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

import CoolProp.CoolProp as CP

from .constants import K_B, R_U
from .gases import Species, Phase

_CP_NAME = {"argon": "Argon", "nitrogen": "Nitrogen", "co2": "CarbonDioxide", "water": "Water"}


def mean_free_path(T: float, p: float, dm: float) -> float:
    """Peters & Paikert 1989 mean free path [m] (dm = mean kinetic diameter)."""
    return K_B * T / (np.sqrt(2.0) * dm**2 * np.pi * p)


def binary_diffusivity(T: float, p: float, cond: Species, carrier: Species) -> float:
    """Chapman-Enskog binary mass diffusivity of condensable in carrier [m2/s].

    Uses MW in g/mol and LJ sigma in Angstrom (Joel's exact expression, line 1077).
    """
    eps_mix = np.sqrt(cond.c.epsilon_k * carrier.c.epsilon_k)     # K
    T_star = T / eps_mix
    omega_D = (1.0 / T_star**0.145) + (1.0 / (T_star + 0.5) ** 2)
    MW_c = cond.c.MW * 1e3       # g/mol
    MW_k = carrier.c.MW * 1e3
    sig_c = cond.c.sigma_LJ * 1e10   # Angstrom
    sig_k = carrier.c.sigma_LJ * 1e10
    return (0.001858e-4 * np.sqrt(T**3 * (1.0 / MW_c + 1.0 / MW_k))
            / ((p / 101325.0) * (0.5 * (sig_c + sig_k)) ** 2 * omega_D))


def _carrier_conductivity(carrier_name: str, T: float, p: float) -> float:
    """Carrier thermal conductivity [W/m/K], CoolProp clamped to triple T."""
    fluid = _CP_NAME[carrier_name]
    Tt = CP.PropsSI("Ttriple", fluid)
    return CP.PropsSI("conductivity", "T", max(T, Tt + 1e-3), "P", p, fluid)


@dataclass
class GrowthPoint:
    dr_dt: float    # [m/s]
    Td: float       # droplet surface temperature [K]
    Kn: float       # Knudsen number [-]
    mfp: float      # mean free path [m]
    D: float        # binary diffusivity [m2/s]


def _gyarmathy_residual(Td, *, r, T, p, p_v, cond, carrier, k_carrier, mfp,
                        R_d, phase, correct_OF):
    rho_d = float(cond.rho_cond(Td, phase)[0])
    vm = cond.c.m_molec / rho_d
    sigma = float(cond.sigma(Td, phase)[0])
    pe = float(cond.p_eq(Td, phase)[0])
    if correct_OF:
        r_crit = 2.0 * sigma * vm / (K_B * Td)
        pe = pe * np.exp(r_crit / r)
    L = float(cond.L(Td, phase)[0])
    denom = r + 1.59 * mfp
    dr_mass = ((p_v - pe) / denom) * (binary_diffusivity(T, p, cond, carrier)
                                      / (rho_d * R_d * T)) * (p / (p - p_v))
    dr_heat = ((Td - T) / denom) * (k_carrier / (rho_d * L))
    return dr_mass - dr_heat


def droplet_growth_rate(
    r: float,
    T: float,
    p: float,
    p_v: float,
    cond: Species,
    carrier: Species,
    phase: Phase = "liquid",
    correct_OF: bool = True,
    dT_bracket: float = 200.0,
) -> GrowthPoint:
    """Gyarmathy 1963 growth rate for one droplet of radius r [m].

    Parameters
    ----------
    r    : droplet radius [m].
    T    : gas temperature [K]; p : total pressure [Pa]; p_v : condensable
           partial pressure [Pa].
    cond : condensable :class:`Species`; carrier : carrier :class:`Species`.
    phase: condensable phase branch.
    """
    carrier_name = carrier.name
    dm = carrier.c.sigma_LJ                      # kinetic diameter ~ LJ sigma
    mfp = mean_free_path(T, p, dm)
    D = binary_diffusivity(T, p, cond, carrier)
    k_carrier = _carrier_conductivity(carrier_name, T, p)
    R_d = R_U / cond.c.MW

    kw = dict(r=r, T=T, p=p, p_v=p_v, cond=cond, carrier=carrier,
              k_carrier=k_carrier, mfp=mfp, R_d=R_d, phase=phase, correct_OF=correct_OF)

    # bracket Td in [T, T + dT_bracket]; residual is +ve at Td=T (heat term 0),
    # decreasing through 0 as the surface warms.
    f_lo = _gyarmathy_residual(T + 1e-6, **kw)
    Td = T
    try:
        f_hi = _gyarmathy_residual(T + dT_bracket, **kw)
        if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi < 0:
            Td = brentq(lambda x: _gyarmathy_residual(x, **kw), T + 1e-6, T + dT_bracket)
    except Exception:
        Td = T
    Td = max(Td, T)

    rho_d = float(cond.rho_cond(Td, phase)[0])
    L = float(cond.L(Td, phase)[0])
    dr_dt = ((Td - T) / (r + 1.59 * mfp)) * (k_carrier / (rho_d * L))
    Kn = mfp / (2.0 * r)
    return GrowthPoint(dr_dt=float(dr_dt), Td=float(Td), Kn=float(Kn), mfp=float(mfp), D=float(D))


def grow_droplets(
    r0: ArrayLike,
    T: ArrayLike,
    p: ArrayLike,
    p_v: ArrayLike,
    cond: Species,
    carrier: Species,
    phase: Phase = "liquid",
    correct_OF: bool = True,
) -> dict[str, NDArray]:
    """Vectorised wrapper: growth rate at each (r0, T, p, p_v) point."""
    r0 = np.atleast_1d(np.asarray(r0, dtype=float))
    T = np.broadcast_to(np.asarray(T, float), r0.shape)
    p = np.broadcast_to(np.asarray(p, float), r0.shape)
    p_v = np.broadcast_to(np.asarray(p_v, float), r0.shape)
    dr_dt = np.empty_like(r0); Td = np.empty_like(r0); Kn = np.empty_like(r0)
    for i in range(r0.size):
        gp = droplet_growth_rate(float(r0[i]), float(T[i]), float(p[i]), float(p_v[i]),
                                 cond, carrier, phase=phase, correct_OF=correct_OF)
        dr_dt[i] = gp.dr_dt; Td[i] = gp.Td; Kn[i] = gp.Kn
    return {"dr_dt": dr_dt, "Td": Td, "Kn": Kn}


__all__ = [
    "GrowthPoint", "mean_free_path", "binary_diffusivity",
    "droplet_growth_rate", "grow_droplets",
]
