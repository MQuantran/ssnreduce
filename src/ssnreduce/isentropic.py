"""Stage 1 -- isentropic reduction of the measured pressure ratio.

Faithful port of the isentropic block of ``analyseSSN_mix5.m`` (lines 344-428):
given the per-station centreline pressure ratio p/p0 and the stagnation state
(p0, T0, mixture gamma0), recover Mach number, static temperature, the local
effective area ratio A/A*, and the choked-throat properties (p*, T*, u*, G*,
rho*).

Equations (with Joel's cited sources):
  M     = sqrt( 2 [ (p/p0)^((1-g)/g) - 1 ] / (g-1) )      Wyslouzil 2000 Eq 5
  T     = T0 / (1 + (g-1)/2 M^2)                           Streletzky 2002 Eq 2
  A/A*  = (1/M) [ (2 + (g-1) M^2)/(g+1) ]^((g+1)/(2g-2))   Streletzky 2002 Eq 3
  p*/p0 = (0.5(1+g))^(-g/(g-1))                            critical-flow
  T*    = 2 T0 / (1+g)
  u*    = sqrt( g R T* / MW )
  G*    = sqrt(MW) p0 ((g+1)/2)^(-g/(g-1)) / sqrt( R T0 / (g (g+1)/2) )
  rho*  = G* / u*

As in Joel's code, gamma is first taken as the constant stagnation value
gamma0, then refined point-by-point as gamma(T,p) of the mixture and the Mach /
area-ratio recomputed (the change is small but kept for fidelity).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from .constants import R_U, BAR_TO_PA
from .mixture import MixtureThermo
from .io import RunData


@dataclass
class IsentropicResult:
    # per-station arrays (aligned with the input stations, sorted by position)
    position: NDArray        # rig stage position [%]  (z[mm] added in Stage 2)
    p_p0: NDArray            # measured centreline pressure ratio [-]
    p: NDArray               # centreline static pressure [Pa]
    M: NDArray               # Mach number (gamma0 first pass) [-]
    T: NDArray               # static temperature [K] (refined)
    T_T0: NDArray            # T/T0 [-]
    gamma: NDArray           # local mixture gamma(T,p) [-]
    A_A_throat: NDArray      # local effective area ratio [-]
    # run-level stagnation scalars
    p0: float                # [Pa]
    T0: float                # [K]
    gamma0: float            # mixture gamma at (T0,p0) [-]
    rho0: float              # mixture density at (T0,p0) [kg/m3]
    cp0: float               # mixture cp at (T0,p0) [J/kg/K]
    MW_mean: float           # mole-weighted mean MW [kg/mol]
    # choked-throat scalars
    p_throat_p0: float
    p_throat: float          # [Pa]
    T_throat: float          # [K]
    u_throat: float          # [m/s]
    G_throat: float          # [kg/m2/s]
    rho_throat: float        # [kg/m3]
    gamma_throat: float
    cp_throat: float         # [J/kg/K]
    # provenance
    mix: MixtureThermo
    smoothed: bool = False   # True if p_p0 was SG-smoothed before reduction


def _mach_from_pp0(pp0: NDArray, gamma: NDArray) -> NDArray:
    """Isentropic Mach from stagnation pressure ratio (Wyslouzil Eq 5)."""
    return np.sqrt(2.0 * (pp0 ** ((1.0 - gamma) / gamma) - 1.0) / (gamma - 1.0))


def _area_ratio(M: NDArray, gamma: NDArray) -> NDArray:
    """A/A* from Mach (Streletzky Eq 3)."""
    return (((2.0 + (gamma - 1.0) * M**2) / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * gamma - 2.0))) / M


def reduce_isentropic(
    run: RunData,
    stations,
    mix: MixtureThermo | None = None,
    refine_gamma: bool = True,
    smooth_pp0: bool = False,
    smooth_window: int = 5,
    smooth_order: int = 2,
) -> IsentropicResult:
    """Reduce per-station p/p0 to (M, T, A/A*) and throat properties.

    Parameters
    ----------
    run          : :class:`ssnreduce.io.RunData` (provides p0, T0, composition).
    stations     : DataFrame from :func:`ssnreduce.io.stations` (needs columns
                   ``position`` and ``P_centreline_bar``).
    mix          : a :class:`MixtureThermo`; built from ``run`` if omitted.
    refine_gamma : do the point-by-point gamma(T,p) refinement (Joel default).
    smooth_pp0   : apply a Savitzky-Golay filter to the p/p0 profile before
                   computing M, T and A/A*. Mirrors Joel's ``smoothData`` flag
                   in ``analyseSSN_mix5.m`` (``smooth(p_p0, 5, 'sgolay', 2)``).
                   Useful when per-station scatter is large. The raw p_p0 is
                   replaced in the result; use ``smooth_pp0=False`` (default) to
                   keep the unsmoothed values for stage-by-stage comparison.
    smooth_window: SG filter window length (odd integer, >= smooth_order+2).
                   Joel uses 5.
    smooth_order : SG polynomial order. Joel uses 2.
    """
    if mix is None:
        mix = MixtureThermo(run.carrier, run.condensable, run.w_condensable)

    p0 = run.p0_bar * BAR_TO_PA
    T0 = run.T0_K

    position = np.asarray(stations["position"], dtype=float)
    pp0 = np.asarray(stations["P_centreline_bar"], dtype=float) / run.p0_bar

    if smooth_pp0 and len(pp0) >= smooth_window:
        pp0 = savgol_filter(pp0, smooth_window, smooth_order)

    p = pp0 * p0

    # stagnation mixture properties
    gamma0 = float(mix.mix_gamma(T0, p0)[0])
    rho0 = float(mix.mix_rho(T0, p0)[0])
    cp0 = float(mix.mix_cp(T0, p0)[0])
    MW_mean = mix.MW_mean_molefrac

    # first pass with constant gamma0
    M = _mach_from_pp0(pp0, np.full_like(pp0, gamma0))
    T = T0 / (1.0 + 0.5 * (gamma0 - 1.0) * M**2)

    if refine_gamma:
        gamma = mix.mix_gamma(T, p)
        Mnew = _mach_from_pp0(pp0, gamma)
        A_A_throat = _area_ratio(M, gamma)
        T = T0 / (1.0 + 0.5 * (gamma - 1.0) * Mnew**2)
    else:
        gamma = np.full_like(pp0, gamma0)
        A_A_throat = _area_ratio(M, gamma)

    T_T0 = T / T0

    # ---- choked throat (gamma at the minimum-area station) ----------------
    idx_throat = int(np.argmin(A_A_throat))
    g_th = float(gamma[idx_throat])
    p_throat_p0 = (0.5 * (1.0 + g_th)) ** (-g_th / (g_th - 1.0))
    p_throat = p_throat_p0 * p0
    T_throat = T0 * 2.0 / (1.0 + g_th)
    u_throat = np.sqrt(g_th * R_U * T_throat / MW_mean)
    G_throat = (
        np.sqrt(MW_mean) * p0 * ((g_th + 1.0) / 2.0) ** (-g_th / (g_th - 1.0))
        / np.sqrt((R_U * T0) / (g_th * ((g_th + 1.0) / 2.0)))
    )
    rho_throat = G_throat / u_throat
    cp_throat = float(mix.mix_cp(T_throat, p_throat)[0])

    return IsentropicResult(
        position=position, p_p0=pp0, p=p, M=M, T=T, T_T0=T_T0, gamma=gamma,
        A_A_throat=A_A_throat,
        p0=p0, T0=T0, gamma0=gamma0, rho0=rho0, cp0=cp0, MW_mean=MW_mean,
        p_throat_p0=p_throat_p0, p_throat=p_throat, T_throat=T_throat,
        u_throat=float(u_throat), G_throat=float(G_throat), rho_throat=float(rho_throat),
        gamma_throat=g_th, cp_throat=cp_throat, mix=mix,
        smoothed=smooth_pp0 and len(pp0) >= smooth_window,
    )


__all__ = ["IsentropicResult", "reduce_isentropic"]
