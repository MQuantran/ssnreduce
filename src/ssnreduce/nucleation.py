"""Stage 3 -- nucleation rates (Joel's ``analyseNucleation``).

Faithful port of the nucleation block of ``compareSSN_4.m`` (lines 677-899):
classical nucleation theory and its standard corrections, plus the full
mean-field kinetic nucleation theory (MKNT, Kalikmanov 2006 / Bhabhe 2012).

Given the temperature T(t) and the condensable partial pressure p_v(t) along a
streamline (from Stage 2), compute at each point:

CNT family
  S        = p_v / p_eq(T)                                supersaturation
  J_CNT    = sqrt(2 sigma/(pi m)) v_m (p_v/kT)^2
             exp( -16 pi v_m^2 sigma^3 / (3 (kT)^3 (ln S)^2) )   (Bhabhe Eq 2.17)
  J_C      = J_CNT / S                                    Courtney (Eq 2.18)
  J_GC     = J_C exp(theta),  theta = sigma s1 / kT        Girshick-Chiu (Eq 2.19)
  J_Wolk   = J_CNT exp(-27.56 + 6500/T)                   Wolk-Strey (H2O)
  J_Hale   = 1e32 exp(-W/kT),
             W/kT = (16 pi/3) Omega^3 (T_c/T - 1)^3 / (ln S)^2,  Omega = 1.44
  n*_CNT   = 32 pi v_m^2 sigma^3 / (3 (kT ln S)^3),  r*_CNT, r_MosesStein

MKNT (Kalikmanov 2006 / Bhabhe 2012)
  hard-sphere diameter from the LJ integral (Eq 2.27); packing fraction eta
  (Eq 2.28); coordination number N1 (Eq 2.29); theta_macro, theta_micro
  (Eq 2.24); omega, lambda (2.32-2.33); q (2.36); pseudospinodal S_psp (2.35);
  per-cluster surface count via the core-radius cubic (Eq 2.28 of Kalikmanov
  2006); formation energy -H(n) (Eq 2.34); critical size and J_MKNT (2.21-2.22).

``phase`` selects the condensable branch (default "liquid", matching Joel; pass
"solid" for the desublimation pathway -- see :mod:`ssnreduce.gases`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

from .constants import K_B
from .gases import Species, Phase

_OMEGA_HALE = 1.44


@dataclass
class NucleationResult:
    T: NDArray
    p_v: NDArray
    S: NDArray
    pe: NDArray
    sigma: NDArray
    v_m: NDArray
    # CNT family (all 1/m3/s)
    J_CNT: NDArray
    J_C: NDArray
    J_GC: NDArray
    J_Wolk: NDArray
    J_Hale: NDArray
    n_star_CNT: NDArray
    r_star_CNT: NDArray
    r_MosesStein: NDArray
    # MKNT
    J_MKNT: NDArray
    n_star_MKNT: NDArray
    r_star_MKNT: NDArray
    theta_macro: NDArray
    theta_micro: NDArray
    omega: NDArray
    lam: NDArray
    qcoeff: NDArray
    S_psp: NDArray
    dHS: NDArray
    eta: NDArray
    N1: NDArray
    phase: str = "liquid"
    meta: dict = field(default_factory=dict)


def _hard_sphere_diameter(sp: Species, T: float) -> float:
    r"""LJ hard-sphere diameter d_HS (Bhabhe Eq 2.27), SI [m].

    d_HS = ( 3 integral_0^rm  r^2 [1 - exp(-beta (u_LJ(r) + eps_k k_B))] dr )^(1/3)
    with rm = sigma_LJ 2^(1/6),  u_LJ = 4 eps [(sigma/r)^12 - (sigma/r)^6].
    """
    sLJ = sp.c.sigma_LJ
    eps = sp.c.epsilon_k * K_B          # J
    beta = 1.0 / (K_B * T)
    rm = sLJ * 2.0 ** (1.0 / 6.0)

    def integrand(r: float) -> float:
        if r <= 0.0:
            return 0.0
        u = 4.0 * eps * ((sLJ / r) ** 12 - (sLJ / r) ** 6)
        return r * r * (1.0 - np.exp(-beta * (u + eps)))

    val, _ = quad(integrand, 0.0, rm, limit=200)
    return (3.0 * val) ** (1.0 / 3.0)


def _core_radius_cubed(n: int, omega: float, lam: float) -> float:
    """Positive real root cubed of the dimensionless core-radius cubic
    x^3 + 3 omega x^2 + 3 omega lambda x - (n - omega lambda^2) = 0
    (Kalikmanov 2006 Eq 28; Joel solves with vpasolve). Returns x^3."""
    coeffs = [1.0, 3.0 * omega, 3.0 * omega * lam, -(n - omega * lam**2)]
    roots = np.roots(coeffs)
    real = roots[np.abs(roots.imag) < 1e-9].real
    real = real[real >= 0.0]
    if real.size == 0:
        return 0.0
    x = float(real.max())
    return x**3


def _mknt_point(sp: Species, T: float, S: float, pe: float, sigma: float,
                vm: float, theta_macro: float, n_max: int, phase: Phase):
    """MKNT at one (T,S). Returns dict of scalars, or zeros if S<=1."""
    m = sp.c.m_molec
    B2 = float(sp.B2(T)[0])
    n_sat = float(sp.n_sat(T, phase)[0])

    dHS = _hard_sphere_diameter(sp, T)
    eta = (np.pi / 6.0) * dHS**3 / vm
    N1 = 5.5116 * eta**2 + 6.1383 * eta + 1.275
    theta_micro = -np.log(-(B2 * pe) / (K_B * T))
    omega = (1.0 / 3.0) * theta_macro / theta_micro
    lam = np.sqrt(N1 / omega - 3.0 / 4.0) - 3.0 / 2.0
    qcoeff = 1.0 + 2.0 * omega + omega * lam
    S_psp = np.exp(theta_micro * (1.0 - 1.0 / qcoeff))

    out = dict(dHS=dHS, eta=eta, N1=N1, theta_micro=theta_micro, omega=omega,
               lam=lam, qcoeff=qcoeff, S_psp=S_psp, J_MKNT=0.0, n_star=0.0, r_star=0.0)
    if not (S > 1.0):
        return out

    n_arr = np.arange(1, n_max + 1)
    N1_round = int(round(N1))
    nSurface = np.empty(n_max, dtype=float)
    for j, n in enumerate(n_arr):
        if n <= N1_round:
            nSurface[j] = n
        else:
            nSurface[j] = n - _core_radius_cubed(int(n), omega, lam)

    neg_Hn = -n_arr * np.log(S) + theta_micro * (nSurface - 1.0)   # -H(n)/kT
    n_star = int(n_arr[int(np.argmax(neg_Hn))])

    s1 = (36.0 * np.pi * vm**2) ** (1.0 / 3.0)                     # monomer area [m2]
    f_sat = (pe * s1 * n_star ** (2.0 / 3.0)) / np.sqrt(2.0 * np.pi * m * K_B * T)
    A_kin = n_sat * f_sat * S
    J_MKNT = A_kin / np.sum(np.exp(neg_Hn))
    r_star = (3.0 * n_star * vm / (4.0 * np.pi)) ** (1.0 / 3.0)

    out.update(J_MKNT=float(J_MKNT), n_star=float(n_star), r_star=float(r_star))
    return out


def nucleation_rates(
    T: ArrayLike,
    p_v: ArrayLike,
    sp: Species,
    phase: Phase = "liquid",
    mknt: bool = True,
    n_max: int = 200,
) -> NucleationResult:
    """Compute all J variants along a streamline.

    Parameters
    ----------
    T    : temperature array [K].
    p_v  : condensable partial pressure array [Pa] (same shape).
    sp   : the condensable :class:`ssnreduce.gases.Species`.
    phase: "liquid" (Joel default, metastable) or "solid" (desublimation).
    mknt : also compute MKNT (slower: a cubic per cluster size per point).
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    p_v = np.broadcast_to(np.asarray(p_v, dtype=float), T.shape).copy()

    pe = sp.p_eq(T, phase)
    sigma = sp.sigma(T, phase)
    vm = sp.v_m(T, phase)
    m = sp.c.m_molec
    S = p_v / pe
    lnS = np.log(S)

    # ---- CNT family (vectorised) ------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        expo = -16.0 * np.pi * vm**2 * sigma**3 / (3.0 * (K_B * T) ** 3 * lnS**2)
        J_CNT = (np.sqrt(2.0 * sigma / (np.pi * m)) * vm
                 * (p_v / (K_B * T)) ** 2 * np.exp(expo))
        J_CNT = np.where(lnS > 0, J_CNT, 0.0)
        n_star_CNT = 32.0 * np.pi * vm**2 * sigma**3 / (3.0 * (K_B * T * lnS) ** 3)
        n_star_CNT = np.where(lnS > 0, n_star_CNT, np.nan)   # undefined when subsaturated
        r_star_CNT = (3.0 * np.maximum(np.nan_to_num(n_star_CNT), 0.0) * vm / (4.0 * np.pi)) ** (1.0 / 3.0)
        r_star_CNT = np.where(lnS > 0, r_star_CNT, np.nan)
        J_C = np.where(S > 0, J_CNT / S, 0.0)
        theta_macro = sigma * (36.0 * np.pi * vm**2) ** (1.0 / 3.0) / (K_B * T)
        J_GC = J_C * np.exp(theta_macro)
        J_Wolk = J_CNT * np.exp(-27.56 + 6.5e3 / T)
        W_kT = (16.0 * np.pi / 3.0) * _OMEGA_HALE**3 * (sp.c.Tc / T - 1.0) ** 3 / lnS**2
        J_Hale = np.where(lnS > 0, 1.0e32 * np.exp(-W_kT), 0.0)
        r_MS = np.where(lnS > 0, 2.0 * sigma * vm / (K_B * T * lnS), np.inf)

    # ---- MKNT (per-point) --------------------------------------------------
    J_MKNT = np.zeros_like(T)
    n_star_MKNT = np.zeros_like(T)
    r_star_MKNT = np.zeros_like(T)
    theta_micro = np.zeros_like(T)
    omega = np.zeros_like(T)
    lam = np.zeros_like(T)
    qcoeff = np.zeros_like(T)
    S_psp = np.zeros_like(T)
    dHS = np.zeros_like(T)
    eta = np.zeros_like(T)
    N1 = np.zeros_like(T)
    if mknt:
        for i in range(T.size):
            r = _mknt_point(sp, float(T[i]), float(S[i]), float(pe[i]), float(sigma[i]),
                            float(vm[i]), float(theta_macro[i]), n_max, phase)
            J_MKNT[i] = r["J_MKNT"]; n_star_MKNT[i] = r["n_star"]; r_star_MKNT[i] = r["r_star"]
            theta_micro[i] = r["theta_micro"]; omega[i] = r["omega"]; lam[i] = r["lam"]
            qcoeff[i] = r["qcoeff"]; S_psp[i] = r["S_psp"]
            dHS[i] = r["dHS"]; eta[i] = r["eta"]; N1[i] = r["N1"]

    return NucleationResult(
        T=T, p_v=p_v, S=S, pe=pe, sigma=sigma, v_m=vm,
        J_CNT=np.nan_to_num(J_CNT), J_C=np.nan_to_num(J_C), J_GC=np.nan_to_num(J_GC),
        J_Wolk=np.nan_to_num(J_Wolk), J_Hale=np.nan_to_num(J_Hale),
        n_star_CNT=n_star_CNT, r_star_CNT=r_star_CNT, r_MosesStein=r_MS,
        J_MKNT=J_MKNT, n_star_MKNT=n_star_MKNT, r_star_MKNT=r_star_MKNT,
        theta_macro=theta_macro, theta_micro=theta_micro, omega=omega, lam=lam,
        qcoeff=qcoeff, S_psp=S_psp, dHS=dHS, eta=eta, N1=N1, phase=phase,
    )


__all__ = ["NucleationResult", "nucleation_rates"]
