"""Stage 2 -- diabatic coupled ODE (Wyslouzil 2000).

Faithful port of ``diabaticEquations.m``. Given the measured static pressure
p(z) and the *effective* area ratio A/A*(z) inferred from a dry companion run,
march the coupled conservation ODE from the throat to the nozzle exit and
recover the wet state: T(z), rho(z), condensed mass fraction g(z), and the
integrated latent heat q(z).

State vector  x = [T, rho, g, q]  (z in mm; all d/dz are per mm).

Right-hand sides (Wyslouzil et al. 2000):
  drho/dz = rho0 [ (1/g_mix)(u*/u)^2 (T0/T*) (dp/dz)/p0 - (rho/rho0) dlnA/dz ]   (Eq 11)
  dg/dz   = [ (rho0/rho)(h - (1/g_mix)(u*/u)^2 (T/T*))(dp/dz)/p0
              + (T/T0) dlnA/dz ] / [ (L/cp_mix - T wg)/T0 ]                      (Eq 17)
  dT/dz   = T0 [ (w0g - (1/g_mix)(u*/u)^2 (T/T*))(rho0/rho)(dp/dz)/p0
              + (T/T0)(dlnA/dz + wg dg/dz) ]                                     (Eq 12)
  dq/dz   = L dg/dz
with the auxiliary closures (mixture molecular-weight bookkeeping)
  u    = u* rho* / (rho A/A*)                                                    (Eq 10)
  h    = w0g - (cp0/cp_mix)(g0-1)/g0                                             (Eq 18)
and mu, w0g, wg from Wyslouzil 1994 / Wegener 5.46.

Solver selection
----------------
Joel's MATLAB uses ``ode15s`` (compareSSN_4.m line 331) with RelTol=1e-6 and
AbsTol=1e-12. ``ode15s`` is a variable-order, variable-step implicit BDF/Adams
solver that automatically handles stiff problems. The default here is ``"LSODA"``
(scipy.integrate.solve_ivp), which is the closest scipy equivalent -- it
auto-switches between Adams and BDF as the problem stiffens near onset.

Use :data:`SOLVERS` to see all available options and :func:`describe_solvers`
(imported from :mod:`ssnreduce.pipeline`) for a formatted summary.

IMPORTANT (apparatus-knowledge gap)
------------------------------------
The ODE integrates in z [mm from throat]. The rig logs probe position in
**% of stage stroke**, so a position(%) -> z(mm) calibration (stage travel per
%, throat offset) is required and is *not yet known* -- get it from Joel before
August. Here z is an explicit input, so the physics is complete and validated;
only the % -> mm map is pending. ``calibrate_z`` is the single place to wire it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Solver catalogue
# ---------------------------------------------------------------------------
SOLVERS: dict[str, tuple[str, str]] = {
    "LSODA": (
        "LSODA (default — recommended)",
        "Adams/BDF auto-switching, variable-order stiff solver. Direct equivalent "
        "of MATLAB ode15s used by Joel (compareSSN_4.m:331, RelTol=1e-6, "
        "AbsTol=1e-12). Handles the stiffness that builds near onset (rapid rise "
        "in g) without step-size collapse.",
    ),
    "BDF": (
        "BDF (Backward Differentiation Formula, order 1–5)",
        "Fully implicit multi-step. Slightly more predictable step-size control "
        "than LSODA but no automatic Adams fallback on non-stiff segments. Good "
        "alternative when LSODA produces warnings.",
    ),
    "Radau": (
        "Radau IIA (implicit Runge-Kutta, order 5)",
        "Highly accurate implicit RK. Slower per step than BDF/LSODA but more "
        "robust to discontinuities. Use for high-accuracy reference runs.",
    ),
    "RK45": (
        "RK45 (Dormand-Prince explicit, order 4(5))",
        "MATLAB ode45 equivalent. Fast on non-stiff segments but takes tiny steps "
        "when the ODE stiffens near onset. Use only for dry runs or very dilute "
        "mixtures (<1 % condensable).",
    ),
    "DOP853": (
        "DOP853 (Dormand-Prince explicit, order 8(5)3)",
        "High-order explicit. More accurate than RK45 per step but same stiffness "
        "limitation. Good for comparing with explicit results from the literature.",
    ),
    "RK23": (
        "RK23 (Bogacki-Shampine explicit, order 2(3))",
        "Low-order explicit. Fastest per step. Use only for quick debugging runs "
        "where accuracy is not critical.",
    ),
}


# ---- thermo interface the ODE needs (MixtureThermo satisfies this) ---------
class ThermoLike(Protocol):
    w_condensable: float
    @property
    def MW_carrier(self) -> float: ...
    @property
    def MW_condensable(self) -> float: ...
    def carrier_cp(self, T, P) -> NDArray: ...
    def carrier_cv(self, T, P) -> NDArray: ...
    def condensable_cp(self, T) -> NDArray: ...
    def condensable_cv(self, T) -> NDArray: ...


@dataclass
class ThroatState:
    """Throat (sonic) boundary condition for the march (from Stage 1)."""
    T: float        # T* [K]
    rho: float      # rho* [kg/m3]
    u: float        # u* [m/s]
    gamma: float    # gamma* [-]


@dataclass
class StagnationState:
    p0: float       # [Pa]
    T0: float       # [K]
    rho0: float     # [kg/m3]
    cp0: float      # [J/kg/K]
    gamma0: float


@dataclass
class DiabaticResult:
    z: NDArray          # [mm]
    T: NDArray          # [K]
    rho: NDArray        # [kg/m3]
    g: NDArray          # condensed mass fraction [-]
    q: NDArray          # integrated latent heat [J/kg]
    u: NDArray          # [m/s]
    p: NDArray          # [Pa]
    T_T0: NDArray
    onset_z: float      # z where condensation onsets (dT/dz sign change), nan if none
    success: bool
    message: str


def calibrate_z(position_pct: NDArray, mm_per_pct: float, z_throat_pct: float) -> NDArray:
    """Map rig stage position [%] to z [mm from throat].

    z = (position - z_throat_pct) * mm_per_pct

    ``mm_per_pct`` (stage travel per % of stroke) and ``z_throat_pct`` (the %
    reading at the geometric throat) are the apparatus constants to obtain from
    Joel. Placeholder until then.
    """
    return (np.asarray(position_pct, dtype=float) - z_throat_pct) * mm_per_pct


def _rhs(z, x, *, thr: ThroatState, stag: StagnationState, thermo: ThermoLike,
         A_At_f, dlnA_f, p_f, dp_f, L_of_T, cp_cond_liq, is_dry):
    T, rho, g, q = x
    if is_dry:
        g = 0.0

    A_At = float(A_At_f(z))
    dlnA = float(dlnA_f(z))
    p = float(p_f(z))
    dp_dz = float(dp_f(z))

    u = thr.u * thr.rho / (rho * A_At)               # Eq 10
    L = float(L_of_T(T))

    w0 = thermo.w_condensable
    MWk = thermo.MW_carrier
    MWc = thermo.MW_condensable
    # mean molecular weight with/without condensate (ratios -> units cancel)
    mu = MWk * MWc * (1.0 - g) / ((1.0 - w0) * MWc + (w0 - g) * MWk)
    mu0 = MWk * MWc / ((1.0 - w0) * MWc + w0 * MWk)
    w0g = mu / (mu0 * (1.0 - g))
    wg = mu / (MWc * (1.0 - g))

    def _s(v):  # robustly extract a python float from scalar-or-1-element-array
        return float(np.asarray(v).ravel()[0])

    cp_k = _s(thermo.carrier_cp(T, p))
    cv_k = _s(thermo.carrier_cv(T, p))
    cp_cv = _s(thermo.condensable_cp(T))
    cv_cv = _s(thermo.condensable_cv(T))
    cp_cl = _s(cp_cond_liq(T))

    cp_mix_v = ((1.0 - w0) * cp_k + (w0 - g) * cp_cv) / (1.0 - g)
    cv_mix_v = ((1.0 - w0) * cv_k + (w0 - g) * cv_cv) / (1.0 - g)
    gamma_mix = cp_mix_v / cv_mix_v
    cp_mix = (1.0 - w0) * cp_k + (w0 - g) * cp_cv + g * cp_cl

    h = w0g - (stag.cp0 / cp_mix) * (stag.gamma0 - 1.0) / stag.gamma0

    uu = (thr.u / u) ** 2
    p0 = stag.p0
    T0 = stag.T0

    drho = stag.rho0 * ((1.0 / gamma_mix) * uu * (T0 / thr.T) * (dp_dz / p0)
                        - (rho / stag.rho0) * dlnA)                       # Eq 11

    latent = (L / cp_mix - T * wg) / T0
    dg = ((stag.rho0 / rho) * (h - (1.0 / gamma_mix) * uu * (T / thr.T)) * (dp_dz / p0)
          + (T / T0) * dlnA) / latent                                    # Eq 17
    if is_dry:
        dg = 0.0

    dT = T0 * ((w0g - (1.0 / gamma_mix) * uu * (T / thr.T)) * (stag.rho0 / rho) * (dp_dz / p0)
               + (T / T0) * (dlnA + wg * dg))                            # Eq 12
    dq = L * dg
    return [dT, drho, dg, dq]


def solve_diabatic(
    z: NDArray,
    A_At: NDArray,
    p: NDArray,
    throat: ThroatState,
    stagnation: StagnationState,
    thermo: ThermoLike,
    L_of_T: Callable[[float], float] = lambda T: 5.9e5,    # CO2 sublimation ~591 kJ/kg
    cp_cond_liq: Callable[[float], float] = lambda T: 1200.0,
    is_dry: bool = False,
    rtol: float = 1e-6,
    atol: float = 1e-12,
    method: str = "LSODA",
    smooth_p: bool = False,
    smooth_window: int = 5,
    smooth_order: int = 2,
) -> DiabaticResult:
    """March the coupled ODE over a z-grid given A/A*(z) and p(z).

    Parameters
    ----------
    z            : monotone z grid [mm], starting at the throat (z[0] ~ 0).
                   Joel uses 0.1 mm spacing (``data.analysisIncrement = 0.1``).
    A_At         : effective area ratio at each z (from the dry companion run).
    p            : measured static pressure at each z [Pa].
    throat       : sonic boundary condition (Stage 1).
    stagnation   : stagnation scalars (Stage 1).
    thermo       : a MixtureThermo (or any ThermoLike).
    L_of_T       : condensable latent heat [J/kg]. Default constant CO2 L_sub.
    cp_cond_liq  : condensed-phase cp [J/kg/K] (only enters once g>0).
    is_dry       : force g==0 (dry isentrope check / pre-onset branch).
    rtol         : relative ODE tolerance (Joel: RelTol=1e-6).
    atol         : absolute ODE tolerance (Joel: AbsTol=1e-12).
    method       : ODE solver. One of :data:`SOLVERS`. Default ``"LSODA"``
                   (auto-stiff, equivalent to Joel's ode15s). See
                   :data:`SOLVERS` for the full list with guidance.
    smooth_p     : apply Savitzky-Golay smoothing to p(z) before integration.
                   Mirrors Joel's ``applySmoothing`` path in compareSSN_4.m
                   and the ``smoothData`` flag in analyseSSN_mix5.m. Useful
                   when the pressure trace is noisy; keeps the trend smooth
                   while dp/dz doesn't oscillate.
    smooth_window: SG filter window length (must be odd, >= smooth_order+2).
                   Joel uses 5-point windows (``smooth(..., 5, 'sgolay', 2)``).
    smooth_order : SG polynomial order. Joel uses order 2.
    """
    z = np.asarray(z, dtype=float)
    A_At = np.asarray(A_At, dtype=float)
    p = np.asarray(p, dtype=float)

    if smooth_p and len(p) >= smooth_window:
        p = savgol_filter(p, smooth_window, smooth_order)

    dlnA = np.gradient(np.log(A_At), z)
    dp = np.gradient(p, z)
    A_At_f = interp1d(z, A_At, kind="linear", fill_value="extrapolate")
    dlnA_f = interp1d(z, dlnA, kind="linear", fill_value="extrapolate")
    p_f = interp1d(z, p, kind="linear", fill_value="extrapolate")
    dp_f = interp1d(z, dp, kind="linear", fill_value="extrapolate")

    if method not in SOLVERS:
        raise ValueError(f"Unknown solver {method!r}. Choose from: {list(SOLVERS)}")

    x0 = [throat.T, throat.rho, 0.0, 0.0]
    sol = solve_ivp(
        lambda zz, xx: _rhs(zz, xx, thr=throat, stag=stagnation, thermo=thermo,
                            A_At_f=A_At_f, dlnA_f=dlnA_f, p_f=p_f, dp_f=dp_f,
                            L_of_T=L_of_T, cp_cond_liq=cp_cond_liq, is_dry=is_dry),
        (z[0], z[-1]), x0, t_eval=z, method=method, rtol=rtol, atol=atol,
    )
    T = sol.y[0]; rho = sol.y[1]; g = sol.y[2]; q = sol.y[3]
    zz = sol.t
    u = throat.u * throat.rho / (rho * A_At_f(zz))
    pp = p_f(zz)

    # onset: first z where dT/dz turns from falling to rising (latent reheat)
    onset_z = float("nan")
    if not is_dry and len(T) > 3:
        dT = np.diff(T)
        idx = np.where((dT[:-1] <= 0) & (dT[1:] > 0))[0]
        if idx.size:
            onset_z = float(zz[idx[0] + 1])

    return DiabaticResult(
        z=zz, T=T, rho=rho, g=g, q=q, u=u, p=pp, T_T0=T / stagnation.T0,
        onset_z=onset_z, success=bool(sol.success), message=str(sol.message),
    )


__all__ = [
    "SOLVERS",
    "ThroatState", "StagnationState", "DiabaticResult",
    "calibrate_z", "solve_diabatic",
]
