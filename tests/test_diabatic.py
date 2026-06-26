"""Stage 2 faithfulness test: the dry ODE must reproduce the analytic isentrope.

We build an exact constant-gamma supersonic nozzle (M(z) prescribed, A/A*(z) and
p(z) from the isentropic relations) and integrate the diabatic ODE with g forced
to zero. The marched T(z), rho(z) must match the algebraic isentropic solution
to ODE tolerance -- this validates Eqs 10/11/12 and the velocity closure
independent of any real-data calibration.
"""
import numpy as np
import pytest

from ssnreduce.constants import R_U
from ssnreduce.diabatic import ThroatState, StagnationState, solve_diabatic


class ConstGammaThermo:
    """Pure-carrier, constant-cp/cv ideal gas (gamma fixed)."""
    def __init__(self, gamma, MW):
        self.gamma = gamma
        self._MW = MW
        Rs = R_U / MW
        self._cp = gamma / (gamma - 1.0) * Rs
        self._cv = Rs / (gamma - 1.0)
        self.w_condensable = 0.0

    @property
    def MW_carrier(self): return self._MW
    @property
    def MW_condensable(self): return 0.044  # unused (w=0)

    def carrier_cp(self, T, P): return np.full_like(np.atleast_1d(T), self._cp, dtype=float)
    def carrier_cv(self, T, P): return np.full_like(np.atleast_1d(T), self._cv, dtype=float)
    def condensable_cp(self, T): return np.full_like(np.atleast_1d(T), self._cp, dtype=float)
    def condensable_cv(self, T): return np.full_like(np.atleast_1d(T), self._cv, dtype=float)


def _build_nozzle(gamma, MW, p0, T0, L=50.0, n=201, Mmax=2.5):
    z = np.linspace(0.0, L, n)
    M = 1.0 + (Mmax - 1.0) * (z / L)
    fac = 1.0 + 0.5 * (gamma - 1.0) * M**2
    A_At = (1.0 / M) * ((2.0 / (gamma + 1.0)) * fac) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    p = p0 * fac ** (-gamma / (gamma - 1.0))
    T = T0 / fac
    rho0 = p0 / ((R_U / MW) * T0)
    rho = rho0 * fac ** (-1.0 / (gamma - 1.0))
    return z, M, A_At, p, T, rho, rho0


def test_dry_ode_recovers_isentrope():
    gamma, MW, p0, T0 = 1.4, 0.039948, 1.0e5, 300.0
    z, M, A_At, p, T_an, rho_an, rho0 = _build_nozzle(gamma, MW, p0, T0)

    Tstar = T0 * 2.0 / (gamma + 1.0)
    rhostar = rho0 * (2.0 / (gamma + 1.0)) ** (1.0 / (gamma - 1.0))
    ustar = np.sqrt(gamma * R_U * Tstar / MW)
    thr = ThroatState(T=Tstar, rho=rhostar, u=ustar, gamma=gamma)
    stag = StagnationState(p0=p0, T0=T0, rho0=rho0,
                           cp0=gamma / (gamma - 1.0) * R_U / MW, gamma0=gamma)
    thermo = ConstGammaThermo(gamma, MW)

    res = solve_diabatic(z, A_At, p, thr, stag, thermo, is_dry=True)
    assert res.success
    # compare away from the throat singularity (first ~2 mm)
    m = res.z > 2.0
    assert np.allclose(res.T[m], T_an[m], rtol=3e-3)
    assert np.allclose(res.rho[m], rho_an[m], rtol=3e-3)
    assert np.all(res.g == 0.0)


def test_dry_ode_velocity_closure():
    """u = u* rho*/(rho A/A*) should match isentropic u = M a."""
    gamma, MW, p0, T0 = 1.667, 0.039948, 1.0e5, 300.0
    z, M, A_At, p, T_an, rho_an, rho0 = _build_nozzle(gamma, MW, p0, T0)
    Tstar = T0 * 2.0 / (gamma + 1.0)
    rhostar = rho0 * (2.0 / (gamma + 1.0)) ** (1.0 / (gamma - 1.0))
    ustar = np.sqrt(gamma * R_U * Tstar / MW)
    thr = ThroatState(T=Tstar, rho=rhostar, u=ustar, gamma=gamma)
    stag = StagnationState(p0=p0, T0=T0, rho0=rho0,
                           cp0=gamma / (gamma - 1.0) * R_U / MW, gamma0=gamma)
    res = solve_diabatic(z, A_At, p, thr, stag, ConstGammaThermo(gamma, MW), is_dry=True)
    a = np.sqrt(gamma * (R_U / MW) * T_an)
    u_is = M * a
    m = res.z > 2.0
    assert np.allclose(res.u[m], u_is[m], rtol=4e-3)
