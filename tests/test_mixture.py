import numpy as np
import pytest

from ssnreduce.mixture import MixtureThermo
from ssnreduce.constants import R_U


def test_argon_gamma_near_monatomic():
    mix = MixtureThermo("argon", "co2", w_condensable=0.0)
    g = mix.mix_gamma(215.0, 1e5)
    assert 1.64 < float(g[0]) < 1.68  # ~5/3 for nearly-ideal Ar


def test_clamp_below_carrier_triple():
    """Below Ar triple (83.8 K) the carrier cp must stay finite (clamped)."""
    mix = MixtureThermo("argon", "co2", 0.04)
    cp_cold = mix.carrier_cp(70.0, 4000.0)   # below triple -> clamped
    cp_trip = mix.carrier_cp(85.0, 4000.0)
    assert np.isfinite(cp_cold).all()
    assert cp_cold[0] == pytest.approx(mix.carrier_cp(83.9, 4000.0)[0], rel=0.2)


def test_condensable_cv_relation():
    mix = MixtureThermo("argon", "co2", 0.04)
    T = np.array([100.0, 150.0, 200.0])
    cp = mix.condensable_cp(T)
    cv = mix.condensable_cv(T)
    assert np.allclose(cp - cv, R_U / mix.MW_condensable)


def test_ideal_gas_condensable_density_monotonic():
    mix = MixtureThermo("argon", "co2", 0.04)
    rho = mix.mix_rho(np.array([90.0, 120.0, 200.0]), 4000.0)
    assert rho[0] > rho[1] > rho[2]  # colder -> denser at fixed p


def test_mw_mean_between_components():
    mix = MixtureThermo("argon", "co2", 0.0434)
    mw = mix.MW_mean_molefrac * 1e3
    assert 39.9 < mw < 40.3  # mostly Ar, slightly heavier from CO2
