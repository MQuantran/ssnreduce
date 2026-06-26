import numpy as np
import pytest

from ssnreduce.gases import species
from ssnreduce.nucleation import nucleation_rates
from ssnreduce.constants import K_B

CO2 = species("co2")


def test_courtney_is_cnt_over_S():
    T = np.array([100.0, 90.0]); p_v = np.array([430.0, 400.0])
    r = nucleation_rates(T, p_v, CO2, mknt=False)
    good = r.S > 1
    assert np.allclose(r.J_C[good], r.J_CNT[good] / r.S[good])


def test_girshick_chiu_factor():
    T = np.array([100.0, 90.0]); p_v = np.array([430.0, 400.0])
    r = nucleation_rates(T, p_v, CO2, mknt=False)
    good = r.J_C > 0
    assert np.allclose(r.J_GC[good] / r.J_C[good], np.exp(r.theta_macro[good]), rtol=1e-6)


def test_no_nucleation_when_subsaturated():
    T = np.array([200.0]); p_v = np.array([100.0])   # S < 1
    r = nucleation_rates(T, p_v, CO2, mknt=True)
    assert r.S[0] < 1.0
    assert r.J_CNT[0] == 0.0
    assert r.J_MKNT[0] == 0.0


def test_cnt_increases_with_supersaturation():
    T = np.full(4, 95.0)
    pe = float(CO2.p_eq(95.0, "liquid")[0])
    S = np.array([5.0, 20.0, 100.0, 500.0])
    r = nucleation_rates(T, S * pe, CO2, mknt=False)
    assert np.all(np.diff(r.J_CNT) > 0)


def test_cnt_kelvin_radius():
    """r*_CNT must satisfy the Kelvin relation 2 sigma v_m/(kT ln S)."""
    T = np.array([95.0]); pe = float(CO2.p_eq(95.0, "liquid")[0])
    p_v = np.array([100.0 * pe])
    r = nucleation_rates(T, p_v, CO2, mknt=False)
    kelvin = 2.0 * r.sigma[0] * r.v_m[0] / (K_B * T[0] * np.log(r.S[0]))
    assert r.r_star_CNT[0] == pytest.approx(kelvin, rel=1e-6)


def test_mknt_positive_and_parameters_sane():
    T = np.array([90.0]); p_v = np.array([400.0])
    r = nucleation_rates(T, p_v, CO2, mknt=True)
    assert r.J_MKNT[0] > 0
    assert 5.0 < r.N1[0] < 13.0          # coordination number ~ bulk 12
    assert 0.3 < r.omega[0] < 1.2
    assert r.theta_micro[0] > 0
    assert r.n_star_MKNT[0] >= 1
