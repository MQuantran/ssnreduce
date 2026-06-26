import numpy as np
import pytest

from ssnreduce.gases import species


def test_solid_psat_below_liquid_metastable():
    """Sublimation pressure must lie below the metastable-liquid line (solid
    is the more stable phase sub-triple)."""
    co2 = species("co2")
    T = np.array([120.0, 150.0, 200.0])
    assert np.all(co2.p_eq(T, "solid") < co2.p_eq(T, "liquid"))


def test_psat_monotonic_in_T():
    co2 = species("co2")
    T = np.array([90.0, 120.0, 160.0, 200.0])
    for ph in ("liquid", "solid"):
        pe = co2.p_eq(T, ph)
        assert np.all(np.diff(pe) > 0)


def test_density_and_vm_consistency():
    co2 = species("co2")
    T = np.array([120.0, 180.0])
    for ph in ("liquid", "solid"):
        rho = co2.rho_cond(T, ph)
        vm = co2.v_m(T, ph)
        assert np.allclose(vm, co2.c.m_molec / rho)


def test_solid_sigma_is_halonen():
    co2 = species("co2")
    assert np.allclose(co2.sigma(np.array([100.0, 150.0]), "solid"), 0.084)


def test_B2_negative_and_grows_cold():
    co2 = species("co2")
    B2 = co2.B2(np.array([100.0, 150.0, 200.0]))
    assert np.all(B2 < 0)
    assert abs(B2[0]) > abs(B2[-1])   # larger magnitude when colder


def test_L_sub_exceeds_L_vap():
    co2 = species("co2")
    T = np.array([150.0])
    assert co2.L(T, "solid")[0] > co2.L(T, "liquid")[0]
