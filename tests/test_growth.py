import numpy as np
import pytest

from ssnreduce.gases import species
from ssnreduce.growth import droplet_growth_rate, mean_free_path, grow_droplets

CO2 = species("co2")
AR = species("argon")


def test_droplet_warmer_than_gas():
    """Latent heat of condensation must warm the droplet: Td >= T."""
    gp = droplet_growth_rate(10e-9, 100.0, 1e4, 430.0, CO2, AR)
    assert gp.Td >= 100.0
    assert gp.dr_dt > 0      # supersaturated -> growing


def test_kelvin_curvature_slows_small_droplets():
    """Smaller droplets see a higher surface vapour pressure (Ostwald-Freundlich)
    -> lower driving force -> cooler surface than large droplets."""
    small = droplet_growth_rate(1e-9, 100.0, 1e4, 430.0, CO2, AR)
    large = droplet_growth_rate(100e-9, 100.0, 1e4, 430.0, CO2, AR)
    assert small.Td < large.Td


def test_knudsen_scales_inverse_radius():
    g1 = droplet_growth_rate(5e-9, 100.0, 1e4, 430.0, CO2, AR)
    g2 = droplet_growth_rate(50e-9, 100.0, 1e4, 430.0, CO2, AR)
    assert g1.Kn == pytest.approx(10.0 * g2.Kn, rel=1e-6)


def test_mean_free_path_scaling():
    # lambda ~ T/p
    assert mean_free_path(200.0, 1e4, 3.4e-10) == pytest.approx(
        2.0 * mean_free_path(100.0, 1e4, 3.4e-10), rel=1e-9)
    assert mean_free_path(100.0, 1e4, 3.4e-10) == pytest.approx(
        2.0 * mean_free_path(100.0, 2e4, 3.4e-10), rel=1e-9)


def test_grow_droplets_vectorised():
    out = grow_droplets([1e-9, 10e-9, 50e-9], 100.0, 1e4, 430.0, CO2, AR)
    assert out["dr_dt"].shape == (3,)
    assert np.all(out["Td"] >= 100.0)
