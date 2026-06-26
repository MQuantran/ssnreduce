import numpy as np
import pytest

from ssnreduce.isentropic import _mach_from_pp0, _area_ratio


def _analytic_pp0(M, g):
    return (1.0 + 0.5 * (g - 1.0) * M**2) ** (-g / (g - 1.0))


def _analytic_AAstar(M, g):
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M**2)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


@pytest.mark.parametrize("g", [1.4, 1.5, 1.667])
def test_mach_inverts_pp0(g):
    """M(p/p0) must invert the isentropic stagnation relation exactly."""
    M = np.linspace(0.2, 3.0, 30)
    pp0 = _analytic_pp0(M, g)
    M_rec = _mach_from_pp0(pp0, np.full_like(pp0, g))
    assert np.allclose(M_rec, M, rtol=1e-9)


@pytest.mark.parametrize("g", [1.4, 1.667])
def test_area_ratio_matches_textbook(g):
    M = np.linspace(0.3, 3.0, 30)
    A = _area_ratio(M, np.full_like(M, g))
    assert np.allclose(A, _analytic_AAstar(M, g), rtol=1e-9)


def test_area_ratio_min_at_M1():
    g = 1.667
    M = np.linspace(0.5, 2.0, 1001)
    A = _area_ratio(M, np.full_like(M, g))
    assert A.min() == pytest.approx(1.0, abs=2e-3)
    assert abs(M[np.argmin(A)] - 1.0) < 5e-3
