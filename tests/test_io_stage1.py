"""Integration test: full Stage 1 on the real 4% run.

Skipped automatically if the run CSV isn't present (e.g. CI without data).
"""
import numpy as np
import pytest

from conftest import data_csv
import ssnreduce as ssn

pytestmark = pytest.mark.skipif(not data_csv().exists(), reason="run CSV not present")


def test_load_identifies_mixture():
    run = ssn.load(data_csv())
    assert run.carrier == "argon"
    assert run.condensable == "co2"
    assert 0.03 < run.x_condensable < 0.05   # ~4%
    assert 0.99 < run.p0_bar < 1.01


def test_stage1_profile_physical():
    res = ssn.reduce_run(data_csv())
    iso = res.isentropic
    assert len(res.stations) > 50
    # p/p0 range matches the postprocessing report (0.104 -> 0.571)
    assert iso.p_p0.min() == pytest.approx(0.104, abs=0.01)
    assert iso.p_p0.max() == pytest.approx(0.574, abs=0.02)
    # supersonic tail
    assert iso.M.max() > 2.0
    assert iso.T.min() < 100.0
    # discharge coefficient sane (real nozzle w/ boundary layer)
    A_throat = (3e-3 * 2.5e-3) - 0.25 * np.pi * (0.5e-3) ** 2
    cd = res.run.mdot_g_s / (iso.G_throat * A_throat * 1e3)
    assert 0.8 < cd < 1.0


def test_throat_is_sonic_branch():
    res = ssn.reduce_run(data_csv())
    iso = res.isentropic
    assert iso.A_A_throat.min() >= 1.0 - 1e-6
    assert 0.45 < iso.p_throat_p0 < 0.55  # ~ (2/(g+1))^(g/(g-1)) for g~1.65
