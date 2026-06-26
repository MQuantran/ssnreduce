"""Stage 1 end-to-end on a real run, with sanity checks.

Run:
    python examples/01_stage1_isentropic.py

Loads the 2026-06-01 4% CO2-in-Ar run, extracts settled stations, reduces
p/p0 -> M, T, A/A* and the choked throat, and prints a summary. Asserts the
p/p0 range and a physically sane supersonic tail (matches the postprocessing
report: p/p0 0.57 -> 0.10, M ~ 2.1, T ~ 88 K).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import ssnreduce as ssn

REPO = Path(__file__).resolve().parents[2]  # D:/PhD
CSV = REPO / "260601_1237_28-Ar, 1 bar, 215 K, 4 percent CO2_long.csv"


def main() -> None:
    run = ssn.load(CSV)
    print(f"Run: {run.carrier} carrier / {run.condensable} condensable")
    print(f"  p0 = {run.p0_bar:.4f} +/- {run.p0_std_bar:.4f} bar")
    print(f"  T0 = {run.T0_K:.2f} +/- {run.T0_std_K:.2f} K")
    print(f"  mdot = {run.mdot_g_s:.3f} g/s")
    print(f"  x_condensable = {run.x_condensable:.4f}, w = {run.w_condensable:.4f}"
          f"  (source: {run.composition_source})")

    st = ssn.stations(run)
    print(f"\nStations: {len(st)}")
    print(f"  position range [{st['position'].min():.1f}, {st['position'].max():.1f}] %")
    print(f"  p/p0 range     [{st['pp0_run'].min():.4f}, {st['pp0_run'].max():.4f}]")

    iso = ssn.reduce_isentropic(run, st)
    print(f"\nMixture: gamma0 = {iso.gamma0:.4f}, rho0 = {iso.rho0:.4f} kg/m3, "
          f"cp0 = {iso.cp0:.1f} J/kg/K, MWmean = {iso.MW_mean*1e3:.3f} g/mol")
    print(f"Throat:  p*/p0 = {iso.p_throat_p0:.4f}, T* = {iso.T_throat:.2f} K, "
          f"u* = {iso.u_throat:.1f} m/s, G* = {iso.G_throat:.2f} kg/m2/s, "
          f"rho* = {iso.rho_throat:.4f} kg/m3")
    print(f"  Mach range [{iso.M.min():.3f}, {iso.M.max():.3f}], "
          f"T range [{iso.T.min():.1f}, {iso.T.max():.1f}] K")

    # theoretical mass flow check (Joel's CD diagnostic)
    A_throat = (3e-3 * 2.5e-3) - 0.25 * np.pi * (0.5e-3) ** 2  # m^2 (Joel geom)
    mdot_theory = iso.G_throat * A_throat * 1e3  # g/s
    print(f"  mdot_theory (geom) = {mdot_theory:.3f} g/s  ->  CD = {run.mdot_g_s/mdot_theory:.3f}")

    # ---- sanity assertions -------------------------------------------------
    assert 0.05 < iso.p_p0.min() < 0.15, iso.p_p0.min()
    assert 0.5 < iso.p_p0.max() < 0.65, iso.p_p0.max()
    assert iso.M.max() > 2.0, iso.M.max()
    assert iso.T.min() < 100.0, iso.T.min()
    assert 1.55 < iso.gamma0 < 1.68, iso.gamma0
    print("\nOK: Stage 1 sanity checks passed.")


if __name__ == "__main__":
    main()
