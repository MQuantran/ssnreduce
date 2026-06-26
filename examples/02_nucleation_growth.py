"""Stage 3 + 4 on a synthetic cooling streamline (no dry trace needed).

Until a dry companion run + the position->z calibration are available, we
demonstrate nucleation and growth on a representative supersonic cooling path:
a monotonic T(z) drop with the CO2 partial pressure p_v = x * p falling with
the static pressure. Plots J(z) for the CNT family + MKNT.

    python examples/02_nucleation_growth.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import ssnreduce as ssn
from ssnreduce.nucleation import nucleation_rates
from ssnreduce.growth import grow_droplets


def main() -> None:
    co2 = ssn.species("co2")
    ar = ssn.species("argon")

    # representative supersonic path: T 180 -> 85 K, p 0.3 -> 0.05 bar, x=4%
    z = np.linspace(0, 40, 60)                 # mm (synthetic)
    T = np.linspace(180.0, 85.0, z.size)
    p = np.linspace(0.30e5, 0.05e5, z.size)
    x_co2 = 0.04
    p_v = x_co2 * p

    nuc = nucleation_rates(T, p_v, co2, phase="liquid", mknt=True)
    print(f"S range:      {nuc.S.min():.2f} .. {nuc.S.max():.2f}")
    print(f"J_CNT  max:   {nuc.J_CNT.max():.3e} 1/m3/s")
    print(f"J_MKNT max:   {nuc.J_MKNT.max():.3e} 1/m3/s")
    iz = int(np.argmax(nuc.J_CNT))
    print(f"CNT peak at z={z[iz]:.1f} mm, T={T[iz]:.1f} K, S={nuc.S[iz]:.1f}, "
          f"n*={nuc.n_star_CNT[iz]:.1f}, r*={nuc.r_star_CNT[iz]*1e9:.2f} nm")

    # growth of a critical-radius droplet at the CNT peak
    r0 = max(nuc.r_star_CNT[iz], 0.5e-9)
    g = grow_droplets(r0, T[iz], p[iz], p_v[iz], co2, ar, phase="liquid")
    print(f"At peak: r0={r0*1e9:.2f} nm -> dr/dt={g['dr_dt'][0]*1e9:.1f} nm/s, "
          f"Td={g['Td'][0]:.1f} K, Kn={g['Kn'][0]:.1f}")

    fig = ssn.plots.plot_nucleation(nuc, Path(__file__).parent.parent / "figures" / "demo_J_of_z.png",
                                    x=z, xlabel="z [mm] (synthetic)",
                                    title="Synthetic streamline: J(z), CO2/Ar liquid branch")
    print(f"wrote {fig}")

    # solid-branch comparison
    nuc_s = nucleation_rates(T, p_v, co2, phase="solid", mknt=True)
    print(f"\nSolid branch: J_CNT max {nuc_s.J_CNT.max():.3e}, "
          f"J_MKNT max {nuc_s.J_MKNT.max():.3e} (desublimation pathway)")


if __name__ == "__main__":
    main()
