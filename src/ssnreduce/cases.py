"""Run/case metadata and dry-wet pairing (Joel's ``getCases``).

A *case* is one rig run plus the bookkeeping the reduction needs that is not in
the CSV: the carrier/condensable, the probe-geometry offsets that turn stage
position into z, which temperature column is T0, and -- crucially -- the *dry
companion* run that supplies the effective area ratio A/A*(z) for Stage 2.

This is intentionally lightweight: a dataclass plus a small registry you append
to as runs accumulate. It mirrors the fields of Joel's `cases.case_N` struct.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Case:
    """Metadata for one SSN run."""
    name: str                      # human label, e.g. "Ar_1bar_215K_4pctCO2"
    csv: str                       # path to the run CSV (absolute or repo-relative)
    carrier: str = "argon"
    condensable: str = "co2"
    condensable_phase: str = "liquid"   # "liquid" (Joel) or "solid" (desublimation)
    t0_col: str = "T0"             # Joel's columnT0
    is_dry: bool = False           # a dry (condensable-off) calibration run?
    dry_case: str | None = None    # name of the paired dry case (for wet runs)
    # position(%) -> z(mm): z = (position - z_throat_pct) * mm_per_pct
    # >>> APPARATUS CONSTANTS TO GET FROM JOEL (see diabatic.calibrate_z) <<<
    mm_per_pct: float | None = None
    z_throat_pct: float | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def has_z_calibration(self) -> bool:
        return self.mm_per_pct is not None and self.z_throat_pct is not None


class CaseRegistry:
    """Tiny name->Case registry with dry/wet lookup."""

    def __init__(self) -> None:
        self._cases: dict[str, Case] = {}

    def add(self, case: Case) -> Case:
        self._cases[case.name] = case
        return case

    def get(self, name: str) -> Case:
        return self._cases[name]

    def dry_for(self, name: str) -> Case | None:
        c = self._cases[name]
        return self._cases.get(c.dry_case) if c.dry_case else None

    def __contains__(self, name: str) -> bool:
        return name in self._cases

    def __iter__(self):
        return iter(self._cases.values())


# A starter registry for the 2026 commissioning runs. The dry pairing and the
# z-calibration are placeholders until a dry companion run + Joel's stage
# constants are available.
REGISTRY = CaseRegistry()
REGISTRY.add(Case(
    name="260603_Ar_1bar_215K_DRY",
    csv="260603_1119_51-Ar, 1 bar, 215 K_long(in).csv",
    carrier="argon", condensable="co2", is_dry=True,
    notes="DRY calibration run (condensable off, x_cond~0). 43 stations, "
          "~4.7 min/station, CD=0.90, M->2.35, throat at -197.4%. Provides the "
          "effective A/A*(position) for Stage 2. Pairs with the 260601/260602 "
          "wet runs (same p0=1 bar, T0=215 K, Ar carrier).",
))
REGISTRY.add(Case(
    name="260601_Ar_1bar_215K_4pctCO2",
    csv="260601_1237_28-Ar, 1 bar, 215 K, 4 percent CO2_long.csv",
    carrier="argon", condensable="co2", condensable_phase="liquid",
    dry_case="260603_Ar_1bar_215K_DRY",
    notes="4% CO2 in Ar, 10 Hz, 67 stations. Clear condensation onset vs the "
          "260603 dry trace at ~22-25% past throat (T reheat from ~100 K).",
))
REGISTRY.add(Case(
    name="260602_Ar_1bar_215K_0p5pctCO2",
    csv="260602_1746_41-Ar, 1 bar, 215 K, 0_dot_5 percent CO2.csv",
    carrier="argon", condensable="co2", condensable_phase="liquid",
    notes="0.5% CO2 in Ar, settling-study run.",
))
REGISTRY.add(Case(
    name="260529_Ar_1bar_215K_1pctCO2",
    csv="260529_1340_28-Ar, 1 bar, 215 K, 1 percent CO2_long.csv",
    carrier="argon", condensable="co2", condensable_phase="liquid",
    notes="1% CO2 in Ar, 0.5 s cadence.",
))


__all__ = ["Case", "CaseRegistry", "REGISTRY"]
