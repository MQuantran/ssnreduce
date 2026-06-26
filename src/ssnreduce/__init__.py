"""ssnreduce -- faithful Python port of Joel Mortimer's SSN reduction pipeline.

Stages (matching ``knowledge/10_ssn_data_reduction_pipeline.md``):
  Stage 1  isentropic   p/p0 -> M, T, A/A*, choked throat       (isentropic.py)
  Stage 2  diabatic     coupled Wyslouzil ODE -> T, rho, g, q   (diabatic.py)
  Stage 3  nucleation   CNT family + MKNT -> J(x)               (nucleation.py)
  Stage 4  growth       Gyarmathy droplet growth                 (growth.py)

Front-end: ``io`` (rig CSV via postprocessing) + ``mixture`` (CoolProp-anchored
Ar+CO2 thermo). High-level orchestration: ``pipeline.reduce_run`` (Stage 1) and
direct calls to ``solve_diabatic`` / ``nucleation_rates`` / ``grow_droplets``
for Stages 2-4.

ODE solver selection (Stage 2)
--------------------------------
Joel's MATLAB uses ode15s (stiff BDF, compareSSN_4.m line 331). The default
here is ``"LSODA"`` -- the scipy equivalent (auto-switching Adams/BDF stiff
solver). To see all options::

    import ssnreduce as ssn
    ssn.describe_solvers()

Phase convention
-----------------
CO2 in the cryogenic SSN sits below its triple point (216.6 K), so the
thermodynamically stable condensate is solid. Joel used the metastable liquid
branch (``phase="liquid"``), which is the default here. Pass ``phase="solid"``
to the nucleation/growth functions to run the desublimation pathway.
"""
from __future__ import annotations

from . import constants
from . import plots
from .io import RunData, load, stations
from .mixture import MixtureThermo
from .isentropic import IsentropicResult, reduce_isentropic
from .diabatic import (
    SOLVERS,
    ThroatState, StagnationState, DiabaticResult, solve_diabatic, calibrate_z,
)
from .gases import Species, species, Phase
from .nucleation import NucleationResult, nucleation_rates
from .growth import GrowthPoint, droplet_growth_rate, grow_droplets
from .cases import Case, CaseRegistry, REGISTRY
from .pipeline import ReductionResult, reduce_run, describe_solvers

__version__ = "0.1.0"

__all__ = [
    "constants", "plots",
    # io
    "RunData", "load", "stations",
    # mixture
    "MixtureThermo",
    # Stage 1
    "IsentropicResult", "reduce_isentropic",
    # Stage 2
    "SOLVERS", "describe_solvers",
    "ThroatState", "StagnationState", "DiabaticResult", "solve_diabatic", "calibrate_z",
    # gases
    "Species", "species", "Phase",
    # Stage 3
    "NucleationResult", "nucleation_rates",
    # Stage 4
    "GrowthPoint", "droplet_growth_rate", "grow_droplets",
    # cases / pipeline
    "Case", "CaseRegistry", "REGISTRY",
    "ReductionResult", "reduce_run",
]
