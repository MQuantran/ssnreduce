"""High-level orchestration -- one call from CSV to reduced profiles + figures.

Chains Stage 1 (isentropic). Stages 2-4 (diabatic, nucleation, growth) are
implemented in their respective modules and can be called directly after
obtaining a Stage 1 result. The pipeline exposes the key knobs that mirror
Joel's MATLAB options:

  ``smooth_pp0``  -- Savitzky-Golay smoothing of p/p0 before Stage 1
                     (mirrors Joel's ``smoothData`` in analyseSSN_mix5.m)
  ``solver``      -- ODE solver for Stage 2 (default "LSODA" = Joel's ode15s)
  ``phase``       -- condensate branch: "liquid" (Joel default) or "solid"

Use :func:`describe_solvers` to print all available ODE solvers with guidance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import io as _io
from .isentropic import IsentropicResult, reduce_isentropic
from .diabatic import SOLVERS
from . import plots


@dataclass
class ReductionResult:
    run: _io.RunData
    stations: pd.DataFrame
    isentropic: IsentropicResult
    figures: dict[str, Path]
    solver: str = "LSODA"      # ODE solver used (Stage 2 onwards)
    phase: str = "liquid"      # condensate branch used for nucleation/growth
    smoothed: bool = False     # True if SG smoothing was applied to p/p0


def describe_solvers() -> None:
    """Print available ODE solvers for Stage 2 with guidance on when to use each.

    Solvers are passed as ``method=`` to :func:`ssnreduce.solve_diabatic`.
    The active default is marked below.

    Example
    -------
    >>> import ssnreduce as ssn
    >>> ssn.describe_solvers()
    """
    print("Available ODE solvers for ssnreduce.solve_diabatic(method=...):\n")
    default = "LSODA"
    for name, (label, desc) in SOLVERS.items():
        marker = "  <-- default (= Joel's ode15s)" if name == default else ""
        print(f"  {name!r:<10s}  {label}{marker}")
        # wrap description at ~70 chars
        words = desc.split()
        line = "             "
        for w in words:
            if len(line) + len(w) + 1 > 79:
                print(line)
                line = "             " + w + " "
            else:
                line += w + " "
        if line.strip():
            print(line)
        print()


def reduce_run(
    csv_path: str | Path,
    t0_col: str = "T0",
    figdir: str | Path | None = None,
    refine_gamma: bool = True,
    position_tol: float = 0.1,
    min_dwell_s: float = 6.0,
    tail_seconds: float = 6.0,
    smooth_pp0: bool = False,
    smooth_window: int = 5,
    smooth_order: int = 2,
    solver: str = "LSODA",
    phase: str = "liquid",
) -> ReductionResult:
    """Load a run, extract settled stations, run Stage 1, optionally plot.

    Parameters
    ----------
    csv_path      : the rig CSV.
    t0_col        : stagnation temperature column (Joel's ``columnT0``).
                    Default ``"T0"`` (upstream RTD).
    figdir        : if given, write ``<stem>_pp0.png`` and
                    ``<stem>_isentropic.png`` to this directory.
    refine_gamma  : do the point-by-point gamma(T,p) iteration (Joel default).
    position_tol  : probe-position tolerance [%] for grouping stations.
    min_dwell_s   : minimum dwell time [s] at a position to count as a station.
    tail_seconds  : take the mean of the last N seconds of each dwell as the
                    settled value (Joel: mean of last 10 % of dwell time).
    smooth_pp0    : Savitzky-Golay smoothing of p/p0 before Stage 1.
                    Mirrors Joel's ``smoothData=1`` path in analyseSSN_mix5.m
                    and ``applySmoothing=1`` in compareSSN_4.m.
                    When True the smoothed values replace the raw stations in the
                    result; the raw stations DataFrame is unchanged.
    smooth_window : SG filter window length (odd, >= smooth_order+2). Joel: 5.
    smooth_order  : SG polynomial order. Joel: 2.
    solver        : ODE solver name for Stage 2 calls. Default ``"LSODA"``
                    (auto-stiff, equivalent to Joel's ode15s). Call
                    :func:`describe_solvers` to see all options.
    phase         : condensate phase branch -- ``"liquid"`` (Joel default,
                    metastable supercooled) or ``"solid"`` (desublimation).
                    Stored in the result so downstream Stage 3/4 calls can
                    inherit it; Stage 1 is phase-independent.

    Notes
    -----
    Stage 1 only. For Stage 2 (diabatic ODE), call :func:`ssnreduce.solve_diabatic`
    with the Stage 1 throat/stagnation state and the A/A*(z) from a dry companion
    run. For Stage 3 (nucleation rates), call :func:`ssnreduce.nucleation_rates`.
    For Stage 4 (droplet growth), call :func:`ssnreduce.grow_droplets`.
    """
    if solver not in SOLVERS:
        raise ValueError(f"Unknown solver {solver!r}. Choose from: {list(SOLVERS)}")
    if phase not in ("liquid", "solid"):
        raise ValueError(f"phase must be 'liquid' or 'solid', got {phase!r}")

    run = _io.load(csv_path, t0_col=t0_col)
    st = _io.stations(run, position_tol=position_tol,
                      min_dwell_s=min_dwell_s, tail_seconds=tail_seconds)
    if st.empty:
        raise RuntimeError(f"No settled stations found in {csv_path}")
    iso = reduce_isentropic(
        run, st, mix=None, refine_gamma=refine_gamma,
        smooth_pp0=smooth_pp0, smooth_window=smooth_window, smooth_order=smooth_order,
    )

    figures: dict[str, Path] = {}
    if figdir is not None:
        stem = Path(csv_path).stem
        smooth_tag = " [SG-smoothed]" if smooth_pp0 else ""
        title = (f"{run.condensable}/{run.carrier}  "
                 f"p0={run.p0_bar:.2f} bar  T0={run.T0_K:.1f} K{smooth_tag}")
        figures["pp0"] = plots.plot_pp0(iso, Path(figdir) / f"{stem}_pp0.png", title=title)
        figures["isentropic"] = plots.plot_isentropic(
            iso, Path(figdir) / f"{stem}_isentropic.png", title=title)

    return ReductionResult(
        run=run, stations=st, isentropic=iso, figures=figures,
        solver=solver, phase=phase, smoothed=iso.smoothed,
    )


__all__ = ["ReductionResult", "reduce_run", "describe_solvers"]
