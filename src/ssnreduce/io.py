"""Rig CSV ingestion for ssnreduce.

This is the front-end that turns a raw SSN logger CSV into the run-level
scalars and per-station pressure ratios that Joel's reduction consumes.

It deliberately *reuses* the proven loader in ``D:/PhD/postprocessing``
(``ssn_postprocess.load_run`` / ``extract_steady_stations``) rather than
re-implementing the 4-row-header parsing, de-duplication and dropout masking.

Mapping to Joel's ``analyseSSN_mix5.m``
--------------------------------------
Joel read fixed CSV *column numbers* (col 11 = p0, col 12 = p_centreline,
col 25 = probe z, col 17/18/44 = mass flows). Our rig CSVs carry *named*
headers, so we bind by name instead (more robust to column reordering):

    P_stagnation        -> p0           (Joel col 11)
    P_centreline        -> p_centreline (Joel col 12)
    T0                  -> T0           (Joel columnT0, default col 2)
    linearStage_pos_actual [%] -> probe position (Joel col 25 was mm)
    mDot_mfc1_actual + mDot_mfc2_actual -> carrier mass flow (Joel col 17+18)
    mDot_coriolis(_CORR)               -> condensable mass flow (Joel col 44)

Run-level p0 and T0 are *whole-run means* (Joel's ``useMeanP0`` / ``mean(T0All)``),
because p0 is PID-controlled and its per-sample noise is not seen on the
attenuated centreline line. Per project policy (``feedback-no-fitted-asymptotes``)
the per-station p_centreline is the mean of the SETTLED window only — that part
is handled inside ``extract_steady_stations``.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --- locate and import the existing postprocessing loader -------------------

def _postprocessing_dir() -> Path:
    env = os.environ.get("SSN_POSTPROCESSING_DIR")
    if env:
        return Path(env)
    # io.py -> ssnreduce -> src -> <repo ssnreduce> -> D:/PhD
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "postprocessing"


_PP = _postprocessing_dir()
if str(_PP) not in sys.path:
    sys.path.insert(0, str(_PP))

try:  # pragma: no cover - exercised indirectly
    import ssn_postprocess as _pp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        f"Could not import ssn_postprocess from {_PP}. Set $SSN_POSTPROCESSING_DIR "
        f"to the folder containing ssn_postprocess.py."
    ) from exc


# --- molecular weights for the carriers/condensables we use [g/mol] ---------
MW = {
    "co2": 44.0095,
    "water": 18.01528,
    "argon": 39.948,
    "ar": 39.948,
    "nitrogen": 28.0134,
    "n2": 28.0134,
    "air": 28.9647,
    "helium": 4.002602,
    "he": 4.002602,
}


def _canon(name: str | None) -> str:
    return (name or "").strip().lower()


def _robust_mean(x: pd.Series | np.ndarray) -> float:
    """Mean after nearest-replacing gross outliers.

    Mirrors MATLAB ``filloutliers(...,'nearest','mean')`` used by Joel on the
    raw columns: points more than 3 scaled-MAD from the median are treated as
    spurious. Pure mean if the series is too short to estimate spread.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 8:
        return float(np.mean(a)) if a.size else float("nan")
    med = np.median(a)
    mad = np.median(np.abs(a - med)) * 1.4826
    if mad == 0:
        return float(np.mean(a))
    good = np.abs(a - med) <= 3.0 * mad
    return float(np.mean(a[good])) if good.any() else float(np.mean(a))


@dataclass
class RunData:
    """One loaded SSN run plus the scalars Joel's reduction needs."""

    path: Path
    df: pd.DataFrame
    units: dict[str, str]
    meta: dict[str, str]
    # identification
    carrier: str          # canonical, e.g. "argon"
    condensable: str      # canonical, e.g. "co2"
    # run-level stagnation scalars (whole-run means)
    p0_bar: float
    T0_K: float
    p0_std_bar: float
    T0_std_K: float
    mdot_g_s: float
    # composition
    x_condensable: float  # mole fraction of condensable in the feed
    w_condensable: float  # mass fraction of condensable in the feed
    # provenance of the composition number
    composition_source: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def MW_carrier(self) -> float:
        return MW[self.carrier]

    @property
    def MW_condensable(self) -> float:
        return MW[self.condensable]


def _composition(df: pd.DataFrame, carrier: str, condensable: str) -> tuple[float, float, str]:
    """Return (mole_frac, mass_frac, source) for the condensable.

    Preference order, most-trustworthy first:
      1. logged actual mole fraction ``x_condensable_actual``
      2. mass-flow ratio from the MFC/Coriolis channels (Joel's method)
      3. logged set mole fraction ``Mole Percent Condensable (set)`` / 100
    """
    mw_c, mw_k = MW[condensable], MW[carrier]

    # 2. mass flows: carrier = MFC1+MFC2, condensable = coriolis(_CORR).
    # Done first because it also detects a DRY run (condensable flow ~ 0).
    cor_col = "mDot_coriolis CORR" if "mDot_coriolis CORR" in df.columns else "mDot_coriolis"
    have_flows = all(c in df.columns for c in ("mDot_mfc1_actual", "mDot_mfc2_actual")) and cor_col in df.columns
    if have_flows:
        m_carrier = max(_robust_mean(df["mDot_mfc1_actual"]), 0.0) + max(_robust_mean(df["mDot_mfc2_actual"]), 0.0)
        m_cond = max(_robust_mean(df[cor_col]), 0.0)
        if m_carrier > 0:
            # dry run: condensable metered flow negligible (< 0.1% of carrier)
            if m_cond <= 1e-3 * m_carrier:
                return 0.0, 0.0, f"dry (no condensable flow, {cor_col})"
            w = m_cond / (m_cond + m_carrier)
            n_c, n_k = m_cond / mw_c, m_carrier / mw_k
            x = n_c / (n_c + n_k)
            return x, w, f"massflow({cor_col})"

    # 1. logged actual mole fraction
    if "x_condensable_actual" in df.columns:
        x = _robust_mean(df["x_condensable_actual"])
        if np.isfinite(x) and 0 <= x < 1:
            w = x * mw_c / (x * mw_c + (1 - x) * mw_k)
            return x, w, ("dry (x_condensable_actual~0)" if x < 1e-4 else "x_condensable_actual")

    # 3. logged set mole percent
    if "Mole Percent Condensable (set)" in df.columns:
        x = _robust_mean(df["Mole Percent Condensable (set)"]) / 100.0
        if np.isfinite(x) and 0 <= x < 1:
            w = x * mw_c / (x * mw_c + (1 - x) * mw_k)
            return x, w, ("dry (mole% set~0)" if x < 1e-4 else "Mole Percent Condensable (set)")

    return float("nan"), float("nan"), "unknown"


def load(csv_path: str | Path, t0_col: str = "T0") -> RunData:
    """Load a run CSV into a :class:`RunData`.

    Parameters
    ----------
    csv_path : path to the SSN long-form CSV.
    t0_col   : which temperature column is the stagnation T0 (Joel's
               ``columnT0``). Default ``"T0"`` (the upstream RTD). Use
               ``"T_coriolis"`` to mirror Joel's occasional col-7 choice.
    """
    csv_path = Path(csv_path)
    df, units, meta = _pp.load_run(csv_path)

    carrier = _canon(meta.get("Carrier"))
    condensable = _canon(meta.get("Condensable"))
    # normalise common aliases
    carrier = {"ar": "argon", "n2": "nitrogen", "he": "helium"}.get(carrier, carrier)
    condensable = {"h2o": "water"}.get(condensable, condensable)
    if carrier not in MW:
        raise ValueError(f"Unknown carrier {carrier!r}; add it to ssnreduce.io.MW")
    if condensable not in MW:
        raise ValueError(f"Unknown condensable {condensable!r}; add it to ssnreduce.io.MW")

    p0 = _robust_mean(df["P_stagnation"])
    p0_std = float(np.nanstd(df["P_stagnation"].to_numpy(dtype=float)))
    if t0_col not in df.columns:
        raise ValueError(f"t0_col {t0_col!r} not in CSV columns")
    T0 = _robust_mean(df[t0_col])
    T0_std = float(np.nanstd(df[t0_col].to_numpy(dtype=float)))

    cor_col = "mDot_coriolis CORR" if "mDot_coriolis CORR" in df.columns else "mDot_coriolis"
    mdot = 0.0
    for c in ("mDot_mfc1_actual", "mDot_mfc2_actual", cor_col):
        if c in df.columns:
            v = _robust_mean(df[c])
            if np.isfinite(v):
                mdot += max(v, 0.0)

    x_c, w_c, src = _composition(df, carrier, condensable)

    return RunData(
        path=csv_path, df=df, units=units, meta=meta,
        carrier=carrier, condensable=condensable,
        p0_bar=p0, T0_K=T0, p0_std_bar=p0_std, T0_std_K=T0_std,
        mdot_g_s=mdot, x_condensable=x_c, w_condensable=w_c,
        composition_source=src,
    )


def stations(
    run: RunData,
    position_tol: float = 0.1,
    min_dwell_s: float = 6.0,
    tail_seconds: float = 6.0,
) -> pd.DataFrame:
    """Per-station settled means via the postprocessing extractor.

    Adds ``pp0_run`` = p_centreline / run-mean-p0 (Joel's ``useMeanP0``
    convention) alongside the per-sample ``pp0`` that the extractor computes
    against the instantaneous p0.
    """
    st = _pp.extract_steady_stations(
        run.df,
        position_tol=position_tol,
        min_dwell_s=min_dwell_s,
        tail_seconds=tail_seconds,
    )
    if not st.empty:
        st = st.copy()
        st["pp0_run"] = st["P_centreline_bar"] / run.p0_bar
    return st


__all__ = ["RunData", "load", "stations", "MW"]
