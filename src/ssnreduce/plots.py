"""Plotting -- reproduce Joel's reduced-profile figures.

Stage 1: p/p0, Mach, T, A/A* along the probe traverse (:func:`plot_isentropic`,
         :func:`plot_pp0`). x-axis is stage position [%] until a z(mm)
         calibration is available; pass ``z`` to switch to mm.
Stage 2: T(z), condensed fraction g(z), velocity u(z) (:func:`plot_diabatic`).
Stage 3: nucleation rate variants J(x) on a log axis (:func:`plot_nucleation`).

All functions use the non-interactive Agg backend and save a PNG, returning the
output Path -- safe to call headless in scripts/CI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .isentropic import IsentropicResult

_BLUE = "#1f4e79"
_RED = "#c0392b"


def _fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_isentropic(
    iso: IsentropicResult,
    savepath: str | Path,
    title: str = "",
    z: np.ndarray | None = None,
) -> Path:
    """Four-panel reduced isentropic profile (p/p0, M, T, A/A*)."""
    plt = _fig()
    x = iso.position if z is None else np.asarray(z)
    xlabel = "stage position [%]" if z is None else "z [mm]"

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    (ax_p, ax_M), (ax_T, ax_A) = axes

    ax_p.plot(x, iso.p_p0, "o-", ms=3, lw=1.2, color=_BLUE)
    ax_p.set_ylabel(r"$p/p_0$")
    ax_p.axhline(iso.p_throat_p0, ls="--", color="grey", lw=1,
                 label=fr"$p^*/p_0={iso.p_throat_p0:.3f}$")
    ax_p.legend(fontsize=8)

    ax_M.plot(x, iso.M, "o-", ms=3, lw=1.2, color=_BLUE)
    ax_M.set_ylabel("Mach $M$")
    ax_M.axhline(1.0, ls="--", color="grey", lw=1)

    ax_T.plot(x, iso.T, "o-", ms=3, lw=1.2, color=_RED)
    ax_T.set_ylabel("static $T$ [K]")
    ax_T.set_xlabel(xlabel)

    ax_A.plot(x, iso.A_A_throat, "o-", ms=3, lw=1.2, color=_BLUE)
    ax_A.set_ylabel(r"$A/A^*$")
    ax_A.set_xlabel(xlabel)
    ax_A.axhline(1.0, ls="--", color="grey", lw=1)

    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(title or "Stage 1: isentropic reduction")
    fig.tight_layout()

    out = Path(savepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_pp0(iso: IsentropicResult, savepath: str | Path, title: str = "",
             z: np.ndarray | None = None) -> Path:
    """Single-panel p/p0 profile with the throat reference line."""
    plt = _fig()
    x = iso.position if z is None else np.asarray(z)
    xlabel = "stage position [%]" if z is None else "z [mm]"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, iso.p_p0, "o-", ms=4, lw=1.2, color=_BLUE)
    ax.axhline(iso.p_throat_p0, ls="--", color="grey", lw=1,
               label=fr"$p^*/p_0={iso.p_throat_p0:.3f}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$p_\mathrm{centreline}/p_0$")
    ax.set_title(title or "Centreline pressure ratio")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = Path(savepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_nucleation(
    nuc,
    savepath: str | Path,
    x: np.ndarray | None = None,
    title: str = "",
    xlabel: str = "z [mm]",
) -> Path:
    """Overlay the nucleation-rate variants J(x) on a log axis.

    ``nuc`` is a :class:`ssnreduce.nucleation.NucleationResult`; ``x`` is the
    abscissa (z or position), defaulting to sample index.
    """
    plt = _fig()
    xx = np.arange(len(nuc.T)) if x is None else np.asarray(x)
    fig, ax = plt.subplots(figsize=(8, 5))
    series = [
        ("CNT", nuc.J_CNT, "-", _BLUE),
        ("Courtney", nuc.J_C, "--", "#2e86c1"),
        ("Girshick-Chiu", nuc.J_GC, "-.", "#27ae60"),
        ("Wolk-Strey", nuc.J_Wolk, ":", "#8e44ad"),
        ("MKNT", nuc.J_MKNT, "-", _RED),
    ]
    for label, J, ls, c in series:
        Jp = np.where(np.asarray(J) > 0, J, np.nan)
        if np.isfinite(Jp).any():
            ax.plot(xx, Jp, ls, color=c, lw=1.4, label=label)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$J$ [m$^{-3}$ s$^{-1}$]")
    ax.set_title(title or f"Nucleation rate ({nuc.phase} branch)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(savepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_diabatic(
    dia,
    savepath: str | Path,
    title: str = "",
) -> Path:
    """Three-panel Stage 2 profile: T(z), condensed fraction g(z), u(z).

    ``dia`` is a :class:`ssnreduce.diabatic.DiabaticResult`.
    z-axis is in mm from the throat (z[0] is the throat).
    """
    plt = _fig()
    z = dia.z
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    ax_T, ax_g, ax_u = axes

    ax_T.plot(z, dia.T, lw=1.4, color=_RED, label="T wet")
    ax_T.set_ylabel("T [K]")
    ax_T.set_xlabel("z [mm]")
    if dia.onset_z is not None and not np.isnan(dia.onset_z):
        for ax in axes:
            ax.axvline(dia.onset_z, ls="--", color="grey", lw=1, label=f"onset z={dia.onset_z:.1f} mm")
        ax_T.legend(fontsize=8)

    ax_g.plot(z, dia.g * 100.0, lw=1.4, color=_BLUE)
    ax_g.set_ylabel("condensed fraction g [%]")
    ax_g.set_xlabel("z [mm]")

    ax_u.plot(z, dia.u, lw=1.4, color=_BLUE)
    ax_u.set_ylabel("u [m/s]")
    ax_u.set_xlabel("z [mm]")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title or "Stage 2: diabatic reduction")
    fig.tight_layout()

    out = Path(savepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


__all__ = ["plot_isentropic", "plot_pp0", "plot_nucleation", "plot_diabatic"]
