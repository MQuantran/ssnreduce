# ssnreduce

A faithful Python port of **Joel Mortimer's supersonic-nozzle (SSN) nucleation
data-reduction pipeline** (the MATLAB `analyseSSN_mix5` / `diabaticEquations` /
`compareSSN` chain), built to analyse and plot the lab's future Ar + CO₂ runs.

It deliberately reuses what already exists in `D:/PhD/`:

- **`postprocessing/`** — the proven rig-CSV loader and settled-station
  extractor (`load_run`, `extract_steady_stations`). `ssnreduce.io` wraps these.
- **`co2ssn/`** — the pure-CO₂ physics library. Kept as an *independent
  cross-check* for the nucleation rates (different EOS/σ choices), not a
  dependency of the reduction.

The new code is the **bridge**: real rig data → Joel's *mixture-aware*
three-stage reduction → Joel's plots.

## Status (v0.1)

| Stage | Module | State |
|------|--------|-------|
| Front-end: rig CSV → run scalars + settled stations | `io.py` | ✅ done, tested |
| Mixture thermo (Ar+CO₂, CoolProp-anchored, sub-triple clamp) | `mixture.py` | ✅ done, tested |
| **Stage 1** isentropic: p/p₀ → M, T, A/A\*, choked throat | `isentropic.py` | ✅ done, validated on real data |
| **Stage 2** diabatic ODE: → T, ρ, g, q | `diabatic.py` | ✅ core done, dry-isentrope self-test; ⚠ needs z-calibration + dry trace for real runs |
| **Stage 3** nucleation: CNT/Courtney/Girshick-Chiu/Wölk/Hale + MKNT | `nucleation.py` | ✅ done, tested (internal invariants) |
| **Stage 4** growth: Gyarmathy 1963 + Ostwald-Freundlich | `growth.py` | ✅ done, tested |
| Property DB (`getGases`), case metadata (`getCases`) | `gases.py`, `cases.py` | ✅ done, tested |
| Plots, pipeline | `plots.py`, `pipeline.py` | ✅ Stage 1 figures + J(z) overlay |

33 pytest invariants green. Publishing roadmap in `PUBLISHING.md` (MIT license +
`CITATION.cff` added; landscape scan found **no existing CNT/MKNT Python package**
— a genuine gap).

Validated on `260601_..._4 percent CO2_long.csv`: p/p₀ 0.104→0.574, M up to
2.10, T down to 87.6 K, discharge coefficient CD ≈ 0.90 — consistent with the
postprocessing report.

## Quick start

```python
import ssnreduce as ssn

# one call: load -> settled stations -> Stage 1 -> figures
res = ssn.reduce_run("260601_..._4 percent CO2_long.csv", figdir="figures")
print(res.run.p0_bar, res.run.T0_K, res.run.x_condensable)
iso = res.isentropic            # M, T, A/A*, throat properties
print(iso.M.max(), iso.T.min(), iso.p_throat_p0)
```

Run the worked example and the tests:

```bash
python examples/01_stage1_isentropic.py
pytest -q
```

## Known gaps to close with Joel (apparatus knowledge)

1. **position(%) → z(mm) calibration** — the rig logs probe position in % of
   stage stroke; the diabatic ODE integrates in mm from the throat. Need the
   stage travel-per-% and the throat % reading. Wired in `diabatic.calibrate_z`.
2. **Dry companion trace** — Stage 2 needs the effective A/A\*(z) from a *dry*
   run paired with each wet run (Joel's `dryTrace` pairing).
3. **`linearStage_steady` flag** is still all-zero in the logs (station
   detection falls back to position dwell — already handled).

See `plan.md` for the full design contract and the next-increment plan.

## Faithfulness

Physics equations and numerical choices follow Joel's MATLAB exactly (Wyslouzil
2000 Eqs 5/10/11/12/17, Streletzky 2002 Eqs 1/2/3, the choked-flow throat
relations, the MKNT cubic to come). The software is rewritten cleanly
(dataclass returns, no `evalin`/globals/`persistent`). Where RefProp was used,
CoolProp replaces it with Joel's own sub-triple fallback (clamp to the carrier
triple temperature). The goal is to reproduce his g(x), J(x) to numerical
tolerance — `tests/` enforces the invariants we can check without his outputs.
