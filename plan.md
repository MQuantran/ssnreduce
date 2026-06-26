# ssnreduce — design contract & build plan

Faithful Python port of Joel Mortimer's SSN reduction MATLAB
(`D:/PhD/MATLAB code/Joel/`), for analysing and plotting future Ar+CO₂ runs.
Read this first; it is the single source of truth for the package.

## 0. Decisions (locked 2026-06-03)

- **Placement:** standalone sibling package `D:/PhD/ssnreduce/`. Imports the
  `postprocessing/` loaders (via a path insert in `io.py`); treats `co2ssn/` as
  an independent nucleation cross-check, not a dependency.
- **Fidelity:** faithful physics (same equations/choices as Joel, reproduce his
  numbers), clean software (dataclasses, no globals/`evalin`/`persistent`).
- **Thermo backend:** CoolProp, with Joel's own sub-triple fallback (clamp
  carrier props to the triple temperature; ideal-gas NASA-7 for condensable
  vapour cp; ideal-gas law for sub-triple densities).

## 1. Module map → Joel's MATLAB

| ssnreduce | ports from | status |
|-----------|-----------|--------|
| `io.py` | CSV read block of `analyseSSN_mix5.m` (cols→names) + `postprocessing` | ✅ |
| `mixture.py` | every `refpropm` carrier call + condensable cp tables | ✅ |
| `isentropic.py` | `analyseSSN_mix5.m` lines 344–428 | ✅ |
| `diabatic.py` | `diabaticEquations.m` + the `analyseCondensation` ODE driver | ✅ core / ⚠ see §3 |
| `nucleation.py` | `analyseNucleation` (CNT/C/GC/Wölk/Hale + MKNT block) | ✅ done, tested |
| `growth.py` | `analyseDropletGrowth` (Gyarmathy 1963 + Ostwald–Freundlich) | ✅ single-droplet law done; population g(z) integration pending |
| `gases.py` | `getGases` property database (ported substance_props.m constants) | ✅ done, tested |
| `cases.py` | `getCases` run metadata + dry/wet pairing | ✅ done |
| `plots.py` | `plotData` figures | ✅ Stage 1 + J(z) overlay / ⏳ growth tracks, g(z), Kn(z) |
| `pipeline.py` | `compareSSN_4.m` top-level driver | ✅ Stage 1 (Stage 2-4 wiring waits on dry trace + z-cal) |

## 2. What is done and validated (v0.1)

- Load any rig CSV → `RunData` (p0, T0, mdot, auto-detected composition from
  mass flows / logged fractions), settled stations via `postprocessing`.
- Stage 1 reduction reproduces the postprocessing report on the 4% run
  (p/p₀ 0.104→0.574, M→2.10, T→87.6 K, CD≈0.90).
- Stage 2 ODE passes a dry-isentrope self-test (marched T, ρ, u match the
  analytic constant-γ isentrope to <0.4%) — Eqs 10/11/12 + velocity closure
  verified without real-data calibration.
- 16 pytest invariants green (Mach inversion exact, A/A\* textbook, triple-T
  clamp, throat sonic branch, …).

## 3. Open items before Stage 2 runs on real data

1. **position(%) → z(mm) calibration** (`diabatic.calibrate_z`). Need from Joel:
   stage travel per % of stroke, and the % reading at the geometric throat.
2. **Dry companion trace.** Each wet run needs a paired dry run to give the
   *effective* A/A\*(z) (boundary-layer-corrected, not CAD area). Mirror Joel's
   `dryTrace`/`dryCaseNumber` pairing in `cases.py`.
3. **Pre/post-onset split** (Joel's case-17 path): integrate dry to onset, then
   wet from the onset state. `solve_diabatic` currently does a single pass; add
   the split once onset detection is exercised on real data.
4. **Latent heat & condensed-phase cp** for CO₂ are placeholders
   (`L_of_T=const 591 kJ/kg`, `cp_cond_liq=const`). Replace with `gases.py`
   sublimation-curve values (CO₂ is below its triple point → desublimation;
   `L_sub`, not `L_vap`).

## 4. Next increment (real-data Stage 2-4 chain + droplet population + publish)

Stages 1-4 physics now exist and are unit-tested. What remains:

1. **Close the apparatus gaps (§3)** so the Stage 2-4 chain runs on a real run:
   the position→z calibration and a dry companion trace. Then wire
   `pipeline.reduce_run` → `reduce_case(case)` that does isentropic → diabatic →
   nucleation → growth end-to-end and reproduces Joel's g(z)/J(z).
2. **Droplet-population g(z) integration** — port the full `analyseDropletGrowth`
   population bookkeeping (per-parcel nucleation dN = J A dz/u, accumulate
   dm_growth, integrate to g(z)) so the growth-model g(z) can be compared with
   the diabatic-inversion g(z) (the cross-check at the heart of the method).
3. **Cross-model validation** — `nucleation.J_CNT` vs `co2ssn.cnt.cnt_standard`
   on a matched (σ, p_sat, v_m) grid (independent implementations of the same
   formula → agree to machine precision).
4. **More plots** — r\*(z), growth tracks, g(z) overlay, Kn(z).
5. **Publish prep** — see `PUBLISHING.md`: push public repo to start the JOSS
   6-month clock, add CI (GitHub Actions), Zenodo DOI, docs; keep apparatus
   constants + raw CSVs private (engine/apparatus split). Confirm naming
   (`ssnreduce`; `nucleation` is taken on PyPI).

## 5. Validation strategy

- **Now:** invariants + the dry-isentrope self-test (no Joel outputs needed).
- **When a wet+dry pair + z-calibration arrive:** reproduce Joel's reduced
  g(z)/T(z)/J(z) for one shared case to numerical tolerance; that closes the
  faithfulness claim. Store the comparison as a regression test.
- **Cross-model:** `nucleation.py` J_CNT vs `co2ssn.cnt.cnt_standard` on a T,S
  grid — independent implementations should agree to ~machine precision on the
  shared formula.
