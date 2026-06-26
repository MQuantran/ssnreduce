# ssnreduce

**ssnreduce** is a Python library for reducing supersonic-nozzle (SSN)
condensation experiments and computing nucleation rates. It implements the
standard four-stage data-reduction pipeline for dilute vapour–carrier gas
mixtures (e.g. CO₂/Ar, H₂O/N₂) developed in the Wyslouzil group
(Wyslouzil et al. 2000) and validated extensively in Bhabhe (2012) for
cryogenic systems.

A landscape scan found **no dedicated, open-source CNT/MKNT nucleation-rate
Python package** — this library fills that gap.

---

## What it does

Given the measured centreline pressure profile p(x) from a supersonic nozzle
experiment, `ssnreduce` recovers the full condensing flow state:

| Stage | Output | Key equations |
|-------|--------|---------------|
| **1 — Isentropic** | M(x), T(x), A/A\*(x), choked-throat state | Wyslouzil 2000 Eq 5; Streletzky 2002 Eqs 2–3 |
| **2 — Diabatic ODE** | T(z), ρ(z), g(z), q(z) | Wyslouzil 2000 Eqs 10–12, 17–18 |
| **3 — Nucleation rates** | J(x) for CNT family + MKNT | Bhabhe 2012 Eqs 2.17–2.34; Kalikmanov 2006 |
| **4 — Droplet growth** | dr/dt(x), Td(x), Kn(x) | Gyarmathy 1963; Peters & Paikert 1989 |

---

## Nucleation models

`nucleation_rates()` computes six rate variants at each streamline point:

| Model | Reference |
|-------|-----------|
| Classical nucleation theory (CNT) | Becker & Döring 1935 |
| Courtney correction | Courtney 1961 |
| Girshick–Chiu (self-consistent) | Girshick & Chiu 1990 |
| Wölk–Strey (empirical H₂O correction) | Wölk & Strey 2001 |
| Hale scaled model | Hale 1986 |
| **Mean-Field Kinetic Nucleation Theory (MKNT)** | Kalikmanov 2006; Bhabhe 2012 |

MKNT is the central modern alternative to CNT for cryogenic gases, validated
for Ar and N₂ within 1–3 orders of magnitude (Bhabhe 2012 Ch 4–5).

---

## Install

```bash
pip install ssnreduce
```

Requires Python ≥ 3.10 and [CoolProp](http://www.coolprop.org/) ≥ 6.6.

---

## Quick start

```python
import numpy as np
import ssnreduce as ssn

# --- nucleation rates on any T(z), p_v(z) streamline ---
co2 = ssn.species("co2")
z   = np.linspace(0, 40, 80)          # mm (synthetic)
T   = np.linspace(200.0, 90.0, 80)    # K
p_v = np.linspace(3000.0, 400.0, 80)  # Pa (partial pressure)

nuc = ssn.nucleation_rates(T, p_v, co2, phase="liquid", mknt=True)
print(f"J_CNT  max: {nuc.J_CNT.max():.2e} m⁻³ s⁻¹")
print(f"J_MKNT max: {nuc.J_MKNT.max():.2e} m⁻³ s⁻¹")

# --- Stage 2: diabatic ODE ---
# see examples/02_nucleation_growth.py for a full worked example

# --- available ODE solvers (Stage 2) ---
ssn.describe_solvers()
```

Run the worked examples:

```bash
python examples/01_stage1_isentropic.py
python examples/02_nucleation_growth.py
pytest -q
```

---

## Phase convention (CO₂ below the triple point)

Cryogenic CO₂ in an SSN sits below its triple point (216.6 K), so the
thermodynamically stable condensate is **solid** (desublimation). Every
property function accepts `phase="liquid"` (metastable supercooled, the
historically used branch) or `phase="solid"` (Span–Wagner sublimation line,
Halonen 2021 MD surface tension). Pass `phase=` consistently — it changes
p_eq, σ, ρ, and L together.

---

## ODE solver selection (Stage 2)

The diabatic coupled ODE is stiff near onset (rapid rise in condensed fraction
g). The default solver is `"LSODA"` (auto-switching Adams/BDF, equivalent to
MATLAB's `ode15s`). All SciPy solvers are available:

```python
ssn.describe_solvers()   # prints the full table with guidance

result = ssn.solve_diabatic(..., method="BDF")     # fully implicit
result = ssn.solve_diabatic(..., method="Radau")   # high-accuracy reference
result = ssn.solve_diabatic(..., method="RK45")    # explicit, dry runs only
```

---

## References

- Bhabhe, A. (2012). *Nucleation and Condensation in Cryogenic Gas Mixtures*. PhD thesis, The Ohio State University.
- Girshick, S. L., & Chiu, C.-P. (1990). Kinetic nucleation theory. *J. Chem. Phys.* 93, 1273.
- Gyarmathy, G. (1963). Zur Wachstumsgeschwindigkeit kleiner Flüssigkeitstropfen. *Z. Angew. Math. Phys.* 14, 280.
- Hale, B. N. (1986). Application of a scaled homogeneous nucleation-rate formalism. *Phys. Rev. A* 33, 4156.
- Halonen, R., et al. (2021). Homogeneous nucleation of CO₂ in supersonic nozzles. *Phys. Chem. Chem. Phys.* 23, 4517.
- Kalikmanov, V. I. (2006). Mean-field kinetic nucleation theory. *J. Chem. Phys.* 124, 124505.
- Peters, F., & Paikert, B. (1989). Nucleation and growth rates of homogeneously condensing water vapor in argon from shock tube experiments. *Exp. Fluids* 7, 521.
- Span, R., & Wagner, W. (1996). A new equation of state for carbon dioxide. *J. Phys. Chem. Ref. Data* 25, 1509.
- Streletzky, K. A., et al. (2002). CO₂ condensation in a supersonic nozzle. *J. Chem. Phys.* 116, 4058.
- Wölk, J., & Strey, R. (2001). Homogeneous nucleation of H₂O and D₂O in comparison. *J. Phys. Chem. B* 105, 11683.
- Wyslouzil, B. E., et al. (2000). Nonisothermal droplet growth in the free molecular regime. *J. Chem. Phys.* 113, 7317.

---

## Status

30 pytest invariants green. MIT licensed. See `PUBLISHING.md` for the JOSS roadmap.
