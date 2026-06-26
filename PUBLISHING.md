# Publishing `ssnreduce` as a public Python library

Synthesised from a 2026-06-03 web-research scan (PyPI/GitHub probed live; JOSS /
packaging.python.org / pyOpenSci as primary sources). This is the plan, not a
commitment — open-sourcing timing must be agreed with Yi Yang and Joel given the
Aug 2026 handover and the unpublished Part-I CO₂ results.

## Why it's worth publishing — there is a real gap

The landscape scan found **no well-maintained, dedicated CNT/MKNT nucleation-rate
Python package**, and **no open-source supersonic-nozzle p(x)→g(x) reduction
pipeline**. Nucleation exists in Python only as a buried term inside atmospheric
aerosol box models (PySDM, particula, PyPartMC — all GPL, all atmospheric) or
inside proprietary CFD wet-steam solvers. The closest theory tool (Preciso) is
metallurgical C++. So `ssnreduce` — a species-agnostic CNT-family + MKNT rate
library *coupled to* a 1-D nozzle condensing-flow reduction — occupies an
unfilled niche.

Caveat: this bounds what is *publicly indexed*; a group's internal unreleased
code can't be ruled out. Worth a one-shot `phd-lit` check on Tanimura/Wyslouzil
for any 2024–2026 data-availability statement pointing to a deposited script.

## Naming

The PyPI distribution name **`nucleation` is already taken** (by a Minecraft
schematic parser, AGPL). Keep the distribution name **`ssnreduce`** (clear,
unclaimed) — or `pynucrate` if a more theory-centric name is wanted later.

## License — MIT

For scientific software meant to be reused across the ecosystem, pyOpenSci and
SciPy recommend permissive **MIT or BSD-3**. We use **MIT** (simplest; matches
CoolProp and PyMieScatt, our likely dependencies). Avoid GPL — it would block
the library from being vendored into permissively-licensed packages (the reason
PySDM/PyPartMC can't be). `LICENSE` added at repo root.

## The JOSS route (note the 2026 scope tightening)

[Journal of Open Source Software](https://joss.theoj.org/) — diamond OA, free,
CrossRef DOI on accept. Its requirements tightened in 2026; the gating items:

- **≥ 6 months of public version history** with releases and public issues/PRs,
  showing *iterative* development — not a single end-of-PhD commit dump.
  **→ Action: start the public repo NOW to bank the 6-month clock**, even while
  the physics is embargoed (JOSS reviews the *software*, not the results).
- **Feature-complete, research-grade** — not a single-function utility. Our four
  stages + CNT/MKNT clear this bar.
- **OSI license** (MIT ✓).
- `paper.md` + `paper.bib` with: Statement of need, State of the field, Software
  design, Research impact, and a **mandatory AI-usage disclosure** (2026 rule).
- Tests, install instructions, example usage, API docs, CONTRIBUTING + Code of
  Conduct. Compile-check `paper.md` via the JOSS GitHub Action before submitting.

## Packaging + release checklist (concrete order)

1. **License + metadata** — `LICENSE` (MIT ✓), fill `pyproject.toml` authors /
   ORCID / URLs, `CITATION.cff` (✓, add DOI once minted).
2. **Public repo** — push to GitHub (private subrepo for apparatus-specific bits,
   see split below). Start the JOSS 6-month clock.
3. **CI (GitHub Actions)** — matrix: 3 OSes × Python 3.10–3.13; run `pytest`
   (+coverage), `ruff` lint; later the JOSS `paper.md` compile action.
4. **TestPyPI dry-run** → real **PyPI via Trusted Publishing (OIDC)** from CI
   (no stored API token — current security norm). hatchling backend (already set);
   single-source the version.
5. **Zenodo** — enable GitHub↔Zenodo; each tagged Release mints a versioned DOI
   + a concept DOI. This is the citable archive independent of JOSS.
6. **Docs** — Sphinx + ReadTheDocs (MyST for Markdown) or mkdocs-material. Either
   satisfies JOSS.
7. **JOSS submission** once ≥6 months public + `paper.md` ready.

## What to open vs hold back (engine / apparatus split)

Open-source the **general, reusable engine** early; gate the **apparatus-specific
and unpublished-result** pieces behind the Part-I paper:

- **Publishable now:** `gases.py`, `mixture.py`, `isentropic.py`, `diabatic.py`,
  `nucleation.py`, `growth.py` (the CNT/MKNT math + the *generic* Wyslouzil 1-D
  reduction). These benefit from the 6-month clock.
- **Hold back (private subrepo / gated):** Joel-specific rig calibrations
  (the position→z stage constants, Ohio-State p₀ correction, throat geometry),
  raw run CSVs, and any unpublished CO₂/N₂ Wilson-point results. `io.py` and
  `cases.py` reference these — keep the *data* and the *fitted constants*
  private; the *code* can be public with placeholders.

## Open verification item

`particula`'s license field was blank on PyPI — check its GitHub `LICENSE`
directly before ever depending on it; it's the one actively-developed package
whose scope edges nearest ours.
