"""Physical constants (SI) and unit helpers for ssnreduce.

Values match Joel Mortimer's MATLAB pipeline to the precision he used, so
that a faithful port reproduces his numbers. Joel used:
    gasConstant   = 8.3145
    boltzmann     = 1.380649e-23
    avogadrosNumber = 6.0221408e23
We use the CODATA-exact values; the difference from Joel's is < 1e-5 relative
and never controls a result.
"""
from __future__ import annotations

K_B = 1.380649e-23          # Boltzmann constant [J/K]
N_A = 6.02214076e23         # Avogadro number [1/mol]
R_U = 8.314462618           # universal gas constant [J/mol/K]

# unit conversions used throughout (kept explicit for readability)
BAR_TO_PA = 1.0e5
PA_TO_BAR = 1.0e-5
KPA_TO_PA = 1.0e3
MM_TO_M = 1.0e-3
ANGSTROM_TO_M = 1.0e-10

__all__ = [
    "K_B", "N_A", "R_U",
    "BAR_TO_PA", "PA_TO_BAR", "KPA_TO_PA", "MM_TO_M", "ANGSTROM_TO_M",
]
