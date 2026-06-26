import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REPO = Path(__file__).resolve().parents[2]  # D:/PhD


def data_csv() -> Path:
    return REPO / "260601_1237_28-Ar, 1 bar, 215 K, 4 percent CO2_long.csv"
