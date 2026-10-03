"""Shared paths. Repository layout: code/ data/ results/ results_mc/ results_mc_regional/ paper_analysis/."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CODE, DATA = ROOT / "code", ROOT / "data"
RES, RMC, RMC_REG = ROOT / "results", ROOT / "results_mc", ROOT / "results_mc_regional"
OUT = ROOT / "paper_outputs"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(CODE))
