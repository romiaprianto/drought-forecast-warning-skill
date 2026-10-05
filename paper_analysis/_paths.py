"""Shared paths. Repository layout: code/ data/ results/ results_mc/ results_earlier_runs/ paper_analysis/."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CODE, DATA = ROOT / "code", ROOT / "data"
RES, RMC, REARLY = ROOT / "results", ROOT / "results_mc", ROOT / "results_earlier_runs"
OUT = ROOT / "paper_outputs"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(CODE))
