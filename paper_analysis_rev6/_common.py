"""Shared paths and settings for the REV6 paper analysis.

Layout expected inside the repository (all paths can be overridden with environment variables):

    code/                         REV6 pipeline (tcn_drought_rev6.py and its REV5 dependencies)
    data/                         NASA POWER cell files (cell_*.csv, cells.csv)
    results_rev6/                 unzipped results_rev6_indomain.zip (must contain parts/)
    results_rev6_transfer/        the three unzipped results_rev6_transfer_*.zip folders, each with parts/
    paper_analysis_rev6/          these scripts
    paper_outputs_rev6/           created by the scripts (tables, figures, re-analysed results)
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("REPO_ROOT", HERE.parent)).resolve()
CODE = ROOT / "code"
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RAW_IN = Path(os.environ.get("RES_INDOMAIN", ROOT / "results_rev6"))
RAW_TR = Path(os.environ.get("RES_TRANSFER", ROOT / "results_rev6_transfer"))

OUT = Path(os.environ.get("PAPER_OUT", ROOT / "paper_outputs_rev6"))
IN = OUT / "indomain_b10k"          # in-domain verification re-run with BOOT_B replicates
TR = OUT / "transfer_merged"        # the ten held-out cells merged and verified together
TAB = OUT / "tables"                # CSV sources of every table in the paper and Online Resource 1
FIG = OUT / "figures"               # Fig1-Fig6 (PNG, 600 dpi) and TIFF copies
GEO = HERE / "_geo"                 # Natural Earth coastlines, downloaded on first use (Fig. 1)

BOOT_B = int(os.environ.get("BOOT_B", 10000))          # block-bootstrap replicates (paper: 10,000)
PERCELL_B = int(os.environ.get("PERCELL_B", 200))      # replicates for per-held-out-cell tables (S7)
NPROC = int(os.environ.get("NPROC", 4))                # parallel workers for the transfer analysis

for p in (OUT, IN, TR, TAB, FIG):
    p.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE))

MODELS = ["RandomForest", "TCN", "LSTM", "Transformer"]
TRANSFER_MODELS = ["RandomForest", "TCN"]
LEADS = [1, 7, 14, 30]
