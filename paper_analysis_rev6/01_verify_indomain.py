"""Step 1. Re-run the in-domain verification of REV6 Run 1 with BOOT_B block-bootstrap replicates.

Input : results_rev6/parts/ (values_indomain__h*.csv.gz, thr_indomain__h*.csv, cont_/pp_ files)
Output: paper_outputs_rev6/indomain_b10k/ (warning_day/onset/episode/regional.csv, continuous_index_space.csv, ...)
Runtime: about 1-2 h on a laptop CPU with BOOT_B=10000 (set BOOT_B=1000 for a quick check).
"""
import glob
import shutil
from _common import RAW_IN, IN, BOOT_B
import tcn_drought_rev6 as R

src = RAW_IN / "parts"
if not src.is_dir():
    raise SystemExit(f"Missing {src}: unzip results_rev6_indomain.zip into {RAW_IN}")
dst = IN / "parts"
dst.mkdir(parents=True, exist_ok=True)
for f in glob.glob(str(src / "*_indomain__h*")):
    shutil.copy2(f, dst)
R.main(dict(R.CONFIG_R6, out_dir=str(IN), phase="analyze", boot_B=BOOT_B))
