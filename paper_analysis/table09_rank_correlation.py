"""Table 9: Spearman rank correlation between continuous skill (skill score vs persistence, GWETROOT) and warning skill
(CSI, OR rule, D1) across the methods at each lead time, single cell. Input: results/rank_correlation.csv
(written by code/tcn_drought_pipeline_v2.py)."""
import pandas as pd
from _paths import RES, OUT
T = pd.read_csv(RES / "rank_correlation.csv"); T.to_csv(OUT / "table09_rank_correlation.csv", index=False); print(T.round(3).to_string(index=False))
