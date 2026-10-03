"""Table 13: predictor ablation for the 14-day soil-moisture forecast (TCN), single cell. Input: results/ablation.csv
(written by code/tcn_drought_pipeline_v2.py)."""
import pandas as pd
from _paths import RES, OUT
T = pd.read_csv(RES / "ablation.csv"); T.to_csv(OUT / "table13_ablation.csv", index=False); print(T.round(3).to_string(index=False))
