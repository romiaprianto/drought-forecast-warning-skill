"""Run-to-run reproducibility quoted in Section 3.6 and in the Limitations: two runs of an earlier configuration of the
multi-cell in-domain phase with identical settings. Inputs: results_earlier_runs/run{1,2}_metrics_indomain.csv and
run{1,2}_dss_multicell.csv."""
import pandas as pd
from _paths import REARLY, OUT
k = ["cell_id", "horizon", "target", "model"]
a = pd.read_csv(REARLY / "run1_metrics_indomain.csv"); b = pd.read_csv(REARLY / "run2_metrics_indomain.csv")
x = a.merge(b, on=k, suffixes=("_1", "_2"))
g = x[x.target == "GWETROOT"].groupby(["model", "horizon"])[["SS_vs_persist_1", "SS_vs_persist_2"]].mean()
g["abs_diff"] = (g.SS_vs_persist_2 - g.SS_vs_persist_1).abs(); g.round(3).to_csv(OUT / "reproducibility_ss.csv")
print("Soil-moisture skill score, mean over cells (run 1 vs run 2):"); print(g.round(3).to_string())
for m in ["TCN", "LSTM", "Transformer"]:
    print(f"  {m}: max change at leads >= 7 d = {g.loc[m].loc[[7, 14, 30], 'abs_diff'].max():.2f}")
da = pd.read_csv(REARLY / "run1_dss_multicell.csv"); db = pd.read_csv(REARLY / "run2_dss_multicell.csv")
y = da.merge(db, on=["horizon", "model", "logic", "level"], suffixes=("_1", "_2")); y["d"] = (y.CSI_2 - y.CSI_1).abs()
print("max change in CSI:", round(y.d.max(), 2), y.loc[y.d.idxmax(), ["model", "horizon", "logic", "level"]].to_dict())
for m in ["Persistence", "Climatology", "RandomForest"]:
    print(f"  {m}: max |change in SS| = {(x[x.model == m].SS_vs_persist_2 - x[x.model == m].SS_vs_persist_1).abs().max():.1e}")
