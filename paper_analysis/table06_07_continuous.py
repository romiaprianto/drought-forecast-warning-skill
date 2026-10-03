"""Tables 6 and 7: continuous performance of the single-cell forecasts (mean +/- sd over five seeds) for GWETROOT (Table 6)
and SPEI-1 (Table 7), with the Benjamini-Hochberg-adjusted Diebold-Mariano p-value against persistence.
An asterisk marks a significant improvement on persistence (filled markers in Figure 7).
Inputs: results/metrics_raw.csv, results/diebold_mariano.csv."""
import pandas as pd
from _paths import RES, OUT
raw = pd.read_csv(RES / "metrics_raw.csv"); dm = pd.read_csv(RES / "diebold_mariano.csv")
dm = dm[dm.comparison.str.endswith("_vs_Persistence")].assign(model=lambda x: x.comparison.str.replace("_vs_Persistence", ""))
pfmt = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
for tab, tg in (("table06", "GWETROOT"), ("table07", "SPEI_1")):
    rows = []
    for m in ["TCN", "LSTM", "Transformer", "RandomForest"]:
        for h in [1, 7, 14, 30]:
            g = raw[(raw.target == tg) & (raw.model == m) & (raw.horizon == h)]; r = dm[(dm.target == tg) & (dm.model == m) & (dm.horizon == h)].iloc[0]
            cell = lambda c, n: f"{g[c].mean():.{n}f} \u00b1 {g[c].std():.{n}f}"
            rows.append(dict(Model=m, Lead=h, RMSE=cell("RMSE", 3), NSE=cell("NSE", 3), KGE=cell("KGE", 3), SS=cell("SS_vs_persist", 2),
                             p_BH=pfmt(r.p_adj_BH) + (" *" if r.beats_persistence else "")))
    T = pd.DataFrame(rows); T.to_csv(OUT / f"{tab}_continuous_{tg}.csv", index=False); print(f"{tab} ({tg}): {int(T.p_BH.str.endswith('*').sum())} significant improvements"); print(T.to_string(index=False))
