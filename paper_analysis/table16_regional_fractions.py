"""Table 16: sensitivity of the regional alert to the aggregation fraction, with the combinations in which a learned
model exceeded persistence. Inputs: results_mc/dss_regional_thresholds.csv, results_mc/metrics_indomain.csv."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from _paths import RMC, OUT
reg = pd.read_csv(RMC / "dss_regional_thresholds.csv"); ind = pd.read_csv(RMC / "metrics_indomain.csv")
cont = ind[ind.target == "GWETROOT"].groupby(["model", "horizon"]).SS_vs_persist.mean().rename("cs").reset_index()
rows, exc = [], []
for fr, g in reg.groupby("threshold"):
    d1 = g[g.level == "D1"]; mean_csi = d1.groupby("model").CSI.mean()
    learned = mean_csi.drop(["Persistence", "Climatology"])
    cnt = {}
    for key, lv in [("D1-D3", ["D1", "D2", "D3"]), ("D0", ["D0"])]:
        best = tot = 0
        for (l, h), x in g[g.level.isin(lv)].groupby(["level", "horizon"]):
            s = x.set_index("model").CSI.fillna(0)
            if s.max() == 0: continue
            tot += 1; best += s["Persistence"] >= s.drop("Persistence").max() - 1e-9
            if s["Persistence"] < s.drop("Persistence").max() - 1e-9: exc.append(dict(fraction=fr, level=l, horizon=h, winner=s.drop("Persistence").idxmax()))
        cnt[key] = f"{best} / {tot}"
    w = d1[["model", "horizon", "CSI"]].merge(cont, on=["model", "horizon"]).dropna()
    rows.append(dict(fraction=fr, base_rate_D1=round(d1.base_rate.mean(), 3), persistence_CSI_D1=round(mean_csi["Persistence"], 2),
                     best_learned_CSI_D1=f"{learned.max():.2f} ({', '.join(learned[np.isclose(learned.round(2), round(learned.max(),2))].index)})",
                     persistence_highest_D1_D3=cnt["D1-D3"], persistence_highest_D0=cnt["D0"], pooled_rho_D1=round(spearmanr(w.cs, w.CSI)[0], 2)))
T = pd.DataFrame(rows); T.to_csv(OUT / "table16_regional_fractions.csv", index=False); print(T.to_string(index=False))
print("\nCombinations in which a learned model exceeded persistence:", len(exc)); pd.DataFrame(exc).to_csv(OUT / "table16_exceptions.csv", index=False)
