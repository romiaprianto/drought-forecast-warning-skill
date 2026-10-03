"""Table 16 (regional aggregation fraction) and the run-to-run reproducibility paragraph of Section 3.6.
Inputs: results_mc_regional/ (second in-domain run, tcn_drought_multicell REV4 or regional_patch) and results_mc/."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from _paths import RMC, RMC_REG, OUT
reg = pd.read_csv(RMC_REG / "dss_regional_thresholds.csv"); ind = pd.read_csv(RMC_REG / "metrics_indomain.csv")
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
print("\nD0 exceptions:", len(exc)); pd.DataFrame(exc).to_csv(OUT / "table16_D0_exceptions.csv", index=False)
old = pd.read_csv(RMC / "metrics_indomain.csv"); k = ["cell_id", "horizon", "target", "model"]
x = ind.merge(old, on=k, suffixes=("_rerun", "_orig")); x["d"] = (x.SS_vs_persist_rerun - x.SS_vs_persist_orig).abs()
print("\nmax |dSS| rerun vs original per model:\n", x.groupby("model").d.max().round(4).to_string())
m = x.groupby(["target", "model", "horizon"])[["SS_vs_persist_orig", "SS_vs_persist_rerun"]].mean().round(3)
m.to_csv(OUT / "reproducibility_ss.csv"); print(m.loc["GWETROOT"].to_string())
