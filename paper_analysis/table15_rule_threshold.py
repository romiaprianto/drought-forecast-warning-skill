"""Table 15 and the seed paragraph of Section 3.6 (single-cell experiment).
Inputs: results/dss_validation.csv, results/metrics_raw.csv."""
import pandas as pd
from scipy.stats import spearmanr
from _paths import RES, OUT
dv = pd.read_csv(RES / "dss_validation.csv"); raw = pd.read_csv(RES / "metrics_raw.csv")
cont = raw[raw.target == "GWETROOT"].groupby(["model", "horizon"]).SS_vs_persist.mean().rename("cs").reset_index()
NM = {"TCN": "TCN", "LSTM": "LSTM", "RandomForest": "RF", "Transformer": "Transformer"}
rows, dom = [], []
for lv in ["D0", "D1"]:
    for lg in ["or", "and"]:
        g = dv[(dv.level == lv) & (dv.logic == lg)]
        pc = lambda h: g[(g.model == "Persistence") & (g.horizon == h)].CSI.iloc[0]
        def best(h):
            s = g[(g.horizon == h) & g.model.isin(NM)].set_index("model").CSI
            return f"{s.max():.2f}" + (f" ({NM[s.idxmax()]})" if s.max() > 0 else "")
        w = g[["model", "horizon", "CSI"]].merge(cont, on=["model", "horizon"])
        rows.append(dict(threshold=lv, rule=lg.upper(), base_rate=round(g.base_rate.mean(), 3),
                         persistence_CSI_1d=round(pc(1), 2), persistence_CSI_30d=round(pc(30), 2),
                         best_learned_1d=best(1), best_learned_30d=best(30), pooled_rho=round(spearmanr(w.cs, w.CSI)[0], 2)))
        for h, x in g.groupby("horizon"):
            s = x.set_index("model").CSI
            dom.append(dict(level=lv, rule=lg, horizon=h, any_detection=s.max() > 0,
                            persistence_best_or_tied=s["Persistence"] >= s.drop("Persistence").max() - 0.005))
T = pd.DataFrame(rows); T.to_csv(OUT / "table15_rule_threshold.csv", index=False); print(T.to_string(index=False))
D = pd.DataFrame(dom); det = D[D.any_detection]
print(f"\npersistence highest or tied (to 2 d.p.): {det.persistence_best_or_tied.sum()} of {len(det)} combinations with detections")
L = raw[(raw.target == "GWETROOT") & raw.model.isin(NM)]
S = L.assign(beats=L.SS_vs_persist > 0).groupby(["model", "horizon"]).beats.sum().unstack()
S.to_csv(OUT / "seed_wins.csv"); print("\nseeds (of 5) beating persistence, soil moisture:\n", S.to_string())
w30 = L[L.horizon == 30].pivot_table(index="seed", columns="model", values="SS_vs_persist")
print("RF > TCN at 30 d in", int((w30.RandomForest > w30.TCN).sum()), "of 5 seeds")
