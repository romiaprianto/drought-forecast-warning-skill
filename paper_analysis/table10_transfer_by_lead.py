"""Table 10: continuous skill under leave-one-cell-out transfer. Skill score relative to persistence averaged over the ten
cells for the TCN and the Random Forest, in-domain -> transfer, with the number of held-out cells in which the transferred
model beats persistence. Inputs: results_mc/metrics_indomain.csv, results_mc/metrics_transfer.csv."""
import pandas as pd
from _paths import RMC, OUT
ind = pd.read_csv(RMC / "metrics_indomain.csv"); tr = pd.read_csv(RMC / "metrics_transfer.csv"); tr = tr[tr.regime == "transfer"] if "regime" in tr else tr
rows = []
for tg in ["GWETROOT", "SPEI_1"]:
    for m in ["TCN", "RandomForest"]:
        r = dict(target=tg, model=m)
        for h in [1, 7, 14, 30]:
            a = ind[(ind.model == m) & (ind.target == tg) & (ind.horizon == h)].SS_vs_persist; b = tr[(tr.model == m) & (tr.target == tg) & (tr.horizon == h)].SS_vs_persist
            r[f"{h} d"] = f"{a.mean():.2f} -> {b.mean():.2f} ({int((b > 0).sum())})"
        rows.append(r)
T = pd.DataFrame(rows); T.to_csv(OUT / "table10_transfer_by_lead.csv", index=False); print(T.to_string(index=False))
