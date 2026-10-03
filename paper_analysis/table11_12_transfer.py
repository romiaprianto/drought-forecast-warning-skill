"""Section 3.4 numbers, Table 11 (skill retention at the 14-day lead) and Table 12 (transfer warning skill)."""
import numpy as np, pandas as pd
from _paths import RMC, OUT
ind = pd.read_csv(RMC / "metrics_indomain.csv"); tr = pd.read_csv(RMC / "metrics_transfer.csv")
for m in ["TCN", "RandomForest"]:
    for tgt in ["GWETROOT", "SPEI_1"]:
        i = ind[(ind.model == m) & (ind.target == tgt)].groupby("horizon").SS_vs_persist.mean()
        t = tr[(tr.model == m) & (tr.target == tgt)].groupby("horizon").SS_vs_persist.mean()
        share = tr[(tr.model == m) & (tr.target == tgt)].groupby("horizon").SS_vs_persist.apply(lambda s: int((s > 0).sum()))
        print(f"{m:12s} {tgt:8s} in-domain {i.round(2).tolist()} | transfer {t.round(2).tolist()} | cells beating persistence {share.tolist()}")
sm = pd.read_csv(RMC / "skill_map.csv")          # TCN, GWETROOT, map_horizon = 14
sm["retention_pct"] = (100 * sm.SS_transfer / sm.SS_indomain).round(0)
sm.sort_values("retention_pct", ascending=False).to_csv(OUT / "table11_transfer_per_cell.csv", index=False)
print("\nTable 11 (14-day lead):\n", sm[["cell_id", "SS_indomain", "SS_transfer", "retention_pct"]].round(3).to_string(index=False))
print(f"mean {sm.SS_indomain.mean():.3f} -> {sm.SS_transfer.mean():.3f}, retention {100*sm.SS_transfer.mean()/sm.SS_indomain.mean():.0f}%, "
      f"r = {np.corrcoef(sm.SS_indomain, sm.SS_transfer)[0,1]:.2f}")
p = pd.read_csv(RMC / "dss_transfer_multicell.csv")
T11 = p[(p.logic == "or") & (p.level == "D1")][["model", "horizon", "POD", "FAR", "CSI", "base_rate"]]
T11.to_csv(OUT / "table12_transfer_warning.csv", index=False); print("\nTable 11:\n", T11.round(2).to_string(index=False))
