"""Rank correlation between continuous skill and warning CSI: multi-cell (pooled, regional 30%) across
targets and levels D0-D3 (Fig. 14, Sections 3.4 and 4.3) and the transfer configuration (Section 3.4).
Uses analyze_dissociation.analyze, the same routine as the single-cell analysis."""
import pandas as pd
from _paths import RMC, OUT, CODE
import analyze_dissociation as A
rows = []
for src, dss in [("multicell", RMC / "dss_multicell.csv"), ("regional", RMC / "dss_regional.csv")]:
    for tgt in ["GWETROOT", "SPEI_1"]:
        for lvl in ["D0", "D1", "D2", "D3"]:
            rt, _ = A.analyze(str(RMC / "metrics_indomain.csv"), str(dss), target=tgt, logic="or", level=lvl)
            rows += [dict(source=src, target=tgt, level=lvl, horizon=str(r.horizon), rho=r.spearman_rho, p=r.p) for r in rt.itertuples()]
M = pd.DataFrame(rows); M.to_csv(OUT / "dissociation_matrix.csv", index=False)
print(M[M.level == "D1"].pivot_table(index=["source", "target"], columns="horizon", values="rho").round(2))
tdss = RMC / "dss_transfer_multicell.csv"
if tdss.exists():
    rt, merged = A.analyze(str(RMC / "metrics_transfer.csv"), str(tdss), target="GWETROOT", logic="or", level="D1")
    rt.to_csv(OUT / "dissociation_transfer_rankcorr.csv", index=False); merged.to_csv(OUT / "dissociation_transfer.csv", index=False)
    print("\ntransfer:\n", rt.round(2).to_string(index=False))
else:
    print("dss_transfer_multicell.csv not found: run tcn_drought_multicell.py REV4 (phase='merge') first")
