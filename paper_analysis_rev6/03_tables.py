"""Step 3. Continuous skill with Diebold-Mariano tests, rank correlation, and the D1 warning tables.

Paper: Table 3 (squared-error skill, DM tests), Section 3.3 (rank correlation), Tables 4 and 6 (CSI with
false-discovery-rate control against raw persistence and against persistence with the same post-processing).
Output: paper_outputs_rev6/tables/T_continuous.csv, T_rankcorr.csv, T_{indomain,transfer}_warning_{day,onset}_D1.csv,
        T_{indomain,transfer}_episode_D1.csv, T_thr_deltas_indomain.csv
"""
import numpy as np
import pandas as pd
import scipy.stats as st
from _common import IN, TR, TAB, MODELS, TRANSFER_MODELS, LEADS
from tcn_drought_pipeline_v2 import benjamini_hochberg as BH

M = ["Persistence", "Climatology"] + MODELS

# ---- continuous skill (pooled over cells) and DM test on the cell-mean loss differential (Table 3)
rows, dm = [], []
for h in LEADS:
    V = pd.read_csv(IN / "parts" / f"values_indomain__h{h}.csv.gz", parse_dates=["date"])
    for tgt, oc, suf in (("GWETROOT", "obs_gwet", "gwet"), ("SPEI_1", "obs_spei", "spei")):
        o = V[oc].values
        pe = V[f"Persistence__{suf}"].values
        ref = np.mean((o - pe) ** 2)
        for m in M:
            f = V[f"{m}__{suf}"].values
            mse = np.mean((o - f) ** 2)
            rows.append(dict(target=tgt, horizon=h, model=m, RMSE=np.sqrt(mse), SS=1 - mse / ref))
            if m == "Persistence":
                continue
            d = (o - f) ** 2 - (o - pe) ** 2
            dd = pd.Series(d).groupby(V.date.values).mean().sort_index().values
            n = len(dd); dbar = dd.mean(); s = np.sum((dd - dbar) ** 2) / n; lag = max(h - 1, 0)
            for k in range(1, lag + 1):                                   # Newey-West, Bartlett weights
                s += 2 * (1 - k / (lag + 1)) * np.sum((dd[k:] - dbar) * (dd[:-k] - dbar)) / n
            stat = dbar / np.sqrt(s / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)   # HLN correction
            dm.append(dict(target=tgt, horizon=h, model=m, DM=stat, p=2 * st.t.cdf(-abs(stat), n - 1)))
cont = pd.DataFrame(rows)
dmd = pd.DataFrame(dm)
rej, padj = BH(dmd.p.values)
dmd["p_BH"] = padj
dmd["beats"] = (dmd.DM < 0) & rej
dmd["worse"] = (dmd.DM > 0) & rej
cont = cont.merge(dmd, on=["target", "horizon", "model"], how="left")
cont.to_csv(TAB / "T_continuous.csv", index=False)
print(cont.pivot_table(index=["target", "model"], columns="horizon", values="SS").round(2))
print("significant (of", len(dmd), "):", int(rej.sum()))

# ---- rank correlation between squared-error skill and day-based CSI (Section 3.3)
day = pd.read_csv(IN / "warning_day.csv")
d1 = day[(day.logic == "or") & (day.level == "D1") & (day.postproc == "raw")]
rk = []
for h in LEADS:
    a = cont[(cont.target == "GWETROOT") & (cont.horizon == h)].set_index("model").SS
    b = d1[d1.horizon == h].set_index("model").CSI
    r, p = st.spearmanr(a[M], b[M]); rk.append(dict(horizon=h, rho=r, p=p))
a = cont[cont.target == "GWETROOT"].set_index(["model", "horizon"]).SS
b = d1.set_index(["model", "horizon"]).CSI
idx = a.index.intersection(b.index)
r, p = st.spearmanr(a[idx], b[idx]); rk.append(dict(horizon="pooled", rho=r, p=p))
pd.DataFrame(rk).to_csv(TAB / "T_rankcorr.csv", index=False)
print(pd.DataFrame(rk).round(3))

# ---- D1 warning tables with BH control against both references (Tables 4 and 6, Fig. 4, Fig. 6)
for exp, dirp, models in (("indomain", IN, MODELS), ("transfer", TR, TRANSFER_MODELS)):
    for f in ("warning_day", "warning_onset"):
        d = pd.read_csv(dirp / f"{f}.csv")
        x = d[(d.logic == "or") & (d.level == "D1")].copy()
        m = x.model.isin(models)
        for ref in ("vsPersRaw", "vsPersSame"):
            _, padj = BH(x.loc[m, f"p_{ref}"].clip(lower=1e-4).values)
            x.loc[m, f"pBH_{ref}"] = padj
        x.to_csv(TAB / f"T_{exp}_{f}_D1.csv", index=False)
        print(exp, f, "comparisons per reference:", int(m.sum()))
    e = pd.read_csv(dirp / "warning_episode.csv")
    e[e.level == "D1"].to_csv(TAB / f"T_{exp}_episode_D1.csv", index=False)
t = pd.read_csv(IN / "thr_deltas.csv")
t[t.logic == "or"].to_csv(TAB / "T_thr_deltas_indomain.csv", index=False)
print("wrote", TAB)
