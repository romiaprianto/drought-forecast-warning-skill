"""Step 5. Robustness checks (Section 3.8; Online Resource 1, Tables S8a, S8b and S9).

  a) fraction of cell-days at or below D1 by year, 1996-2025, recomputed from the NASA POWER input data;
  b) day, onset and episode verification split into the dry years 2023-2024 and the wet years 2021, 2022 and 2025;
  c) symmetric extremal dependence index (SEDI; Ferro and Stephenson 2011) and relative economic value of the
     deterministic alerts (Richardson 2000) for the in-domain D1 alerts.
Output: tables/T_baserate_year.csv, T_byyears.csv, T_sedi_rev.csv
"""
import numpy as np
import pandas as pd
from _common import IN, TAB, DATA, MODELS
import tcn_drought_rev6 as R
import tcn_drought_multicell as Mc

cfg = dict(R.CONFIG_R6, data_dir=str(DATA))

# ---- a) base rates by year
meta, raw = Mc.load_cells(cfg)
rows = []
for cid, rdf in raw.items():
    mr = meta.loc[meta.cell_id == cid].iloc[0]
    df = Mc.process_cell(rdf, float(mr.lat), float(mr.elev), cfg)
    spei = df[[c for c in df.columns if c.startswith("SPEI")][0]].values
    rows.append(pd.DataFrame({"cell": cid, "year": df.index.year,
                              "d1": R.status(spei, df["SSI"].values, "or") >= 2,
                              "sp": R.sev(spei) >= 2, "ss": R.sev(df["SSI"].values) >= 2}))
Yb = pd.concat(rows)
Yb.groupby("year")[["d1", "sp", "ss"]].mean().to_csv(TAB / "T_baserate_year.csv")
for a, b in ((1996, 2020), (2017, 2020), (2021, 2025)):
    print(a, b, Yb[(Yb.year >= a) & (Yb.year <= b)][["d1", "sp", "ss"]].mean().round(3).to_dict())


# ---- b) and c)
def csi(H, F, M):
    return H / (H + F + M) if H + F + M > 0 else np.nan


def sedi(H, F, M, n):
    CN = n - H - F - M; Hr = H / (H + M); Fr = F / (F + CN)
    if Hr in (0, 1) or Fr in (0, 1):
        return np.nan
    return ((np.log(Fr) - np.log(Hr) - np.log(1 - Fr) + np.log(1 - Hr))
            / (np.log(Fr) + np.log(Hr) + np.log(1 - Fr) + np.log(1 - Hr)))


def rev(H, F, M, n, a):
    """Relative economic value for cost-loss ratio a (static deterministic alerts)."""
    CN = n - H - F - M; s = (H + M) / n; Hr = H / (H + M); Fr = F / (F + CN)
    e_clim = min(a, s); e_perf = s * a; e_fc = Fr * (1 - s) * a + Hr * s * a + (1 - Hr) * s
    return (e_clim - e_fc) / (e_clim - e_perf)


yr, sc = [], []
for h in (7, 14, 30):
    V = pd.read_csv(IN / "parts" / f"values_indomain__h{h}.csv.gz", parse_dates=["date"])
    V = V.sort_values(["cell_id", "date"]).reset_index(drop=True)
    thr = pd.read_csv(IN / "parts" / f"thr_indomain__h{h}.csv")
    o = R.status(V.obs_spei.values, V.obs_ssi.values, "or") >= 2
    below = R.status(V.iss_spei.values, V.iss_ssi.values, "or") < 2
    grp = np.where(np.isin(V.date.dt.year.values, [2023, 2024]), "2023-24", "2021-22,25")
    for m, pp in [("Persistence", "raw"), ("Persistence", "THR")] + [(m, "THR") for m in MODELS]:
        sp, ss = R.variant(V, m, pp)
        dl = R.thr_shift(V, thr, m, "or", 2) if pp == "THR" else np.zeros(len(V))
        f = R.status(sp, ss, "or", dl) >= 2
        for scheme, mask in (("day", np.ones(len(V), bool)), ("onset", below)):
            H = (o & f & mask).sum(); F = (~o & f & mask).sum(); M_ = (o & ~f & mask).sum(); n = mask.sum()
            r = dict(h=h, model=m, pp=pp, scheme=scheme, CSI=csi(H, F, M_), SEDI=sedi(H, F, M_, n), base=(o & mask).sum() / n)
            for a in (0.05, 0.1, 0.2, 0.3, 0.5):
                r[f"V{a}"] = rev(H, F, M_, n, a)
            r["Vmax"] = max(rev(H, F, M_, n, a) for a in np.linspace(0.01, 0.99, 99))
            sc.append(r)
            for g in ("2023-24", "2021-22,25"):
                k = mask & (grp == g)
                H = (o & f & k).sum(); F = (~o & f & k).sum(); M_ = (o & ~f & k).sum()
                yr.append(dict(h=h, model=m, pp=pp, scheme=scheme, years=g, events=int((o & k).sum()), n=int(k.sum()),
                               H=int(H), F=int(F), M=int(M_), CSI=csi(H, F, M_)))
        for cid, g in V.assign(_o=o, _f=f).groupby("cell_id"):
            oo = [s for s, _ in R.runs(g._o.values, cfg["ep_merge_gap"], cfg["ep_min_dur"]) if s > 0]
            fo = [s for s, _ in R.runs(g._f.values, cfg["ep_merge_gap"], cfg["ep_min_dur"]) if s > 0]
            ys = g.date.dt.year.values
            for gname, dry in (("2023-24", True), ("2021-22,25", False)):
                keep = lambda s: (ys[s] in (2023, 2024)) == dry
                a, b, c, _ = R.match_onsets([s for s in oo if keep(s)], [s for s in fo if keep(s)], cfg["ep_tol"])
                yr.append(dict(h=h, model=m, pp=pp, scheme="episode", years=gname, cell=cid, H=a, M=b, F=c))
Y = pd.DataFrame(yr)
Ye = Y[Y.scheme == "episode"].groupby(["h", "model", "pp", "scheme", "years"])[["H", "M", "F"]].sum().reset_index()
Ye["events"] = Ye.H + Ye.M
Ye["CSI"] = Ye.H / (Ye.H + Ye.M + Ye.F)
pd.concat([Y[Y.scheme != "episode"], Ye]).to_csv(TAB / "T_byyears.csv", index=False)
pd.DataFrame(sc).to_csv(TAB / "T_sedi_rev.csv", index=False)
print("wrote T_baserate_year.csv, T_byyears.csv, T_sedi_rev.csv")
