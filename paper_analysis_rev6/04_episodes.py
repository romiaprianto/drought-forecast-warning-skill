"""Step 4. Episode-onset verification with uncertainty (Section 3.5, Table 5, Fig. 5, Online Resource 1 S5 and S10).

  a) matches observed and forecast episode onsets cell by cell (D1, OR rule; episodes >= 7 d, gaps < 10 d merged,
     +-7 d tolerance), in-domain and transfer;
  b) cell bootstrap (ten cells resampled, 10,000 replicates) of CSI, POD, FAR, frequency bias and the CSI difference
     from raw and from threshold-shifted persistence, with BH control over the episode comparisons;
  c) sensitivity to single test years: each year 2021-2025 omitted in turn.
Output: tables/episode_percell_D1.csv, T_episode_cellboot.csv, T_episode_vsPersTHR.csv, T_episode_leaveoneyear.csv
"""
import glob
import numpy as np
import pandas as pd
from _common import IN, TR, TAB, MODELS, LEADS
import tcn_drought_rev6 as R
from tcn_drought_pipeline_v2 import benjamini_hochberg as BH

cfg = R.CONFIG_R6
B = 10000
EP = dict(merge_gap=cfg["ep_merge_gap"], min_dur=cfg["ep_min_dur"], tol=cfg["ep_tol"])


def load(exp, h):
    if exp == "indomain":
        V = pd.read_csv(IN / "parts" / f"values_indomain__h{h}.csv.gz", parse_dates=["date"])
        thr = pd.read_csv(IN / "parts" / f"thr_indomain__h{h}.csv")
    else:
        fs = sorted(glob.glob(str(TR / "parts" / f"values_transfer__*__h{h}.csv.gz")))
        V = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in fs])
        thr = pd.concat([pd.read_csv(f.replace("values_", "thr_").replace(".csv.gz", ".csv")) for f in fs])
    return V.sort_values(["cell_id", "date"]).reset_index(drop=True), thr


def onsets(ev):
    return [s for s, _ in R.runs(ev, EP["merge_gap"], EP["min_dur"]) if s > 0]


# ---- a) per-cell matching
out = []
for exp in ("indomain", "transfer"):
    for h in LEADS:
        V, thr = load(exp, h)
        o_ev = R.status(V.obs_spei.values, V.obs_ssi.values, "or") >= 2
        for m in R.methods_in(V):
            for pp in ("raw", "THR"):
                sp, ss = R.variant(V, m, pp)
                dl = R.thr_shift(V, thr, m, "or", 2) if pp == "THR" else np.zeros(len(V))
                f_ev = R.status(sp, ss, "or", dl) >= 2
                for cid, g in V.assign(_o=o_ev, _f=f_ev).groupby("cell_id"):
                    a, b, c, _ = R.match_onsets(onsets(g._o.values), onsets(g._f.values), EP["tol"])
                    out.append(dict(experiment=exp, horizon=h, model=m, postproc=pp, cell_id=cid, hits=a, misses=b, fa=c))
E = pd.DataFrame(out)
E.to_csv(TAB / "episode_percell_D1.csv", index=False)

# ---- b) cell bootstrap against raw persistence (S5 legacy) and threshold-shifted persistence (main text)
csi = lambda H, Mi, F: H / np.maximum(H + Mi + F, 1)
for ref, fname, seed in (("raw", "T_episode_cellboot.csv", 7), ("THR", "T_episode_vsPersTHR.csv", 11)):
    rng = np.random.default_rng(seed)
    res = []
    for (exp, h), g in E.groupby(["experiment", "horizon"]):
        cells = sorted(g.cell_id.unique())
        idx = rng.integers(0, len(cells), (B, len(cells)))

        def arr(m, pp):
            s = g[(g.model == m) & (g.postproc == pp)].set_index("cell_id").loc[cells]
            return s.hits.values, s.misses.values, s.fa.values
        pH, pM, pF = arr("Persistence", ref)
        pb = csi(pH[idx].sum(1), pM[idx].sum(1), pF[idx].sum(1))
        p0 = pH.sum() / max(pH.sum() + pM.sum() + pF.sum(), 1)
        for m in g.model.unique():
            for pp in ("raw", "THR"):
                H, Mi, F = arr(m, pp)
                Hb, Mb, Fb = H[idx].sum(1), Mi[idx].sum(1), F[idx].sum(1)
                cb = csi(Hb, Mb, Fb)
                c0 = H.sum() / max(H.sum() + Mi.sum() + F.sum(), 1)
                pct = lambda x, q: float(np.percentile(x, q))
                res.append(dict(experiment=exp, horizon=h, model=m, postproc=pp, hits=H.sum(), fa=F.sum(),
                                obs=H.sum() + Mi.sum(), fc=H.sum() + F.sum(),
                                POD=H.sum() / max(H.sum() + Mi.sum(), 1),
                                POD_lo=pct(Hb / np.maximum(Hb + Mb, 1), 2.5), POD_hi=pct(Hb / np.maximum(Hb + Mb, 1), 97.5),
                                FAR=F.sum() / max(H.sum() + F.sum(), 1),
                                FAR_lo=pct(Fb / np.maximum(Hb + Fb, 1), 2.5), FAR_hi=pct(Fb / np.maximum(Hb + Fb, 1), 97.5),
                                bias=(H.sum() + F.sum()) / max(H.sum() + Mi.sum(), 1),
                                bias_lo=pct((Hb + Fb) / np.maximum(Hb + Mb, 1), 2.5),
                                bias_hi=pct((Hb + Fb) / np.maximum(Hb + Mb, 1), 97.5),
                                CSI=c0, CSI_lo=pct(cb, 2.5), CSI_hi=pct(cb, 97.5),
                                dCSI=c0 - p0, d_lo=pct(cb - pb, 2.5), d_hi=pct(cb - pb, 97.5), p=float(np.mean(cb - pb <= 0))))
    Rt = pd.DataFrame(res)
    if ref == "THR":
        Rt = Rt.rename(columns={"dCSI": "dCSI_vsPersTHR"})
    for exp in ("indomain", "transfer"):                       # BH over the episode comparisons of each experiment
        m = (Rt.experiment == exp) & Rt.model.isin(MODELS)
        _, padj = BH(Rt.loc[m, "p"].clip(lower=1e-4).values)
        Rt.loc[m, "pBH"] = padj
    Rt.to_csv(TAB / fname, index=False)
x = pd.read_csv(TAB / "T_episode_vsPersTHR.csv")
print(x[(x.horizon == 14) & (x.postproc == "THR")][["experiment", "model", "hits", "fa", "FAR", "CSI", "dCSI_vsPersTHR", "d_lo", "d_hi", "pBH"]].round(3).to_string())

# ---- c) leave-one-test-year-out (in-domain, 7 and 14 d)
rows = []
for h in (7, 14):
    V, thr = load("indomain", h)
    o = R.status(V.obs_spei.values, V.obs_ssi.values, "or") >= 2
    for m, pp in [("Persistence", "THR"), ("Persistence", "raw")] + [(m, "THR") for m in MODELS]:
        sp, ss = R.variant(V, m, pp)
        dl = R.thr_shift(V, thr, m, "or", 2) if pp == "THR" else np.zeros(len(V))
        f = R.status(sp, ss, "or", dl) >= 2
        for cid, g in V.assign(_o=o, _f=f).groupby("cell_id"):
            oo, fo, ys = onsets(g._o.values), onsets(g._f.values), g.date.dt.year.values
            for drop in [None, 2021, 2022, 2023, 2024, 2025]:
                a, b, c, _ = R.match_onsets([s for s in oo if ys[s] != drop], [s for s in fo if ys[s] != drop], EP["tol"])
                rows.append(dict(h=h, model=m, pp=pp, cell=cid, drop=drop if drop else "none", H=a, M=b, F=c))
Y = pd.DataFrame(rows).groupby(["h", "model", "pp", "drop"])[["H", "M", "F"]].sum().reset_index()
Y["obs"] = Y.H + Y.M
Y["CSI"] = Y.H / (Y.H + Y.M + Y.F)
Y["FAR"] = Y.F / (Y.H + Y.F)
P = Y[(Y.model == "Persistence") & (Y.pp == "THR")][["h", "drop", "CSI"]].rename(columns={"CSI": "pCSI"})
Y = Y.merge(P, on=["h", "drop"])
Y["dCSI"] = Y.CSI - Y.pCSI
Y.to_csv(TAB / "T_episode_leaveoneyear.csv", index=False)
print(Y[Y.h == 14].pivot_table(index=["model", "pp"], columns="drop", values="dCSI").round(2))
