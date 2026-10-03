"""
================================================================================
MULTI-CELL DROUGHT FORECASTING  (transfer / spatial generalization)  -- REV4
--------------------------------------------------------------------------------
Pooled, cross-cell analysis (NOT ten independent single-cell models):
  * in-domain: ONE pooled model trained on windows from ALL cells (with static
    covariates), tested on each cell's 2021-2025;                                [2.16]
  * transfer (leave-one-cell-out): pooled model trained on all cells except cell
    k, tested on cell k (never seen);                                            [2.16]
  * static covariates (lat, lon, elev, mean annual P, mean dry-season JJAS P,
    mean soil moisture, aridity P/PET), z-scored across the training cells;
  * a REGIONAL early-warning rule: alert when >= regional_frac of cells cross
    the threshold; POD/FAR/CSI on the regional alert series.                     [2.12]
All physics / index / model / DSS code is imported from tcn_drought_pipeline_v2.

REV4 (release) folds into this module every notebook-level patch used to produce the
published multi-cell results, so that a plain run reproduces them without extra cells:
  * Random Forest with max_features="sqrt" (rf_max_features);
  * transfer Random Forest with 100 trees (transfer_rf_trees); in-domain keeps rf_trees;
  * warning verification at levels D0-D3 (levels) in the in-domain phase;
  * warning verification in the transfer phase (per held-out cell, OR and AND, D0-D3),
    merged into dss_transfer_percell.csv and pooled into dss_transfer_multicell.csv;
  * Climatology continuous metrics in the transfer phase;
  * regional alert scored at several aggregation fractions (regional_fracs), written to
    dss_regional_thresholds.csv; dss_regional.csv keeps the primary fraction (regional_frac);
  * optional export of daily observed and forecast SPEI/SSI per cell (save_values), so any
    further fraction, level or rule can be scored without retraining.

SHARDING (run the long job in short, resumable pieces with IDENTICAL results)
----------------------------------------------------------------------------
The transfer loop is independent per held-out cell, and in-domain is independent
per horizon, so the run can be split and merged exactly:
  CONFIG keys:
    phase         : "all" (default) | "indomain" | "transfer" | "merge"
    only_held     : None | "<cell_id>"  (transfer: run just this held-out cell)
    only_horizons : None | [h, ...]     (restrict horizons for this job)
  Shards write partial CSVs under <out_dir>/parts/ ; phase="merge" concatenates
  them into the final metrics_indomain.csv / metrics_transfer.csv / dss_*.csv /
  skill_map.csv. A plain run (phase="all", no only_*) behaves exactly as before.

Inputs: data/cells.csv + data/cell_<id>.csv (with WS2M) from the downloader.
TensorFlow is optional - RF + baselines alone run the whole thing.
Usage:  python tcn_drought_multicell.py
================================================================================
"""
import os, glob
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tcn_drought_pipeline_v2 as P          # canonical functions (single source of truth)

CONFIG_MC = dict(
    data_dir="data", out_dir="results_mc",
    train_end="2016-12-31", val_end="2020-12-31",         # 3-way split; calib = train+val
    test_start="2021-01-01", test_end="2025-12-31",
    targets=["GWETROOT", "SPEI_1"],
    static_feats=["lat", "lon", "elev", "clim_precip", "clim_dryP", "clim_gwet", "aridity"],
    lookback=90, horizons=[1, 7, 14, 30], seeds=[0, 1, 2],
    epochs=60, batch_size=64, rf_trees=200, spei_clip=3.0,
    spei_scale=30, doy_halfwin=15, wind_var="WS2M", wind_default=2.0,
    transfer_models=["TCN", "RandomForest"],
    map_horizon=14, map_target="GWETROOT",
    regional_frac=0.30,                                   # >=30% of cells -> regional alert
    regional_fracs=(0.20, 0.30, 0.40, 0.50),              # sensitivity of the regional alert
    levels=(1, 2, 3, 4),                                  # verified drought levels D0..D3
    rf_max_features="sqrt",                               # predictors considered per split
    transfer_rf_trees=100,                                # RF trees in the transfer phase
    save_values=True,                                     # daily SPEI/SSI export per cell
    # ---- sharding controls ----
    phase="all", only_held=None, only_horizons=None,
)

def _parts(cfg):
    p = os.path.join(cfg["out_dir"], "parts"); os.makedirs(p, exist_ok=True); return p
def _horizons(cfg):
    return cfg["only_horizons"] if cfg.get("only_horizons") else cfg["horizons"]
def _htag(cfg):
    return "h" + "_".join(str(h) for h in _horizons(cfg))

# ------------------------------- data loading ---------------------------------
def load_cells(cfg):
    meta = pd.read_csv(os.path.join(cfg["data_dir"], "cells.csv"))
    raw = {}
    for _, r in meta.iterrows():
        path = os.path.join(cfg["data_dir"], f"cell_{r['cell_id']}.csv")
        if not os.path.exists(path):
            print(f"  [skip] missing {path}"); continue
        df = pd.read_csv(path, parse_dates=["DATE"]).set_index("DATE").sort_index()
        raw[r["cell_id"]] = df
    return meta, raw

def process_cell(rdf, lat, elev, cfg):
    df = rdf.copy().interpolate("time").ffill().bfill()
    df = P.calculate_pet(df, lat, elev, cfg["wind_var"], cfg["wind_default"])
    df = P.calculate_spei(df, cfg["val_end"], cfg["spei_clip"], cfg["spei_scale"], cfg["doy_halfwin"])
    cm = df.index <= pd.Timestamp(cfg["val_end"])
    df["SSI"] = P.standardized_soil_moisture_index(df["GWETROOT"], cm, cfg["spei_clip"], cfg["doy_halfwin"])
    return df

def static_raw(df, meta_row, cfg):
    cal = df.loc[:cfg["val_end"]]
    p_ann, pet_ann = float(cal["PRECTOTCORR"].mean()), float(cal["PET"].mean())
    jjas = float(cal[cal.index.month.isin([6, 7, 8, 9])]["PRECTOTCORR"].mean())
    return dict(lat=float(meta_row["lat"]), lon=float(meta_row["lon"]), elev=float(meta_row["elev"]),
                clim_precip=p_ann, clim_dryP=jjas, clim_gwet=float(cal["GWETROOT"].mean()),
                aridity=(p_ann / pet_ann if pet_ann else np.nan))

def prepare(cfg):
    meta, raw = load_cells(cfg)
    has_ws = len(raw) > 0 and all("WS2M" in rdf.columns for rdf in raw.values())
    cfg = dict(cfg); cfg["features"] = P.BASE_FEATURES + (["WS2M"] if has_ws else [])
    print(f"Features ({'with' if has_ws else 'no'} WS2M): {cfg['features']}")
    cells = {}
    for cid, rdf in raw.items():
        mr = meta.loc[meta.cell_id == cid].iloc[0]
        df = process_cell(rdf, float(mr["lat"]), float(mr["elev"]), cfg)
        d = P.split_scale(df, cfg)
        cm = df.index <= pd.Timestamp(cfg["val_end"])
        cells[cid] = dict(df=df, d=d, ssi_clim=P.fit_ssi_climatology(df["GWETROOT"], cm, cfg["doy_halfwin"]),
                          static=static_raw(df, mr, cfg), meta=mr)
    return meta, cells, cfg

# --------------------- static covariates (fold-specific z-score) --------------
def zscore_static(cells, train_ids, cfg):
    F = cfg["static_feats"]
    M = np.array([[cells[c]["static"][f] for f in F] for c in train_ids])
    mu, sd = M.mean(0), M.std(0); sd[sd == 0] = 1.0
    return {c: (np.array([cells[c]["static"][f] for f in F]) - mu) / sd for c in cells}

def cell_windows(cells, cid, zstat, h, cfg, split):
    d = cells[cid]["d"]; L = cfg["lookback"]
    Xd, Y = (d["Xtr"], d["Ytr"]) if split == "train" else (d["Xte"], d["Yte"])
    Xfull = np.hstack([Xd, np.tile(zstat[cid], (len(Xd), 1))])
    return P.build_windows(Xfull, Y, L, h)

# ------------------------------ pooled fit/predict ----------------------------
def fit_pool(Xtr, Ytr, cfg, models, n_trees=None):
    fitted = {}
    if "RandomForest" in models:
        fitted["RandomForest"] = []
        for s in cfg["seeds"]:
            rf = RandomForestRegressor(n_estimators=n_trees or cfg["rf_trees"],
                                       max_features=cfg.get("rf_max_features", "sqrt"),
                                       random_state=s, n_jobs=-1)
            rf.fit(Xtr.reshape(len(Xtr), -1), Ytr); fitted["RandomForest"].append(("rf", rf))
    if P.HAS_TF:
        in_shape = (cfg["lookback"], Xtr.shape[2])
        bmap = {"TCN": P.build_tcn, "LSTM": P.build_lstm, "Transformer": P.build_transformer}
        for mname in models:
            if mname not in bmap:
                continue
            fitted[mname] = [("keras", P.train_keras(bmap[mname], Xtr, Ytr, in_shape, cfg, s))
                             for s in cfg["seeds"]]
    return fitted

def predict_pool(fitted, Xte):
    out = {}
    for mname, models in fitted.items():
        ps = [m.predict(Xte.reshape(len(Xte), -1)) if kind == "rf" else m.predict(Xte, verbose=0)
              for kind, m in models]
        out[mname] = np.mean(ps, axis=0)
    return out

# ------------------------------- in-domain run --------------------------------
def _alert_scores(o, f):
    hits = int((f & o).sum()); fa = int((f & ~o).sum()); miss = int((~f & o).sum()); cn = int((~f & ~o).sum())
    return dict(POD=hits / (hits + miss) if (hits + miss) else np.nan,
                FAR=fa / (hits + fa) if (hits + fa) else np.nan,
                CSI=hits / (hits + miss + fa) if (hits + miss + fa) else np.nan,
                base_rate=(hits + miss) / len(o) if len(o) else np.nan,
                hits=hits, false_alarms=fa, misses=miss, correct_neg=cn)

def regional_rows(status_obs, status_fc, h, fracs, levels):
    """Regional alert: fires when >= frac of cells are at or above a level (OR rule).
    status_obs: DataFrame date x cell of integer severity; status_fc: {model: same}."""
    rows = []
    for frac in fracs:
        for level in levels:
            OBS = (status_obs >= level).mean(axis=1) >= frac
            for m, sf in status_fc.items():
                FC = (sf >= level).mean(axis=1) >= frac
                a, b = OBS.align(FC, join="inner")
                rows.append(dict(horizon=h, model=m, logic="or", level=P.LEVEL_NAMES[level], threshold=frac,
                                 **_alert_scores(a.values, b.values)))
    return rows

def run_indomain(cells, cfg):
    ids = list(cells.keys())
    z = zscore_static(cells, ids, cfg)
    gi, si = cfg["targets"].index("GWETROOT"), cfg["targets"].index("SPEI_1")
    learned = ["RandomForest"] + (["TCN", "LSTM", "Transformer"] if P.HAS_TF else [])
    levels = tuple(cfg.get("levels", (1, 2)))
    fracs = tuple(cfg.get("regional_fracs", (cfg["regional_frac"],)))
    rows, dss_pool, reg_rows = [], [], []

    for h in _horizons(cfg):
        Xtr, Ytr = [], []
        for cid in ids:
            Xw, Yw, _ = cell_windows(cells, cid, z, h, cfg, "train"); Xtr.append(Xw); Ytr.append(Yw)
        fitted = fit_pool(np.concatenate(Xtr), np.concatenate(Ytr), cfg, learned)

        models_all = ["Persistence", "Climatology"] + learned
        pool_obs = {"spei": [], "ssi": []}
        pool_fc = {m: {"spei": [], "ssi": []} for m in models_all}
        st_obs, st_fc, values = {}, {m: {} for m in models_all}, []

        for cid in ids:
            Xte, Yte, Ylast = cell_windows(cells, cid, z, h, cfg, "test")
            d = cells[cid]["d"]; inv = d["sY"].inverse_transform
            Yte_o, Ylast_o = inv(Yte), inv(Ylast)
            tdates = d["te_index"][cfg["lookback"] + h - 1: cfg["lookback"] + h - 1 + len(Yte)]
            tdoy = np.array([dt.dayofyear for dt in tdates]); clim = cells[cid]["ssi_clim"]

            preds = {"Persistence": {gi: Ylast_o[:, gi], si: Ylast_o[:, si]},
                     "Climatology": {gi: P.climatology_pred(d["tr_index"], d["tr_raw"][:, gi], tdates),
                                     si: P.climatology_pred(d["tr_index"], d["tr_raw"][:, si], tdates)}}
            scaled = predict_pool(fitted, Xte)
            for m in learned:
                po = inv(scaled[m]); preds[m] = {gi: po[:, gi], si: po[:, si]}

            for m, pr in preds.items():
                for ti, tname in [(gi, "GWETROOT"), (si, "SPEI_1")]:
                    rows.append(dict(cell_id=cid, horizon=h, target=tname, model=m,
                                     **P.all_metrics(Yte_o[:, ti], pr[ti], Ylast_o[:, ti])))

            idx = pd.DatetimeIndex(tdates)
            oss = P.apply_ssi(Yte_o[:, gi], tdoy, clim, cfg["spei_clip"]); osp = Yte_o[:, si]
            pool_obs["spei"].append(osp); pool_obs["ssi"].append(oss)
            st_obs[cid] = pd.Series(P.status_series(osp, oss, "or"), index=idx)
            vrow = pd.DataFrame({"date": idx, "cell_id": cid, "obs_spei": osp, "obs_ssi": oss})
            for m, pr in preds.items():
                fss = P.apply_ssi(pr[gi], tdoy, clim, cfg["spei_clip"]); fsp = pr[si]
                pool_fc[m]["spei"].append(fsp); pool_fc[m]["ssi"].append(fss)
                st_fc[m][cid] = pd.Series(P.status_series(fsp, fss, "or"), index=idx)
                vrow[f"{m}_spei"] = fsp; vrow[f"{m}_ssi"] = fss
            values.append(vrow)

        if cfg.get("save_values", False):
            sdir = os.path.join(cfg["out_dir"], "status"); os.makedirs(sdir, exist_ok=True)
            pd.concat(values).to_csv(os.path.join(sdir, f"values__h{h}.csv.gz"), index=False)

        OS, OG = np.concatenate(pool_obs["spei"]), np.concatenate(pool_obs["ssi"])
        for m in models_all:
            FS, FG = np.concatenate(pool_fc[m]["spei"]), np.concatenate(pool_fc[m]["ssi"])
            for logic in ("or", "and"):
                for level in levels:
                    dss_pool.append(dict(horizon=h, model=m, logic=logic, level=P.LEVEL_NAMES[level],
                                         **P.validate_forecast_dss(OS, OG, FS, FG, logic, level)))
        reg_rows += regional_rows(pd.DataFrame(st_obs), {m: pd.DataFrame(st_fc[m]) for m in models_all},
                                  h, fracs, levels)
    return pd.DataFrame(rows), pd.DataFrame(dss_pool), pd.DataFrame(reg_rows)

# ----------------------- leave-one-cell-out transfer --------------------------
def run_transfer(cells, cfg):
    """Leave-one-cell-out transfer. Returns (continuous metrics, per-cell warning verification)."""
    ids = list(cells.keys())
    held_list = [cfg["only_held"]] if cfg.get("only_held") else ids
    for hc in held_list:
        if hc not in ids:
            raise ValueError(f"only_held='{hc}' not among cells {ids}")
    gi, si = cfg["targets"].index("GWETROOT"), cfg["targets"].index("SPEI_1")
    models = cfg["transfer_models"]; levels = tuple(cfg.get("levels", (1, 2)))
    rows, dss_rows = [], []
    for held in held_list:
        train_ids = [c for c in ids if c != held]
        z = zscore_static(cells, train_ids, cfg)
        for h in _horizons(cfg):
            Xtr, Ytr = [], []
            for cid in train_ids:
                Xw, Yw, _ = cell_windows(cells, cid, z, h, cfg, "train"); Xtr.append(Xw); Ytr.append(Yw)
            fitted = fit_pool(np.concatenate(Xtr), np.concatenate(Ytr), cfg, models,
                              n_trees=cfg.get("transfer_rf_trees"))
            Xte, Yte, Ylast = cell_windows(cells, held, z, h, cfg, "test")
            d = cells[held]["d"]; inv = d["sY"].inverse_transform
            Yte_o, Ylast_o = inv(Yte), inv(Ylast)
            scaled = predict_pool(fitted, Xte)
            tdates = d["te_index"][cfg["lookback"] + h - 1: cfg["lookback"] + h - 1 + len(Yte)]
            tdoy = np.array([dt.dayofyear for dt in tdates]); clim = cells[held]["ssi_clim"]

            preds = {}
            for m in models:
                if m in scaled:
                    po = inv(scaled[m]); preds[m] = {gi: po[:, gi], si: po[:, si]}
            preds["Persistence"] = {gi: Ylast_o[:, gi], si: Ylast_o[:, si]}
            preds["Climatology"] = {gi: P.climatology_pred(d["tr_index"], d["tr_raw"][:, gi], tdates),
                                    si: P.climatology_pred(d["tr_index"], d["tr_raw"][:, si], tdates)}
            for m, pr in preds.items():
                for ti, tname in [(gi, "GWETROOT"), (si, "SPEI_1")]:
                    rows.append(dict(held_out=held, horizon=h, target=tname, model=m, regime="transfer",
                                     **P.all_metrics(Yte_o[:, ti], pr[ti], Ylast_o[:, ti])))
            oss = P.apply_ssi(Yte_o[:, gi], tdoy, clim, cfg["spei_clip"]); osp = Yte_o[:, si]
            for m, pr in preds.items():
                fss = P.apply_ssi(pr[gi], tdoy, clim, cfg["spei_clip"]); fsp = pr[si]
                for logic in ("or", "and"):
                    for level in levels:
                        dss_rows.append(dict(held_out=held, horizon=h, model=m, logic=logic,
                                             level=P.LEVEL_NAMES[level], regime="transfer",
                                             **P.validate_forecast_dss(osp, oss, fsp, fss, logic, level)))
    return pd.DataFrame(rows), pd.DataFrame(dss_rows)

def pool_transfer_dss(per_cell):
    """Pool per-held-out-cell verification by summing contingency counts (not averaging ratios)."""
    k = ["horizon", "model", "logic", "level"]
    p = per_cell.groupby(k)[["hits", "false_alarms", "misses", "correct_neg"]].sum().reset_index()
    n = p.hits + p.false_alarms + p.misses + p.correct_neg
    p["POD"] = p.hits / (p.hits + p.misses); p["FAR"] = p.false_alarms / (p.hits + p.false_alarms)
    p["CSI"] = p.hits / (p.hits + p.misses + p.false_alarms); p["base_rate"] = (p.hits + p.misses) / n
    p["freq_bias"] = (p.hits + p.false_alarms) / (p.hits + p.misses)
    return p

# ------------------------------- skill map / merge ----------------------------
def build_skill_map(meta, ind, tr, cfg):
    primary = "TCN" if P.HAS_TF else "RandomForest"
    mh = cfg["map_horizon"] if cfg["map_horizon"] in cfg["horizons"] else cfg["horizons"][0]
    mt = cfg["map_target"]
    i_s = (ind[(ind.horizon == mh) & (ind.target == mt) & (ind.model == primary)]
           [["cell_id", "SS_vs_persist"]].rename(columns={"SS_vs_persist": "SS_indomain"}))
    t_s = (tr[(tr.horizon == mh) & (tr.target == mt) & (tr.model == primary)]
           [["held_out", "SS_vs_persist"]].rename(columns={"held_out": "cell_id",
                                                            "SS_vs_persist": "SS_transfer"}))
    return (meta[["cell_id", "name", "lat", "lon"]].merge(i_s, on="cell_id", how="left")
            .merge(t_s, on="cell_id", how="left"))

def _concat(prefix, cfg):
    fs = sorted(glob.glob(os.path.join(cfg["out_dir"], "parts", prefix + "*.csv")))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

def merge_shards(cfg):
    meta = pd.read_csv(os.path.join(cfg["data_dir"], "cells.csv"))
    ind = _concat("indomain__", cfg)
    if ind.empty and os.path.exists(f"{cfg['out_dir']}/metrics_indomain.csv"):
        ind = pd.read_csv(f"{cfg['out_dir']}/metrics_indomain.csv")
    dss = _concat("dssmc__", cfg)
    reg = _concat("dssreg__", cfg)
    dtr = _concat("dsstr__", cfg)
    tr = _concat("transfer__", cfg)
    if tr.empty and os.path.exists(f"{cfg['out_dir']}/metrics_transfer.csv"):
        tr = pd.read_csv(f"{cfg['out_dir']}/metrics_transfer.csv")
    # de-duplicate (in case shards overlap)
    if not ind.empty: ind = ind.drop_duplicates(["cell_id", "horizon", "target", "model"])
    if not tr.empty:  tr  = tr.drop_duplicates(["held_out", "horizon", "target", "model"])
    if not dss.empty: dss = dss.drop_duplicates(["horizon", "model", "logic", "level"])
    if not reg.empty:
        if "threshold" not in reg.columns: reg["threshold"] = cfg["regional_frac"]
        reg = reg.drop_duplicates(["horizon", "model", "level", "threshold"])
    if not dtr.empty: dtr = dtr.drop_duplicates(["held_out", "horizon", "model", "logic", "level"])

    od = cfg["out_dir"]
    if not ind.empty: ind.to_csv(f"{od}/metrics_indomain.csv", index=False)
    if not dss.empty: dss.to_csv(f"{od}/dss_multicell.csv", index=False)
    if not reg.empty:
        reg[np.isclose(reg.threshold, cfg["regional_frac"])].to_csv(f"{od}/dss_regional.csv", index=False)
        reg.to_csv(f"{od}/dss_regional_thresholds.csv", index=False)
    if not dtr.empty:
        dtr.to_csv(f"{od}/dss_transfer_percell.csv", index=False)
        pool_transfer_dss(dtr).to_csv(f"{od}/dss_transfer_multicell.csv", index=False)
    if not tr.empty:  tr.to_csv(f"{od}/metrics_transfer.csv", index=False)
    if not ind.empty and not tr.empty:
        build_skill_map(meta, ind, tr, cfg).to_csv(f"{od}/skill_map.csv", index=False)
    print(f"[merge] indomain rows={len(ind)} transfer rows={len(tr)} dss={len(dss)} regional={len(reg)} "
          f"transfer-dss={len(dtr)} "
          f"-> {od}/ (metrics_indomain, metrics_transfer, dss_multicell, dss_regional, skill_map)")

# ------------------------------------ main ------------------------------------
def main(cfg=CONFIG_MC):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    phase = cfg.get("phase", "all")
    sharding = (cfg.get("only_held") is not None) or (cfg.get("only_horizons") is not None)

    if phase == "merge":
        merge_shards(cfg); return

    meta, cells, cfg = prepare(cfg)
    print(f"Cells loaded: {list(cells.keys())}  |  TensorFlow: {P.HAS_TF}  |  phase={phase}  "
          f"only_held={cfg.get('only_held')}  horizons={_horizons(cfg)}")

    if phase in ("all", "indomain"):
        ind, dss, reg = run_indomain(cells, cfg)
        if sharding:
            tag = _htag(cfg); pd_ = _parts(cfg)
            ind.to_csv(f"{pd_}/indomain__{tag}.csv", index=False)
            dss.to_csv(f"{pd_}/dssmc__{tag}.csv", index=False)
            reg.to_csv(f"{pd_}/dssreg__{tag}.csv", index=False)
            print(f"[indomain shard] wrote parts/*__{tag}.csv")
        else:
            ind.to_csv(f"{cfg['out_dir']}/metrics_indomain.csv", index=False)
            dss.to_csv(f"{cfg['out_dir']}/dss_multicell.csv", index=False)
            reg[np.isclose(reg.threshold, cfg["regional_frac"])].to_csv(f"{cfg['out_dir']}/dss_regional.csv", index=False)
            reg.to_csv(f"{cfg['out_dir']}/dss_regional_thresholds.csv", index=False)
            print("\n=== POOLED DSS (OR, D1) ===")
            print(dss[(dss.logic == "or") & (dss.level == "D1")]
                  [["horizon", "model", "POD", "FAR", "CSI", "base_rate", "hits"]].round(3).to_string(index=False))
            print(f"\n=== REGIONAL >= {int(cfg['regional_frac']*100)}% ALERT (OR, D1) ===")
            print(reg[(reg.level == "D1") & np.isclose(reg.threshold, cfg["regional_frac"])]
                  [["horizon", "model", "POD", "FAR", "CSI", "base_rate", "hits"]]
                  .round(3).to_string(index=False))

    if phase in ("all", "transfer"):
        tr, dtr = run_transfer(cells, cfg)
        if sharding:
            tag = (cfg.get("only_held") or "allcells") + "__" + _htag(cfg); pd_ = _parts(cfg)
            tr.to_csv(f"{pd_}/transfer__{tag}.csv", index=False)
            dtr.to_csv(f"{pd_}/dsstr__{tag}.csv", index=False)
            print(f"[transfer shard] wrote parts/transfer__{tag}.csv and parts/dsstr__{tag}.csv")
        else:
            tr.to_csv(f"{cfg['out_dir']}/metrics_transfer.csv", index=False)
            dtr.to_csv(f"{cfg['out_dir']}/dss_transfer_percell.csv", index=False)
            pool_transfer_dss(dtr).to_csv(f"{cfg['out_dir']}/dss_transfer_multicell.csv", index=False)

    if phase == "all":
        smap = build_skill_map(meta, ind, tr, cfg)
        smap.to_csv(f"{cfg['out_dir']}/skill_map.csv", index=False)
        print(f"\n=== SKILL MAP (in-domain vs LOCO transfer) ===\n{smap.round(3).to_string(index=False)}")
        print(f"\nWrote canonical CSVs -> {cfg['out_dir']}/")
    elif sharding:
        print("Shard done. After all shards: run with phase='merge' to assemble final CSVs.")

if __name__ == "__main__":
    main()
