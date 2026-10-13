"""tcn_drought_rev6.py  --  REV6: from forecast skill to warning skill (multi-cell)
================================================================================
REV6 adds four things to the REV5 multi-cell experiment. REV5 code and results are
left untouched (this module imports them); everything new is written to results_rev6/.

 1. Post-processing that restores the forecast distribution, fitted out of sample.
    A development model is trained on 1996-2016 and forecasts the validation years
    2017-2020; the post-processing is fitted on those validation forecasts and then
    applied to the test forecasts (2021-2025) of the final model (trained on 1996-2020,
    exactly as in REV5). The test period is never used for fitting.
      raw : the model output (as in REV5)
      QM  : empirical quantile mapping of each index (SPEI, SSI) onto its observed
            validation distribution
      VI  : variance inflation (mean and standard deviation matched to observation)
      THR : alert threshold shifted by a delta chosen on validation to maximise CSI
            (per model, lead, logic and level); the forecast itself is unchanged
    The same treatments are applied to persistence and climatology, so every method
    is compared on equal terms.

 2. Onset verification, where persistence cannot simply carry today's drought forward.
      onset (conditional): only issue days on which the observed state is BELOW the
            alert level; the event is that the level is reached at the target day.
            (Persistence can still alert here, but only through the seasonal change
            of the soil-moisture climatology between issue and target day.)
      episode: drought episodes (runs >= ep_min_dur days, gaps < ep_merge_gap merged);
            an observed onset is a hit if a forecast onset lies within +/- ep_tol days.
 3. Moving-block bootstrap (circular, blocks of boot_block days, same dates for every
    cell and method, i.e. paired) for CSI and for the CSI difference against persistence.
 4. Transformer with sinusoidal positional encoding and last-step + mean pooling
    (REV5 attention had no positional information). Reported as "Transformer".

Phases (CONFIG_R6["phase"]):
  "indomain" : train dev + final pooled models per lead, save values (resumable by lead)
  "transfer" : leave-one-cell-out for transfer_models (resumable by held-out cell / lead)
  "analyze"  : read the saved values and write every verification table (CPU, minutes)
  "all"      : indomain + transfer + analyze
Usage:  python tcn_drought_rev6.py      (edit CONFIG_R6 or pass a dict to main())
================================================================================
"""
import os, glob, json, gc
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tcn_drought_pipeline_v2 as P
import tcn_drought_multicell as M

CONFIG_R6 = dict(
    M.CONFIG_MC,
    out_dir="results_rev6",
    val_start="2017-01-01",                 # validation years used to fit the post-processing
    learned=["RandomForest", "TCN", "LSTM", "Transformer"],
    transfer_models=["TCN", "RandomForest"],
    postproc=("raw", "QM", "VI", "THR"),
    thr_grid=tuple(np.round(np.arange(-0.50, 1.501, 0.05), 2)),
    qm_quantiles=201,
    boot_B=1000, boot_block=60, boot_seed=2026,
    ep_min_dur=7, ep_merge_gap=10, ep_tol=7,
    only_cells=None,                        # testing only: restrict to a list of cell ids
    phase="indomain", only_held=None, only_horizons=None,
)

LEVEL_THR = np.array([-0.50, -0.84, -1.28, -1.64])           # USDM D0..D3, as P.DROUGHT_LEVELS
LNAME = {1: "D0", 2: "D1", 3: "D2", 4: "D3"}

# =============================== Transformer + PE ===============================
if P.HAS_TF:
    class SinusoidalPE(P.layers.Layer):
        """Adds the fixed sinusoidal position code of Vaswani et al. (2017)."""
        def build(self, input_shape):
            L, d = int(input_shape[1]), int(input_shape[2])
            pos = np.arange(L)[:, None]; i = np.arange(d)[None, :]
            ang = pos / np.power(10000.0, (2 * (i // 2)) / d)
            pe = np.where(i % 2 == 0, np.sin(ang), np.cos(ang)).astype("float32")
            self.pe = P.tf.constant(pe[None, :, :])
            super().build(input_shape)
        def call(self, x):
            return x + self.pe

    def build_transformer_pe(in_shape, out=2, d_model=64, heads=4):
        """REV5 encoder block (same width, heads, normalisation) + positional encoding;
        pooled by concatenating the last time step with the mean over the window."""
        L = P.layers
        inp = L.Input(shape=in_shape)
        x = SinusoidalPE()(L.Dense(d_model)(inp))
        a = L.MultiHeadAttention(num_heads=heads, key_dim=d_model)(x, x)
        x = L.LayerNormalization()(x + a)
        f = L.Dense(d_model, activation="relu")(x)
        x = L.LayerNormalization()(x + f)
        last = L.Lambda(lambda z: z[:, -1, :])(x)
        x = L.Concatenate()([last, L.GlobalAveragePooling1D()(x)])
        m = P.Model(inp, L.Dense(out)(x)); m.compile(optimizer="adam", loss="mse"); return m

    BUILDERS = {"TCN": P.build_tcn, "LSTM": P.build_lstm, "Transformer": build_transformer_pe}
else:
    BUILDERS = {}

# ================================ data / windows ================================
def prepare(cfg):
    meta, cells, cfg = M.prepare(cfg)
    if cfg.get("only_cells"):
        cells = {c: v for c, v in cells.items() if c in cfg["only_cells"]}
        meta = meta[meta.cell_id.isin(cfg["only_cells"])]
    dev = dict(cfg, val_end=cfg["train_end"], test_start=cfg["val_start"], test_end=cfg["val_end"])
    for cid in cells:                       # development split: scaler on <=2016, "test" = 2017-2020
        cells[cid]["d_dev"] = P.split_scale(cells[cid]["df"], dev)
    return meta, cells, cfg

def windows(d, zvec, h, cfg, split):
    Xd, Y = (d["Xtr"], d["Ytr"]) if split == "train" else (d["Xte"], d["Yte"])
    return P.build_windows(np.hstack([Xd, np.tile(zvec, (len(Xd), 1))]), Y, cfg["lookback"], h)

def fit_models(cells, ids, z, h, cfg, models, dkey, n_trees=None):
    """Pooled fit on the 'train' part of cells[cid][dkey] (dkey='d' final, 'd_dev' development).
    Neural models stop early on the last val_frac of each cell (REV5 rule)."""
    Xs, Ys = [], []
    for cid in ids:
        Xw, Yw, _ = windows(cells[cid][dkey], z[cid], h, cfg, "train"); Xs.append(Xw); Ys.append(Yw)
    fitted = {}
    if "RandomForest" in models:
        Xtr, Ytr = np.concatenate(Xs), np.concatenate(Ys)
        Xflat = Xtr.reshape(len(Xtr), -1)
        fitted["RandomForest"] = []
        for s in cfg["seeds"]:
            rf = RandomForestRegressor(n_estimators=n_trees or cfg["rf_trees"],
                                       max_features=cfg.get("rf_max_features", "sqrt"),
                                       random_state=s, n_jobs=-1)
            rf.fit(Xflat, Ytr); fitted["RandomForest"].append(("rf", rf))
        del Xtr, Xflat
    neural = [m for m in models if m in BUILDERS]
    if neural:
        Xf, Yf, Xv, Yv = M.split_per_cell(Xs, Ys, cfg["val_frac"])
        shape = (cfg["lookback"], Xf.shape[2])
        for m in neural:
            fitted[m] = [("keras", M.train_keras_val(BUILDERS[m], Xf, Yf, Xv, Yv, shape, cfg, s))
                         for s in cfg["seeds"]]
    return fitted

def forecast_frame(cells, cid, dkey, zvec, h, fitted, cfg):
    """One row per target day: observed indices at the target and issue day, and every
    method's forecast (GWETROOT, SPEI, SSI)."""
    gi, si = cfg["targets"].index("GWETROOT"), cfg["targets"].index("SPEI_1")
    d = cells[cid][dkey]; clim = cells[cid]["ssi_clim"]; L = cfg["lookback"]
    Xte, Yte, Ylast = windows(d, zvec, h, cfg, "test")
    inv = d["sY"].inverse_transform; Yo, Yl = inv(Yte), inv(Ylast)
    tdates = pd.DatetimeIndex(d["te_index"][L + h - 1: L + h - 1 + len(Yte)])
    idates = pd.DatetimeIndex(d["te_index"][L - 1: L - 1 + len(Yte)])
    tdoy, idoy = tdates.dayofyear.values, idates.dayofyear.values
    ssi = lambda g, doy: P.apply_ssi(g, doy, clim, cfg["spei_clip"])
    V = pd.DataFrame({"cell_id": cid, "date": tdates, "issue_date": idates,
                      "obs_gwet": Yo[:, gi], "obs_spei": Yo[:, si], "obs_ssi": ssi(Yo[:, gi], tdoy),
                      "iss_spei": Yl[:, si], "iss_ssi": ssi(Yl[:, gi], idoy)})
    fc = {"Persistence": Yl,
          "Climatology": np.column_stack([P.climatology_pred(d["tr_index"], d["tr_raw"][:, j], tdates)
                                          for j in range(Yo.shape[1])])}
    for m, mods in fitted.items():
        ps = [mm.predict(Xte.reshape(len(Xte), -1)) if kind == "rf" else mm.predict(Xte, verbose=0)
              for kind, mm in mods]
        fc[m] = inv(np.mean(ps, axis=0))
    for m, F in fc.items():
        V[f"{m}__gwet"] = F[:, gi]; V[f"{m}__spei"] = F[:, si]; V[f"{m}__ssi"] = ssi(F[:, gi], tdoy)
    return V

def methods_in(V):
    return [c[:-6] for c in V.columns if c.endswith("__spei")]

# ============================== post-processing ================================
def sev(x, shift=0.0):
    """Severity 0..4 (None, D0..D3) as P.severity, vectorised; NaN -> 0. `shift` (scalar or
    one value per row) moves every threshold up by delta (alert earlier) for THR."""
    x = np.asarray(x, float); shift = np.asarray(shift, float)
    thr = LEVEL_THR[None, :] + (shift[:, None] if shift.ndim else shift)
    return (x[:, None] <= thr).sum(1)

def status(spei, ssi, logic, shift=0.0):
    a, b = sev(spei, shift), sev(ssi, shift)
    return np.maximum(a, b) if logic == "or" else np.minimum(a, b)

def fit_qm(f, o, nq):
    q = np.linspace(0, 1, nq); ok = np.isfinite(f) & np.isfinite(o)
    fq, oq = np.quantile(f[ok], q), np.quantile(o[ok], q)
    fq = np.maximum.accumulate(fq) + np.arange(nq) * 1e-9               # strictly increasing
    return fq, oq

def apply_qm(x, par, clip):
    fq, oq = par; x = np.asarray(x, float); y = np.interp(x, fq, oq)
    lo, hi = x < fq[0], x > fq[-1]
    y[lo] = oq[0] + (x[lo] - fq[0]); y[hi] = oq[-1] + (x[hi] - fq[-1])
    return np.clip(y, -clip, clip)

def fit_vi(f, o):
    ok = np.isfinite(f) & np.isfinite(o)
    return float(f[ok].mean()), float(f[ok].std()), float(o[ok].mean()), float(o[ok].std())

def apply_vi(x, par, clip):
    mf, sf, mo, so = par
    return np.clip(mo + (np.asarray(x, float) - mf) * (so / sf if sf > 1e-9 else 1.0), -clip, clip)

def csi_counts(o, f):
    h = int((o & f).sum()); fa = int((~o & f).sum()); mi = int((o & ~f).sum())
    return h, fa, mi

def fit_postproc(Vval, cfg):
    """Fit QM / VI per method and index, and the THR delta per method, logic and level,
    on the validation forecasts of the development model. Returns (params, thr table)."""
    params, thr_rows = {}, []
    ostat = {lg: status(Vval.obs_spei.values, Vval.obs_ssi.values, lg) for lg in ("or", "and")}
    for m in methods_in(Vval):
        params[m] = {}
        for ix in ("spei", "ssi"):
            f, o = Vval[f"{m}__{ix}"].values, Vval[f"obs_{ix}"].values
            params[m][ix] = dict(QM=fit_qm(f, o, cfg["qm_quantiles"]), VI=fit_vi(f, o))
        for lg in ("or", "and"):
            for lev in cfg["levels"]:
                o = ostat[lg] >= lev; best = (-1.0, 0.0)
                for dl in cfg["thr_grid"]:
                    f = status(Vval[f"{m}__spei"].values, Vval[f"{m}__ssi"].values, lg, dl) >= lev
                    h, fa, mi = csi_counts(o, f); c = h / (h + fa + mi) if (h + fa + mi) else 0.0
                    if c > best[0] + 1e-12 or (abs(c - best[0]) <= 1e-12 and abs(dl) < abs(best[1])):
                        best = (c, float(dl))
                thr_rows.append(dict(model=m, logic=lg, level=LNAME[lev], delta=best[1], CSI_val=best[0]))
    return params, pd.DataFrame(thr_rows)

def add_postproc(V, params, cfg):
    clip = cfg["spei_clip"]
    for m, pm in params.items():
        for ix in ("spei", "ssi"):
            V[f"{m}__{ix}__QM"] = apply_qm(V[f"{m}__{ix}"].values, pm[ix]["QM"], clip)
            V[f"{m}__{ix}__VI"] = apply_vi(V[f"{m}__{ix}"].values, pm[ix]["VI"], clip)
    return V

def params_table(params):
    rows = []
    for m, pm in params.items():
        for ix in ("spei", "ssi"):
            mf, sf, mo, so = pm[ix]["VI"]
            rows.append(dict(model=m, index=ix, mean_fc=mf, sd_fc=sf, mean_obs=mo, sd_obs=so,
                             qm_fc_q05=float(np.interp(0.05, np.linspace(0, 1, len(pm[ix]["QM"][0])), pm[ix]["QM"][0])),
                             qm_obs_q05=float(np.interp(0.05, np.linspace(0, 1, len(pm[ix]["QM"][1])), pm[ix]["QM"][1]))))
    return pd.DataFrame(rows)

# ================================= training ====================================
def _free():
    gc.collect()
    if P.HAS_TF:
        P.tf.keras.backend.clear_session()

def _parts(cfg):
    p = os.path.join(cfg["out_dir"], "parts"); os.makedirs(p, exist_ok=True); return p

def _horizons(cfg):
    return cfg["only_horizons"] if cfg.get("only_horizons") else cfg["horizons"]

def continuous_rows(V, key):
    rows = []
    for cid, g in V.groupby("cell_id"):
        for m in methods_in(V):
            for tname, oc, pc, rc in [("GWETROOT", "obs_gwet", f"{m}__gwet", "Persistence__gwet"),
                                      ("SPEI_1", "obs_spei", f"{m}__spei", "Persistence__spei")]:
                rows.append(dict(**key, cell_id=cid, target=tname, model=m,
                                 **P.all_metrics(g[oc].values, g[pc].values, g[rc].values)))
    return rows

def run_indomain(cells, cfg):
    ids = list(cells.keys()); z = M.zscore_static(cells, ids, cfg)
    learned = [m for m in cfg["learned"] if m == "RandomForest" or m in BUILDERS]
    for h in _horizons(cfg):
        vfile = f"{_parts(cfg)}/values_indomain__h{h}.csv.gz"
        if os.path.exists(vfile):
            print(f"[indomain] h={h} already done"); continue
        fd = fit_models(cells, ids, z, h, cfg, learned, "d_dev")                     # 1996-2016
        Vval = pd.concat([forecast_frame(cells, c, "d_dev", z[c], h, fd, cfg) for c in ids])
        del fd; _free()
        params, thr = fit_postproc(Vval, cfg)
        ff = fit_models(cells, ids, z, h, cfg, learned, "d")                          # 1996-2020
        V = pd.concat([forecast_frame(cells, c, "d", z[c], h, ff, cfg) for c in ids])
        del ff; _free()
        V = add_postproc(V, params, cfg)
        key = dict(experiment="indomain", horizon=h)
        pd.DataFrame(continuous_rows(V, key)).to_csv(f"{_parts(cfg)}/cont_indomain__h{h}.csv", index=False)
        thr.assign(**key).to_csv(f"{_parts(cfg)}/thr_indomain__h{h}.csv", index=False)
        params_table(params).assign(**key).to_csv(f"{_parts(cfg)}/pp_indomain__h{h}.csv", index=False)
        add_postproc(Vval, params, cfg).to_csv(f"{_parts(cfg)}/valvalues_indomain__h{h}.csv.gz", index=False)
        V.to_csv(vfile, index=False)
        print(f"[indomain] h={h} done: {len(V)} target days x {V.cell_id.nunique()} cells", flush=True)

def run_transfer(cells, cfg):
    ids = list(cells.keys())
    held_list = [cfg["only_held"]] if cfg.get("only_held") else ids
    models = [m for m in cfg["transfer_models"] if m == "RandomForest" or m in BUILDERS]
    for held in held_list:
        train_ids = [c for c in ids if c != held]; z = M.zscore_static(cells, train_ids, cfg)
        for h in _horizons(cfg):
            vfile = f"{_parts(cfg)}/values_transfer__{held}__h{h}.csv.gz"
            if os.path.exists(vfile):
                print(f"[transfer] {held} h={h} already done"); continue
            nt = cfg.get("transfer_rf_trees")
            fd = fit_models(cells, train_ids, z, h, cfg, models, "d_dev", nt)
            # post-processing is fitted on the TRAINING cells only: the held-out cell stays unseen
            Vval = pd.concat([forecast_frame(cells, c, "d_dev", z[c], h, fd, cfg) for c in train_ids])
            del fd; _free()
            params, thr = fit_postproc(Vval, cfg)
            ff = fit_models(cells, train_ids, z, h, cfg, models, "d", nt)
            V = add_postproc(forecast_frame(cells, held, "d", z[held], h, ff, cfg), params, cfg)
            del ff; _free()
            key = dict(experiment="transfer", horizon=h)
            pd.DataFrame(continuous_rows(V, key)).assign(held_out=held).to_csv(
                f"{_parts(cfg)}/cont_transfer__{held}__h{h}.csv", index=False)
            thr.assign(**key, held_out=held).to_csv(f"{_parts(cfg)}/thr_transfer__{held}__h{h}.csv", index=False)
            V.to_csv(vfile, index=False)
            print(f"[transfer] held={held} h={h} done", flush=True)

# ================================= analysis ====================================
def variant(V, m, pp):
    """(spei, ssi) of method m under post-processing pp (THR uses the raw forecast)."""
    if pp in ("QM", "VI"):
        return V[f"{m}__spei__{pp}"].values, V[f"{m}__ssi__{pp}"].values
    return V[f"{m}__spei"].values, V[f"{m}__ssi"].values

def block_indices(n, B, block, seed):
    rng = np.random.default_rng(seed); nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(B, -1)[:, :n]

def per_date_counts(dates_inv, n_dates, o, f, mask=None):
    w = np.ones(len(o), bool) if mask is None else mask
    H = np.bincount(dates_inv, weights=(o & f & w).astype(float), minlength=n_dates)
    F = np.bincount(dates_inv, weights=(~o & f & w).astype(float), minlength=n_dates)
    Mi = np.bincount(dates_inv, weights=(o & ~f & w).astype(float), minlength=n_dates)
    return H, F, Mi

def scores(h, fa, mi, n=None):
    h, fa, mi = float(h), float(fa), float(mi)
    out = dict(hits=h, false_alarms=fa, misses=mi,
               POD=h / (h + mi) if (h + mi) else np.nan, FAR=fa / (h + fa) if (h + fa) else np.nan,
               CSI=h / (h + fa + mi) if (h + fa + mi) else np.nan,
               freq_bias=(h + fa) / (h + mi) if (h + mi) else np.nan)
    if n: out["base_rate"] = (h + mi) / n
    return out

def boot_csi(Hd, Fd, Md, idx):
    h, f, m = Hd[idx].sum(1), Fd[idx].sum(1), Md[idx].sum(1); den = h + f + m
    return np.where(den > 0, h / np.where(den > 0, den, 1), np.nan)

def runs(ev, merge_gap, min_dur):
    ev = np.asarray(ev, bool); idx = np.flatnonzero(ev)
    if len(idx) == 0: return []
    br = np.flatnonzero(np.diff(idx) > 1)
    st, en = np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]
    merged = [[st[0], en[0]]]
    for s, e in zip(st[1:], en[1:]):
        if s - merged[-1][1] - 1 < merge_gap: merged[-1][1] = e
        else: merged.append([s, e])
    return [(s, e) for s, e in merged if e - s + 1 >= min_dur]

def match_onsets(obs_on, fc_on, tol):
    used, hits, dts = set(), 0, []
    for o in obs_on:
        cand = [(abs(f - o), j) for j, f in enumerate(fc_on) if j not in used and abs(f - o) <= tol]
        if cand:
            _, j = min(cand); used.add(j); hits += 1; dts.append(fc_on[j] - o)
    return hits, len(obs_on) - hits, len(fc_on) - len(used), dts

def thr_shift(V, thr, m, lg, lev):
    """Per-row THR delta: a constant from `thr` (one model fit) or, when the frame pools
    several held-out cells, each cell's own delta (thr has a held_out column)."""
    t = thr[(thr.model == m) & (thr.logic == lg) & (thr.level == LNAME[lev])]
    if len(t) == 0:
        return np.zeros(len(V))
    if "held_out" in t.columns and t.held_out.notna().any():
        return V.cell_id.map(dict(zip(t.held_out, t.delta))).fillna(0.0).values.astype(float)
    return np.full(len(V), float(t.delta.iloc[0]))

def episode_scores(V, o_ev, f_ev, cfg):
    hh = mm = ff = 0; dts = []
    for _, g in V.assign(_o=o_ev, _f=f_ev).groupby("cell_id", sort=False):
        oo = [s for s, _ in runs(g._o.values, cfg["ep_merge_gap"], cfg["ep_min_dur"]) if s > 0]
        fo = [s for s, _ in runs(g._f.values, cfg["ep_merge_gap"], cfg["ep_min_dur"]) if s > 0]
        a, b, c, d = match_onsets(oo, fo, cfg["ep_tol"]); hh += a; mm += b; ff += c; dts += d
    return dict(obs_episodes=hh + mm, fc_episodes=hh + ff, **scores(hh, ff, mm),
                timing_mean=float(np.mean(dts)) if dts else np.nan,
                timing_median=float(np.median(dts)) if dts else np.nan)

def analyze_frame(V, exp, h, thr, cfg, regional=True):
    """All verification for one experiment and lead. V holds one or many cells (sorted by
    cell and date inside). Bootstrap resamples blocks of target DATES, shared by all cells
    and methods (paired)."""
    V = V.sort_values(["cell_id", "date"]).reset_index(drop=True)
    udates, inv = np.unique(pd.to_datetime(V["date"]).values, return_inverse=True); nd = len(udates)
    bidx = block_indices(nd, cfg["boot_B"], cfg["boot_block"], cfg["boot_seed"] + h)
    methods = methods_in(V); pps = cfg["postproc"]
    out = dict(cont=[], day=[], onset=[], episode=[], regional=[])

    # continuous, index space: the price of post-processing in squared error, and dispersion
    for ix in ("spei", "ssi"):
        o = V[f"obs_{ix}"].values; ok = np.isfinite(o)
        ref = float(np.mean((o - V[f"Persistence__{ix}"].values)[ok] ** 2))
        for m in methods:
            for pp in ("raw", "QM", "VI"):
                f = V[f"{m}__{ix}" if pp == "raw" else f"{m}__{ix}__{pp}"].values; k = ok & np.isfinite(f)
                mse = float(np.mean((o[k] - f[k]) ** 2))
                out["cont"].append(dict(experiment=exp, horizon=h, index=ix, model=m, postproc=pp, MSE=mse,
                                        SS_vs_persist_raw=1 - mse / ref if ref > 0 else np.nan,
                                        sd_ratio=float(np.std(f[k]) / np.std(o[k]))))

    for lg in ("or", "and"):
        o_stat = status(V.obs_spei.values, V.obs_ssi.values, lg)
        i_stat = status(V.iss_spei.values, V.iss_ssi.values, lg)
        for lev in cfg["levels"]:
            o_ev = o_stat >= lev; below = i_stat < lev
            boot = {}
            for pp in pps:
                for m in methods:
                    sp, ss = variant(V, m, pp)
                    dl = thr_shift(V, thr, m, lg, lev) if pp == "THR" else np.zeros(len(V))
                    f_ev = status(sp, ss, lg, dl) >= lev
                    for scheme, mask in (("day", None), ("onset", below)):
                        Hd, Fd, Md = per_date_counts(inv, nd, o_ev, f_ev, mask)
                        n = int(len(o_ev) if mask is None else mask.sum())
                        row = dict(experiment=exp, horizon=h, model=m, postproc=pp, logic=lg, level=LNAME[lev],
                                   delta=float(np.mean(dl)), n=n, **scores(Hd.sum(), Fd.sum(), Md.sum(), n))
                        if lg == "or":
                            cb = boot_csi(Hd, Fd, Md, bidx)
                            row.update(CSI_lo=np.nanpercentile(cb, 2.5), CSI_hi=np.nanpercentile(cb, 97.5))
                            boot[(scheme, m, pp)] = (cb, row)
                        out[scheme].append(row)
                    if lg == "or":
                        out["episode"].append(dict(experiment=exp, horizon=h, model=m, postproc=pp,
                                                   level=LNAME[lev], delta=float(np.mean(dl)),
                                                   **episode_scores(V, o_ev, f_ev, cfg)))
            if lg == "or":          # paired bootstrap of CSI differences against persistence
                for (scheme, m, pp), (cb, row) in boot.items():
                    for tag, refk in (("vsPersRaw", (scheme, "Persistence", "raw")),
                                      ("vsPersSame", (scheme, "Persistence", pp))):
                        rb, rrow = boot[refk]; diff = np.nan_to_num(cb) - np.nan_to_num(rb)
                        row[f"dCSI_{tag}"] = np.nan_to_num(row["CSI"]) - np.nan_to_num(rrow["CSI"])
                        row[f"dCSI_{tag}_lo"] = float(np.percentile(diff, 2.5))
                        row[f"dCSI_{tag}_hi"] = float(np.percentile(diff, 97.5))
                        row[f"p_{tag}"] = float(np.mean(diff <= 0))     # one-sided: model not better

    # regional alert (in-domain: all cells forecast by the same pooled model)
    if regional and exp == "indomain" and V.cell_id.nunique() > 1:
        frac = cfg["regional_frac"]
        piv = lambda s: pd.Series(np.asarray(s), index=[V.date, V.cell_id]).unstack()
        for lev in cfg["levels"]:
            O = ((piv(status(V.obs_spei.values, V.obs_ssi.values, "or")) >= lev).mean(1) >= frac).values
            I = ((piv(status(V.iss_spei.values, V.iss_ssi.values, "or")) >= lev).mean(1) >= frac).values
            for pp in pps:
                for m in methods:
                    sp, ss = variant(V, m, pp)
                    dl = thr_shift(V, thr, m, "or", lev) if pp == "THR" else np.zeros(len(V))
                    Fr = ((piv(status(sp, ss, "or", dl)) >= lev).mean(1) >= frac).values
                    for scheme, w in (("day", np.ones(len(O), bool)), ("onset", ~I)):
                        out["regional"].append(dict(experiment=exp, horizon=h, model=m, postproc=pp, scheme=scheme,
                                                    level=LNAME[lev], threshold=frac, n=int(w.sum()),
                                                    **scores((O & Fr & w).sum(), (~O & Fr & w).sum(),
                                                             (O & ~Fr & w).sum(), int(w.sum()))))
    return {k: pd.DataFrame(v) for k, v in out.items()}

def analyze(cfg):
    od = cfg["out_dir"]; pd_ = os.path.join(od, "parts"); tabs = {}
    def collect(res, suffix=""):
        for k, v in res.items():
            if len(v): tabs.setdefault(k + suffix, []).append(v)
    hz = lambda f: int(f.split("__h")[-1].split(".")[0])
    for f in sorted(glob.glob(f"{pd_}/values_indomain__h*.csv.gz"), key=hz):
        h = hz(f); V = pd.read_csv(f, parse_dates=["date", "issue_date"])
        collect(analyze_frame(V, "indomain", h, pd.read_csv(f"{pd_}/thr_indomain__h{h}.csv"), cfg))
        print(f"[analyze] indomain h={h}", flush=True)
    tfiles = sorted(glob.glob(f"{pd_}/values_transfer__*__h*.csv.gz"))
    for h in sorted({hz(f) for f in tfiles}):
        fs = [f for f in tfiles if hz(f) == h]
        V = pd.concat([pd.read_csv(f, parse_dates=["date", "issue_date"]) for f in fs], ignore_index=True)
        thr = pd.concat([pd.read_csv(f.replace("values_", "thr_").replace(".csv.gz", ".csv")) for f in fs])
        collect(analyze_frame(V, "transfer", h, thr, cfg))                       # pooled over held-out cells
        for held, g in V.groupby("cell_id"):                                     # and per held-out cell
            collect({k: v.assign(held_out=held) for k, v in
                     analyze_frame(g, "transfer", h, thr[thr.held_out == held], cfg).items()
                     if k in ("day", "onset", "episode")}, "_percell")
        print(f"[analyze] transfer h={h} ({V.cell_id.nunique()} held-out cells)", flush=True)
    names = dict(cont="continuous_index_space", day="warning_day", onset="warning_onset",
                 episode="warning_episode", regional="warning_regional")
    for k, lst in tabs.items():
        base, _, suf = k.partition("_percell")
        fname = names[base] + ("_transfer_percell" if k.endswith("_percell") else "")
        pd.concat(lst, ignore_index=True).to_csv(f"{od}/{fname}.csv", index=False)
    for kind, fname in (("cont", "metrics_original_scale"), ("thr", "thr_deltas"), ("pp", "postproc_params")):
        fs = sorted(glob.glob(f"{pd_}/{kind}_*.csv"))
        if fs:
            pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).to_csv(f"{od}/{fname}.csv", index=False)
    print(f"[analyze] wrote tables to {od}/")
    summary(cfg)

def summary(cfg):
    od = cfg["out_dir"]
    for name, scheme in (("warning_day.csv", "DAY-BASED"), ("warning_onset.csv", "ONSET (issue day below level)")):
        f = os.path.join(od, name)
        if not os.path.exists(f): continue
        d = pd.read_csv(f); d = d[(d.logic == "or") & (d.level == "D1")]
        for exp, g in d.groupby("experiment"):
            print(f"\n=== {exp}: {scheme}, OR, D1 -- CSI [95% block-bootstrap CI] ===")
            g = g.assign(cell=g.apply(lambda r: f"{r.CSI:.2f} [{r.CSI_lo:.2f},{r.CSI_hi:.2f}]", axis=1))
            print(g.pivot_table(index=["model", "postproc"], columns="horizon", values="cell", aggfunc="first")
                  .to_string())
    f = os.path.join(od, "warning_episode.csv")
    if os.path.exists(f):
        d = pd.read_csv(f); d = d[d.level == "D1"]
        for exp, g in d.groupby("experiment"):
            print(f"\n=== {exp}: EPISODE ONSETS, D1 -- CSI (hits/obs episodes) ===")
            g = g.assign(cell=g.apply(lambda r: f"{r.CSI:.2f} ({int(r.hits)}/{int(r.obs_episodes)})", axis=1))
            print(g.pivot_table(index=["model", "postproc"], columns="horizon", values="cell", aggfunc="first")
                  .to_string())

# =================================== main ======================================
def main(cfg=CONFIG_R6):
    cfg = dict(cfg); os.makedirs(cfg["out_dir"], exist_ok=True)
    phase = cfg.get("phase", "indomain")
    if phase in ("indomain", "transfer", "all"):
        meta, cells, cfg = prepare(cfg)
        print(f"REV6 | cells={list(cells)} | TF={P.HAS_TF} | phase={phase} | horizons={_horizons(cfg)} "
              f"| held={cfg.get('only_held')}")
        with open(os.path.join(cfg["out_dir"], "config_rev6.json"), "w") as fh:
            json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()
                       if isinstance(v, (str, int, float, list, tuple, type(None)))}, fh, indent=1)
        if phase in ("indomain", "all"): run_indomain(cells, cfg)
        if phase in ("transfer", "all"): run_transfer(cells, cfg)
    if phase in ("analyze", "all"):
        analyze(cfg)

if __name__ == "__main__":
    main()
