"""
================================================================================
MULTIVARIATE DROUGHT FORECASTING PIPELINE  (single cell)  --  REV3, Methods-aligned
--------------------------------------------------------------------------------
Targets: root-zone soil moisture (GWETROOT) and SPEI-1; forecasts drive a DSS
whose warnings are verified with POD/FAR/CSI. Built to match the manuscript's
Materials & Methods exactly:

  * SPEI-1 = rolling 30-day climatic water balance D = P - PET, standardized by
    DAY OF YEAR with a +/-15-day window (empirical CDF -> standard normal).    [2.6]
  * SSI    = root-zone soil moisture standardized the SAME way (day-of-year).  [2.6]
  * Chronological split: TRAIN 1996-2016 | VALIDATION 2017-2020 | TEST 2021-25.
    Preprocessing (index climatologies + Min-Max scaler) is estimated on TRAIN
    during development and RE-ESTIMATED on TRAIN+VALIDATION for the final model;
    the TEST period is used once and never for any fitting.                    [2.4, 2.8]
  * FAO-56 Penman-Monteith PET using WS2M wind when present (else fixed 2 m/s,
    with a wind-sensitivity check in robustness()).                            [2.5, 2.17]
  * Diebold-Mariano with Newey-West (Bartlett) HAC variance and the Harvey-
    Leybourne-Newbold small-sample correction; Benjamini-Hochberg FDR control
    across the family of comparisons.                                          [2.11]
  * Forecast-skill-vs-warning-skill rank correlation.                          [2.14]
  * Input-contribution ablation (multivariate / univariate / no-soil-moisture). [2.15]
  * Robustness: wind, rolling-vs-monthly SPEI, threshold, clipping, window.    [2.17]

Self-contained: no climate_indices dependency (Kaggle internet may be off).
TensorFlow is optional; with RF + baselines alone the whole pipeline still runs.
Usage:  python tcn_drought_pipeline_v2.py
================================================================================
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import scipy.stats as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except Exception:
    HAS_TF = False

# ================================ configuration ===============================
BASE_FEATURES = ["PRECTOTCORR", "T2M", "RH2M", "ALLSKY_SFC_SW_DWN", "GWETROOT"]
CONFIG = dict(
    data_path   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cell_sumbawaC.csv"),
    out_dir     = "results",
    train_end   = "2016-12-31",     # development training period end
    val_end     = "2020-12-31",     # validation period end (= final-model calibration end)
    test_start  = "2021-01-01",
    test_end    = "2025-12-31",
    features    = None,             # set at runtime (BASE_FEATURES [+ WS2M if available])
    targets     = ["GWETROOT", "SPEI_1"],   # primary first
    lookback    = 90,
    horizons    = [1, 7, 14, 30],
    seeds       = [0, 1, 2, 3, 4],
    epochs      = 60,
    batch_size  = 64,
    lat_deg     = -8.5,             # representative latitude of the 0.5-deg cell
    elev_m      = 210.0,            # NASA POWER site elevation (Central Sumbawa); document
    rf_trees    = 200,
    spei_clip   = 3.0,
    spei_scale  = 30,               # rolling window (days) for the water balance
    doy_halfwin = 15,               # +/-15-day day-of-year smoothing window
    wind_var    = "WS2M",
    wind_default= 2.0,
)

# =================================== 1. data ==================================
def load_data(path):
    """Read a cell CSV. Accepts the legacy single-cell format (sep=';', decimal
    comma, dates DD/MM/YYYY, column YEAR) or the multi-cell format (DATE, comma)."""
    head = open(path, "r", encoding="utf-8", errors="ignore").readline()
    if ";" in head:                                            # legacy Dataset.csv
        df = pd.read_csv(path, sep=";", decimal=",")
        df["DATE"] = pd.to_datetime(df["YEAR"], format="%d/%m/%Y")
        df = df.drop(columns=["YEAR"]).set_index("DATE").sort_index()
    else:                                                      # cell_<id>.csv
        df = pd.read_csv(path, parse_dates=["DATE"]).set_index("DATE").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").interpolate("time").ffill().bfill()
    assert df.isnull().sum().sum() == 0, "Missing values remain after gap-filling."
    return df

def set_features(df, cfg):
    """6 variables when WS2M is present, else the 5-variable legacy set."""
    cfg = dict(cfg)
    cfg["features"] = BASE_FEATURES + (["WS2M"] if "WS2M" in df.columns else [])
    return cfg

# ============================ 2. FAO-56 PET ===================================
GSC = 0.0820
def extraterrestrial_radiation(idx, lat_deg):
    J = idx.dayofyear.values; phi = np.radians(lat_deg)
    dr = 1 + 0.033*np.cos(2*np.pi*J/365.0)
    dec = 0.409*np.sin(2*np.pi*J/365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi)*np.tan(dec), -1, 1))
    Ra = (24*60/np.pi)*GSC*dr*(ws*np.sin(phi)*np.sin(dec) + np.cos(phi)*np.cos(dec)*np.sin(ws))
    return pd.Series(Ra, index=idx)

def calculate_pet(df, lat_deg, elev_m, wind_var="WS2M", wind_default=2.0):
    """FAO-56 Penman-Monteith. Uses WS2M (2-m wind) when the column exists,
    otherwise a fixed reference wind (documented; varied in robustness())."""
    df = df.copy()
    alpha = 0.23
    u2 = df[wind_var].clip(lower=0.1) if wind_var in df.columns \
         else pd.Series(wind_default, index=df.index)
    T = df["T2M"]
    es = 0.6108*np.exp((17.27*T)/(T+237.3))
    ea = es*(df["RH2M"]/100.0)
    delta = (4098*es)/((T+237.3)**2)
    P0 = 101.3*((293 - 0.0065*elev_m)/293)**5.26
    gamma = 0.000665*P0
    Rs = df["ALLSKY_SFC_SW_DWN"]
    Ra = extraterrestrial_radiation(df.index, lat_deg)
    Rso = (0.75 + 2e-5*elev_m)*Ra
    Rns = (1 - alpha)*Rs
    sigma = 4.903e-9
    rel = np.clip(Rs/Rso.replace(0, np.nan), 0.0, 1.0).fillna(0.6)
    Rnl = sigma*((T+273.16)**4)*(0.34 - 0.14*np.sqrt(ea.clip(lower=0)))*(1.35*rel - 0.35)
    Rn = (Rns - Rnl).clip(lower=0); G = 0.0
    num = 0.408*delta*(Rn - G) + gamma*(900/(T+273))*u2*(es - ea)
    den = delta + gamma*(1 + 0.34*u2)
    df["PET"] = (num/den).clip(lower=0)
    return df

# =================== 3. day-of-year standardization machinery =================
def fit_doy_climatology(series, calib_mask, halfwin=15, min_n=40):
    """For each day-of-year (1..366) store the sorted pool of values within
    +/-halfwin days across the CALIBRATION years (empirical CDF support)."""
    s = series[calib_mask].dropna()
    doy = s.index.dayofyear.values; vals = s.values
    clim = {}
    for d in range(1, 367):
        diff = np.abs(doy - d); diff = np.minimum(diff, 366 - diff)
        pool = np.sort(vals[diff <= halfwin])
        clim[d] = pool if len(pool) >= min_n else None
    return clim

def apply_doy(values, doys, clim, clip=3.0):
    """Map (value, day-of-year) through the stored empirical CDF -> standard normal.
    Works for observed AND forecast values (same climatology)."""
    values = np.asarray(values, float); doys = np.asarray(doys, int)
    out = np.full(len(values), np.nan)
    for d in np.unique(doys):
        pool = clim.get(int(d))
        if pool is None:
            continue
        idx = np.where(doys == d)[0]
        v = values[idx]
        r = np.searchsorted(pool, v, side="right")
        p = (r + 0.5)/(len(pool) + 1.0)
        out[idx] = np.clip(st.norm.ppf(np.clip(p, 1e-4, 1 - 1e-4)), -clip, clip)
    return out

# =============================== 3b. SPEI / SSI ===============================
def calculate_spei(df, calib_end, clip=3.0, scale=30, halfwin=15):
    """Rolling-`scale`-day SPEI: D = P - PET, 30-day running sum, standardized by
    day-of-year (+/-halfwin). Calibration uses data up to `calib_end` only."""
    df = df.copy()
    D = (df["PRECTOTCORR"] - df["PET"])
    D30 = D.rolling(scale, min_periods=scale).sum()
    calib = (df.index <= pd.Timestamp(calib_end)) & D30.notna()
    clim = fit_doy_climatology(D30, calib, halfwin)
    df["SPEI_1"] = apply_doy(D30.values, df.index.dayofyear.values, clim, clip)
    df["SPEI_1"] = df["SPEI_1"].interpolate().bfill().ffill()   # fill the first `scale` days
    return df

def calculate_spei_monthly(df, calib_end, clip=3.0):
    """OLD monthly SPEI (month-end resample, per-calendar-month standardization,
    broadcast to daily). Kept ONLY for the rolling-vs-monthly robustness check."""
    m = df.resample("ME").agg({"PRECTOTCORR": "sum", "PET": "sum"})
    m["D"] = m["PRECTOTCORR"] - m["PET"]; train = m.index <= pd.Timestamp(calib_end)
    spei_m = pd.Series(np.nan, index=m.index)
    for mo in range(1, 13):
        sel = m.index.month == mo
        tr = np.sort(m["D"][sel & train].dropna().values)
        if len(tr) < 5:
            continue
        pp = (np.arange(1, len(tr)+1) - 0.5)/len(tr)
        spei_m[sel] = np.clip(st.norm.ppf(np.clip(np.interp(m["D"][sel].values, tr, pp),
                                                  1e-4, 1-1e-4)), -clip, clip)
    out = spei_m.reindex(df.index, method="ffill")
    return out

def fit_ssi_climatology(daily, train_mask, halfwin=15):
    return fit_doy_climatology(daily, train_mask, halfwin)

def apply_ssi(values, doys, clim, clip=3.0):
    return apply_doy(values, doys, clim, clip)

def standardized_soil_moisture_index(daily, train_mask, clip=3.0, halfwin=15):
    clim = fit_ssi_climatology(daily, train_mask, halfwin)
    return pd.Series(apply_ssi(daily.values, daily.index.dayofyear.values, clim, clip),
                     index=daily.index)

# ===================== 4. split / scale / windows =============================
def split_scale(df, cfg):
    """Min-Max scaler fit on the CALIBRATION span (train+validation, <= val_end);
    transform calibration and test. Targets scaled separately for clean inversion."""
    calib_end = cfg.get("val_end", cfg["train_end"])
    feats, tgts = cfg["features"], cfg["targets"]
    cal = df.loc[:calib_end]; te = df.loc[cfg["test_start"]:cfg["test_end"]]
    sX, sY = MinMaxScaler(), MinMaxScaler()
    Xtr = sX.fit_transform(cal[feats].values); Xte = sX.transform(te[feats].values)
    Ytr = sY.fit_transform(cal[tgts].values); Yte = sY.transform(te[tgts].values)
    return dict(Xtr=Xtr, Ytr=Ytr, Xte=Xte, Yte=Yte, sX=sX, sY=sY,
                tr_index=cal.index, te_index=te.index,
                tr_raw=cal[tgts].values, te_raw=te[tgts].values)

def build_windows(X, Y, lookback, horizon):
    """Sliding windows: X[t-L+1..t] -> Y[t+h]; also return the last in-window Y
    (the persistence forecast = value at t)."""
    Xs, Ys, Ylast = [], [], []
    n = len(X)
    for t in range(lookback - 1, n - horizon):
        Xs.append(X[t - lookback + 1:t + 1])
        Ys.append(Y[t + horizon])
        Ylast.append(Y[t])
    return np.array(Xs), np.array(Ys), np.array(Ylast)

# ================================ 5. metrics ==================================
def rmse(o, p): return float(np.sqrt(np.mean((o - p)**2)))
def mae (o, p): return float(np.mean(np.abs(o - p)))
def nse (o, p): return float(1 - np.sum((o - p)**2)/np.sum((o - o.mean())**2))
def r2  (o, p):
    if np.std(o) < 1e-12 or np.std(p) < 1e-12: return float("nan")
    return float(np.corrcoef(o, p)[0, 1]**2)             # coefficient of determination
def kge (o, p):
    if abs(o.mean()) < 1e-9 or abs(p.mean()) < 1e-9:     # ill-defined for zero-mean (e.g. SPEI)
        return float("nan")
    r = np.corrcoef(o, p)[0, 1] if (np.std(o) > 0 and np.std(p) > 0) else 0.0
    beta = p.mean()/o.mean()
    gamma = (p.std()/p.mean())/(o.std()/o.mean())
    return float(1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2))
def skill_score(o, p, p_ref):
    den = np.mean((o - p_ref)**2)
    return float(1 - np.mean((o - p)**2)/den) if den > 0 else float("nan")

def diebold_mariano(o, p1, p2, h=1):
    """Two-sided DM (squared-error loss) with Newey-West (Bartlett) HAC variance
    truncated at lag h-1, and the Harvey-Leybourne-Newbold small-sample factor.
    DM<0 favours p1 (model under test) over p2 (reference)."""
    d = (o - p1)**2 - (o - p2)**2
    n = len(d); dbar = d.mean()
    g0 = np.sum((d - dbar)**2)/n
    s = g0; L = max(h - 1, 0)
    for k in range(1, L + 1):
        w = 1.0 - k/(L + 1.0)                             # Bartlett weight
        gk = np.sum((d[k:] - dbar)*(d[:-k] - dbar))/n
        s += 2.0*w*gk
    Vd = s/n
    if Vd <= 0:
        return float("nan"), float("nan")
    dm = dbar/np.sqrt(Vd)
    dm *= np.sqrt((n + 1 - 2*h + h*(h - 1)/n)/n)          # HLN correction
    pval = 2*st.t.cdf(-abs(dm), df=n - 1)
    return float(dm), float(pval)

def benjamini_hochberg(pvals, alpha=0.05):
    """Return (reject flags, BH-adjusted p-values) for FDR control at `alpha`."""
    p = np.asarray(pvals, float); ok = np.isfinite(p)
    m = int(ok.sum()); reject = np.zeros(len(p), bool); padj = np.full(len(p), np.nan)
    if m == 0:
        return reject, padj
    idx = np.where(ok)[0]; order = idx[np.argsort(p[idx])]; ps = p[order]
    adj = np.minimum.accumulate((ps*m/np.arange(1, m + 1))[::-1])[::-1]
    padj[order] = np.clip(adj, 0, 1)
    crit = np.where(ps <= alpha*np.arange(1, m + 1)/m)[0]
    if len(crit):
        thr = ps[crit.max()]; reject[idx] = p[idx] <= thr
    return reject, padj

def all_metrics(o, p, p_ref):
    return dict(RMSE=rmse(o, p), MAE=mae(o, p), R2=r2(o, p), NSE=nse(o, p),
                KGE=kge(o, p), SS_vs_persist=skill_score(o, p, p_ref))

def climatology_pred(train_index, train_target, target_dates):
    """Smoothed day-of-year mean of the training target, looked up by date."""
    s = pd.Series(train_target, index=train_index)
    doy_mean = s.groupby(s.index.dayofyear).mean()
    doy_mean = doy_mean.reindex(range(1, 367)).interpolate().bfill().ffill()
    return np.array([doy_mean.loc[min(dt.dayofyear, 366)] for dt in target_dates])

# ============================ 6. deep models ==================================
def _set_seed(s):
    np.random.seed(s)
    if HAS_TF: tf.random.set_seed(s)

def build_tcn(in_shape, out=2, filters=64, k=3, drop=0.2, dilations=(1, 2, 4, 8, 16)):
    inp = layers.Input(shape=in_shape); x = inp
    for d in dilations:
        prev = x
        x = layers.Conv1D(filters, k, padding="causal", dilation_rate=d, activation="relu")(x)
        x = layers.Dropout(drop)(x)
        x = layers.Conv1D(filters, k, padding="causal", dilation_rate=d, activation="relu")(x)
        x = layers.Dropout(drop)(x)
        if prev.shape[-1] != filters:
            prev = layers.Conv1D(filters, 1, padding="same")(prev)
        x = layers.add([prev, x])
    x = layers.Lambda(lambda z: z[:, -1, :])(x)
    out_layer = layers.Dense(out)(x)
    m = Model(inp, out_layer); m.compile(optimizer="adam", loss="mse"); return m

def build_lstm(in_shape, out=2, units=64):
    inp = layers.Input(shape=in_shape)
    x = layers.LSTM(units)(inp); x = layers.Dropout(0.2)(x)
    m = Model(inp, layers.Dense(out)(x)); m.compile(optimizer="adam", loss="mse"); return m

def build_transformer(in_shape, out=2, d_model=64, heads=4):
    inp = layers.Input(shape=in_shape)
    x = layers.Dense(d_model)(inp)
    a = layers.MultiHeadAttention(num_heads=heads, key_dim=d_model)(x, x)
    x = layers.LayerNormalization()(x + a)
    f = layers.Dense(d_model, activation="relu")(x)
    x = layers.LayerNormalization()(x + f)
    x = layers.GlobalAveragePooling1D()(x)
    m = Model(inp, layers.Dense(out)(x)); m.compile(optimizer="adam", loss="mse"); return m

def train_keras(builder, Xtr, Ytr, in_shape, cfg, seed):
    _set_seed(seed)
    model = builder(in_shape)
    es = EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")
    model.fit(Xtr, Ytr, validation_split=0.1, epochs=cfg["epochs"],
              batch_size=cfg["batch_size"], verbose=0, callbacks=[es])
    return model

# ============================== 7. DSS taxonomy ===============================
DROUGHT_LEVELS = [(-0.50, 1), (-0.84, 2), (-1.28, 3), (-1.64, 4)]   # USDM D0..D3
LEVEL_NAMES = {0: "None", 1: "D0", 2: "D1", 3: "D2", 4: "D3"}

def severity(value):
    s = 0
    for thr, lev in DROUGHT_LEVELS:
        if value <= thr: s = lev
    return s

def dss_status(spei, ssi, logic="or"):
    a, b = severity(spei), severity(ssi)
    if logic == "and":      return min(a, b)
    if logic == "or":       return max(a, b)
    return severity((spei + ssi)/2.0)                    # 'combined'

def status_series(spei, ssi, logic="or"):
    return np.array([dss_status(s, g, logic) for s, g in zip(np.asarray(spei), np.asarray(ssi))])

def validate_forecast_dss(obs_spei, obs_ssi, fc_spei, fc_ssi, logic, level):
    obs = status_series(obs_spei, obs_ssi, logic) >= level
    fc  = status_series(fc_spei, fc_ssi, logic) >= level
    hits = int(np.sum(fc & obs)); fa = int(np.sum(fc & ~obs))
    miss = int(np.sum(~fc & obs)); cn = int(np.sum(~fc & ~obs))
    pod = hits/(hits + miss) if (hits + miss) else np.nan
    far = fa/(hits + fa)     if (hits + fa)   else np.nan
    csi = hits/(hits + miss + fa) if (hits + miss + fa) else np.nan
    bias = (hits + fa)/(hits + miss) if (hits + miss) else np.nan
    base = (hits + miss)/len(obs) if len(obs) else np.nan
    return dict(POD=pod, FAR=far, CSI=csi, freq_bias=bias, base_rate=base,
                hits=hits, false_alarms=fa, misses=miss, correct_neg=cn)

def events_from_catalogue(dates, intervals):
    """Optional: boolean drought-day mask from an independent (BNPB/EM-DAT) list
    of (start, end) intervals; used only as ancillary qualitative corroboration."""
    mask = pd.Series(False, index=pd.DatetimeIndex(dates))
    for a, b in intervals:
        mask.loc[(mask.index >= pd.Timestamp(a)) & (mask.index <= pd.Timestamp(b))] = True
    return mask.values

# ============================== 8. evaluation =================================
def evaluate(df, cfg):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    calib_end = cfg["val_end"]; calib_mask = df.index <= pd.Timestamp(calib_end)
    ssi_clim = fit_ssi_climatology(df["GWETROOT"], calib_mask, cfg["doy_halfwin"])
    gi, si = cfg["targets"].index("GWETROOT"), cfg["targets"].index("SPEI_1")
    d = split_scale(df, cfg)
    learned = ["RandomForest"] + (["TCN", "LSTM", "Transformer"] if HAS_TF else [])

    rows, dm_rows, dss_rows, pred_rows = [], [], [], []
    for h in cfg["horizons"]:
        Xtr, Ytr, _     = build_windows(d["Xtr"], d["Ytr"], cfg["lookback"], h)
        Xte, Yte, Ylast = build_windows(d["Xte"], d["Yte"], cfg["lookback"], h)
        inv = d["sY"].inverse_transform
        Yte_o, Ylast_o = inv(Yte), inv(Ylast)
        tdates = d["te_index"][cfg["lookback"] + h - 1: cfg["lookback"] + h - 1 + len(Yte)]
        tdoy = np.array([dt.dayofyear for dt in tdates])

        preds = {"Persistence": {gi: Ylast_o[:, gi], si: Ylast_o[:, si]},
                 "Climatology": {gi: climatology_pred(d["tr_index"], d["tr_raw"][:, gi], tdates),
                                 si: climatology_pred(d["tr_index"], d["tr_raw"][:, si], tdates)}}
        # learned models (seed-averaged forecasts)
        for mname in learned:
            seeds_pred = []
            for s in cfg["seeds"]:
                if mname == "RandomForest":
                    rf = RandomForestRegressor(n_estimators=cfg["rf_trees"], random_state=s, n_jobs=-1)
                    rf.fit(Xtr.reshape(len(Xtr), -1), Ytr)
                    seeds_pred.append(inv(rf.predict(Xte.reshape(len(Xte), -1))))
                else:
                    bmap = {"TCN": build_tcn, "LSTM": build_lstm, "Transformer": build_transformer}
                    mdl = train_keras(bmap[mname], Xtr, Ytr, (cfg["lookback"], Xtr.shape[2]), cfg, s)
                    seeds_pred.append(inv(mdl.predict(Xte, verbose=0)))
                po = seeds_pred[-1]
                rows.append(dict(target="GWETROOT", model=mname, horizon=h, seed=s,
                                 **all_metrics(Yte_o[:, gi], po[:, gi], Ylast_o[:, gi])))
                rows.append(dict(target="SPEI_1", model=mname, horizon=h, seed=s,
                                 **all_metrics(Yte_o[:, si], po[:, si], Ylast_o[:, si])))
            mean_pred = np.mean(seeds_pred, axis=0)
            preds[mname] = {gi: mean_pred[:, gi], si: mean_pred[:, si]}
        # baselines (single row)
        for mname in ("Persistence", "Climatology"):
            rows.append(dict(target="GWETROOT", model=mname, horizon=h, seed=0,
                             **all_metrics(Yte_o[:, gi], preds[mname][gi], Ylast_o[:, gi])))
            rows.append(dict(target="SPEI_1", model=mname, horizon=h, seed=0,
                             **all_metrics(Yte_o[:, si], preds[mname][si], Ylast_o[:, si])))

        # Diebold-Mariano: each model vs persistence
        for mname in learned + ["Climatology"]:
            for ti, tname in [(gi, "GWETROOT"), (si, "SPEI_1")]:
                dm, pv = diebold_mariano(Yte_o[:, ti], preds[mname][ti], preds["Persistence"][ti], h=h)
                dm_rows.append(dict(horizon=h, target=tname, comparison=f"{mname}_vs_Persistence",
                                    DM=dm, p_raw=pv))
        # DSS validation
        obs_spei = Yte_o[:, si]
        obs_ssi  = apply_ssi(Yte_o[:, gi], tdoy, ssi_clim, cfg["spei_clip"])
        for mname, pr in preds.items():
            fc_spei = pr[si]
            fc_ssi  = apply_ssi(pr[gi], tdoy, ssi_clim, cfg["spei_clip"])
            for logic in ("or", "and"):
                for level in (1, 2):
                    m = validate_forecast_dss(obs_spei, obs_ssi, fc_spei, fc_ssi, logic, level)
                    dss_rows.append(dict(horizon=h, model=mname, logic=logic,
                                         level=LEVEL_NAMES[level], **m))
            if h == cfg["horizons"][len(cfg["horizons"]) // 2]:           # export one slice
                for k, dt in enumerate(tdates):
                    pred_rows.append(dict(date=dt, target="GWETROOT", horizon=h, model=mname,
                                          observed=Yte_o[k, gi], forecast=pr[gi][k]))
                    pred_rows.append(dict(date=dt, target="SPEI_1", horizon=h, model=mname,
                                          observed=Yte_o[k, si], forecast=pr[si][k]))

    raw = pd.DataFrame(rows)
    dm_df = pd.DataFrame(dm_rows)
    rej, padj = benjamini_hochberg(dm_df["p_raw"].values, alpha=0.05)   # FDR across the family
    dm_df["p_adj_BH"] = padj; dm_df["reject_BH"] = rej
    dm_df["beats_persistence"] = (dm_df["DM"] < 0) & dm_df["reject_BH"]

    raw.to_csv(f"{cfg['out_dir']}/metrics_raw.csv", index=False)
    dm_df.to_csv(f"{cfg['out_dir']}/diebold_mariano.csv", index=False)
    pd.DataFrame(dss_rows).to_csv(f"{cfg['out_dir']}/dss_validation.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(f"{cfg['out_dir']}/predictions.csv", index=False)
    return raw, dm_df, pd.DataFrame(dss_rows)

# =================== 9. forecast skill vs warning skill [2.14] =================
def forecast_vs_warning_rank(raw, dss, cfg, target="GWETROOT", logic="or", level="D1"):
    """Spearman rank correlation between continuous skill (skill score, higher=better)
    and warning skill (CSI, higher=better) across models, per horizon and pooled."""
    cont = (raw[raw.target == target].groupby(["model", "horizon"])["SS_vs_persist"]
            .mean().reset_index())
    warn = dss[(dss.logic == logic) & (dss.level == level)][["model", "horizon", "CSI"]]
    merged = cont.merge(warn, on=["model", "horizon"], how="inner").dropna()
    out = []
    for h in sorted(merged["horizon"].unique()):
        s = merged[merged.horizon == h]
        if s["model"].nunique() >= 3:
            rho, p = st.spearmanr(s["SS_vs_persist"], s["CSI"])
            out.append(dict(horizon=h, n_models=s["model"].nunique(), spearman_rho=rho, p=p))
    if len(merged) >= 3:
        rho, p = st.spearmanr(merged["SS_vs_persist"], merged["CSI"])
        out.append(dict(horizon="pooled", n_models=len(merged), spearman_rho=rho, p=p))
    res = pd.DataFrame(out)
    res.to_csv(f"{cfg['out_dir']}/rank_correlation.csv", index=False)
    return res

# =============================== 10. ablation [2.15] ==========================
def ablation(df, cfg, horizon=14):
    """Multivariate vs univariate (GWETROOT only) vs no-soil-moisture input."""
    if not HAS_TF:
        return pd.DataFrame()
    calib_mask = df.index <= pd.Timestamp(cfg["val_end"])
    ssi_clim = fit_ssi_climatology(df["GWETROOT"], calib_mask, cfg["doy_halfwin"])
    gi = cfg["targets"].index("GWETROOT"); si = cfg["targets"].index("SPEI_1")
    variants = {"multivariate_all": cfg["features"],
                "univariate_gwet" : ["GWETROOT"],
                "no_gwet_input"   : [f for f in cfg["features"] if f != "GWETROOT"]}
    out = []
    for name, feats in variants.items():
        c = dict(cfg); c["features"] = feats
        d = split_scale(df, c)
        Xtr, Ytr, _     = build_windows(d["Xtr"], d["Ytr"], c["lookback"], horizon)
        Xte, Yte, Ylast = build_windows(d["Xte"], d["Yte"], c["lookback"], horizon)
        inv = d["sY"].inverse_transform; o = inv(Yte); persist = inv(Ylast)
        tdates = d["te_index"][c["lookback"] + horizon - 1: c["lookback"] + horizon - 1 + len(Yte)]
        tdoy = np.array([dt.dayofyear for dt in tdates])
        preds = [inv(train_keras(build_tcn, Xtr, Ytr, (c["lookback"], len(feats)), c, s).predict(Xte, verbose=0))
                 for s in cfg["seeds"]]
        p = np.mean(preds, axis=0)
        obs_ssi = apply_ssi(o[:, gi], tdoy, ssi_clim, c["spei_clip"])
        fc_ssi  = apply_ssi(p[:, gi], tdoy, ssi_clim, c["spei_clip"])
        m = validate_forecast_dss(o[:, si], obs_ssi, p[:, si], fc_ssi, "or", 1)
        out.append(dict(variant=name, horizon=horizon, target="GWETROOT",
                        RMSE=rmse(o[:, gi], p[:, gi]), NSE=nse(o[:, gi], p[:, gi]),
                        SS_vs_persist=skill_score(o[:, gi], p[:, gi], persist[:, gi]),
                        CSI_D0_or=m["CSI"]))
    res = pd.DataFrame(out); res.to_csv(f"{cfg['out_dir']}/ablation.csv", index=False)
    return res

# ============================== 11. robustness [2.17] =========================
def robustness(df_raw, cfg):
    """Light (RF, GWETROOT, h=14) sensitivity checks that the main conclusions
    survive: input window length, wind in PET, and rolling-vs-monthly SPEI."""
    out = []; h = 14; gi = cfg["targets"].index("GWETROOT")
    def rf_skill(df, c):
        d = split_scale(df, c)
        Xtr, Ytr, _ = build_windows(d["Xtr"], d["Ytr"], c["lookback"], h)
        Xte, Yte, Yl = build_windows(d["Xte"], d["Yte"], c["lookback"], h)
        inv = d["sY"].inverse_transform; o, pl = inv(Yte), inv(Yl)
        rf = RandomForestRegressor(n_estimators=cfg["rf_trees"], random_state=0, n_jobs=-1)
        rf.fit(Xtr.reshape(len(Xtr), -1), Ytr)
        p = inv(rf.predict(Xte.reshape(len(Xte), -1)))
        return skill_score(o[:, gi], p[:, gi], pl[:, gi])
    # (i) input window length
    for L in (30, 60, 90, 120):
        c = dict(cfg); c["lookback"] = L
        df = calculate_spei(calculate_pet(df_raw, cfg["lat_deg"], cfg["elev_m"],
                                          cfg["wind_var"], cfg["wind_default"]),
                            cfg["val_end"], cfg["spei_clip"], cfg["spei_scale"], cfg["doy_halfwin"])
        out.append(dict(check="window_length", setting=L, skill_gwet_h14=rf_skill(df, c)))
    # (ii) PET wind: WS2M (if present) vs fixed 2 m/s
    for tag, wv in [("wind_WS2M_or_default", cfg["wind_var"]), ("wind_fixed_2ms", "__none__")]:
        dfp = calculate_pet(df_raw, cfg["lat_deg"], cfg["elev_m"], wv, cfg["wind_default"])
        df = calculate_spei(dfp, cfg["val_end"], cfg["spei_clip"], cfg["spei_scale"], cfg["doy_halfwin"])
        out.append(dict(check="pet_wind", setting=tag, skill_gwet_h14=rf_skill(df, cfg)))
    # (iii) rolling-30-day vs monthly SPEI: agreement on the test window
    dfp = calculate_pet(df_raw, cfg["lat_deg"], cfg["elev_m"], cfg["wind_var"], cfg["wind_default"])
    roll = calculate_spei(dfp, cfg["val_end"], cfg["spei_clip"], cfg["spei_scale"], cfg["doy_halfwin"])["SPEI_1"]
    mon = calculate_spei_monthly(dfp, cfg["val_end"], cfg["spei_clip"])
    te = slice(cfg["test_start"], cfg["test_end"])
    both = pd.concat([roll.loc[te].rename("roll"), mon.loc[te].rename("mon")], axis=1).dropna()
    corr = float(both["roll"].corr(both["mon"])) if len(both) > 2 else float("nan")
    out.append(dict(check="rolling_vs_monthly_spei", setting="pearson_corr_test", skill_gwet_h14=corr))
    res = pd.DataFrame(out); res.to_csv(f"{cfg['out_dir']}/robustness.csv", index=False)
    return res

# ================================== main ======================================
def main(cfg=CONFIG):
    df0 = load_data(cfg["data_path"])
    cfg = set_features(df0, cfg)
    print(f"Variables: {cfg['features']}  ({'WS2M present' if 'WS2M' in df0.columns else 'WS2M absent -> u2=2 m/s'})")
    df = calculate_pet(df0, cfg["lat_deg"], cfg["elev_m"], cfg["wind_var"], cfg["wind_default"])
    df = calculate_spei(df, cfg["val_end"], cfg["spei_clip"], cfg["spei_scale"], cfg["doy_halfwin"])
    print(f"TensorFlow: {HAS_TF}  |  split: TRAIN<= {cfg['train_end']} | VAL<= {cfg['val_end']} | TEST {cfg['test_start']}..{cfg['test_end']}")

    raw, dm_df, dss = evaluate(df, cfg)
    print("\n=== Continuous skill vs persistence (seed-mean) ===")
    piv = (raw[raw.model.isin(["RandomForest", "TCN", "LSTM", "Transformer"])]
           .groupby(["target", "model", "horizon"])["SS_vs_persist"].mean().round(3))
    print(piv.to_string())
    print("\n=== DSS CSI (OR, D1) ===")
    print(dss[(dss.logic == "or") & (dss.level == "D1")][["horizon", "model", "POD", "FAR", "CSI", "base_rate"]]
          .round(3).to_string(index=False))
    rk = forecast_vs_warning_rank(raw, dss, cfg)
    print("\n=== Forecast-skill vs warning-skill rank correlation [2.14] ===")
    print(rk.round(3).to_string(index=False))
    ab = ablation(df, cfg)
    if not ab.empty:
        print("\n=== Ablation [2.15] ===\n", ab.round(3).to_string(index=False))
    rb = robustness(df0, cfg)
    print("\n=== Robustness [2.17] ===\n", rb.round(3).to_string(index=False))
    print(f"\nWrote CSVs to {cfg['out_dir']}/ : metrics_raw, diebold_mariano, dss_validation, "
          f"predictions, rank_correlation, ablation, robustness")

if __name__ == "__main__":
    main()
