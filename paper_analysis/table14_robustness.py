"""Table 14: robustness of continuous AND warning skill (single cell, RF, one seed, 14-day lead).
Same data, index construction and RF settings as tcn_drought_pipeline_v2.robustness(), plus the
event-based verification used for Table 8. The forest is grown in resumable warm-start chunks
(identical to a single fit), so long runs survive interruptions.
Usage: python robustness_warning.py [--trees 200] [--chunk 60] [--only window_30 ...]
Full run: about 1 h on one CPU core."""
import argparse, os, time, joblib, numpy as np, pandas as pd
from _paths import DATA, OUT
import tcn_drought_pipeline_v2 as P
from sklearn.ensemble import RandomForestRegressor
ap = argparse.ArgumentParser(); ap.add_argument("--trees", type=int, default=None)
ap.add_argument("--chunk", type=int, default=60); ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--out", default=str(OUT / "robustness_warning.csv")); a = ap.parse_args()
CONFIGS = {"window_90_baseline": (90, "WS2M", False), "window_30": (30, "WS2M", False), "window_60": (60, "WS2M", False),
           "window_120": (120, "WS2M", False), "wind_fixed_2ms": (90, "__none__", False), "spei_monthly": (90, "WS2M", True)}
df0 = P.load_data(str(DATA / "cell_sumbawaC.csv")); base = P.set_features(df0, dict(P.CONFIG)); h, gi, si = 14, 0, 1
ntrees = a.trees or base["rf_trees"]; ck = OUT / "_ckpt"; ck.mkdir(exist_ok=True)
done = set(pd.read_csv(a.out).config) if os.path.exists(a.out) else set()
for tag, (L, wind, monthly) in CONFIGS.items():
    if (a.only and tag not in a.only) or tag in done: continue
    cfg = dict(base); cfg["lookback"] = L
    pet = P.calculate_pet(df0, cfg["lat_deg"], cfg["elev_m"], wind, cfg["wind_default"])
    df = P.calculate_spei(pet, cfg["val_end"], cfg["spei_clip"], cfg["spei_scale"], cfg["doy_halfwin"])
    if monthly:
        df["SPEI_1"] = P.calculate_spei_monthly(pet, cfg["val_end"], cfg["spei_clip"]); df = df.dropna(subset=["SPEI_1"])
    clim = P.fit_ssi_climatology(df["GWETROOT"], df.index <= pd.Timestamp(cfg["val_end"]), cfg["doy_halfwin"])
    d = P.split_scale(df, cfg)
    Xtr, Ytr, _ = P.build_windows(d["Xtr"], d["Ytr"], L, h); Xte, Yte, Yl = P.build_windows(d["Xte"], d["Yte"], L, h)
    f = ck / f"{tag}_{ntrees}.joblib"
    rf = joblib.load(f) if f.exists() else RandomForestRegressor(n_estimators=0, random_state=0, n_jobs=-1, warm_start=True)
    while rf.n_estimators < ntrees:
        t = time.time(); rf.n_estimators = min(ntrees, rf.n_estimators + a.chunk); rf.fit(Xtr.reshape(len(Xtr), -1), Ytr)
        joblib.dump(rf, f); print(f"{tag}: {rf.n_estimators}/{ntrees} trees ({time.time()-t:.0f} s)", flush=True)
    inv = d["sY"].inverse_transform; o, pl = inv(Yte), inv(Yl); p = inv(rf.predict(Xte.reshape(len(Xte), -1)))
    td = d["te_index"][L + h - 1: L + h - 1 + len(Yte)]; doy = np.array([x.dayofyear for x in td])
    oss = P.apply_ssi(o[:, gi], doy, clim, cfg["spei_clip"]); rows = []
    for m, fc in [("RandomForest", p), ("Persistence", pl)]:
        fss = P.apply_ssi(fc[:, gi], doy, clim, cfg["spei_clip"])
        r = dict(config=tag, lookback=L, wind=wind, spei="monthly" if monthly else "rolling30", model=m, trees=ntrees,
                 SS_gwet=P.skill_score(o[:, gi], fc[:, gi], pl[:, gi]))
        for lg, lv, lab in [("or", 2, "CSI_OR_D1"), ("or", 1, "CSI_OR_D0"), ("and", 2, "CSI_AND_D1")]:
            r[lab] = P.validate_forecast_dss(o[:, si], oss, fc[:, si], fss, lg, lv)["CSI"]
        rows.append(r)
    pd.DataFrame(rows).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False); f.unlink(missing_ok=True)
    print(f"DONE {tag}: RF SS={rows[0]['SS_gwet']:.4f} | CSI OR-D1 RF={rows[0]['CSI_OR_D1']:.3f} persistence={rows[1]['CSI_OR_D1']:.3f}")
