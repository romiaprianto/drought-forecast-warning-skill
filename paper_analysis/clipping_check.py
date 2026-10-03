"""Clipping check (Methods 2.17 iv, Results 3.6): share of days beyond +/-3 and drought-status changes.
Input: data/cell_sumbawaC.csv (the single-cell experiment)."""
import numpy as np, pandas as pd, json
from _paths import DATA, OUT
import tcn_drought_pipeline_v2 as P
df0 = P.load_data(str(DATA / "cell_sumbawaC.csv")); cfg = P.set_features(df0, dict(P.CONFIG))
pet = P.calculate_pet(df0, cfg["lat_deg"], cfg["elev_m"], cfg["wind_var"], cfg["wind_default"])
out = {}
for tag, clip in [("clipped", 3.0), ("unclipped", 99.0)]:
    d = P.calculate_spei(pet.copy(), cfg["val_end"], clip, cfg["spei_scale"], cfg["doy_halfwin"])
    calib = d.index <= pd.Timestamp(cfg["val_end"]); clim = P.fit_ssi_climatology(d["GWETROOT"], calib, cfg["doy_halfwin"])
    out[tag] = dict(spei=d["SPEI_1"].values, ssi=P.apply_ssi(d["GWETROOT"].values, d.index.dayofyear.values, clim, clip), idx=d.index)
te = out["clipped"]["idx"] >= pd.Timestamp(cfg["test_start"]); res = {}
for nm in ["spei", "ssi"]:
    u = out["unclipped"][nm]; ok = np.isfinite(u)
    res[nm] = dict(days=int(ok.sum()), beyond3=int((np.abs(u[ok]) > 3).sum()), pct=round(100 * (np.abs(u[ok]) > 3).mean(), 2),
                   test_beyond3=int((np.abs(u[ok & te]) > 3).sum()), min=round(float(np.nanmin(u)), 2), max=round(float(np.nanmax(u)), 2))
diff = 0
for logic in ["or", "and"]:
    m = np.isfinite(out["unclipped"]["spei"]) & np.isfinite(out["clipped"]["spei"])
    a = P.status_series(out["clipped"]["spei"], out["clipped"]["ssi"], logic)
    b = P.status_series(out["unclipped"]["spei"], out["unclipped"]["ssi"], logic)
    diff += int((a[m] != b[m]).sum())
res["status_changes"] = diff
json.dump(res, open(OUT / "clipping_check.json", "w"), indent=1); print(json.dumps(res, indent=1))
