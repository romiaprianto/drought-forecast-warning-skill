"""
================================================================================
NASA POWER MULTI-CELL DOWNLOADER  (run LOCALLY; needs internet)
--------------------------------------------------------------------------------
Pulls the SIX daily variables used by the framework for a set of grid cells that
span the Nusa Tenggara DRY-SEASON gradient. Note: every cell here is humid in the
annual mean (P/PET > 0.65); the gradient the study exploits is SEASONAL - the
June-September (JJAS) dry-season rainfall falls from ~140 mm in the west (NTB /
W Flores) to ~35 mm in the southeast (W Timor), and the number of dry months
rises from 3-4 to 6. Cells are NOT arranged by annual aridity.

Outputs (created under OUT_DIR):
  data/cell_<id>.csv   columns: DATE, PRECTOTCORR, T2M, RH2M, ALLSKY_SFC_SW_DWN,
                                WS2M, GWETROOT
  data/cells.csv       columns: cell_id, name, lat, lon, elev, n_days, gwet_mean,
                                 gwet_std, pct_missing   (use for screening)

WS2M (2-m wind) is included so PET can use observed wind (FAO-56) with a fixed-
wind sensitivity check; the pipeline falls back to 2 m/s only if WS2M is absent.

Only standard-library HTTP is used (urllib), so no extra install beyond pandas.
NASA POWER cannot be reached from the analysis sandbox; run this on your machine.

Screening note: at 0.5deg some island cells contain a large ocean fraction, where
GWETROOT may be flat or filled. After running, open data/cells.csv and DROP any
cell whose gwet_std is ~0 or whose pct_missing is high; those cells are unreliable.

Usage:  python download_nasapower_cells.py
================================================================================
"""
import json, time, os, urllib.parse, urllib.request, urllib.error
import numpy as np, pandas as pd

# ----------------------------- configuration ----------------------------------
OUT_DIR = "data"
START, END = "19960101", "20251231"
VARS = ["PRECTOTCORR", "T2M", "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M", "GWETROOT"]
FILL = -999.0                                   # NASA POWER missing-value flag
PAUSE_S = 1.5                                   # politeness pause between requests
RETRIES = 3
BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Cells ordered along the JJAS dry-season gradient (wettest dry season -> driest).
# All are humid annually; the label reflects geography, not an aridity class.
# Coordinates are land points; the POWER point API snaps each to its 0.5 deg cell.
# Keep cell_id short and unique (these ids are referenced throughout the project).
CELLS = [
    dict(cell_id="lombok",   name="Lombok (NTB)",             lat=-8.6,  lon=116.3),
    dict(cell_id="sumbawaB", name="West Sumbawa (NTB)",       lat=-8.8,  lon=116.9),
    dict(cell_id="sumbawaC", name="Central Sumbawa (base)",   lat=-8.5,  lon=117.4),
    dict(cell_id="bima",     name="Bima/Dompu (E Sumbawa)",   lat=-8.5,  lon=118.5),
    dict(cell_id="manggarai",name="Manggarai (W Flores)",     lat=-8.6,  lon=120.4),
    dict(cell_id="ende",     name="Ende (C Flores)",          lat=-8.7,  lon=121.6),
    dict(cell_id="floresT",  name="East Flores",              lat=-8.4,  lon=122.9),
    dict(cell_id="sumba",    name="Sumba",                    lat=-9.6,  lon=120.2),
    dict(cell_id="kupang",   name="Kupang (W Timor)",         lat=-10.1, lon=123.6),
    dict(cell_id="soe",      name="Soe (Timor interior)",     lat=-9.9,  lon=124.2),
]

# ------------------------------- downloader -----------------------------------
def fetch_cell(lat, lon):
    """Return (DataFrame indexed by date with VARS columns, elevation_m)."""
    q = urllib.parse.urlencode({
        "parameters": ",".join(VARS), "community": "AG",
        "longitude": lon, "latitude": lat, "start": START, "end": END,
        "format": "JSON",
    })
    url = f"{BASE}?{q}"
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                doc = json.load(r)
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last = e; time.sleep(3 * attempt)
    else:
        raise RuntimeError(f"download failed for ({lat},{lon}): {last}")

    elev = None
    try:
        elev = float(doc["geometry"]["coordinates"][2])      # POWER returns [lon, lat, elev]
    except Exception:
        pass
    param = doc["properties"]["parameter"]
    df = pd.DataFrame({v: pd.Series(param[v]) for v in VARS})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.sort_index().replace(FILL, np.nan)
    return df, elev

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = []
    for c in CELLS:
        print(f"-> {c['cell_id']:10s} ({c['lat']}, {c['lon']}) ...", end=" ", flush=True)
        try:
            df, elev = fetch_cell(c["lat"], c["lon"])
        except Exception as e:
            print(f"SKIP ({e})"); continue
        df.index.name = "DATE"
        df.to_csv(os.path.join(OUT_DIR, f"cell_{c['cell_id']}.csv"))
        g = df["GWETROOT"]
        meta.append(dict(cell_id=c["cell_id"], name=c["name"], lat=c["lat"], lon=c["lon"],
                         elev=(round(elev, 1) if elev is not None else np.nan),
                         n_days=len(df),
                         gwet_mean=round(float(g.mean()), 3),
                         gwet_std=round(float(g.std()), 3),
                         pct_missing=round(float(df.isna().mean().mean()) * 100, 2)))
        print(f"ok  n={len(df)}  GWETROOT std={meta[-1]['gwet_std']}  miss={meta[-1]['pct_missing']}%")
        time.sleep(PAUSE_S)

    cells = pd.DataFrame(meta)
    cells.to_csv(os.path.join(OUT_DIR, "cells.csv"), index=False)
    print("\n================ SCREENING TABLE (data/cells.csv) ================")
    print(cells.to_string(index=False))
    flag = cells[(cells["gwet_std"] < 0.02) | (cells["pct_missing"] > 5)]
    if not flag.empty:
        print("\nWARNING - inspect/drop these cells (flat GWETROOT or many gaps):")
        print(flag[["cell_id", "name", "gwet_std", "pct_missing"]].to_string(index=False))
    else:
        print("\nAll cells pass the basic screen (GWETROOT varies; few gaps).")

if __name__ == "__main__":
    main()
