"""Step 2. Merge the three leave-one-cell-out shards of REV6 Run 2 and verify the ten held-out cells together.

Pooled scores use BOOT_B replicates; the per-held-out-cell scores (Online Resource 1, Table S7) use PERCELL_B.
Input : results_rev6_transfer/*/parts/ (values_transfer__<cell>__h*.csv.gz, thr_transfer__..., cont_transfer__...)
Output: paper_outputs_rev6/transfer_merged/ (same file names as the REV6 module writes)
Runtime: about 1 h with NPROC=4 and BOOT_B=10000.
"""
import glob
import os
import shutil
from multiprocessing import Pool
import pandas as pd
from _common import RAW_TR, TR, BOOT_B, PERCELL_B, NPROC, LEADS
import tcn_drought_rev6 as R

PARTS = TR / "parts"
BYH = TR / "byh"


def collect_parts():
    shards = sorted(glob.glob(str(RAW_TR / "*" / "parts")))
    if not shards:
        raise SystemExit(f"Missing shards: unzip the three results_rev6_transfer_*.zip into {RAW_TR}")
    PARTS.mkdir(parents=True, exist_ok=True)
    for s in shards:
        for f in glob.glob(os.path.join(s, "*_transfer__*")):
            shutil.copy2(f, PARTS)
    n = len(set(os.path.basename(f).split("__")[1] for f in glob.glob(str(PARTS / "values_transfer__*"))))
    print(f"collected parts for {n} held-out cells from {len(shards)} shards")


def run_lead(h):
    fs = sorted(glob.glob(str(PARTS / f"values_transfer__*__h{h}.csv.gz")))
    V = pd.concat([pd.read_csv(f, parse_dates=["date", "issue_date"]) for f in fs], ignore_index=True)
    thr = pd.concat([pd.read_csv(f.replace("values_", "thr_").replace(".csv.gz", ".csv")) for f in fs])
    res = R.analyze_frame(V, "transfer", h, thr, dict(R.CONFIG_R6, boot_B=BOOT_B))
    for k, v in res.items():
        if len(v):
            v.to_csv(BYH / f"{k}__h{h}.csv", index=False)
    cfg = dict(R.CONFIG_R6, boot_B=PERCELL_B)
    for held, g in V.groupby("cell_id"):
        r = R.analyze_frame(g, "transfer", h, thr[thr.held_out == held], cfg)
        for k in ("day", "onset", "episode"):
            r[k].assign(held_out=held).to_csv(BYH / f"{k}_percell__{held}__h{h}.csv", index=False)
    return h


def merge():
    names = dict(cont="continuous_index_space", day="warning_day", onset="warning_onset", episode="warning_episode")
    for k, fname in names.items():
        fs = sorted(glob.glob(str(BYH / f"{k}__h*.csv")))
        if fs:
            pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).to_csv(TR / f"{fname}.csv", index=False)
        fs = sorted(glob.glob(str(BYH / f"{k}_percell__*.csv")))
        if fs:
            pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).to_csv(
                TR / f"{fname}_transfer_percell.csv", index=False)
    for kind, fname in (("cont", "metrics_original_scale"), ("thr", "thr_deltas")):
        fs = sorted(glob.glob(str(PARTS / f"{kind}_transfer__*.csv")))
        if fs:
            pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).to_csv(TR / f"{fname}.csv", index=False)


if __name__ == "__main__":
    collect_parts()
    BYH.mkdir(parents=True, exist_ok=True)
    with Pool(min(NPROC, len(LEADS))) as pool:
        for h in pool.imap_unordered(run_lead, LEADS):
            print(f"transfer lead {h} d done", flush=True)
    merge()
    print(f"wrote {TR}")
