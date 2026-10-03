"""
================================================================================
FORECAST-SKILL-vs-WARNING-SKILL DISSOCIATION  (Section 2.14)  -- standalone CLI
--------------------------------------------------------------------------------
Rebuilt companion script called by the multi-cell runbook (Step B5). It applies
the SAME Section-2.14 analysis used in the single-cell pipeline
(tcn_drought_pipeline_v2.forecast_vs_warning_rank) to the MULTI-CELL outputs, so
no model is re-run: it only reads two CSVs that the multi-cell pipeline already
wrote and reports whether continuous forecast skill and warning skill agree.

It pairs, for every (model, horizon):
    * continuous skill : mean SS_vs_persist  (skill score vs persistence; higher
                         is better) from a metrics file
                         (metrics_indomain.csv or metrics_transfer.csv), and
    * warning skill    : CSI (higher is better) from a DSS file
                         (dss_multicell.csv or dss_regional.csv),
filtered to one (logic, level) cell of the DSS table, then computes the Spearman
rank correlation per horizon and pooled. A LOW or NEGATIVE rho is the formal
evidence that "forecast skill is not warning skill": a model can rank well on
continuous accuracy yet rank poorly as a warning instrument.

It also prints the model that is MOST and LEAST dissociated (best continuous rank
but worst warning rank, and vice versa), and writes a tidy merged table.

Design notes
------------
* Single source of truth: it imports tcn_drought_pipeline_v2 if available and
  reuses its Spearman call; if that import fails (e.g. TensorFlow missing on a
  bare machine) it falls back to scipy.stats.spearmanr directly. Either path
  gives identical numbers.
* Column names match what the pipelines write:
    metrics_*.csv : model, horizon, target, SS_vs_persist
    dss_*.csv     : model, horizon, logic, level, CSI
  The regional DSS file (dss_regional.csv) has logic='or' and no 'and' rows; the
  defaults (--logic or --level D1) therefore work for both DSS files.
* The metrics file may aggregate over cells (in-domain has a 'cell_id' column,
  transfer has 'held_out'); both are averaged into one (model, horizon) score,
  which is exactly the per-condition mean the dissociation test needs.

Usage
-----
  python analyze_dissociation.py METRICS_CSV DSS_CSV
  python analyze_dissociation.py results_mc/metrics_indomain.csv results_mc/dss_multicell.csv
  python analyze_dissociation.py results_mc/metrics_indomain.csv results_mc/dss_regional.csv \
         --logic or --level D1 --target GWETROOT --out results_mc/dissociation_indomain.csv

Exit status is 0 even when a horizon has too few models to correlate; the script
reports the limitation instead of failing, so it never breaks the runbook.
================================================================================
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

# --- Spearman: reuse the pipeline's dependency if importable, else scipy direct ---
try:
    import tcn_drought_pipeline_v2 as _P          # single source of truth
    _spearman = _P.st.spearmanr                   # scipy.stats imported as st there
    _SRC = "tcn_drought_pipeline_v2 (canonical)"
except Exception:
    from scipy.stats import spearmanr as _spearman
    _SRC = "scipy.stats (fallback)"


# ------------------------------ helpers ---------------------------------------
def _read_csv(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {what}: {path}\n"
            f"  Run the multi-cell pipeline first (tcn_drought_multicell.py) so this CSV exists."
        )
    return pd.read_csv(path)


def _require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{name} is missing column(s) {missing}. Found columns: {list(df.columns)}"
        )


def load_continuous(metrics_path, target):
    """Mean continuous skill (SS_vs_persist) per (model, horizon).

    Works for metrics_indomain.csv (has cell_id) and metrics_transfer.csv (has
    held_out): both are averaged across whatever cell dimension is present.
    """
    raw = _read_csv(metrics_path, "metrics CSV")
    _require_cols(raw, ["model", "horizon", "target", "SS_vs_persist"], os.path.basename(metrics_path))
    if target is not None and target in set(raw["target"].unique()):
        raw = raw[raw["target"] == target]
    elif target is not None:
        avail = sorted(raw["target"].unique())
        print(f"  [warn] target='{target}' not in metrics file; available {avail} -> using all targets")
    cont = (raw.groupby(["model", "horizon"])["SS_vs_persist"]
               .mean().reset_index().rename(columns={"SS_vs_persist": "cont_skill"}))
    return cont


def load_warning(dss_path, logic, level):
    """CSI per (model, horizon) for one (logic, level) slice of a DSS table."""
    dss = _read_csv(dss_path, "DSS CSV")
    _require_cols(dss, ["model", "horizon", "logic", "level", "CSI"], os.path.basename(dss_path))
    sub = dss[(dss["logic"] == logic) & (dss["level"] == level)].copy()
    if sub.empty:
        avail = (dss[["logic", "level"]].drop_duplicates()
                 .sort_values(["logic", "level"]).to_dict("records"))
        raise ValueError(
            f"No rows in {os.path.basename(dss_path)} for logic='{logic}', level='{level}'.\n"
            f"  Available (logic, level) pairs: {avail}"
        )
    warn = sub[["model", "horizon", "CSI"]].rename(columns={"CSI": "warn_skill"})
    return warn


def spearman_safe(x, y):
    """Spearman rho with guards for <3 points or zero-variance input."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, len(x)
    rho, p = _spearman(x, y)
    return float(rho), float(p), len(x)


# ------------------------------ core analysis ---------------------------------
def analyze(metrics_path, dss_path, target="GWETROOT", logic="or", level="D1"):
    cont = load_continuous(metrics_path, target)
    warn = load_warning(dss_path, logic, level)
    merged = cont.merge(warn, on=["model", "horizon"], how="inner").dropna(
        subset=["cont_skill", "warn_skill"]
    )
    if merged.empty:
        raise ValueError(
            "No common (model, horizon) rows between the metrics and DSS files after "
            "filtering. Check that both came from the same run / phase."
        )

    rows = []
    for h in sorted(merged["horizon"].unique()):
        s = merged[merged["horizon"] == h]
        rho, p, n = spearman_safe(s["cont_skill"], s["warn_skill"])
        rows.append(dict(horizon=h, n_models=int(s["model"].nunique()),
                         spearman_rho=rho, p=p, n_used=n))
    rho, p, n = spearman_safe(merged["cont_skill"], merged["warn_skill"])
    rows.append(dict(horizon="pooled", n_models=int(merged["model"].nunique()),
                     spearman_rho=rho, p=p, n_used=n))
    rank_tbl = pd.DataFrame(rows)

    # --- identify the most dissociated conditions (rank gap, pooled) ----------
    m = merged.copy()
    m["cont_rank"] = m["cont_skill"].rank(ascending=False, method="min")   # 1 = best accuracy
    m["warn_rank"] = m["warn_skill"].rank(ascending=False, method="min")   # 1 = best warning
    m["rank_gap"] = m["warn_rank"] - m["cont_rank"]   # +: good accuracy but poor warning
    return rank_tbl, m.sort_values("rank_gap", ascending=False)


# ------------------------------ reporting -------------------------------------
def report(rank_tbl, merged, metrics_path, dss_path, logic, level, target):
    print("=" * 78)
    print("FORECAST SKILL vs WARNING SKILL  (Section 2.14, multi-cell)")
    print("=" * 78)
    print(f"  metrics : {metrics_path}")
    print(f"  dss     : {dss_path}")
    print(f"  slice   : target={target}  logic={logic}  level={level}")
    print(f"  spearman: {_SRC}")

    print("\n--- Spearman rank correlation (continuous skill vs CSI) ---")
    show = rank_tbl.copy()
    show["spearman_rho"] = show["spearman_rho"].map(
        lambda v: f"{v:+.3f}" if pd.notna(v) else "n/a")
    show["p"] = show["p"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
    print(show.to_string(index=False))

    pooled = rank_tbl[rank_tbl["horizon"].astype(str) == "pooled"]
    if not pooled.empty and pd.notna(pooled["spearman_rho"].iloc[0]):
        rho = pooled["spearman_rho"].iloc[0]; pv = pooled["p"].iloc[0]
        if rho < 0.3:
            verdict = ("LOW/NEGATIVE rho -> forecast skill and warning skill are DISSOCIATED: "
                       "ranking models by continuous accuracy does NOT recover the warning ranking.")
        elif rho < 0.7:
            verdict = "MODERATE rho -> partial agreement; continuous accuracy only weakly predicts warning skill."
        else:
            verdict = "HIGH rho -> in this slice the two axes largely agree."
        print(f"\n  Pooled rho = {rho:+.3f} (p = {pv:.3f}).  {verdict}")

    print("\n--- Most dissociated conditions (high continuous rank, low warning rank) ---")
    cols = ["model", "horizon", "cont_skill", "warn_skill", "cont_rank", "warn_rank", "rank_gap"]
    head = merged[cols].head(5).copy()
    for c in ("cont_skill", "warn_skill"):
        head[c] = head[c].map(lambda v: f"{v:+.3f}")
    for c in ("cont_rank", "warn_rank", "rank_gap"):
        head[c] = head[c].astype(int)
    print(head.to_string(index=False))
    print("\n  (rank_gap > 0 means: better continuous accuracy than its warning usefulness "
          "would suggest -- the core 'forecast skill is not warning skill' case.)")


def main():
    ap = argparse.ArgumentParser(
        description="Section 2.14 dissociation analysis for multi-cell DSS outputs "
                    "(continuous skill vs warning CSI, Spearman rank correlation).")
    ap.add_argument("metrics_csv", help="metrics_indomain.csv or metrics_transfer.csv")
    ap.add_argument("dss_csv", help="dss_multicell.csv or dss_regional.csv")
    ap.add_argument("--target", default="GWETROOT",
                    help="continuous target to score (default: GWETROOT; use 'all' to pool targets)")
    ap.add_argument("--logic", default="or", choices=["or", "and"],
                    help="DSS alert logic to read (default: or; regional file only has 'or')")
    ap.add_argument("--level", default="D1",
                    help="drought level to read from the DSS file (default: D1)")
    ap.add_argument("--out", default=None,
                    help="optional path to write the merged per-(model,horizon) table as CSV")
    args = ap.parse_args()

    target = None if str(args.target).lower() == "all" else args.target
    try:
        rank_tbl, merged = analyze(args.metrics_csv, args.dss_csv,
                                   target=target, logic=args.logic, level=args.level)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print("ERROR:", e)
        sys.exit(1)

    report(rank_tbl, merged, args.metrics_csv, args.dss_csv,
           args.logic, args.level, args.target)

    out = args.out
    if out is None:
        base = os.path.dirname(os.path.abspath(args.metrics_csv)) or "."
        tag = "regional" if "regional" in os.path.basename(args.dss_csv) else "pooled"
        out = os.path.join(base, f"dissociation_{tag}.csv")
    merged_out = merged.drop(columns=["cont_rank", "warn_rank", "rank_gap"], errors="ignore")
    merged_out.to_csv(out, index=False)
    rank_out = os.path.splitext(out)[0] + "_rankcorr.csv"
    rank_tbl.to_csv(rank_out, index=False)
    print(f"\nWrote: {out}\n       {rank_out}")


if __name__ == "__main__":
    main()
