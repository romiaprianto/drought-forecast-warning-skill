# REV6 — from forecast skill to warning skill

REV6 is an addition, not a replacement. `tcn_drought_pipeline_v2.py` and `tcn_drought_multicell.py` (REV5) are
unchanged, and so are the REV5 results in `results/` and `results_mc/`. The new module `tcn_drought_rev6.py` imports
them and writes to `results_rev6/`.

## Why
REV5 showed that models selected by squared error issue poor warnings and that persistence has the highest
day-by-day CSI. Three objections to that result are answered here:

1. *"The mechanism (conditional-mean forecasts are under-dispersed) is known, and the obvious fixes were not tried."*
   REV6 tries them.
2. *"Persistence wins day-by-day CSI only because the index is autocorrelated; it hardly warns of a new drought."*
   REV6 verifies onsets as well as days.
3. *"The CSI values and the rank correlations carry no uncertainty."* REV6 adds a paired block bootstrap.

## What is new
| # | Change | Where |
|---|---|---|
| 1 | **Post-processing fitted out of sample.** A development model (trained 1996–2016) forecasts 2017–2020; quantile mapping (QM), variance inflation (VI) and an alert-threshold shift (THR, chosen to maximise CSI) are fitted on those forecasts and applied to the test forecasts (2021–2025) of the final model (trained 1996–2020, as in REV5). The same treatments are applied to persistence and climatology. | `fit_postproc`, `add_postproc` |
| 2 | **Onset verification.** *Onset*: only issue days on which the observed state is below the alert level count; persistence can alert there only through the seasonal change of the soil-moisture climatology between issue and target day. *Episode*: drought episodes (≥ 7 days, gaps < 10 days merged); an observed onset is a hit if a forecast onset lies within ±7 days. | `analyze_frame`, `runs`, `match_onsets` |
| 3 | **Uncertainty.** Circular moving-block bootstrap (blocks of 60 days, 1000 replicates), resampling target dates jointly for all cells and methods (paired): 95 % intervals for CSI and for the CSI difference against persistence, and a one-sided bootstrap p-value. | `block_indices`, `boot_csi` |
| 4 | **Transformer with positional encoding.** Sinusoidal positional encoding after the input projection; output pooled from the last time step and the window mean. Width, heads and normalisation as in REV5. | `build_transformer_pe` |
| 5 | **Index-space continuous scores and dispersion.** MSE, skill against raw persistence, and the ratio of forecast to observed standard deviation, for raw, QM and VI forecasts. This shows directly whether post-processing trades squared error for warning skill. | `analyze_frame` (cont) |

## Unchanged
Data, indices (SPEI-1, SSI), thresholds (USDM D0–D3), OR/AND rules, inputs and static covariates, lookback (90 d),
leads (1, 7, 14, 30 d), seeds (0, 1, 2), TCN/LSTM/RF hyper-parameters, Random Forest trees (200 in-domain, 100 transfer),
per-cell chronological early stopping (REV5), regional rule (≥ 30 % of cells).

## Leakage control
* The test years are used only for verification.
* Post-processing is fitted on the 2017–2020 forecasts of a model that never saw 2017–2020.
* In the transfer experiment the post-processing is fitted on the **training cells only**; the held-out cell is never used.
* The SPEI and SSI climatologies are calibrated on 1996–2020, as in REV5. The validation-year indices therefore use the
  same definition as the targets of the final model.

## Outputs (`results_rev6/`)
* `warning_day.csv`, `warning_onset.csv`: POD, FAR, CSI, frequency bias and base rate by experiment, lead, method,
  post-processing, logic and level, plus bootstrap intervals and CSI differences against persistence (OR logic).
* `warning_episode.csv`: episode-onset hits, misses, false alarms, CSI and timing error in days.
* `warning_regional.csv`: regional alert (in-domain), day-based and onset.
* `continuous_index_space.csv`: MSE, skill and dispersion ratio for raw, QM and VI.
* `metrics_original_scale.csv`: the REV5 continuous metrics (GWETROOT, SPEI-1) per cell for the REV6 models.
* `thr_deltas.csv`, `postproc_params.csv`: the fitted post-processing.
* `parts/values_*.csv.gz`: daily observed and forecast indices (raw, QM, VI) for every cell, lead and method, so that
  every table can be recomputed without retraining (`phase="analyze"`, a few minutes on CPU).

## Compute
In-domain: about twice REV5 run 1 (a development fit and a final fit per lead), i.e. about 5 h on a Kaggle GPU, resumable
per lead. Transfer (TCN and Random Forest, 10 folds): resumable per held-out cell.
