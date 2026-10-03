# Forecast skill is not warning skill — code, data and results

Companion repository for the article *"Forecast skill is not warning skill: an open, transferable framework for
data-driven drought forecasting and decision support"* (submitted to *Environmental Modelling & Software*).

DOI of this archive: [REPOSITORY DOI] · Licence: code under the MIT licence (`LICENSE`); data and results under CC BY 4.0 (`LICENSE-DATA`)

The repository contains the forecasting pipeline, the NASA POWER input data for the ten grid cells, the result files
behind every table and figure of the article, and one script per table or figure that regenerates it from those files.

## Structure

| Folder | Content |
|---|---|
| `code/` | Forecasting and verification pipeline (`tcn_drought_pipeline_v2.py`, single cell; `tcn_drought_multicell.py`, REV4, multi-cell in-domain and leave-one-cell-out transfer), dissociation analysis, NASA POWER download script, shared drawing code for Figures 1, 4 and 6 |
| `data/` | `cells.csv` (cell coordinates and elevation) and `cell_<id>.csv` (daily NASA POWER series, 1996–2025) |
| `notebooks/` | Kaggle runbooks used for the GPU runs: single cell, multi-cell, and the second multi-cell run for the regional aggregation fractions |
| `results/` | Single-cell outputs (Central Sumbawa, C03) |
| `results_mc/` | Multi-cell outputs (in-domain and transfer; per-fold parts in `parts/`) |
| `results_mc_regional/` | Second multi-cell run with regional alerts at 20–50 % aggregation fractions |
| `paper_analysis/` | One script per table or figure; outputs are written to `paper_outputs/` |

## Requirements

Python 3.10 or later and the packages in `requirements.txt`. TensorFlow is needed only to retrain the neural models;
every script in `paper_analysis/` runs on a CPU from the stored result files. `fig02_study_area.py` downloads Natural
Earth coastlines (public domain) on first use.

## Reproducing the results

1. **Input data** (optional, already included): `python code/download_nasapower_cells.py`.
2. **Forecasts and verification**: run `notebooks/runbook_kaggle.ipynb` (single cell) and
   `notebooks/runbook_kaggle_multisel.ipynb` (multi-cell) on a GPU; `notebooks/regional_threshold_kaggle.ipynb` produces
   `results_mc_regional/`. Persistence, climatology and the Random Forest are deterministic; the TCN, LSTM and
   Transformer are trained on a GPU and are not bit-for-bit reproducible between runs.
3. **Tables and figures**: from `paper_analysis/`, run the script listed below.

## Tables and figures

| Item | Script (`paper_analysis/`) | Input |
|---|---|---|
| Table 1 | — (cell descriptors; `data/cells.csv`, static covariates computed in `code/tcn_drought_multicell.py`) | `data/` |
| Tables 2–5 | — (descriptive: variables, thresholds, model settings, contingency table) | — |
| Tables 6, 7 | `table06_07_continuous.py` | `results/metrics_raw.csv`, `results/diebold_mariano.csv` |
| Table 8 | `table08_warning.py` | `results/dss_validation.csv` |
| Table 9 | `table09_rank_correlation.py` | `results/rank_correlation.csv` |
| Table 10 | `table10_transfer_by_lead.py` | `results_mc/metrics_indomain.csv`, `results_mc/metrics_transfer.csv` |
| Tables 11, 12 | `table11_12_transfer.py` | `results_mc/skill_map.csv`, `results_mc/dss_transfer_multicell.csv` |
| Table 13 | `table13_ablation.py` | `results/ablation.csv` |
| Table 14 | `table14_robustness.py` (retrains the Random Forest; about 1 h on a CPU, resumable) | `data/`, `results/robustness.csv` |
| Table 15 | `table15_rule_threshold.py` | `results/` |
| Table 16 | `table16_regional_fractions.py` | `results_mc/`, `results_mc_regional/` |
| Figure 1 | `fig01_framework.py` | — |
| Figure 2 | `fig02_study_area.py` | Natural Earth coastlines |
| Figures 3, 5 | `fig03_05_design.py` | — |
| Figures 4, 6 | `fig04_06_acf_tcn.py` | `data/cell_sumbawaC.csv` |
| Figure 7 | `fig07_skill_vs_lead.py` | `results/metrics_raw.csv`, `results/diebold_mariano.csv` |
| Figure 11 | `fig11_performance_diagram.py` | `results/dss_validation.csv` |
| Figure 8 | `fig08_pred_vs_obs.py` | `results/predictions.csv` |
| Figures 9, 10 | `fig09_10_all_models.py` | `results/predictions.csv` |
| Figure 12 | `fig12_forecast_vs_warning.py` | `results/` |
| Figure 13 | `fig13_transfer_map.py` | `results_mc/skill_map.csv` |
| Figure 14 | `dissociation_matrix.py`, then `fig14_dissociation_heatmap.py` | `results_mc/` |
| Graphical abstract | `graphical_abstract.py` | `results/`, `results_mc/` |
| Clipping statement (Section 2.17) | `clipping_check.py` | `data/cell_sumbawaC.csv` |

Every number in Tables 6–13, 15 and 16 of the article is reproduced by these scripts from the stored result files.
Every figure of the article is produced by the script listed for it.

## Data source

Daily meteorological and land-surface variables from the NASA POWER project (https://power.larc.nasa.gov/), MERRA-2
based, retrieved for the ten 0.5° grid cells listed in `data/cells.csv`.
