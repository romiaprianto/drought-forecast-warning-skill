# paper_analysis_rev6 — tables and figures of the Natural Hazards article (REV6)

Scripts that regenerate every number, table and figure of

> Aprianto R, Tawaqqal A, Syaharuddin. *Persistence wins on days, calibrated machine-learning forecasts anticipate more
> drought onsets: verifying data-driven drought warnings in eastern Indonesia* (submitted to *Natural Hazards*)

from the daily forecasts saved by the REV6 runs. No model is retrained; everything runs on a CPU.

## 1. Inputs

The results of both REV6 runs are part of the repository:

```
results_rev6/                 in-domain Run 1 (contents of results_rev6_indomain.zip; see its README)
results_rev6_transfer/        leave-one-cell-out Run 2, three shards, each with parts/
```

The forecasts themselves were produced on a Kaggle GPU with `notebooks/rev6_run1_indomain_kaggle.ipynb` and
`notebooks/rev6_run2_transfer_kaggle.ipynb` (module `code/tcn_drought_rev6.py`; see `code/CHANGES_REV6.md`).

Other locations can be set with environment variables: `RES_INDOMAIN`, `RES_TRANSFER`, `DATA_DIR`, `PAPER_OUT`.

## 2. Running

```
pip install -r requirements.txt
cd paper_analysis_rev6
python run_all.py            # all six steps; "python run_all.py 3" resumes at step 3
```

| Step | Script | What it does | Runtime (laptop CPU) |
|---|---|---|---|
| 1 | `01_verify_indomain.py` | Re-runs the in-domain verification (day, onset, episode, regional, continuous) with 10,000 block-bootstrap replicates | 1–2 h |
| 2 | `02_verify_transfer.py` | Merges the three leave-one-cell-out shards and verifies the ten held-out cells together (10,000 replicates; 200 per held-out cell) | ≈ 1 h with 4 workers |
| 3 | `03_tables.py` | Squared-error skill and Diebold–Mariano tests, rank correlation, D1 warning tables with false-discovery-rate control against raw and against equally post-processed persistence | 1 min |
| 4 | `04_episodes.py` | Episode-onset matching, cell bootstrap (CSI, POD, FAR, bias, ΔCSI), FDR control, leave-one-test-year-out | 2 min |
| 5 | `05_robustness.py` | Base rates by year 1996–2025, dry/wet test-year split, SEDI and relative economic value | 2 min |
| 6 | `06_figures.py` | Figures 1–6 (PNG, 600 dpi) and TIFF copies; downloads Natural Earth coastlines once for Fig. 1 | 1 min |

For a quick check of the whole chain, set fewer bootstrap replicates: `BOOT_B=200 PERCELL_B=50 python run_all.py`.
Point estimates (CSI, hits, false alarms, skill scores) are unchanged; only the bootstrap intervals and p-values differ.
The default settings use the same fixed random seeds as the published analysis. A test run with reduced
replicates reproduced every point estimate in the published tables exactly.

Outputs are written to `paper_outputs_rev6/` (`indomain_b10k/`, `transfer_merged/`, `tables/`, `figures/`).

## 3. Where each item of the article comes from

| Article item | Output (`paper_outputs_rev6/`) | Script |
|---|---|---|
| Table 1 (cells) | `data/cells.csv`; static descriptors in `code/tcn_drought_multicell.py` | — |
| Table 2 (model settings) | descriptive | — |
| Table 3, Section 3.1 (squared-error skill, DM tests) | `tables/T_continuous.csv` | 03 |
| Section 3.2, Fig. 3, Table S6 (dispersion, price of QM) | `indomain_b10k/continuous_index_space.csv` | 01, 06 |
| Section 3.3 (rank correlation) | `tables/T_rankcorr.csv` | 03 |
| Table 4, Fig. 4, Tables S1, S3 (day and onset CSI) | `tables/T_indomain_warning_{day,onset}_D1.csv`; all levels in `indomain_b10k/warning_{day,onset}.csv` | 01, 03 |
| Table S2 (threshold shifts) | `tables/T_thr_deltas_indomain.csv` | 03 |
| Table 5, Fig. 5, Table S5 (episodes, false alarms) | `tables/T_episode_vsPersTHR.csv`, `T_indomain_episode_D1.csv` | 04 |
| Table S10, Section 3.8 (leave-one-year-out) | `tables/T_episode_leaveoneyear.csv` | 04 |
| Section 3.6, Table S4 (severity, regional alerts) | `indomain_b10k/warning_day.csv`, `warning_regional.csv` | 01 |
| Table 6, Fig. 6, Table S7 (transfer) | `tables/T_transfer_*.csv`, `transfer_merged/*_transfer_percell.csv` | 02, 03, 04 |
| Tables S8a, S8b, Section 3.8 (test years) | `tables/T_baserate_year.csv`, `T_byyears.csv` | 05 |
| Table S9 (SEDI, economic value) | `tables/T_sedi_rev.csv` | 05 |
| Figs. 1–6 | `figures/Fig1_study_area.png` … `Fig6_transfer.png`, `Fig1.tif` … `Fig6.tif` | 06 |
