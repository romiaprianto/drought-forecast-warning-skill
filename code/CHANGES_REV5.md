# REV5 — per-cell chronological validation for the neural models

**Problem in REV4.** `fit_pool` passed the pooled calibration windows of all cells to Keras, and
`tcn_drought_pipeline_v2.train_keras` used `validation_split=0.1`. Keras takes the validation set from the tail of the
array, which in the pooled array (cells concatenated in the order of `data/cells.csv`) is almost all of the last cell.
As a result the TCN, LSTM and Transformer
* in the in-domain experiment never trained on the Soe cell and stopped training on Soe's error alone;
* in each leave-one-cell-out transfer fold never trained on the last training cell (Soe, or Kupang when Soe was held out).

**Change in REV5 (only this).** `split_per_cell` takes the last 10 % (`val_frac`) of *each* cell's chronological
calibration windows as the early-stopping set; the neural models are fitted on the first 90 % of every cell
(`train_keras_val`, otherwise identical to `train_keras`: patience 8, restore best weights, same epochs, batch size and
seeds). The same split is used in the in-domain and transfer experiments.

**Unchanged.** Inputs, architectures, seeds, horizons, verification, the Random Forest (trained on all calibration
windows, no early stopping), persistence and climatology. A side-by-side run confirmed that the Random Forest,
persistence and climatology results of REV5 are identical to REV4.

**Not affected.** The single-cell experiment (`tcn_drought_pipeline_v2.py`): with one cell, `validation_split` already
holds out the last 10 % of its chronological windows.
