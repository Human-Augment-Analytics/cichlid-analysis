# Interval models

Detect **when** circling happens. Each writes `outputs/predictions/<name>/<video>.csv`
(`VideoID, StartFrame, EndFrame`, inclusive frames, plus model columns) and artifacts to `outputs/<name>/`.

| Model | Method | Output names |
|---|---|---|
| [`rule_based`](rule_based/README.md) | Closest-pair distance + orbital-rotation rules, no training | `rule_based` |
| [`logistic_regression`](logistic_regression/README.md) | Per-frame L2 logistic regression, 15 hand spans | `logistic_regression` |
| [`xgboost`](xgboost/README.md) | XGBoost on 121-frame windows of 5 cleaned signals | `xgboost` |
| [`xgboost_frame`](xgboost_frame/README.md) | Per-frame XGBoost, shared segments | `xgboost_<features>` |
| [`logistic_regression_cv`](logistic_regression_cv/README.md) | Per-frame logistic regression, C by grouped CV, shared segments | `lrcv_<features>` |
| [`mstcn_dlc2action`](mstcn_dlc2action/README.md) | DLC2Action MS-TCN3 segmentation, shared segments | `mstcn_<features>` |
| [`actionformer`](actionformer/README.md) | ActionFormer temporal localization, scored intervals | `actionformer_<features>` |
| [`sartab`](sartab/README.md) | 60 s window logistic regression (fish occupancy + pair motion) | `sartab` |

Add `Direction`/`Cycles` to any of these: [`../direction/rule_based`](../direction/rule_based/README.md).

## Shared protocol (`models/common/segments.py`)
Used by `xgboost_frame`, `logistic_regression_cv`, `mstcn_dlc2action`, `actionformer`:
- **Labels**: `CIRCLING_GROUND_TRUTH` (default `merged_spans.csv`); unlabeled frames are negative.
- **Rows**: pose converted at stride 2 (every 2nd frame). **Segments**: 128 rows (256 frames), no overlap.
- **Sampling** per video: every segment containing circling + the same number without (seed 42);
  a video without circling contributes negatives equal to the median positive-segment count.
- **Split**: stratified random 80/20 of segments, `random_state=42`. Same split for all four models.
- **Scores to spans** (`common/predict.py`, frame-score models): optional moving average, then
  sustained spans (≥ `min_fraction` positive, strictly longer than `min_seconds`), then gap merge.
  Settings: CLI flag > `outputs/<name>/postprocess.json` > defaults threshold 0.5, 7.5 s, 0.9, merge 0, smooth 0.
- **OOF tuning** (`xgboost_frame`, `logistic_regression_cv`): 3 video-disjoint folds; grid threshold
  {0.25, 0.35, 0.5, 0.65, 0.8} × smooth {0, 1, 2} s × min {2, 4, 7.5} s × merge {0, 1, 2} s;
  maximizes mean per-video (frame F1 + F1@50) / 2.
