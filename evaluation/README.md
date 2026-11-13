# evaluation

`evaluate.py` is a marimo notebook that compares predicted circling intervals with
ground truth, per video and overall. The metric functions live in `metrics.py`
and have no marimo or polars dependency.

## Usage

```sh
uv run marimo edit evaluation/evaluate.py
```

Pick a model from `outputs/predictions/`; the per-video CSV files in its folder
are concatenated and compared with the finalized
`ground_truth_annotations/finalized_annotations/circling_annotations_vid*.csv`
files. CSV filenames declare the evaluation-video scope, including a
header-only file for a video with no detections.

Other notebooks can import these functions, as `model_performance_report.py` does:

```python
from evaluation.evaluate import (evaluate, prediction_models, prediction_video_ids,
                                 read_ground_truth, read_ground_truth_labels,
                                 read_predictions, summary_view)

gt = read_ground_truth()
gt_labels = read_ground_truth_labels()  # direction and cycles of annotated events
for name, folder in prediction_models().items():  # e.g. {'rule_based': Path(...)}
    predicted = read_predictions(folder, labels=True, scores=True)
    results, warnings = evaluate(
        gt, predicted, gt_labels=gt_labels, videos=prediction_video_ids(folder))
    summary_view(results, warnings)
```

`predicted` has `video_id`, `start` and `end` columns, plus `direction` and
`cycles` with `labels=True` and `score` with `scores=True`. `evaluate` also takes an optional `num_frames`
mapping of video ID to frame count; direction and cycle metrics are only added
when `gt_labels` is given.

## Input format

Each model writes one file per video to `outputs/predictions/<model_name>/<video_id>.csv`.
Write a file for every video, even if it only has the header row. Files are CSV
with a video ID column plus start/end frame columns, e.g. `VideoID`, `StartFrame`,
`EndFrame` (header case, spaces and punctuation are ignored, so `Video ID` and
`Start Frame` also work). Optionally add a `Direction` column (`CW` or `CCW`),
a `Cycles` column (number of circles, may be fractional), and a finite numeric
`Score` (higher means more confident; required for mAP). Other columns are
ignored. Frame ranges are zero-based and inclusive.

A model that only finds intervals can borrow the rule-based direction and cycle
count: `uv run python -m models.rule_based.episode_labels outputs/predictions/<model>`
writes a copy with `Direction` and `Cycles` to `outputs/predictions/<model>_rule_labels/`,
which is then evaluated as its own model. For that estimator (and the oracle-span
helper below), **positive `d_orbital_rad_s` is clockwise (CW) on screen** with
image y pointing down.

Joint metrics (`direction_f1`, cycle MAE at IoU ≥ 0.1) are the end-to-end product
score (Layer C). Use the strict-IoU (Layer B) and oracle-span tables to judge
attribute quality without detection noise.

## Metrics

- **MoF**: fraction of all frames (circling and background) labelled correctly.
  It needs each video's total frame count, which comes from `num_frames`, else
  `VIDEO_FRAMES`, else `data/<video>.mp4`, else the last frame in any interval + 1 (with a warning).
- **Edit score**: `100 × (1 − Levenshtein / max segments)` over the ordered
  circling/background segment sequence, so it ignores exact timing. The overall
  figure is the average across videos.
- **Frame precision / recall / F1**: for the circling class, pooled across videos
  for the overall figure.
- **Frame IoU**: `|A ∩ B| / |A ∪ B|`, where A is all frames predicted as circling
  in a video and B is all frames annotated as circling, so it is `TP / (TP + FP + FN)`
  over frames. The overall figure is the average across videos.
- **IoU** (intersection over union) of a predicted segment P and a ground-truth
  segment G, both inclusive frame ranges:

  ```
  |P ∩ G| = max(0, min(P.end, G.end) − max(P.start, G.start) + 1)
  |P ∪ G| = |P| + |G| − |P ∩ G|        where |P| = P.end − P.start + 1
  IoU     = |P ∩ G| / |P ∪ G|
  ```

  1 is an exact match and 0 means no overlap. For example, G = 0–9 and
  P = 5–14 share 5 of 15 frames, so IoU = 0.33.
- **F1@τ** for τ ∈ {10, 25, 50} (segmental F1, as in MS-TCN):
  1. Take the predicted segments in order. Compare each with the ground-truth
     segment it has the highest IoU with.
  2. It is a true positive (TP) if that IoU ≥ τ/100 and no earlier prediction
     has claimed that ground-truth segment; otherwise it is a false positive.
     So at most one prediction counts per ground-truth segment.
  3. Ground-truth segments left unclaimed are false negatives.
  4. `precision = TP / predicted segments`, `recall = TP / ground-truth segments`,
     `F1@τ = 2 · precision · recall / (precision + recall)` (0 when TP = 0).

  In the example above the prediction is a TP at τ = 10 and 25 but not at 50.
  Lower τ rewards finding events; higher τ also requires accurate boundaries.
  True positives (`tp@τ`) and segment counts are pooled across videos for the
  overall figure.
- **Temporal mAP** follows the FSH-TAL temporal action-localization protocol.
  Predictions are sorted by `Score`, greedily matched one-to-one to the
  highest-IoU unmatched ground-truth event, and integrated with all-point
  interpolated precision-recall. The report shows mAP@0.1, 0.3, 0.5 and 0.7;
  `mAP@0.1:0.7` averages all seven 0.1-spaced thresholds and then videos.
  Inclusive frame intervals become half-open time intervals
  `[start / fps, (end + 1) / fps)`. Without genuine scores, mAP is unavailable;
  row order is never treated as confidence. FSH-TAL's published score is not
  directly comparable because its behaviors, point-derived ground truth,
  supervision, camera and species differ.

- **Joint direction precision / recall / F1**, for models with a `Direction` column.
  Predicted and ground-truth segments are paired as in F1@10 (`LABEL_TAU`). For
  each direction d, a predicted d segment paired with a ground-truth d segment is
  a TP; any other predicted d segment is a FP (except one paired with a segment of
  unknown direction, which is ignored); ground-truth d segments without a TP are
  FNs. `direction_f1` is the mean of `cw_f1` and `ccw_f1`, so it also penalises
  missed and spurious events; `direction_accuracy` is the fraction of paired
  segments with the right direction, so it judges direction alone.
- **Cycle MAE / RMSE / bias**, for models with a `Cycles` column: error
  (predicted − annotated circles) over paired segments; bias > 0 means
  over-counting. `gt_cycles` and `pred_cycles` are the totals per video. Errors are
  pooled across videos for the overall figure.
- **Strict attribute quality (IoU ≥ 0.5)** reports direction accuracy and cycle
  MAE/RMSE/bias only on well-localized matched spans. Missed and spurious events
  are excluded because detection quality is already measured by F1/mAP.
  Direction coverage and balanced accuracy prevent a model from hiding hard
  labels or exploiting CW/CCW imbalance. Count coverage, WAPE, and fractions
  within ±0.5/±1 circle complement MAE with scale-free and tolerance-based views.
- **Oracle-span attribute quality** accepts a second prediction folder whose
  intervals exactly match ground truth. This is the cleanest measurement of the
  direction/count estimator independent of detection; non-exact boundaries are
  rejected rather than silently called oracle.

Ground-truth direction and cycles come from the annotated events in
`circling_annotations_vid*.csv`. Each
merged span takes the direction covering most of its frames and the circles of
the events it overlaps, prorated by the fraction of each event inside it
(`label_spans`). Predicted segments are labelled the same way when overlapping
predictions are merged. Spans with no attribute annotation are left out with a
warning; spans overlapping events of both directions take the majority direction
and sum the prorated cycle counts, also with a warning.

For the rule-based attribute estimator, oracle labels can be generated from a
multi-video interval CSV:

```sh
uv run python -m models.rule_based.episode_labels path/to/ground_truth_intervals.csv \
  --output-dir outputs/predictions/rule_based_oracle_labels
```

Metrics for a label a model does not predict are NaN, with counts left empty.

Predictions for videos without ground truth are ignored with a warning.

## Tests

```sh
uv run python -m unittest tests.test_evaluation
```
