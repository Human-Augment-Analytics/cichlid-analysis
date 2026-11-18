# evaluation

`evaluate.py` is a marimo notebook that compares predicted circling intervals with
ground truth, per video and overall. The metric functions live in `metrics.py`
and have no marimo or polars dependency.

## Usage

```sh
uv run marimo edit evaluation/evaluate.py
```

Pick a model from `outputs/predictions/`; the per-video CSV files in its folder
are concatenated and compared with
`ground_truth_annotations/finalized_annotations/merged_spans.csv`.

Other notebooks can import these functions, as `model_performance_report.py` does:

```python
from evaluation.evaluate import (GROUND_TRUTH, evaluate, prediction_models, read_intervals,
                                 read_predictions, summary_view)

gt = read_intervals(GROUND_TRUTH)
for name, folder in prediction_models().items():  # e.g. {'rule_based': Path(...)}
    results, warnings = evaluate(gt, read_predictions(folder))
    summary_view(results, warnings)
```

`predicted` has `video_id`, `start` and `end` columns. `evaluate` also takes an
optional `num_frames` mapping of video ID to frame count.

## Input format

Each model writes one file per video to `outputs/predictions/<model_name>/<video_id>.csv`.
Write a file for every video, even if it only has the header row. Files are CSV
with a video ID column plus start/end frame columns, e.g. `VideoID`, `StartFrame`,
`EndFrame` (header case, spaces and punctuation are ignored, so `Video ID` and
`Start Frame` also work). Other columns are ignored by the standard metrics and
every row is used, so ground truth must contain only valid circling events.
ActionFormer additionally writes `Confidence`; `tal_report.py` uses that native
score for mAP@IoU ranking. Frame ranges are zero-based and inclusive.

## Metrics

- **MoF**: fraction of all frames (circling and background) labelled correctly.
  It needs each video's total frame count, which comes from `num_frames`, else
  `data/<video>.mp4`, else the last frame in any interval + 1 (with a warning).
- **Edit score**: `100 × (1 − Levenshtein / max segments)` over the ordered
  circling/background segment sequence, so it ignores exact timing. The overall
  figure is the average across videos.
- **Frame precision / recall / F1**: for the circling class, pooled across videos
  for the overall figure.
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

Predictions outside the evaluation set are ignored with a warning. When
`num_frames` is supplied, its keys define that set; therefore an explicitly
selected negative-only video is retained and any predicted frames count as
false positives. `python -m evaluation.evaluate_models` prints family metrics by
default and writes overall, per-video, held-out, and 60 s window tables to
`outputs/figures/family_metrics_*.csv`. Raw per-model audit tables are written to
`outputs/figures/details/model_metrics_*.csv`; `--details` additionally prints
the raw overall and held-out variant tables.

Family values are equal-weight arithmetic means over available model variants,
calculated separately for each metric and video. They are descriptive summaries,
not predictions from a family ensemble. The visual report writes complete
versions to `outputs/figures/without_actionformer/` and
`outputs/figures/with_actionformer/`, restricted to Rule-based, Deep learning
(MS-TCN), XGBoost, and (in the second version) ActionFormer. In addition to the
existing comparisons, each version contains `family_metric_profile.csv/.png`,
which puts whole-video metrics and TAL mAP on a common percentage scale, and
`family_video_distributions.png`, which shows per-video Frame F1 and F1@50 spread.

## Tests

```sh
uv run python -m unittest tests.test_evaluation
```
