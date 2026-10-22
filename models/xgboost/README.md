# XGBoost circling classifier

Open the training and evaluation notebook from the repository root:

```sh
uv sync
uv run marimo edit models/xgboost/notebook.py
```

On macOS, install the OpenMP runtime with `brew install libomp` before using
XGBoost. Inputs use `CIRCLING_DATA_DIR`, `data/`, or `~/Downloads`, with the same
pose/pair Parquet filenames as the rule-based detector.

The notebook loads the finalized positive intervals in
`ground_truth_annotations/finalized_annotations/circling_annotations_vid0028.csv` and
`circling_annotations_vid0031.csv` (`evaluation.evaluate.GROUND_TRUTH`). It reuses the
rule-based artifact cleaning and closest-pair selection, retaining
`centroid_distance_px` and `d_orbital_rad_s`. Each row contains these two signals
across 121 centered frames; its label is the middle frame's label.
Missing distances and context beyond video boundaries are NaN. Missing rates or
rates exceeding 15 rad/s in magnitude become zero, matching the rule baseline.

Three additional XGBoost-only per-frame signals bring the total to 605 features
(five signals × 121 offsets, from -60 to +60). No window means are computed:

- `centroid_distance_rate_px_s`: signed frame-to-frame distance change
  multiplied by 30 fps. Positive means the pair moves apart.
- `d_orbital_rate_rad_s2`: signed frame-to-frame change of
  `d_orbital_rad_s` multiplied by 30 fps (angular acceleration).
- `intersection_midpoint_speed_px_s`: frame-to-frame Euclidean
  displacement of the head-to-head/tail-to-tail intersection multiplied by 30 fps.
  The finite segments use `Head_x/y` and `TailTip_x/y` for the selected pair. The
  intersection is the user's defined midpoint; it need not be the geometric center.

Each signal has a separate column for every offset, for example
`d_orbital_rate_rad_s2_offset_+0`. The rate at frame t uses t-1 and t; the
window's leftmost rate can use one preceding context frame. Missing measurements,
pair switches and episode changes produce NaN rates. Invalid orbital rates stay missing when deriving acceleration;
their zero placeholders in the raw features do not create artificial changes.
Parallel, collinear, degenerate or non-intersecting segments and missing keypoints
produce NaN intersections. Shared endpoints count as intersections. The movement
rate requires a valid intersection at both consecutive frames. Invalid steps
remain NaN for XGBoost's missing-value handling. These additions do not alter
the shared feature-engineering pipeline or its dictionary.

Sampling includes every positive frame and ceil(segment length / 2) frames on
each side of each merged positive segment. Buffers are clipped to the video and
unioned. Positive labels take precedence over negative buffers, and rows are
unique by video and center frame. Overlap can reduce the negative proportion.

A reproducible stratified 70/30 frame split is used. XGBoost fits only the 70%
training rows, with fixed hyperparameters and threshold 0.5. The notebook displays
holdout metrics, a confusion matrix, a precision–recall curve, feature importance,
and frame/segment evaluation over the entire length of both videos. Full-video
scores use bounded batches and retain 60 frames of context across batch boundaries.
Full-video predictions are smoothed with a centered 30-frame majority vote over
scores thresholded at 0.5. A frame is positive when at least 50% of the available
window predictions are positive (15 of 30, including ties). The even window uses
offsets -15..+14; video boundaries use only available frames. Voting is applied
after joining scoring batches, independently per video. Contiguous processed
positive frames become inclusive exported spans. This bridges short gaps and
suppresses isolated bursts, rather than merging every gap up to 30 frames.

The notebook and `metrics.json` compare raw and processed full-video metrics.
Processed AP/ROC AUC use the positive-vote fraction as the ranking score. Holdout
metrics use raw predictions because sampled holdout rows are not contiguous.
Per-frame score files retain both raw and processed predictions.

**Interpretation:** neighboring windows overlap across the random split, so the
holdout is not independent-event or unseen-video validation. Full-video metrics
include training frames and assume all unannotated frames are negative; ground
truth may be incomplete. Training scores are not calibrated for full-video
prevalence. Centered windows require future frames.

Files follow the model folder layout:

- `features.py`: cleaning, sampling, labels, and window features.
- `train.py`: reusable training, scoring, evaluation, and artifact export.
- `postprocess.py`: centered 30-frame majority voting.
- `notebook.py`: marimo workflow and results.
- `outputs/xgboost/`: `model.ubj`, `sampled_frames.parquet`,
  `test_predictions.parquet`, `feature_importance.csv`, `metrics.json`, and
  `scores/<video>_scores.parquet`.
- `outputs/predictions/xgboost/<video>.csv`: processed prediction files usable by
  `evaluation/evaluate.py` and `model_performance_report.py`.

The saved model is the exact holdout-evaluated model; no full-data refit is made.
Re-run without marimo with:

```sh
uv run python -m models.xgboost.train
uv run python -m unittest tests.test_xgboost
```
