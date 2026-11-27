# logistic_regression: per-frame logistic regression baseline

- **Outputs**: intervals. `outputs/predictions/logistic_regression/<video>.csv`
  (`VideoID, StartFrame, EndFrame, duration_seconds, positive_fraction`); model, metrics and report in
  `outputs/circling_baseline/`, per-frame score cache in `outputs/circling_baseline/scores/`.
- **Train / predict**:
  ```sh
  uv run python -m models.intervals.logistic_regression.train   [--data-dir data] [--output-dir outputs/circling_baseline]
  uv run python -m models.intervals.logistic_regression.predict [--threshold 0.5] [--min-seconds 7.5] [--min-fraction 0.9]
      [--model ...model.joblib] [--data-dir data] [--scores-dir DIR] [--output-dir DIR]
  ```
- **Training setup**:
  - Labels: the 15 hand spans in `CIRCLING_SPANS_BY_VIDEO` (`train.py`), videos `0028_vid`, `0031_vid` only.
  - Features (`features.py`): 50 pose features with a causal 30-frame history + 52 per-frame pair
    aggregates (pair count; mean/min/max of 6 distances, 3 rates, 4 angles as sin/cos) over all pairs.
  - Sampling: every positive frame + the same number of random negative frames per video (seed 42).
  - Split: stratified random 80/20 of the sampled frames, `random_state=42`.
  - Model: median imputation + indicators → `StandardScaler` → `LogisticRegression(C=1, lbfgs, max_iter=1000)`,
    no class weights, threshold 0.5. Nothing tuned on test labels.
  - Spans (`postprocess.py`): greedily merged windows strictly longer than `--min-seconds` with
    ≥ `--min-fraction` frames scoring ≥ `--threshold`.
- Also the source of `frame_labels`, `THRESHOLD`, `VIDEO_FRAMES` and the pose/pair features reused by
  `common/` and the `*_pose_pair` / `*_turns` variants.
