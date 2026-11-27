# sartab: SARTAB-inspired window classifier + pair motion

- **Outputs**: intervals. `outputs/predictions/sartab/<video>.csv`
  (`VideoID, StartFrame, EndFrame, duration_seconds, positive_fraction`); `model.joblib`, `windows.parquet`,
  `roi.json`, report and per-frame `scores/` in `outputs/sartab/`.
- **Train / predict**:
  ```sh
  uv run python -m models.intervals.sartab.train      # no options
  uv run python -m models.intervals.sartab.predict [--threshold T] [--min-seconds S] [--min-fraction F]
      [--merge-seconds M] [--smooth-seconds W]
  ```
- **Training setup** (own windows, not the shared segments):
  - Units: 60 s windows (1800 frames) every 30 s. Label: window overlaps any `CIRCLING_GROUND_TRUTH` interval.
  - Features (12): fraction of sampled frames (every 6th) with exactly two fish in a 400 px ROI (the densest
    400 px square of male positions, 20 px bins), plus 60 s means of 11 `pm_*` pair-motion columns
    (distance, speed, closing speed, orbital rate, 4 s close/facing/coverage/turns/corotation).
  - Split: stratified random 80/20 of the non-overlapping windows, `random_state=42`. Overlapping windows
    are scored but never fitted.
  - Model: median imputation + indicators → `StandardScaler` → `LogisticRegressionCV` (balanced,
    11 C values `logspace(-3, 2)`, average precision, `max_iter=2000`).
  - Frame scores: linear interpolation between window centers. Spans: `common/predict.py` defaults
    (0.5, > 7.5 s, 0.9); no OOF tuning.
- Not a strict SARTAB replication (ROI choice and pair-motion features added).
