# logistic_regression_cv: per-frame logistic regression, C by CV

- **Outputs**: intervals. `outputs/predictions/lrcv_<features>/<video>.csv`
  (`VideoID, StartFrame, EndFrame, duration_seconds, positive_fraction`); model, report and
  per-frame `scores/` in `outputs/lrcv_<features>/`.
- **Train / predict**:
  ```sh
  uv run python -m models.intervals.logistic_regression_cv.train   [--features pairmotion] [--skip-postprocess-tuning]
  uv run python -m models.intervals.logistic_regression_cv.predict [--features pairmotion] [--threshold T] [--min-seconds S]
                                                                   [--min-fraction F] [--merge-seconds M] [--smooth-seconds W]
  ```
  First run `uv run python -m models.intervals.mstcn_dlc2action.train --prepare-only --features <f>`.
- **Training setup**: [shared protocol](../README.md#shared-protocol-modelscommonsegmentspy); every frame of the
  train segments is a row; labels per frame.
  - Pipeline: median imputation with missingness indicators → `StandardScaler` → `LogisticRegressionCV`
    (L2, lbfgs, `max_iter=1500`, 13 C values `logspace(-4, 2)`, scoring average precision).
  - C chosen by `StratifiedGroupKFold(5, shuffle=True, random_state=42)` over train frames grouped by segment.
  - Post-processing: OOF-tuned span rule (shared protocol) unless `--skip-postprocess-tuning`.

| `--features` | Output name |
|---|---|
| `pair` / `pairemb` / `pairvmae` | `lrcv_pose_pair` / `lrcv_dinov2` / `lrcv_videomae` |
| `pairturns` / `pairturnsvmae` | `lrcv_turns` / `lrcv_turns_videomae` (interval model; "turns" = feature set) |
| `pairmotion` (default) / `pairmotionemb` / `pairmotionvmae` | `lrcv_pairmotion` / `lrcv_pairmotion_dinov2` / `lrcv_pairmotion_videomae` |

Feature sets are the same as [xgboost_frame](../xgboost_frame/README.md).
