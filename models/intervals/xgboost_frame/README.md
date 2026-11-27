# xgboost_frame: per-frame XGBoost

- **Outputs**: intervals. `outputs/predictions/xgboost_<features>/<video>.csv`
  (`VideoID, StartFrame, EndFrame, duration_seconds, positive_fraction`); model, report and
  per-frame `scores/` in `outputs/xgboost_<features>/`.
- **Train / predict**:
  ```sh
  uv run python -m models.intervals.xgboost_frame.train   [--features pairmotion] [--skip-postprocess-tuning]
  uv run python -m models.intervals.xgboost_frame.predict [--features pairmotion] [--threshold T] [--min-seconds S]
                                                          [--min-fraction F] [--merge-seconds M] [--smooth-seconds W]
  ```
  First run `uv run python -m models.intervals.mstcn_dlc2action.train --prepare-only --features <f>` (converts
  pose, builds pair/embedding files).
- **Training setup**: [shared protocol](../README.md#shared-protocol-modelscommonsegmentspy); every frame of the
  train segments is a row; labels per frame.
  - `XGBClassifier(n_estimators=1500, max_depth=4, learning_rate=0.03, min_child_weight=5, subsample=0.9,
    colsample_bytree=0.8, reg_lambda=5.0, tree_method='hist', random_state=42)`; NaNs left to the trees.
  - Early stopping: 60 rounds on AUC-PR, validation = one `StratifiedGroupKFold(5, shuffle, seed 42)` fold of
    the train frames grouped by segment. Test segments never pick trees.
  - Post-processing: OOF-tuned span rule (shared protocol) unless `--skip-postprocess-tuning`.

| `--features` | Output name | Inputs |
|---|---|---|
| `pair` | `xgboost_pose_pair` | 50 pose + 52 pair frame features |
| `pairemb` / `pairvmae` | `xgboost_dinov2` / `xgboost_videomae` | `pair` + 32 DINOv2 / VideoMAE PCs |
| `pairturns` | `xgboost_turns` | `pair` + rule-detector inputs (distance, orbit rate, turns over 2/4/8 s) |
| `pairturnsvmae` | `xgboost_turns_videomae` | `pairturns` + 32 VideoMAE PCs |
| `pairmotion` (default) | `xgboost_pairmotion` | 50 pose + normalized male/partner pair motion, 0.5–16 s windows |
| `pairmotionemb` / `pairmotionvmae` | `xgboost_pairmotion_dinov2` / `_videomae` | `pairmotion` + 32 DINOv2 / VideoMAE PCs |
