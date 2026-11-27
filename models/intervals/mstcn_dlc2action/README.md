# mstcn_dlc2action: MS-TCN action segmentation (DLC2Action)

- **Outputs**: intervals. `outputs/predictions/mstcn_<features>/<video>.csv`
  (`VideoID, StartFrame, EndFrame, duration_seconds, positive_fraction`); report and per-frame
  `scores/` in `outputs/mstcn_<features>/`; DLC2Action data/project in `outputs/mstcn_dlc2action/`.
- **Train / predict**:
  ```sh
  uv run python -m models.intervals.mstcn_dlc2action.train [--features pairmotion] [--num-epochs 100] [--prepare-only] [--fresh]
  uv run python -m models.intervals.mstcn_dlc2action.predict [--features pairmotion] [--threshold T] [--min-seconds S] [--min-fraction F]
  ```
  `--prepare-only` converts pose (stride 2) and builds pair / pair-motion / embedding files; the frame
  models and ActionFormer need it too. Training resumes: an identical finished run is skipped unless `--fresh`.
- **Training setup**: [shared protocol](../README.md#shared-protocol-modelscommonsegmentspy).
  - DLC2Action 1.0 MS-TCN3, `--preset aug` (top-down rotation, mirror, shift, noise), 100 epochs,
    segment 128 rows, overlap 0, canvas 1296x972.
  - DLC2Action's validation subset = the test segments, logging only (no early stopping or selection).
  - Post-processing: no OOF tuning; predict uses the defaults (0.5, > 7.5 s, 0.9) unless flags are given.
- `scripts/`: DLC2Action CLI (`dlc2action_circling.py`), DINOv2 (`embed_probe.py`) and VideoMAE
  (`videomae_extract.py`) extraction, `preflight_dlc2action.py` checks. `dlc_yolo_converter/`: pose/pair → DLC2Action.

| `--features` | Output name | Inputs |
|---|---|---|
| `pose` | `mstcn_pose` | male and partner pose |
| `pair` | `mstcn_pose_pair` | pose + 17 pair features |
| `pairemb` / `pairvmae` | `mstcn_dinov2` / `mstcn_videomae` | `pair` + 32 PCs of DINOv2 male crops / VideoMAE scene clips |
| `pairmotion` (default) | `mstcn_pairmotion` | pose + normalized pair motion, 0.5–16 s windows |
| `pairmotionemb` / `pairmotionvmae` | `mstcn_pairmotion_dinov2` / `_videomae` | `pairmotion` + 32 DINOv2 / VideoMAE PCs |
