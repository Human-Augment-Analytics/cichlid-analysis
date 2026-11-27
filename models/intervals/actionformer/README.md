# actionformer: ActionFormer temporal action localization

- **Outputs**: intervals with confidence. `outputs/predictions/actionformer_<features>/<video>.csv`
  (`VideoID, StartFrame, EndFrame, duration_seconds, positive_fraction, Confidence`); `model.pt`,
  `postprocess.json`, `inference.json`, `proposals/` in `outputs/actionformer_<features>/`.
- **Train / predict**:
  ```sh
  uv run python -m models.intervals.actionformer.train [--features pairmotion] [--epochs 60] [--batch-size 4]
      [--learning-rate 1e-4] [--validation-fraction 0.2] [--patience 6] [--skip-oof-tuning] [--device cuda|cpu]
  uv run python -m models.intervals.actionformer.predict [--features pairmotion] [--min-score S] [--min-seconds S]
      [--merge-seconds M] [--output-dir DIR] [--device D]
  ```
  Needs `mstcn_dlc2action.train --prepare-only --features <f>` first.
- **Training setup**: [shared protocol](../README.md#shared-protocol-modelscommonsegmentspy) segments (256 frames each).
  - Model: vendored official ActionFormer (`vendor/`, upstream code), convTransformer backbone, 128-dim
    embedding, attention window 16, 4 pyramid levels, Soft-NMS. One class.
  - Features standardized with train-frame mean/std; NaN → train mean.
  - AdamW, lr 1e-4, weight decay 1e-4, 2-epoch warmup + cosine, grad clip 5, seed 42.
  - Checkpoint: best validation loss on a stratified 20% of train segments (from epoch 5); early stop
    after 6 stale epochs once ≥ 10 epochs. Test segments never used for selection.
  - Post-processing (min score, min seconds, merge gap): OOF-tuned on 3 video-disjoint folds, grid min score
    0.0001–0.8 (13 values) × min {0, 0.5, 1, 2, 4, 7.5} s × merge {0, 1, 2, 4} s, objective mean
    (frame F1 + F1@50 + edit/100) / 3. `--skip-oof-tuning`: tune on the validation segments instead.
  - Inference: 256-frame windows, stride 128; cross-window hard NMS (IoU 0.3).

| `--features` | Output name |
|---|---|
| `pairmotion` (default) | `actionformer_pairmotion` |
| `pairmotionemb` / `pairmotionvmae` | `actionformer_pairmotion_dinov2` / `actionformer_pairmotion_videomae` |
