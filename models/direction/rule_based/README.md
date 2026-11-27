# rule_based: rule-based direction + cycle count

- **Outputs**: direction/cycles. Copies each interval CSV with `Direction` (`CW`/`CCW`) and `Cycles`
  columns (existing ones replaced). Folder input → `<input_dir>_rule_labels/`; one multi-video CSV →
  `outputs/predictions/rule_based_oracle_labels/` (one file per video, overlapping spans merged).
- **Run** (no training):
  ```sh
  uv run python -m models.direction.rule_based.episode_labels outputs/predictions/<model>   # -> <model>_rule_labels/
  uv run python -m models.direction.rule_based.episode_labels path/to/intervals.csv         # oracle spans
      [--output-dir DIR] [--data-dir DIR] [--raw]   # --raw: tracking data before cleaning
  ```
  Data: as `models/common/circling_cleaning.py` (`CIRCLING_DATA_DIR`, `data/`, `~/Downloads`);
  per-video cache in `<data>/rule_detector_cache/`.
- **Rules** (`episode_labels.py`, 30 fps, closest cleaned pair):
  - Per-frame rotation = `d_orbital_rad_s / 30` while distance < 160 px, else 0; rates > 15 rad/s → 0.
    Positive = clockwise on screen.
  - Direction: sign of the rotation summed over a centered 4 s window per frame; the interval takes the
    direction covering ≥ 50% of its frames (ties → CW).
  - Cycles: |net rotation| of each same-direction stretch, summed, / 2π. Jitter cancels; reversals count both ways.
- Also used by the interval detector [`intervals/rule_based`](../../intervals/rule_based/README.md).
