# rule_based: rule-based circling detector (no training)

- **Outputs**: intervals **plus** `Direction`/`Cycles` (from [`direction/rule_based`](../../direction/rule_based/README.md)).
  `outputs/predictions/rule_based/<video>.csv`: `VideoID, EventID, StartFrame, EndFrame, start_s, end_s,
  duration_s, Direction, Cycles, detector_name, overlaps_label`. Videos `0031`, `0028` only.
- **Run**:
  ```sh
  uv run python models/intervals/rule_based/detector.py        # writes the predictions
  uv run marimo edit models/intervals/rule_based/detector.py   # same, with explanations and figures
  uv run marimo edit models/intervals/rule_based/data_cleaning.py  # tracking artifacts removed first
  ```
  Data: `CIRCLING_DATA_DIR`, `data/` or `~/Downloads`; cache in `<data>/rule_detector_cache/`.
- **Rules** (closest cleaned pair per frame, 30 fps, centered 4 s windows):
  - Close: distance < 160 px for ≥ 60% of the window. Orbital rates above 15 rad/s count as 0.
  - Start: ≥ 0.75 net turns (close frames only) in the window. Keep going: ≥ 0.4 turns once started.
  - Split runs at frame gaps > 1 s, keep runs ≥ 2 s, then join episodes ≤ 3 s apart.
  - The keep/bridge settings were tuned on video 0031 only, then checked on 0028.
- **No training**: the hand labels in `detector.py` are for display / `overlaps_label` only.
- Files: `detector.py` (marimo), `data_cleaning.py` (marimo, figures in `figures/`, see
  [figures/README.md](figures/README.md)). Cleaning code: `models/common/circling_cleaning.py`.
