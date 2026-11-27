# models

| Dir | Predicts | Writes |
|---|---|---|
| [`intervals/`](intervals/README.md) | **When** circling happens | `outputs/predictions/<name>/<video>.csv`: `VideoID, StartFrame, EndFrame` (+ model columns) |
| [`direction/`](direction/README.md) | **Direction** (CW/CCW) and **turns/cycles** of given intervals | `Direction`, `Cycles` columns, or per-event CSVs |
| `common/` | Shared code for both groups | — |

Naming trap: `xgboost_turns` / `lrcv_turns` are **interval** models. "turns" is their
feature set (`--features pairturns`, the rule detector's turn-count inputs); they output spans only.

`common/`:
- `config.py`: paths and env vars. `CIRCLING_DATA_DIR` (default `data/`, else `../data`), `CIRCLING_GROUND_TRUTH`
  (default `ground_truth_annotations/finalized_annotations/merged_spans.csv`), `CIRCLING_VIDEOS`,
  `CIRCLING_LABELS=ground_truth|baseline` (baseline adds name suffix `_15spans`),
  `CIRCLING_HARD_NEGATIVES=0..1` (adds suffix `_hn<percent>`).
- `circling_cleaning.py`: drops reflections, duplicate detections, keypoint jumps (pictures:
  `intervals/rule_based/data_cleaning.py`). `uv run python -m models.common.circling_cleaning [DATA_DIR]`.
- `segments.py`: shared segment sample and 80/20 split, see [intervals/README.md](intervals/README.md).
- `frame_model.py`, `predict.py`, `postprocess_tuning.py`: per-frame training, scores to spans, OOF span-rule tuning.
- `pair_motion.py` (`pairmotion` features), `turns.py` (`pairturns` features), `report.py` (metrics + report).
