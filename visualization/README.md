# Cichlid visualization

Review full recordings with pose overlays, selected-pair movement plots, human annotations and detector predictions. Optionally export short annotated MP4s for presentations.

These tools display existing project results. They do not train a model, estimate poses from raw video, or run a live behavior detector. OpenCV, DLC2Action and Marimo are not required.

## Start the full-video player

From the **repository root**, on macOS:

```bash
python3.11 -m venv .venv-viz  # only if the environment does not exist
source .venv-viz/bin/activate
python -m pip install -r visualization/requirements.txt
python -m visualization.live visualization/live0028.json
```

Open **http://localhost:8001** and leave Terminal running. Stop with Ctrl+C. Do not launch a separate `http.server`.

FFprobe must be available. If missing, install FFmpeg with `brew install ffmpeg`.

The 0028 example opens near frame 206206, but the player covers the entire recording. For 0031, stop the server and run:

```bash
python -m visualization.live visualization/live0031.json
```

## Inputs

| Input | 0028 example | Purpose |
|---|---|---|
| Original video | `data/0028_vid.mp4` | Full recording |
| Individual pose Parquet | `data/0028_vid.parquet` | TrackUID, centers and keypoint coordinates |
| Human annotations | `ground_truth_annotations/finalized_annotations/merged_spans.csv` | Accepted reference intervals |
| Existing predictions | `outputs/predictions/rule_based/0028_vid.csv` | Detector intervals |

The videos/Parquets must be obtained separately and kept local. The CSVs are shared repository outputs. No pairwise Parquet is required by the viewer; it calculates diagnostic distance and angular velocity from centers. The upstream detector may itself use pairwise files.

Annotations and predictions are optional: remove their entries from `tracks` for pose-only viewing. Numeric frame scores are optional and must be supplied explicitly. See [LIVE_GUIDE.md](LIVE_GUIDE.md) for another recording, additional detector tracks and scores.

## Optional: export an annotated MP4

```bash
python -m pip install -r visualization/requirements-export.txt
python -m visualization.export_video visualization/export0028.json --check
python -m visualization.export_video visualization/export0028.json
open outputs/visualization/exports/0028_rule_results.mp4
```

The example exports a 30-second excerpt. This is separate from full-recording playback and can be slow because the exporter decodes from the beginning to preserve source-frame selection. See [EXPORT_GUIDE.md](EXPORT_GUIDE.md).

## Files

| Files | Purpose |
|---|---|
| `live.py`, `live.html`, `live_intervals.py` | Local video server, browser player and interval adapter |
| `live0028.json`, `live0031.json` | Full-recording configurations |
| `export_video.py`, `export0028.json`, `export0031.json` | Optional MP4 exporter and examples |
| `demo_export.py` | Clearly synthetic exporter demo |
| `test_live.py`, `test_export_video.py` | Automated checks |
| `requirements*.txt` | Player and optional export dependencies |
| `LIVE_GUIDE.md`, `EXPORT_GUIDE.md` | Detailed usage |

## Interpretation

- Frame intervals default to zero-based inclusive endpoints. Confirm the producer convention.
- Browser frame counters/steps are time-based estimates, not decoder-verified source indices.
- Unlabeled time is unknown, not a verified negative.
- Track IDs can fragment; sex and biological identity are not inferred by this viewer.
- A scene-level prediction does not identify the selected pair as the participants.
- Numeric model scores are not necessarily calibrated probabilities.
- Playback and pose-section loading depend on the video encoding, disk and Parquet row groups.

## Checks

Install the optional export dependencies before running both suites:

```bash
python -m unittest visualization.test_live visualization.test_export_video -v
```

Tests cover video byte ranges, pose gaps, interval handling, pair geometry, automatic pair selection and synthetic MP4 export. Real fish-video alignment and long-recording performance must be checked locally. This is software validation, not a detector accuracy benchmark.

## Scope and credit

This folder supersedes the earlier visualization bundle: proximity experiments, optional training, the old extracted-clip viewer and the standalone annotation player are intentionally omitted. They remain recoverable from earlier Git commits; their old commands do not apply to this folder.

Video/poses, annotations and detector predictions are the respective project contributors' work. Visualization code was prepared with OpenAI Codex assistance. This complements the project's Marimo review/evaluation tools rather than replacing their formal evaluation.
