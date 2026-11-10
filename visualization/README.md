# Cichlid visualization

A local browser viewer for complete recordings, pose overlays, diagnostic pair movement, human annotations, detector timelines and saved model scores. It supports local video files and direct Dropbox playback without downloading the entire MP4 first.

## Install and open

From the **repository root** on your visualization branch, merge the latest main first so the shared annotations and model outputs are available. Replace the previous `visualization/` folder with this folder (back up custom configurations first).

```bash
source .venv-viz/bin/activate
python -m pip install -r visualization/requirements.txt
python -m visualization.live
```

Open **http://localhost:8001** and leave Terminal running. Stop with Ctrl+C. Do not run a separate `http.server`. If the virtual environment does not exist, create it with `python3.11 -m venv .venv-viz` first. FFprobe is needed for local video metadata: `brew install ffmpeg` if missing.

## Cloud deployment (Render)

This viewer can run as a hosted Python web service. The server must bind to `0.0.0.0` and use Render's `PORT` environment variable; the deployment-compatible `live.py` supports both.

**Data requirements:** Render only receives files committed to the deployed branch (and any external sources reachable from the service). Local-only Parquet files and ignored outputs are not uploaded automatically. The catalog and recording paths must be checked against the deployed filesystem; the original video can stream from configured Dropbox links. A working site does not guarantee pose/score availability.

## What's new

- **Recording selector:** choose registered recordings without restarting. The catalog includes 0028 and 0031; it also discovers local `data/*_vid.mp4` recordings.
- **Local / Dropbox / Auto:** Auto prefers a local video and uses the configured remote source if no local file exists. A local playback error can trigger one remote fallback. You can explicitly choose Dropbox even when a local copy exists.
- **Multiple detectors:** standard `outputs/predictions/<model>/<video_id>.csv` files are discovered for the two registered seed recordings. This includes rule-based, logistic regression, XGBoost, and other folders you obtain later.
- **Model scores:** dropdown and synchronized score plot discover `outputs/<run>/scores/<video_id>_scores.parquet`. Logistic-regression scores on main work directly. MS-TCN/DINOv2 scores work when you obtain that branch's output files; they are not bundled or generated here.
- **Shared pairwise feature plots:** if `outputs/pair_motion/pair_features_frame.parquet` exists, select shared distance, angular velocity, proximity fraction, turns, facing or speed. Values are matched to the displayed pair by frame and TrackUIDs.
- **Original / Cleaned poses:** optional cleaned-layer selector if a matching cleaned pose file is configured and exists. Default clean paths are `data/clean/<video_id>.parquet`; change them to your actual cleaning output directory.
- **Availability and input notes:** missing optional files do not prevent playback. Refresh after adding outputs or editing the catalog.

No model is trained or run by the viewer. Streaming a saved Dropbox video is not live camera behavior detection.

## Inputs

| Input | Purpose | Required? |
|---|---|---|
| Local video or direct remote video URL | Full recording playback | One video source |
| `data/<video_id>.parquet` | Skeletons, IDs, box-center geometry | Optional; otherwise video only |
| Finalized annotation CSV | Reference timeline/event list | Optional |
| `outputs/predictions/<model>/<video_id>.csv` | Detector status/timeline/event list | Optional |
| Per-frame score CSV/Parquet | Named score readout and graph | Optional |
| Shared pair feature Parquet | Additional team-feature graph | Optional |
| Cleaned individual pose Parquet | Inspect cleaned overlays | Optional |

The default human reference is `ground_truth_annotations/finalized_annotations/merged_spans.csv`. No research video, Parquet or model artifacts are included in this download. Merge main for its committed outputs; obtain branch-specific or ignored outputs from the producing teammate or run their pipeline separately.

## Daily use

1. Select the recording and source. Auto chooses local first; Dropbox streams the recording via the local server.
2. Play, seek, change speed, or jump to a known source frame. The whole-recording timeline and event list also seek the video.
3. Select a score source to inspect saved scores. Predictions remain scene-level unless the producer explicitly supplies actor assignments; this viewer does not infer those assignments.
4. Pick two tracks or click **Closest pair now**. Disable All tracks to show only that pair. Pair choice stays fixed across playback until changed; old IDs may become absent.
5. If a cleaned pose file exists, switch layers. This changes skeletons and box-center plots only, not saved detector predictions or the independent shared feature table.

Details: [LIVE_GUIDE.md](LIVE_GUIDE.md).

## Optional MP4 export

The separate exporter is retained for shareable short clips:

```bash
python -m pip install -r visualization/requirements-export.txt
python -m visualization.export_video visualization/export0028.json --check
python -m visualization.export_video visualization/export0028.json
```

See [EXPORT_GUIDE.md](EXPORT_GUIDE.md). This exporter still uses local source files and selected short excerpts; Dropbox/catalog features apply to the interactive player, not the exporter.

## Checks and scope

```bash
python -m unittest visualization.test_live visualization.test_export_video -v
```

Install optional export dependencies and FFmpeg to run both suites. Integration tests cover catalog isolation, optional inputs, pose identity metadata, byte-range seeking and proxy behavior. Synthetic browser checks cover playback, recording switches, pose layers, detector/score selectors, shared features, missing-data gaps and proxy seeking. Actual Dropbox links were tested with small range requests, not ten-hour playback or research pose alignment.

Your final annotation boundaries still need exact-frame verification. Model scores are not necessarily calibrated probabilities or held-out results; consult each model's run report. Unlabeled reference time is unknown. See LIVE_GUIDE.md for further interpretation and performance limitations.

## Files and credit

`live.py` serves the player/video; `catalog.py` loads catalog and scoped data; `live_intervals.py` adapts interval CSVs; `live.html` renders the interface. `recordings.json` is the main catalog. `live0028.json` and `live0031.json` retain single-recording launches. The exporter, demo and tests are separate.

This supersedes older experiment/training and standalone annotation-player bundles; those older workflows remain in Git history. Existing team Marimo evaluation/review tools are unchanged.

Visualization implementation prepared with OpenAI Codex assistance. Videos, tracking, cleaned data, annotations, features, models and predictions belong to their respective project contributors. The default source links follow the repository's `video_source.py`; this browser viewer uses its own HTTP range proxy and does not import OpenCV. No OpenCV, Marimo, PyTorch or DLC2Action is required to view existing outputs.
