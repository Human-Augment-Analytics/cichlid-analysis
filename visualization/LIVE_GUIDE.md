# Full-recording visualization

This extends the visualization folder with an interactive player. The browser plays the original MP4 and draws poses on a canvas. Python serves byte ranges for video seeking and reads ten-second pose sections on demand. There is no clip extraction, training, or MP4 export before playback.

## Install and run on Mac

Keep your visualization branch updated with main. Copy the files from this package's `visualization` directory into your existing `visualization` directory. These are new files; keep your earlier viewer and exporter.

From the repository root:

```bash
source .venv-viz/bin/activate
python -m pip install -r visualization/requirements-live.txt
python -m visualization.live visualization/live0028.json
```

Open **http://localhost:8001**. Keep Terminal running; Ctrl+C stops it. Do not run `python -m http.server` for this player: it needs the pose API and video range-serving support in `visualization.live`.

FFprobe is required to read video metadata. If missing: `brew install ffmpeg`. No PyTorch, DLC2Action, Marimo, or browser package is needed.

## Required inputs

The ready configuration uses:

- `data/0028_vid.mp4` — full original recording, local only.
- `data/0028_vid.parquet` — individual fish poses, local only. The pairs Parquet is not needed.
- `ground_truth_annotations/finalized_annotations/merged_spans.csv` — human reference from main.
- `outputs/predictions/rule_based/0028_vid.csv` — existing detector intervals from main.

The video initially opens near the earlier example (frame 206206), but the seek bar covers the **entire recording**. Set `start_frame` to 0 to start at the beginning.

## Controls

Play/pause, speed, and the seek bar work across the recording. Jump by source frame, or select a reference/prediction event. Previous/next uses the chosen event source. Click the whole-recording timeline to seek.

All tracks are displayed by default, with stable colors derived from TrackUID. Turn off "All tracks" to show only the selected pair. Choose two IDs from the current ten-second section, or click "Closest pair now". The initial closest pair is a diagnostic selection, not identification of courtship participants. Selection stays fixed until you change it; old IDs may disappear due to track fragmentation.

Distance is between box centers in source pixels. Angular velocity measures rotation of the line connecting centers, using wrapped differences over consecutive frames. Positive is clockwise with image y downward. Missing tracks produce gaps. Plots cover the current ten-second section, not the full recording.

Predictions and reference labels are synchronized by source-frame intervals. Scene-level predictions do not assign behavior to the selected pair. Gray/unlabeled time is unknown, not verified negative. Reference rows with ShouldUse=false remain unknown. Approximate circling is unknown; conflicting known reference labels are reported as unknown. Review skipped incomplete-row counts in `/api/meta` if labels appear absent.

## Another recording

Copy `live0028.json` to a new JSON file. Change `video_id`, `expected_project_id`, video path, poses path and optional start_frame. Paths are relative to the JSON file. Set `tracks` to `[]` for video plus poses only, or point tracks at that recording's real annotations/predictions. Do not reuse 0028 predictions for another recording.

`live0031.json` is also included. It needs the corresponding 0031 video, individual pose Parquet and prediction CSV.

To display a score, add an optional `scores` object:

```json
"scores": {"path":"../outputs/my_scores.csv", "video":"video_id", "frame":"frame", "score":"score"}
```

Scores must be finite numbers in [0,1] with unique source frames per recording. The player does not manufacture probabilities from predicted intervals or run model inference. A model score is not necessarily calibrated probability.

## Precision and limitations

Frame numbers estimate the zero-based source index by rounding presented-frame media time × FPS to the nearest integer. Fallback browsers use floor(playback time × FPS). Browser stepping seeks by 1/FPS seconds, not an independently verified decoded frame index. This assumes constant-rate source video aligned with Parquet FrameNum. Average FPS is checked, but does not prove constant frame rate. Use an exact-frame annotation workflow to finalize event boundaries.

Playback speed and seek latency depend on source encoding, keyframe placement, disk speed and browser. Chunk reads use Parquet predicates; large row groups can still take time to decompress. Video remains playable while pose data loads, and missing/loading poses are never replaced by stale overlays. This is a local viewer, not a real-time detector.

H.264 MP4 is normally browser-compatible. Unsupported codecs require a separate compatible proxy with a verified time/frame mapping. No raw videos are bundled or uploaded by this program. The server binds only to 127.0.0.1 and exposes just the configured video and APIs, not the repository directory.

## Validation and credit

Run `python -m unittest visualization.test_live`. Automated checks cover byte-range seeking, missing pose values, identity mismatch and unknown labels. A synthetic 24-second WebM browser check passed playback, seeking across three chunks, pose gaps, pair geometry, prediction/reference labels, stepping controls and event jumps without JavaScript errors; real recordings were not available to the developer, so verify overlay alignment locally before reporting it as validated.

Implementation prepared with OpenAI Codex. Pose data, human annotations, and detector outputs belong to their respective project contributors. The interval adapter follows this project's earlier visualization review code. No teammate detector is reimplemented here.

Commit the new Python, HTML, JSON, requirements, tests and this guide. Keep data and generated outputs out of Git.
