# Optional annotated MP4 export

Use this when you want a shareable clip; use the live player for interactive full-recording review.

## Setup and run

From the repository root with `.venv-viz` active:

```bash
python -m pip install -r visualization/requirements-export.txt
python -m visualization.export_video visualization/export0028.json --check
python -m visualization.export_video visualization/export0028.json
open outputs/visualization/exports/0028_rule_results.mp4
```

FFmpeg and FFprobe must be installed (`brew install ffmpeg` if missing). No separate HTTP server is needed. Existing outputs are protected; append `--overwrite` to regenerate.

For 0031 use `visualization/export0031.json` and its matching local source files. The configuration's `out` field gives the output path.

## Inputs and output

Inputs are the original video, individual pose Parquet, detector interval CSV and optional accepted human annotations. Default 0028 paths match those documented in README.md. Numeric scores are optional. No pose/behavior inference or training occurs.

Output is a silent H.264 MP4 with video, selected-pair pose overlays, distance/angular-velocity plots, reference/prediction status and interval timelines. Sidecar `.provenance.json`, `.frames.csv` and `.features.csv` record settings, label states and measurements. Keep generated files local under `outputs/visualization/`.

The default 0028 example covers frames 206206–207105, inclusive (30 seconds at 30 FPS). It automatically chooses a diagnostic pair within that excerpt. This is not actor identification. The default 0031 example specifies its pair explicitly; do not reuse old IDs on other recordings.

## Change settings

Copy an example JSON and update recording/project IDs, video/pose paths, source-frame bounds, detector name and prediction CSV. Paths are relative to the JSON file. Use `pair: "auto"` or specify two TrackUIDs. Run `--check` before rendering to review the selected pair and available intervals. This exporter permits up to 18001 frames; it is not a full ten-hour recording renderer.

Frame CSVs support `VideoID`/`video_id`/`video`, and `StartFrame`/`EndFrame`, spaced `Start Frame`/`End Frame`, or lowercase `start_frame`/`end_frame`. Endpoints default to inclusive; set `end_exclusive: true` in a source specification when required. Timestamp-only source labels must first be converted using an agreed frame mapping.

For merged references, `ShouldUse=true` and `Circling=true/false` supply explicit positive/negative decisions. Rejected/incomplete/approximate or conflicting references are not silently treated as verified negatives. For another accepted label source, use `Verified` fields or explicitly set `verified: true` in its annotation specification. Confirm the label schema before doing so. Missing annotations mean unknown.

Optional numeric scores can be configured as:

```json
"scores": {
  "path": "../outputs/my_model/frame_scores.csv",
  "video": "video_id",
  "frame": "frame",
  "score": "score"
}
```

Use scores and interval predictions from the same model/run. Do not substitute interval averages for per-frame scores or interpret uncalibrated scores as certain probabilities.

## Performance and limitations

The exporter decodes from source frame zero to reach the excerpt, then draws overlays with Pillow and encodes a new MP4. Late clips can take minutes even when the final clip is short. Prefer the live player to navigate recordings without export.

FPS is checked against video average FPS, but that does not prove constant-rate source timestamps or correct pose alignment. Missing geometry remains missing. Pair movement is calculated independently of upstream detector cleaning and smoothing. Circling predictions do not confirm spawning/fertilization. Direction and circle counts are not inferred here.

## Synthetic example and checks

```bash
python -m visualization.demo_export
python -m unittest visualization.test_export_video -v
```

The demo generates synthetic video, poses and invented predictions under `outputs/visualization/export_demo/`. It is a software demonstration, not research evidence. The package does not bundle source research data or a pre-rendered demo.
