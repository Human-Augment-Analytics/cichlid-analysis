# Shareable pipeline-results videos

For a clean 0028 setup, start with [SETUP_0028.md](SETUP_0028.md).

The exporter turns existing team data into a silent, presentation-ready H.264 MP4. A recipient only needs a video player. It does not train a detector, generate poses from pixels, or improve model performance.

## Install

Copy the additive files in this update into the existing visualization/ package. Run from the repository root:

```bash
source .venv-viz/bin/activate
python -m pip install -r visualization/requirements-export.txt
```

If that environment does not exist, create it first with `python3.11 -m venv .venv-viz`. FFmpeg and ffprobe must be installed and on PATH (`brew install ffmpeg` on macOS if needed). OpenCV, Marimo, DLC2Action and PyTorch are not required to export already-existing results. This file uses Pillow to draw text, so FFmpeg's drawtext filter is not needed.

## Generate the configured 0031 result video

Required local files:

| Input | Default path | Purpose |
| --- | --- | --- |
| Original recording | data/0031_vid.mp4 | Video pixels |
| Matching individual pose Parquet | data/0031_vid.parquet | Centers, keypoints and TrackUIDs |
| Team's rule-based predictions | outputs/predictions/rule_based/0031_vid.csv | Predicted circling intervals |
| Human reference decisions | ground_truth_annotations/finalized_annotations/merged_spans.csv | Accepted labels and explicit unknowns |

The CSVs are from the shared repository; the large video/Parquet must be local. No pairs Parquet is required: the two displayed diagnostic features are computed from box centers. The default verifies pose project_id MCYHF1-P-613_t011_tr2, not only the video basename.

```bash
python -m visualization.export_video visualization/export0031.json
```

Open `outputs/visualization/exports/0031_rule_results.mp4` in QuickTime, VLC, a browser, or a slide deck. No localhost server is needed to watch it.

The configured excerpt is 30 seconds: source frames 97830–98729, approximately 00:54:21–00:54:51. Endpoints are inclusive. TrackUIDs 30004551 and 30004572 are the explicitly selected diagnostic pair; they need not remain visible for the whole excerpt. Missing poses/geometry remain missing, rather than changing fish identities silently.

The file shows:
- Video with the selected fish's skeletons and IDs in consistent colors.
- Original source frame and elapsed time.
- Scene-level detector interval status (predicted circling / no predicted interval).
- Human reference status (circling / explicit non-circling / unknown).
- Full-excerpt distance and angular-velocity plots with synchronized cursors.
- Human and predicted interval timelines.
- A numeric score only if an actual per-frame score file is supplied.

The output is a static MP4 with the overlays burned in, unlike the interactive HTML viewers. It is suitable for demonstrations, reports, and short error-analysis clips. The `.provenance.json`, `.frames.csv`, and `.features.csv` beside it record inputs/settings and support inspection. Keep large results and source data out of source-code commits unless the team agrees on artifact storage.

## Adjust a clip, model, or recording

Copy export0031.json to a local JSON configuration. Every path is relative to the JSON file. Set video_id, expected_project_id (or omit only after confirming identity), video/pose/output paths, source-frame bounds, selected TrackUIDs (or pair="auto"), detector name and prediction file. Use --check to validate inputs and print the resolved pair/export plan before decoding video. Never reuse old IDs or labels on a new recording. Use the existing review command with --list-tracks if ID discovery is needed, or inspect the matching pose file.

Frame-based CSV inputs support VideoID/video_id/video; StartFrame/EndFrame, Start Frame/End Frame, or start_frame/end_frame. Defaults are zero-based inclusive bounds; set end_exclusive=true inside the relevant predictions or annotations object for exclusive-end producer output. Timestamp-only annotation CSVs are deliberately not guessed here; use a reviewed frame-based export.

For merged review files, ShouldUse=true and Circling=true/false explicitly determine positive/negative reference status. Rejected rows are unknown. For other human-label files, set verified=true inside annotations only when that source is an accepted human-reviewed export, or provide Verified/verified fields. Primary label/behavior, Behavior and behavior are supported. Approximate circling remains unknown here; it is not silently converted to a negative. Conflicting positive/negative/unknown overlaps render as unknown.

Delete the annotations entry if no human reference exists; the video then displays reference unknown, not an accuracy judgement. New videos require matching poses and existing detector predictions. The exporter does not run a model on a new video.

For a trained model's real per-frame scores, optionally add:

```json
"scores": {
  "path": "../outputs/visualization/learning/predictions/predictions.csv",
  "video": "video_id",
  "frame": "frame",
  "score": "score"
}
```

Also point predictions to that same run's predicted_segments.csv and label detector_name accordingly. Scores and intervals should come from the same model/run; the exporter cannot establish this automatically. Interval mean_score and positive_fraction are not frame probabilities and must not be substituted. Rule-based interval files have no probability: “Model score: not supplied” is intentional.

The prediction scope may be scene-level, while geometry is for one selected pair. Do not claim that the displayed pair produced the detector output. Neither circling nor a high score confirms spawning/fertilization. Direction and turn counts are not inferred by this exporter.

## Performance and output conventions

FFmpeg decodes from source frame zero to select exact source indices; it stops after the excerpt is decoded. A late clip in a 10-hour recording can therefore take several minutes. Start with the 30-second example; do not launch hundreds of late clips independently. Streaming avoids retaining all video frames in RAM, but this version does not implement optimized multi-clip batching. Output plays at the configured nominal FPS and contains no audio. FPS must agree with the video average within .01; nominal agreement does not prove every timestamp is uniformly spaced. Pose/video alignment still needs visual verification on research footage.

Existing outputs are protected unless --overwrite is passed. Do not use an output path equal to a source. Model/pose inference is expected upstream. The renderer uses exactly one detector for a clean presentation; the separate comparison dashboard supports multiple-detector analysis.

## Try without research data

The download includes a pre-rendered, clearly marked synthetic results_demo.mp4. It demonstrates the layout, not real fish or trained-model results. To regenerate:

```bash
python -m visualization.demo_export
```

The generated synthetic source/labels/predictions live under outputs/visualization/export_demo/. Five tests (`python -m unittest visualization.test_export_video -v`) verify label states, missing-data geometry, approximate-label policy and FFmpeg output frame count/range. Actual footage is not available here; alignment must be checked locally.

## Attribution

Video and tracking data are from the lab; annotation and detector results are the respective contributors' work. Credit the detector by name and source version in a presentation. The exporter and example were generated with OpenAI Codex assistance, using Pillow, NumPy/pandas/pyarrow, and FFmpeg. This is a visualization/export contribution, not a new behavioral model. Existing repository licensing applies.
