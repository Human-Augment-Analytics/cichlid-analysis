# Optional annotated MP4 export

This remains a separate short-clip tool. It uses local files; catalog selection and Dropbox streaming apply to the interactive player, not this exporter.

From the repository root, with `.venv-viz` active:

```bash
python -m pip install -r visualization/requirements-export.txt
python -m visualization.export_video visualization/export0028.json --check
python -m visualization.export_video visualization/export0028.json
open outputs/visualization/exports/0028_rule_results.mp4
```

FFmpeg and FFprobe are required (`brew install ffmpeg` if missing). Existing files are protected; use `--overwrite` to regenerate. For 0031 use `export0031.json` and the corresponding local inputs. Output settings and paths are relative to the JSON file.

Inputs are original local video, individual pose Parquet, predicted intervals and optional accepted human reference/score files. The default 0028 clip contains frames 206206–207105 (30 seconds at 30 FPS) and automatically chooses a diagnostic pair. Prediction/reference intervals default to zero-based inclusive source frames. The exporter does not perform pose estimation, training, or behavior inference.

Outputs are silent H.264 MP4 plus provenance, frame-label and feature sidecars. Missing poses remain missing, scores must be actually supplied, unknown reference time is not negative, and scene-level predictions do not identify the displayed pair. Local frame/time alignment must be checked. Default 0031 IDs are explicit and must not be reused on another recording.

Rendering can be slow: the exporter decodes from frame zero to reach its exact excerpt, draws overlays and encodes the result. Use the interactive player for full-recording exploration. The renderer limits an excerpt to 18001 frames; it is not a ten-hour video export workflow.

For a clearly synthetic software demonstration, run `python -m visualization.demo_export`. Run `python -m unittest visualization.test_export_video -v` for synthetic export/geometry checks. No research data or pre-rendered demo is bundled. Keep raw data, model artifacts and generated clips out of visualization source commits.
