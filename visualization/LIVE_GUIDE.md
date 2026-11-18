# Full-recording player configuration

Start from the repository root with `python -m visualization.live`. This loads `visualization/recordings.json`. For the earlier single-recording workflow, `python -m visualization.live visualization/live0028.json` still works. Append `--port 8002` to change ports.

## Catalog and discovery

`recordings.json` contains a `recordings` list and a default recording ID. Each entry has a unique `id`, a source `video_id`, the expected project ID, local paths, optional remote URL and known video metadata. All relative paths resolve against this JSON file. The catalog is configuration, not a generated result.

The supplied 0028/0031 entries include direct shared-file links for Dropbox, original pose paths, optional cleaned paths, and accepted reference CSVs. `discover_predictions: true` enables matching prediction/score folders for these known seed videos. The loader also reads the current repo's VIDEO_URLS dictionary in `video_source.py` if an entry has no explicit URL, without importing its OpenCV/model dependencies.

With `discover_local: true`, the catalog lists additional `data/*_vid.mp4` files and looks for a same-basename individual Parquet. These get no seed annotations/predictions attached automatically. They default to 30 FPS and are marked identity-unconfirmed: verify FPS, project and coordinate alignment by making an explicit catalog entry. Nested project folders and remote Dropbox folders are not automatically enumerated; register those recordings explicitly.

**Basenames are reused across projects.** Use a unique catalog `id` such as `project538_day05_0007`, the true `video_id` inside that file, and `expected_project_id`. Disable automatic output discovery for reused basenames, and explicitly configure the correct output paths. Video footage identity still needs visual confirmation; dimensions alone cannot identify a recording.

Example extra entry (replace every example value with real source information):

```json
{
  "id": "unique_project_recording",
  "video_id": "0007_vid",
  "expected_project_id": "YOUR_PROJECT_ID",
  "label": "Project name · day · video 0007",
  "video": "../data/project_name/0007_vid.mp4",
  "video_url": "https://www.dropbox.com/scl/fi/SHARED_FILE/0007_vid.mp4?raw=1&rlkey=YOUR_KEY",
  "poses": "../data/project_name/0007_vid.parquet",
  "fps": 30,
  "width": 1296,
  "height": 972,
  "frame_count": 1080000,
  "discover_predictions": false,
  "tracks": []
}
```

Use the real dimensions/FPS/frame count, not the example defaults. The browser updates duration and dimensions from the actual video after it opens. For local video, FFprobe also checks average FPS; remote video FPS remains a configured assumption. Click **Refresh available outputs** after changing configuration or adding files. Refresh reloads inputs; it does not run training or prediction commands.

## Dropbox playback

Choose **Dropbox / remote** to stream even if a local file exists. **Auto** chooses local if present, otherwise the registered remote video. A local decode error can trigger one remote fallback; missing local ffprobe or invalid local metadata instead needs fixing or an explicit switch to Dropbox.

The server forwards video byte-range requests and streams responses in bounded buffers. It does not save a full copy. Watching transfers data, and the browser may buffer ahead. Seek performance depends on connection, source keyframes, and Dropbox's range support. Only the two seed links are preconfigured; arbitrary folder-preview URLs are not a playable video inventory. The proxy converts standard Dropbox shared-file URLs to raw links; a web-page response or ignored seek range produces an error instead of misaligned video.

Keep matching Parquets/CSV outputs local initially. No Dropbox login is handled by this tool; links must be directly accessible in the environment running Python. Shared links can expire or change permissions. Update the URL or choose a local file if remote playback fails. Avoid committing private signed URLs for restricted data unless the team approves their use in the repository. The supplied defaults match links already present in the shared repository.

This is streaming of **recorded footage**, not live camera capture or model inference. The server binds to 127.0.0.1 and exposes only configured recording sources and APIs, not directory browsing. Remote URLs come only from trusted local configuration, never from browser-supplied arbitrary URLs.

## Prediction intervals and scores

For enabled seed recordings, interval discovery scans `outputs/predictions/*/<video_id>.csv`. Add an explicit track when using another location:

```json
{"name":"Model/run name", "kind":"prediction", "path":"../outputs/custom_run/0028_vid.csv"}
```

Required frame columns default to `StartFrame`/`EndFrame`, spaced `Start Frame`/`End Frame`, or lowercase `start_frame`/`end_frame`. Video selectors recognize `VideoID`, `video_id`, or `video`. Defaults are zero-based inclusive; set `end_exclusive: true` for exclusive-end output. Timestamp inputs need the agreed FPS/time mapping and can be explicitly specified via start/end column names and units. Do not guess a producer's frame convention.

Annotation tracks use `kind: "annotation"`. In the merged reference file, `ShouldUse=true` plus `Circling=true/false` defines accepted positives/negatives; rejected or missing decisions stay unknown. Other label sources require explicit Verified fields or a deliberate source-level `verified: true` policy. Approximate circling stays unknown. Conflicting accepted positive/negative intervals display as unknown. A non-overlapping model prediction is a review candidate, not automatically a false positive.

Per-frame score discovery scans `outputs/*/scores/<video_id>_scores.parquet` and recognizes `FrameNum`/`circling_score` (or `frame`/`score`). Add an explicit score source when needed:

```json
"scores": [
  {"name":"Custom model/run", "path":"../outputs/custom/scores.parquet", "video":"VideoID", "frame":"FrameNum", "score":"circling_score"}
]
```

Scores must be finite values in [0,1], with a unique frame per recording. Missing values stay unavailable. Only a ten-second score section is read at a time. CSV score sources are also supported but are reread per section; Parquet is preferable for long recordings. Select which source to plot; all interval tracks remain visible. Score/interval run consistency is the user's responsibility: name the run clearly and consult provenance/metrics before comparing. Interval averages and binary detections are not probabilities. Existing models may have different splits, training overlap, postprocessing and calibration.

## Shared pairwise features

The default shared source is `outputs/pair_motion/pair_features_frame.parquet` when present for enabled seed recordings. Generate it with the feature team's existing pipeline or obtain it from them. The viewer never generates it. The small sample CSV in the repo is not a full-recording feature table and is not substituted.

Set `features` to another Parquet path if required. It needs `FrameNum`, `TrackUID_1`, `TrackUID_2` and supported feature columns. A shared multi-recording table needs `day_label`/`VideoID`/`video_id`/`video`; `project_id` is filtered when expected_project_id is supplied. Values are matched to the selected pair with UID ordering. Pair changes, missing feature rows or different ID namespaces leave plots blank.

Available columns include shared centroid distance (pixels/body lengths), orbital velocity and cleaned velocity, four-second close fraction, net/absolute turns, mutual facing, pair speed and closing speed. The table can use midline centroids; the viewer's original distance/angular plots use box centers. They need not agree exactly. Some feature windows are centered and use future observations; displaying them is not proof of live-ready processing. Shared turn features are window summaries, not validated event circle-count labels.

## Optional cleaned poses

Set `cleaned_poses` to a matching individual cleaned Parquet. Default examples point at `data/clean/<video_id>.parquet`; if Aaron's script writes elsewhere, update the path. The cleaned selector appears only if the file exists and has the required schema. The new validation scripts are not bundled; obtain them through the team's branch/merge workflow. Cleaning and pair-file rebuilding remain upstream tasks.

The viewer uses FrameNum, TrackUID, X_center/Y_center and available keypoint coordinate pairs. Missing keypoints remain missing. Duplicate frame/UID rows fail the section rather than drawing ambiguous poses. Parquet identity metadata is checked when present; record/project predicates isolate multi-recording tables. Missing or rejected pose files permit video-only viewing and report a warning.

Changing layers changes overlay and box-center geometry only. Predictions, scores and shared features stay as produced; an actual before/after detector evaluation needs two separately generated runs. Track IDs can fragment and sex is not inferred here. Selecting a closest pair does not identify courtship actors.

## Timing, loading and limits

The browser uses presented-frame media time when supported; otherwise playback time. Frame indices are estimates based on configured FPS. ±1 frame seeks in time and is not independently verified decoder stepping. Verify constant-rate video/Parquet alignment before reporting exact boundary accuracy.

Ten-second sections use Parquet predicates and small in-memory caches. Large row groups can still decompress slowly; poses clear while a new section loads. A recording switch resets data so old annotations/poses are not reused. The full-recording timeline compares saved event intervals; movement/score plots show the current section.

The tool does not: train models, run pose or behavior inference, generate alerts, calculate formal model benchmarks, infer direction/circle counts from footage, or automatically label negatives. Those can be integrated once upstream outputs and scope are agreed.
