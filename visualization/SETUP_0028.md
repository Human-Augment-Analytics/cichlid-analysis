# Start here: export a 0028 results video

This is an additive update to the visualization folder already in the shared GitHub repository. It renders a shareable MP4 from existing team outputs. It does not train or run a detector.

## 1. Copy the code

Extract the download and copy the files inside its visualization/ folder into the repository's existing visualization/ folder. Keep the existing folder and files. If prompted, replace these export-specific files with the updated versions; no viewer or training files need to be removed.

The new main files are export_video.py, export0028.json and requirements-export.txt. Additional docs/tests and an optional synthetic example are supplied. Do not nest the folder as visualization/visualization/.

Expected layout:

```text
cichlid-analysis/
  visualization/
    __init__.py
    export_video.py
    export0028.json
    requirements-export.txt
    ...existing files...
  data/
    0028_vid.mp4
    0028_vid.parquet
  outputs/
    predictions/
      rule_based/
        0028_vid.csv
  ground_truth_annotations/
    finalized_annotations/
      merged_spans.csv
```

The video and Parquet are the existing research files. The two CSVs belong to the team and exist on origin/main; they are not supplied by this code update. The Parquet must be the individual pose table, not 0028_vid_pairs.parquet. Keep raw data out of git.

## 2. Install only the export dependencies

Open Terminal in the local cichlid-analysis repository root:

```bash
source .venv-viz/bin/activate
python -m pip install -r visualization/requirements-export.txt
ffmpeg -version
ffprobe -version
```

If .venv-viz does not exist, first run `python3.11 -m venv .venv-viz`. If FFmpeg/ffprobe are missing, install with `brew install ffmpeg`. No OpenCV, Marimo, DLC2Action or PyTorch is needed for this exporter.

## 3. Check the inputs before rendering

```bash
python -m visualization.export_video visualization/export0028.json --check
```

Expected: a printed export plan and “Input check passed. No video was rendered.” The plan lists the recording/project ID, selected TrackUID pair, valid geometry frames, reference/prediction intervals in the excerpt, output duration, and destination. Confirm these are sensible before sharing any results.

The selected pair is the most frequently closest valid pair during predicted frames in the excerpt (or across the excerpt if no predicted frames are available). This is a diagnostic convenience, not biological identification or attribution of scene-level circling. It can select reflections or incorrect tracks. It does not select a new partner at each playback frame. Missing selected tracks remain missing. Pair metadata is saved in the export provenance. Set pair to an explicit two-item list to override automatic selection.

If a file is missing, the check prints its resolved path. Place the correct data there, or edit that path in export0028.json. Paths are relative to the JSON file, so ../data refers to the repository's data directory. Do not move video files into visualization/.

If a required CSV is absent locally but origin/main has it, fetch first and inspect the paths:

```bash
git fetch origin
git ls-tree -r --name-only origin/main -- outputs/predictions/rule_based/0028_vid.csv ground_truth_annotations/finalized_annotations/merged_spans.csv
```

Update the branch from main through the team's normal merge workflow, or copy only the missing CSVs from that ref after preserving any local edits. Fetch alone does not change the working tree. Do not overwrite a CSV containing annotation work just to run the exporter.

## 4. Render the MP4

```bash
python -m visualization.export_video visualization/export0028.json
```

The command reads frames from the original recording, draws poses and results, and encodes the MP4. The example is a 30-second excerpt around a reference/predicted event:

- Source frames: 206206–207105, zero-based inclusive.
- Approximately 01:54:33.533–01:55:03.500; output duration exactly 30 seconds at 30 FPS.
- Expected project_id: MC_920_t001_tr1.
- Existing rule-based detector intervals, not a new model.
- Reference labels from merged_spans.csv, filtered to 0028_vid.

To preserve source-index alignment, the decoder starts at the beginning and stops after the excerpt. Because this section is almost two hours into the source, it can take several minutes before rendering starts. Progress messages appear every 300 rendered frames. Do not assume it is stuck merely because output frames have not started yet.

## 5. Watch and share

```bash
open outputs/visualization/exports/0028_rule_results.mp4
```

Use QuickTime/VLC or insert the file into a slide. No localhost server is needed. Check that skeletons follow the intended fish, timings align, and reference/prediction transitions make sense. The code has synthetic validation; the real 0028 video was not available during development.

The MP4 shows video, selected-pair keypoints/IDs, source frame/time, predicted-interval status, human reference status, distance/angular-velocity plots, and synchronized timelines. With rule-based interval input there is no numeric probability: “Model score: not supplied” is intentional. A numeric score requires a compatible actual frame-score file from the same model run.

Alongside the MP4, .provenance.json, .frames.csv and .features.csv preserve the rendering settings and frame states. The saved config and resolved pair make the selection reproducible.

To rerender an existing destination:

```bash
python -m visualization.export_video visualization/export0028.json --overwrite
```

## Changing the output

The 30-second range is a starter for checking alignment and preparing a presentation example; it is not the entire recording. Change start_frame/end_frame and the output filename for a different/longer excerpt. Current exports allow up to 18001 frames (about 10 minutes). A full-recording MP4 needs the duration guard and long-run strategy revised; this package does not claim full-recording export or interactive playback.

For 0031, the separate export0031.json is also included; it uses explicit known IDs. For another recording, supply its video, individual pose Parquet, detector CSV, and matching settings. Human reference is optional (remove annotations to display unknown). Different frame conventions require an explicit end_exclusive setting; do not assume all annotation exports use the same convention. See EXPORT_GUIDE.md.

## Tests and GitHub

```bash
python -m unittest visualization.test_export_video -v
```

Five tests cover automatic pair selection, missing observations, label uncertainty, source frame range/count, and input-check behavior. Tests include an actual synthetic FFmpeg export. The bundled example MP4 is clearly synthetic, not research evidence. The export-specific files can be committed to the existing visualization branch after review; data, environments, and large generated results should remain outside the code commit.

## How to describe this work

“I’m producing shareable annotated videos from the team's existing outputs. The MP4 combines the source footage, tracked keypoints, predicted circling intervals, human reference labels, and synchronized movement plots. My contribution is the visualization/export workflow; the underlying tracking, annotations and detector predictions come from the team.”

Next milestone: render and verify the 0028 example, then choose a small set of representative real clips for the presentation. Only claim real-data alignment or successful model detections after checking the footage and reference labels.
