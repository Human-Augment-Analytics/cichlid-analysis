# courtship-analysis

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed, then run from the repository root:

```sh
uv sync
```

`uv sync` installs the locked dependencies and the required Python version. Open the
local URL printed by marimo to use the notebook. On first launch, `analysis.py`
downloads the source videos and Parquet files into `data/`; the two videos are
about 31 GB each, so allow time and disk space for the initial download.

```sh
uv run marimo edit analysis.py
```
You can edit the marimo notebook from the browser with the above command, or just work on it in
VSCode with the Marimo extension, just like Jupyter.

## Guidelines for Circling Annotation and Span adjustment

1. Run `analysis.py` to force download of raw videos and dropbox files
2. Open/run `merged_spans_review.py`
3. Refine start and end frames, mark `Verified` column in `00xx_vid_merged_spans.csv` as `true` or `false`
    - Notebook is reactive and new preview will be rendered immediately after widget changes, but updating the csv file requires a manual run of the loading cell.
4. Commit changes and submit a pull request.

## notes

Export all 15 annotated circling spans with colored pose points and a corner TrackID legend:

```sh
uv run python circling_clips.py
```

Clips are saved in `outputs/circling_clips/`, with frame ranges and durations in
`clips.json`. Each clip includes both endpoint frames, at the source resolution
and frame rate. Keypoints have no text labels. Colors stay consistent by TrackID; frames without detections
remain in the video. The notebook displays all exported clips.

Train the circling logistic regression baseline:

```sh
uv run python circling_model.py train
```

Uses all 7,818 positive frames and 7,818 negatives sampled without replacement
(equal counts within each video, seed 42), followed by a stratified 80/20 split.
The 102 features combine the original 50 pose features (motion, bend, turning,
track count and trailing 30-frame means) with 52 pair features from the matching
`*_pairs.parquet`: mean/min/max of six distances, sine/cosine of four angles,
and three angular rates, plus pair count. Frames without pairs remain included. Imputation and scaling are fit
on training frames only. All frames outside annotated spans are negative.

Artifacts are in `outputs/circling_baseline/`: `report.md`, `metrics.json`,
`precision_recall.png`, the trained `model.joblib`, `coefficients.csv`,
`sampled_frames.parquet` (features, labels and split), and
`test_predictions.parquet`. The saved model is trained on the training split
only. Adjacent frames can occur in both splits, so test results do not establish
performance on unseen videos or independent circling events.

Predict a specific zero-based frame from a pose Parquet file:

```sh
uv run python circling_model.py predict --poses data/0031_vid.parquet --frame 265510
```

The command reads up to 30 preceding pose frames and the matching pair Parquet
file. Use `--pairs PATH` to override the automatically selected pair file. The reported
score reflects the balanced sampled dataset, not the real full-video prevalence.

Run feature/label checks:

```sh
uv run python -m unittest discover -s tests -v
```

In `analysis.py`, **Circling ground truth vs. logistic regression** displays aligned
teal ground-truth and purple prediction bars. Choose either video, an annotated
span or the full timeline, then adjust the frame range and prediction threshold.
Gray denotes non-circling; dashed lines mark annotation boundaries. Predictions
are generated for every displayed frame and cached while changing the threshold.
This review includes training frames, so it is separate from test-set evaluation.

The notebook's **Sustained positive predictions** section scans the entire video
selected in the timeline, using its current prediction threshold. It finds
301-frame windows with at least 90% positive classifications and greedily merges
overlapping/touching windows while preserving that fraction. Returned spans are
strictly longer than 10 seconds at nominal 30 FPS. The table lists every match;
a dropdown renders a selected span with colored pose points and a TrackID legend.
Each row also includes the video and event IDs, start and end frames and times, predicted
behavior, observed fish TrackIDs, and a positive-frame fraction with a note that
fish involvement is unverified. End time is the boundary after the last frame.
Full-video scores are cached as Parquet in `outputs/predicted_spans/`.
The span table loads from its CSV there when present; otherwise the notebook computes
it and saves both CSV and Parquet files. Delete the CSV to recompute the spans;
full-resolution clips and compact previews are generated on selection.
