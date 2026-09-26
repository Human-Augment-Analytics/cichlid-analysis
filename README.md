# courtship-analysis

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
Each row also includes the video and event IDs, start and end times, predicted
behavior, observed fish TrackIDs, and a positive-frame fraction with a note that
fish involvement is unverified. End time is the boundary after the last frame.
Full-video scores and span tables are cached in `outputs/predicted_spans/`;
full-resolution clips and compact previews are generated on selection.
