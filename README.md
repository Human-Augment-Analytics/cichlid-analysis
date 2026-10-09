# cichlid-analysis

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

## DLC2Action segmentation notebook

```sh
uv sync
uv run marimo edit dlc2action_notebook.py
```

The notebook loads saved results and lets you rerun a small DLC2Action 1.0
MS-TCN experiment, inspect training loss, select a held-out merged span, and
compare reviewed labels with frame scores and predicted segments. It uses the
DLC2Action model API with a custom PyTorch training loop and the existing 102
pose/pair features, without requiring another pose-file conversion.

All rows from `outputs/merged_spans/*_merged_spans.csv` define the excerpts.
`Verified=true` spans are positive, with inclusive endpoints. Rejected or blank
rows are ignored in the loss and metrics. Frames outside the CSV spans but
within the context buffer are **assumed background**, not verified negatives.
The default buffer is 150 frames per side. Overlapping excerpts are united;
512-frame model windows never join disjoint excerpts. Padding is masked.

The complete `0028_vid` excerpts train the model; `0031_vid` excerpts are held
out. Median imputation and scaling fit only supervised training frames. The
small two-stage model trains for five CPU epochs with seed 42, class-weighted
cross-entropy at both stages, and no test-based checkpoint selection. Predictions
are video-level circling, without assigning actions to individual fish.

Run the same experiment without opening marimo:

```sh
uv run python dlc2action_segmentation.py --epochs 5
```

Results go to `outputs/dlc2action_test/`: an annotation/split snapshot in
`span_manifest.csv`, `metrics.json`, `report.md`, `predicted_segments.csv`,
`test_predictions.parquet`, model weights in `model.pt`, and fitted preprocessing
in `preprocessing.joblib`. Binary artifacts are ignored by git and regenerated
by the command. The notebook warns if the source CSVs differ from the saved
experiment and uses the saved labels for its plots. Source pose/pair file sizes
and modification times are recorded in the metrics file.

The initial test used 153 rows (140 accepted, 13 ignored) and evaluated 54,484
held-out frames: precision 0.9195, recall 0.8864, F1 0.9026, AP 0.9611. This is
an excerpt-based smoke test with assumed background labels, not a full-video
performance estimate. Predictions use a fixed 0.5 threshold without smoothing
or minimum-duration filtering, so short segments and window-edge artifacts
can occur.

Check the adapter and export an executed, static notebook:

```sh
uv run python -m unittest discover -s tests -p 'test_dlc2action_segmentation.py' -v
uv run marimo check dlc2action_notebook.py
uv run marimo export html dlc2action_notebook.py -o outputs/dlc2action_test/notebook.html
```

## VideoMAE → MS-TCN visual segmentation

```sh
uv sync
uv run python videomae_pipeline.py
uv run marimo edit videomae_notebook.py
```

This pipeline adapts a pretrained VideoMAE visual encoder to the seed clips,
extracts visual embeddings, and trains DLC2Action's MS-TCN on those embeddings.
It uses the buffered excerpts defined by **all merged-span rows**, with the same
label policy and video split as the pose baseline. Only `0028_vid` is used for
training either model. `0031_vid` stays held out, including during VideoMAE
adaptation. These are excerpt-based results, not predictions across the entire
20 hours of source footage. CSV boundaries determine excerpt selection, so this
is not a blind search of full videos.

The encoder starts from `MCG-NJU/videomae-base-finetuned-kinetics`, pinned to
revision `488eb9a0565f257b32866000305c8178965eb9f6`. Its last two transformer
blocks, pooling normalization, and a new binary classification head are
fine-tuned with supervised circling/background clip labels. The earlier layers
are frozen. This is **supervised fine-tuning**, not masked-reconstruction
pretraining from scratch. Defaults: 256 class-balanced training clips sampled
across training excerpts, three epochs, AdamW at `1e-5`, weight decay `0.01`,
batch size 2, gradient norm limit 1, and seed 42. A saved weight-change check
confirms that the visual encoder itself was updated. MPS/CUDA is selected when
available; accelerator arithmetic is not guaranteed bitwise reproducible.

Each clip contains 16 RGB frames sampled every two source frames (approximately
one second). Frames are letterboxed to 224×224 with the full scene retained,
then normalized with the checkpoint's ImageNet mean/std. Clip edges replicate
frames within the excerpt. Embeddings use mean-pooled final hidden tokens and
the learned pooling normalization, yielding 768 features. There are no pose or
pair features in the model input. Pose overlays are used only for review videos.

Embeddings are sampled every 16 source frames, plus the last cached frame of
each excerpt, and linearly interpolated to original frame resolution within
each excerpt. The last source frame uses endpoint extension if needed.
This is offline, noncausal inference; no interpolation crosses excerpt gaps.
The embedding spacing limits boundary precision even though output rows are
per-frame. MS-TCN uses the same small architecture as the pose baseline,
512-frame windows, and 30 training epochs by default. Imputation and scaling
use only supervised training frames. No test-based model selection is used.

```sh
uv run python videomae_pipeline.py --videomae-epochs 3 --samples 256 --stride 16 --epochs 30
```

Artifacts are in `outputs/videomae_mstcn/`. In addition to the standard metrics,
frame predictions, and predicted segments, `encoders/` contains the adapted
VideoMAE weights, configuration, training history and training-clip manifest;
`embeddings/` contains 768-D vectors with original frame anchors. `frames/`
contains reusable RGB frame caches. Completed encoder/embedding caches are
keyed by settings and input signatures and reused on rerun. These large caches,
weights, per-frame predictions, and previews are ignored by git. Keep them to
avoid recomputation; the CLI regenerates them if absent.

The new notebook includes both training curves, segmentation metrics, a
comparison with saved pose-baseline results when evaluation labels match, and
segment video previews. Changing the form's settings reruns the pipeline,
reusing compatible caches. This implementation is our seed-video experiment,
not an exact reproduction of the paper's unspecified VideoMAE extractor.

### Temporal evaluation

Both segmentation pipelines now compute these metrics in addition to existing
frame precision/recall/F1, AP and ROC-AUC:

- **MoF:** percentage of correctly labeled frames, including background.
- **Edit Score:** 100 × (1 − normalized Levenshtein distance) between sequences
  of run-collapsed action labels. Background is excluded. Scores are averaged
  over contiguous evaluation runs; two empty action sequences score 100.
- **F1@10, F1@25, F1@50:** segment-level F1 at temporal intersection-over-union
  thresholds 0.10, 0.25 and 0.50. Matches are one-to-one and require the same
  class. Background segments are excluded; TP/FP/FN are pooled over runs.
- **Edit including background:** a supplementary score retaining background
  runs. Foreground-only Edit has limited ordering information for one action
  class and largely reflects fragmentation or missing segments.

All new scores use the 0–100 scale; existing frame metrics remain on 0–1.
Ignored annotations and missing-frame/excerpt gaps split evaluation into
independent contiguous runs. Segments intersecting ignored intervals are thus
clipped at those boundaries; unknown regions cannot join or penalize matches.
Assumed background labels remain a limitation of every evaluation metric.
The implementation follows the [MS-TCN evaluation conventions](https://github.com/yabufarha/ms-tcn/blob/master/eval.py),
with explicit handling of ignored labels and excerpt gaps.

```sh
uv run python -m unittest discover -s tests -p 'test_temporal_metrics.py' -v
uv run python -m unittest discover -s tests -p 'test_videomae_pipeline.py' -v
uv run marimo check videomae_notebook.py
```

Re-evaluate saved visual predictions and replay the saved MS-TCN checkpoint from
cached embeddings without retraining:

```sh
uv run python evaluate_videomae.py
```

This writes `evaluation.md`, `evaluation_checks.json`, and
`evaluation_comparison.csv` in `outputs/videomae_mstcn/`. If the saved pose run
uses different excerpts, the comparison uses only shared frames with identical
labels; full visual-run metrics remain in `metrics.json` and `report.md`.
The completed visual run scores MoF 54.18%, Edit 40.16%, F1@10/25/50
18.92/8.78/4.05%, and frame F1 0.4691 on 54,484 labeled held-out frames.
It underperforms the pose baseline; encoder training accuracy is not evidence
of held-out segmentation quality.
