# Visualization and Experiment User Guide

Run commands from the shared cichlid-analysis repository root, after adding this download's visualization/ folder. Current main supplies the annotation/prediction CSVs referenced by examples. Do not install the repository's full modeling environment just to use this module.

## Install once

```bash
python3.11 -m venv .venv-viz
source .venv-viz/bin/activate
python -m pip install -r visualization/requirements.txt
```

Install FFmpeg if missing: `brew install ffmpeg`. Reuse .venv-viz on later sessions by activating it. Training additionally needs:

```bash
python -m pip install -r visualization/requirements-learning.txt
```

Training dependencies are much larger. CPU is the default and supported Mac starting point; no GPU/MPS is required. The model imports DLC2Action's MS_TCN3 with a custom PyTorch loop; this does not run the DLC2Action GUI or full Project workflow. No SSL, active learning, or video-pixel model is implemented.

## 1 — Visualize existing data

Need: video + matching individual pose Parquet. Annotation and detector CSVs are optional, but the supplied example references files from current main.

```bash
python -m visualization.review visualization/seed0031.json
python -m http.server 8000
```

Open http://localhost:8000/outputs/visualization/0031_first_event/ . Shows poses, IDs, pair distance/angular velocity/proximity, separate human/model intervals, comments, and review notes. Nothing is trained. Use --list-tracks with a review config to find tracks in another excerpt. Config paths are relative to that config; outputs named by command-line options are relative to the working directory.

## 2 — Test a proximity hypothesis

Need: individual pose Parquet + human annotations. Video is optional for metrics and required for actual video clips.

```bash
python -m visualization.experiment visualization/hypothesis0031.json
python -m visualization.experiment visualization/hypothesis0031.json --clips 3
```

Open http://localhost:8000/outputs/visualization/hypothesis0031/ . The first command only computes results; the second also extracts up to three example clips. The supplied scope is two minutes, not the whole video. Sweep distance/duration settings and inspect retained/partial/missed events, positive-frame coverage, and review load. Extra candidates outside annotated time remain unverified. Choose a rule on development data and freeze it before a held-out assessment.

## 3 — Train a model, predict, and visualize

Need for training: `data/0028_vid_pairs.parquet`, `data/0031_vid_pairs.parquet`, and reviewed annotations. For learned video review also need `data/0031_vid.mp4` and `data/0031_vid.parquet`. The training adapter currently requires pairs tables with the original distance/angle/rate fields. A new individual pose file alone supports paths 1/2, but requires upstream pair-feature generation before this training path.

The supplied config uses main's merged_spans.csv (accepted event decisions) and explicitly assigns 0028=train, 0031=test. Both currently have positives only. Strict mode will not invent negatives:

```bash
python -m visualization.training prepare visualization/training_seed.json
python -m visualization.training train --epochs 10
```

The training command will stop with a clear error if the prepared training set has no labeled negatives. Add independently reviewed non-circling intervals to your own annotation CSV and point the config to it. Supported frame columns: VideoID, StartFrame, EndFrame, Circling, ShouldUse, Notes. Use Circling=false and ShouldUse=true for accepted negatives; positive rows use both true; rejected rows remain unknown. Do not overwrite shared source annotations to invent labels. Include both positive and negative rows in that selected source.

### Explicit exploratory option (same assumption as the old seed-learn command)
If you want to reproduce positive-only experimentation, explicitly prepare with:

```bash
python -m visualization.training prepare visualization/training_seed.json --assume-background
python -m visualization.training train --epochs 10
python -m visualization.training predict --split test
python -m visualization.training review visualization/seed0031.json
```

Open http://localhost:8000/outputs/visualization/learned_review/ . The learned viewer contains the same poses/plots plus frame scores and learned interval tracks. A visible score confirms a prediction exists for that frame, not biological correctness.

**Assumed background means unannotated sampled context is treated as negative, for both train and test labels.** It can include missed circling. Metrics are labeled as assumed-background and are exploratory, not verified precision. Unknown/rejected/conflicting intervals stay ignored. Default sampling is annotation intervals plus five seconds on either side, so reported results are context-sampled, not a whole-video benchmark. Set annotation_context_seconds to null and appropriate frame scope in your config for full-range inference/evaluation; account for time and memory. Full evaluation still requires independent negatives/background review.

Training creates outputs/visualization/learning/{prepared,model,predictions}. Important outputs: model/model.pt, model/history.json, predictions/predictions.csv, predictions/predicted_segments.csv, predictions/metrics.json, setup_provenance.json. Preprocessing is fitted on training known frames only. Predictions use 256-frame windows with half overlap, last-stage softmax, and center-weighted averaging. Scores are uncalibrated and scene-level; they do not identify the responsible fish pair. The model uses 52 all-pair aggregate features (104 after missing-value indicators), not the viewer's selected-pair feature vector. Checkpoints from other pipelines are not automatically migrated.

### Predict a new video without retraining

Create a training-style JSON containing just the new video with split=test, matching pairs path, and no annotations. Set annotation_context_seconds=null. Prepare into a separate run, then point prediction to the existing model:

```bash
python -m visualization.training prepare visualization/local_predict.json --out outputs/visualization/new_prediction
python -m visualization.training predict --run outputs/visualization/new_prediction --model outputs/visualization/learning/model --split test
python -m visualization.training review visualization/local_new_video.json --run outputs/visualization/new_prediction --out outputs/visualization/new_learned_review
```

The review config needs the new individual pose file, video, scope, and pair IDs; remove old source timelines. No annotations are needed to predict, but metrics cannot measure correctness without them. All input fields/units must match training. Prediction on unfamiliar species/recording conditions requires validation. Same video/source must not appear under different split names or IDs; hash checks catch identical inputs, but cannot detect all renamed/trimmed variants.

## Workflow responsibilities

- Visualization: inspect data and any teammate's outputs.
- Hypothesis test: quantify a specific candidate-generation rule.
- Optional learning: train a scene-level model and inspect its outputs using the common viewer.

You can compare rule and learned interval tracks, but this package does not automatically produce a fair model-versus-rule precision comparison. Match scopes/labels and use independent reviewed negatives before making that claim. Keep hyperparameters fixed before looking at test results.

## Synthetic training smoke test

```bash
python -m visualization.demo_training
python -m visualization.training prepare outputs/visualization/training_demo_inputs/training.json --out outputs/visualization/training_demo
python -m visualization.training train --run outputs/visualization/training_demo --epochs 1
python -m visualization.training predict --run outputs/visualization/training_demo --split test
python -m visualization.training review outputs/visualization/training_demo_inputs/review.json --run outputs/visualization/training_demo --out outputs/visualization/training_demo_review
```

This learned synthetic viewer has no video pixels, but does show invented poses/geometry/labels/model scores. It is a software check, not fish research evidence. Use the separate visualization.demo for the bundled synthetic video example.

## Verification and credit

Thirteen adapter/preparation tests pass. An actual one-epoch DLC2Action MS-TCN synthetic train → held-out prediction → learned viewer export passed. Model scores were present in the exported viewer data. This is functionality verification, not real fish performance. Real source video alignment and browser playback still require local checking.

Generated/assembled with OpenAI Codex assistance. The training backend reuses the earlier AI-assisted cichlid-viz prototype; scene_features.py preserves its original aggregation. Human annotation and existing detector outputs belong to their respective project contributors. No teammate detector implementation was copied into this module. DLC2Action and PyTorch supply the model/runtime; preserve their respective licenses. DLC2Action 1.0 is AGPL-licensed; this package does not bundle its source or weights, and users install the dependency separately. The repository's own license continues to apply to the contribution.
