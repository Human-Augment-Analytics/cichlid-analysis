# cichlid-analysis

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed, then run from the repository root:

```sh
uv sync --locked
```

`uv sync` installs the locked dependencies and the required Python version. Open the
local URL printed by marimo to use the notebook. `model_performance_report.py` evaluates
the committed predictions in `outputs/predictions/`. On first launch it downloads the pose and
pair Parquet files into `data/` (about 260 MB per video). Clip previews stream only the frames
they need from the source videos (about 13 s to start, then cached in `outputs/clips/`); a video
already in `data/` is used instead. Scripts that scan whole videos still need local copies.

```sh
uv run marimo edit model_performance_report.py
```
You can edit the marimo notebook from the browser with the above command, or just work on it in
VSCode with the Marimo extension, just like Jupyter.

## Repository layout

- `data/`: video and parquet files
- `outputs/`: generated artifacts, model files, etc
  - `outputs/predictions/<model_name>/<video_id>.csv`: predicted circling spans, one file per
    video (`VideoID`, `StartFrame`, `EndFrame`, plus any model-specific columns).
- `ground_truth_annotations/`: annotation workflow, finalized intervals, and supporting notebooks.
- `feature_engineering/`: reusable feature pipelines. Each feature group has its own package,
  documentation, checked-in sample data, and ignored generated outputs.
- `tests/`: regression tests for the model feature, span and evaluation functions.
- `evaluation/`: library for action segmentation evaluation
- `experiments/shared_features/`: leave-one-video-out comparison of feature sets and
  learners on the shared pair-motion features (additive; the `models/` detectors are
  unchanged). See [experiments/README.md](experiments/README.md).
- `models/`: see [models/README.md](models/README.md).
  - `models/intervals/`: **when** circling happens; each model writes `outputs/predictions/<name>/`
    ([list](models/intervals/README.md)): rule-based detector, logistic regression, XGBoost,
    per-frame XGBoost / logistic-regression CV, MS-TCN (DLC2Action), ActionFormer, SARTAB.
  - `models/direction/`: rotation **direction** and **turns/cycles** of given intervals
    ([list](models/direction/README.md)): rule-based episode labels, optical-flow / pose / SAM 2 methods.
  - `models/common/`: shared config, data cleaning, segment sampling, frame features, span post-processing.
- `model_performance_report.py`: evaluation metrics for every model in `outputs/predictions/`,
  using the functions from `evaluation/evaluate.py`, plus pose-overlay clip previews of
  predicted spans.
- `merged_spans_review.py`: review ground-truth circling spans against the source video,
  with buffered pose-overlay previews (drawing functions imported from
  `model_performance_report.py`).

### Logistic regression

The committed model is `outputs/circling_baseline/model.joblib`. To retrain it (writes the
model, metrics and report to `outputs/circling_baseline/`):

```sh
uv run python -m models.intervals.logistic_regression.train
```

To score every frame (cached in `outputs/circling_baseline/scores/`) and write the
predicted spans to `outputs/predictions/logistic_regression/`:

```sh
uv run python -m models.intervals.logistic_regression.predict [--threshold 0.5] [--min-seconds 7.5] [--min-fraction 0.9]
```

### Rule-based detector

```sh
uv run marimo edit models/intervals/rule_based/detector.py       # detector and results
uv run marimo edit models/intervals/rule_based/data_cleaning.py  # tracking artifacts removed first
```

Running the detector (`uv run python models/intervals/rule_based/detector.py`, or opening it in
marimo) writes its episodes to `outputs/predictions/rule_based/`.

To add rule-based `Direction` and `Cycles` columns to any model's predictions (writes
`outputs/predictions/<model>_rule_labels/`):

```sh
uv run python -m models.direction.rule_based.episode_labels outputs/predictions/<model>
```

### XGBoost

```sh
uv run marimo edit models/intervals/xgboost/notebook.py
```

The notebook samples all positive centers plus nearby negative frames, trains on
70% of the sampled rows, evaluates the 30% holdout, and scores both full videos.
It writes artifacts to `outputs/xgboost/` and standard prediction CSVs to
`outputs/predictions/xgboost/`. On macOS, run `brew install libomp` first.
See [models/intervals/xgboost/README.md](models/intervals/xgboost/README.md) for feature and evaluation details.

### Pair-motion feature pipeline

`feature_engineering/pair_motion/` creates a frame-level pair table. It covers
body-length-normalized distance, facing, speed and closing rate, angular movement
(orbital and heading rates, corotation), and 0.5, 1, 2 and 4 s window summaries
(close fraction, orbit turns, mean distance). It discovers matching
`<video>.parquet` and `<video>_pairs.parquet` inputs in `data/`; they are local
inputs and are not committed.

```sh
uv run python -m feature_engineering.pair_motion.build_features --data-dir data
```

The full table is written to `outputs/pair_motion/` (ignored). Feature definitions,
quality checks, and parameters are documented alongside the pipeline. The build
also regenerates a small committed sample CSV. See [feature_engineering/pair_motion/README.md](feature_engineering/pair_motion/README.md).

Pair-motion model variants use these feature definitions through
`models/common/pair_motion.py`; `feature_groups.py` lists the columns by group.

### MS-TCN

The default feature set is `pairmotion`. Use `--features pairemb` for pose/pair
features with DINOv2 embeddings (see `--help` for the other feature sets).

```sh
uv run python -m models.intervals.mstcn_dlc2action.train
uv run python -m models.intervals.mstcn_dlc2action.predict [--threshold 0.5] [--min-seconds 7.5] [--min-fraction 0.9]
```

Other interval models (per-frame XGBoost, logistic regression CV, ActionFormer, SARTAB) and
their train/predict commands: [models/intervals/README.md](models/intervals/README.md).

### Shared-feature experiments

`experiments/shared_features/` scores these features, and the existing models' feature
sets, with leave-one-video-out validation (each video held out once) using the
same `evaluation/` metrics and the same `outputs/predictions/<name>/` format.

```sh
uv run python -m experiments.shared_features.dataset --data-dir ~/Downloads
uv run python -m experiments.shared_features.lovo --variants all
```

See [experiments/README.md](experiments/README.md) for the protocol, variants and results.

## Tests

Run the regression suite and visualization tests from the repository root:

```sh
uv run --with pytest pytest tests visualization/test_export_video.py visualization/test_live.py
```

CI (`.github/workflows/ci.yml`) runs the lightweight repository checks on every
pull request. Run them locally before pushing:

```sh
uv lock --check                 # uv.lock is valid and matches pyproject.toml
uvx ruff@0.14.0 check .         # unused imports and syntax errors (e.g. conflict markers)
python scripts/check_repo.py    # conflict markers, pasted-twice doc lines, committed/ignored claims
uv run --no-project --with pytest pytest tests/test_repo_checks.py
```
