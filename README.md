# cichlid-analysis

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed, then run from the repository root:

```sh
uv sync --locked
```

`uv sync` installs the locked dependencies and the required Python version. Open the
local URL printed by marimo to use the notebook. `model_performance_report.py` evaluates
the committed predictions in `outputs/predictions/`. On first launch it downloads the source
videos and the pose and pair Parquet files into `data/` for clip previews; the two videos are
about 31 GB each, so allow time and disk space for the initial download.

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
- `models/logistic_regression/`: frame-level logistic regression: features, training,
  per-frame scoring and post-processing into sustained spans.
- `models/rule_based/`: rule-based detector (closest-pair distance plus rotation) and the
  data-cleaning rules and notebook it relies on.
- `models/xgboost/`: marimo training notebook using centered 121-frame windows of
  the cleaned rule-based distance and rotation features, with 70/30 holdout and full-video evaluation.
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
uv run python -m models.logistic_regression.train
```

To score every frame (cached in `outputs/circling_baseline/scores/`) and write the
predicted spans to `outputs/predictions/logistic_regression/`:

```sh
uv run python -m models.logistic_regression.predict [--threshold 0.5] [--min-seconds 7.5] [--min-fraction 0.9]
```

### Rule-based detector

```sh
uv run marimo edit models/rule_based/detector.py       # detector and results
uv run marimo edit models/rule_based/data_cleaning.py  # tracking artifacts removed first
```

Running the detector (`uv run python models/rule_based/detector.py`, or opening it in
marimo) writes its episodes to `outputs/predictions/rule_based/`.

### XGBoost

```sh
uv run marimo edit models/xgboost/notebook.py
```

The notebook samples all positive centers plus nearby negative frames, trains on
70% of the sampled rows, evaluates the 30% holdout, and scores both full videos.
It writes artifacts to `outputs/xgboost/` and standard prediction CSVs to
`outputs/predictions/xgboost/`. On macOS, run `brew install libomp` first.
See [models/xgboost/README.md](models/xgboost/README.md) for feature and evaluation details.

### Pair-motion feature pipeline

`feature_engineering/pair_motion/` creates a frame-level pair table. It covers
body-length-normalized distance, facing, speed and closing rate, angular movement
(orbital and heading rates, corotation), and 0.5, 1, 2 and 4 s window summaries
(close fraction, orbit turns, mean distance). It expects the four validated pose
parquets in `data/`; they are local inputs and are not committed.

```sh
uv run python -m feature_engineering.pair_motion.build_features --data-dir data
```

The full table is written to `outputs/pair_motion/` (ignored). Feature definitions,
quality checks, parameters, and a small committed sample are kept alongside the
pipeline. See [feature_engineering/pair_motion/README.md](feature_engineering/pair_motion/README.md).
Nothing in `models/` imports these features; any model can read the parquet and use
the columns it wants (`feature_groups.py` lists them by group).

### Shared-feature experiments

`experiments/shared_features/` scores these features, and the existing models' feature
sets, with leave-one-video-out validation (each video held out once) using the
same `evaluation/` metrics and the same `outputs/predictions/<name>/` format.

```sh
uv run python -m experiments.shared_features.dataset --data-dir ~/Downloads
uv run python -m experiments.shared_features.lovo --variants all
```

See [experiments/README.md](experiments/README.md) for the protocol, variants and results.
