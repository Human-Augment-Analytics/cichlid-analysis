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

## Repository layout

- `data/`: video and parquet files
- `outputs/`: generated artifacts, model files, etc
- `ground_truth_annotations/`: annotation workflow, finalized intervals, and supporting notebooks.
- `feature_engineering/`: reusable feature pipelines. Each feature group has its own package,
  documentation, checked-in sample data, and ignored generated outputs.
- `tests/`: regression tests for shared model and interval utilities.
- `evaluation/`: library for action segmentation evaluation
- `analysis.py`: interactive analysis, labels, baseline models, and clip review.

### Pair-motion feature pipeline

`feature_engineering/pair_motion/` creates a frame-level pair table for distance,
orientation, and speed. It expects the four validated pose parquets in `data/`;
they are local inputs and are not committed.

```sh
uv run python -m feature_engineering.pair_motion.build_features --data-dir data
```

The full table is written to `outputs/pair_motion/` (ignored). Feature definitions,
quality checks, parameters, and a small committed sample are kept alongside the
pipeline. See [feature_engineering/pair_motion/README.md](feature_engineering/pair_motion/README.md).
