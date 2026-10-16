# Feature engineering

Reusable feature pipelines live here rather than in the repository root. Each
pipeline is a Python package with its own documentation, sample data, and
generated-output location.

Current packages:

- `pair_motion/`: pair distance, proximity, orientation, and speed features
  from validated pose parquets.

Raw parquets stay in the repository-level `data/` directory, which is ignored.
Generated full-size outputs stay under `outputs/<pipeline>/`, also ignored.
