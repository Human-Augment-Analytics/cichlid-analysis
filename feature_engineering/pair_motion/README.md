# Pair-motion features

Builds a frame-level table for pairs of cichlid tracks. It preserves source
measurements and adds body-length-normalized distance, proximity, facing,
pair-speed summaries, closing rate, and short gap-aware smoothing. It also adds
angular movement (orbital and heading rates, corotation, bend) and 0.5, 1, 2 and
4 s window summaries (close fraction, orbit turns, and more) that stop at pair
episode and gap boundaries. Windows are centered by default; `trailing` is
available in `angular_temporal.add_window_features` for causal use.

## Inputs

Place these validated pose parquets in the repository-level `data/` directory:

- `0028_vid.parquet`
- `0031_vid.parquet`
- `0028_vid_pairs.parquet`
- `0031_vid_pairs.parquet`

They are ignored because they are large research inputs. The CLI accepts another
directory when the files live elsewhere.

## Run

From the repository root:

```sh
uv run python -m feature_engineering.pair_motion.build_features --data-dir data
```

For the existing local files in Downloads:

```sh
uv run python -m feature_engineering.pair_motion.build_features --data-dir ~/Downloads
```

## Outputs

- `outputs/pair_motion/pair_features_frame.parquet`: full feature table; ignored.
- `feature_engineering/pair_motion/data/sample_pair_features.csv`: small,
  committed sample with one representative episode per video.
- `feature_engineering/pair_motion/docs/qc_summary.md`: regenerated data checks.
- `feature_engineering/pair_motion/docs/feature_params.json`: generated
  threshold and smoothing parameters.

`feature_groups.py` lists the model-facing columns by group (distance, orientation,
speed, angular, temporal). Labels are never added here; see `experiments/` for
the label-joined tables and the leave-one-video-out comparison.

See `docs/feature_dictionary.md` for column definitions. The build runs
synthetic checks and full-table invariants before it writes outputs.
