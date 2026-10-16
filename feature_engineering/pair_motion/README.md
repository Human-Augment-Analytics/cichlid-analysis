# Pair-motion features

Builds a frame-level table for pairs of cichlid tracks. It preserves source
measurements and adds body-length-normalized distance, proximity, facing,
pair-speed summaries, closing rate, and short gap-aware smoothing.

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

See `docs/feature_dictionary.md` for column definitions. The build runs
synthetic checks and full-table invariants before it writes outputs.
