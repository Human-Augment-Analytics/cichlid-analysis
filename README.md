# cichlid-analysis

Frame-level distance, orientation, and speed features for cichlid-pair pose data.

The source data is the local parquet files `0028_vid.parquet`, `0031_vid.parquet`, `0028_vid_pairs.parquet`, and `0031_vid_pairs.parquet` (by default in `~/Downloads`). GitHub does not host those tables. They stay out of git because the track files exceed GitHub's file size limit.

## Run

From this directory, with the `cvat` conda env:

```bash
python -m src.build_features --data-dir ~/Downloads
```

That writes:

- `outputs/pair_features_frame.parquet` — one row per pair per frame
- `data/sample_pair_features.csv` — a short episode from each video
- `docs/qc_summary.md` — track, pair, gap, and join checks
- `docs/feature_params.json` — proximity cutoff, facing cutoff, and smooth window

Column definitions are in [docs/feature_dictionary.md](docs/feature_dictionary.md).
The QC results are in [docs/qc_summary.md](docs/qc_summary.md).
