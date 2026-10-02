"""Build the frame-level distance, orientation, and speed table.

Run from the repository root:

    python -m src.build_features --data-dir ~/Downloads
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .features_motion import PARAMETERS, build_motion_features, run_synthetic_checks, spot_check_features
from .io_pose import DEFAULT_DATA_DIR, load_pose
from .qc import build_qc_summary, write_qc_summary


def write_feature_table(features, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(features, preserve_index=False)
    existing = dict(table.schema.metadata or {})
    custom = {key: str(value) for key, value in PARAMETERS.items()}
    custom["feature_set"] = "distance_orientation_speed_v1"
    merged = {**existing, **{key.encode(): value.encode() for key, value in custom.items()}}
    table = table.replace_schema_metadata(merged)
    pq.write_table(table, path)


def write_sample(features, path: Path, rows_per_video: int = 80) -> None:
    parts = []
    for _, video in features.groupby("day_label", sort=False):
        sizes = video.groupby("pair_episode_id").size()
        episode_id = (sizes - sizes.median()).abs().idxmin()
        parts.append(video.loc[video["pair_episode_id"] == episode_id].head(rows_per_video))
    sample = pd.concat(parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build first-half circling motion features")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=Path("outputs/pair_features_frame.parquet"))
    parser.add_argument("--qc-out", type=Path, default=Path("docs/qc_summary.md"))
    parser.add_argument("--params-out", type=Path, default=Path("docs/feature_params.json"))
    parser.add_argument("--sample-out", type=Path, default=Path("data/sample_pair_features.csv"))
    args = parser.parse_args()

    print("Checking feature definitions...", flush=True)
    run_synthetic_checks()

    print(f"Loading pose tables from {args.data_dir}...", flush=True)
    tracks, pairs, metadata = load_pose(args.data_dir)
    print(f"Tracks {len(tracks):,}  pairs {len(pairs):,}", flush=True)

    print("Building distance, orientation, and speed features...", flush=True)
    features, join_stats = build_motion_features(pairs, tracks)
    print(f"Feature rows {len(features):,}  columns {features.shape[1]}", flush=True)

    print("Spot-checking invariants...", flush=True)
    spot_check_features(features)

    write_feature_table(features, args.out)
    write_sample(features, args.sample_out)
    summary = build_qc_summary(tracks, pairs, join_stats)
    text = write_qc_summary(summary, args.qc_out)
    args.params_out.parent.mkdir(parents=True, exist_ok=True)
    args.params_out.write_text(json.dumps(PARAMETERS, indent=2) + "\n")

    print(text)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.sample_out}")
    print(f"Wrote {args.qc_out}")
    print(f"Source files: {', '.join(item['file'] for item in metadata)}")


if __name__ == "__main__":
    main()
