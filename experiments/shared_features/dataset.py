"""Frame-level, label-joined tables built from the shared pair-motion features.

Labels are attached here, never inside the reusable feature calculation.
One row per video frame that has at least one valid pair:

* closest valid pair's features (the input of the rule-based and XGBoost detectors),
* all-pair aggregates over the valid pairs of that frame (for logistic regression),
* ``label`` (1 inside an annotated circling span) and cleaning metadata.

A pair is valid when the rule-based cleaning keeps it (no mirror reflection, no
duplicate detection, no keypoint jump). Frames without a valid pair have no row;
detectors score them as not circling.

    uv run python -m experiments.shared_features.dataset --data-dir ~/Downloads
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.pair_motion.feature_groups import ALL_MODEL_FEATURES, FEATURE_GROUPS
from models.rule_based import circling_cleaning as cc

from .common import FEATURE_TABLE, MODEL_FRAMES, VIDEO_FRAMES, VIDEOS, frame_labels, read_spans

# (source column, statistic) pairs aggregated over all valid pairs in a frame.
AGGREGATES = [
    ("centroid_distance_bl", "min"),
    ("centroid_distance_bl", "mean"),
    ("abs_d_orbital_rad_s", "max"),
    ("abs_d_orbital_rad_s", "mean"),
    ("speed_mean_bl_s", "mean"),
    ("closing_speed_bl_s", "mean"),
    ("w4s_close_frac", "max"),
    ("w4s_orbit_turns_abs", "max"),
    ("w4s_mutual_facing_frac", "max"),
    ("w4s_abs_d_orbital_mean", "max"),
]
AGGREGATE_COLUMNS = [f"agg_{stat}_{column}" for column, stat in AGGREGATES] + ["agg_n_pairs"]
META_COLUMNS = [
    "VideoID", "FrameNum", "label", "TrackID_1", "TrackID_2", "pair_episode_id",
    "qc_bad_body_length", "qc_angular_rate_clipped",
]


def valid_pair_keys(data_dir: Path, video: str) -> pd.DataFrame:
    """FrameNum/TrackID pairs that survive the rule-based artifact cleaning."""
    video_id = video.removesuffix("_vid")
    fish = cc.flag_fish(pd.read_parquet(cc.fish_path(data_dir, video_id), columns=cc.FISH_COLS), video_id)
    pairs = pd.read_parquet(
        cc.pairs_path(data_dir, video_id),
        columns=["FrameNum", "TrackID_1", "TrackID_2", "centroid_distance_px"],
    )
    flagged = cc.flag_pairs(pairs, fish)
    return flagged.loc[flagged["keep"], ["FrameNum", "TrackID_1", "TrackID_2"]]


def build_video_frames(features: pd.DataFrame, keys: pd.DataFrame, video: str, spans) -> pd.DataFrame:
    rows = features.loc[features["day_label"] == video]
    rows = rows.merge(keys, on=["FrameNum", "TrackID_1", "TrackID_2"], how="inner", validate="many_to_one")
    for column in ALL_MODEL_FEATURES:
        if rows[column].dtype == "boolean" or rows[column].dtype == bool:
            rows[column] = rows[column].astype("float32")
    rows = rows.sort_values(["FrameNum", "centroid_distance_px"], kind="mergesort")

    grouped = rows.groupby("FrameNum", sort=True)
    aggregates = {"agg_n_pairs": grouped.size().astype("float32")}
    for column, stat in AGGREGATES:
        aggregates[f"agg_{stat}_{column}"] = grouped[column].agg(stat).astype("float32")
    aggregate_table = pd.DataFrame(aggregates)

    closest = rows.drop_duplicates("FrameNum", keep="first").set_index("FrameNum")
    out = closest[
        ["TrackID_1", "TrackID_2", "pair_episode_id", "qc_bad_body_length", "qc_angular_rate_clipped"]
        + ALL_MODEL_FEATURES
    ].astype({column: "float32" for column in ALL_MODEL_FEATURES})
    out = out.join(aggregate_table)
    labels = frame_labels(VIDEO_FRAMES[video], spans[video])
    out.insert(0, "label", labels[out.index.to_numpy()])
    out.insert(0, "VideoID", video)
    return out.reset_index()[META_COLUMNS + ALL_MODEL_FEATURES + AGGREGATE_COLUMNS]


def build_model_frames(data_dir: Path, feature_table: Path = FEATURE_TABLE) -> pd.DataFrame:
    features = pd.read_parquet(feature_table)
    spans = read_spans()
    parts = []
    for video in VIDEOS:
        print(f"Building frame table: {video}", flush=True)
        parts.append(build_video_frames(features, valid_pair_keys(data_dir, video), video, spans))
    return pd.concat(parts, ignore_index=True)


def load_model_frames(path: Path = MODEL_FRAMES) -> pd.DataFrame:
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=cc.default_data_dir(cc.ROOT))
    parser.add_argument("--features", type=Path, default=FEATURE_TABLE)
    parser.add_argument("--out", type=Path, default=MODEL_FRAMES)
    args = parser.parse_args()
    table = build_model_frames(args.data_dir, args.features)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    summary = table.groupby("VideoID").agg(frames=("FrameNum", "size"), positive=("label", "sum"))
    print(summary)
    print(f"Feature groups: { {k: len(v) for k, v in FEATURE_GROUPS.items()} }")
    print(f"Wrote {args.out} ({len(table):,} rows, {table.shape[1]} columns)")


if __name__ == "__main__":
    main()
