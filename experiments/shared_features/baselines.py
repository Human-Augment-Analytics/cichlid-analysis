"""The existing detectors' feature sets, on the same frames as the shared features.

* ``their_lr102``: the logistic regression's 102 per-frame features
  (50 causal pose features + 52 all-pair aggregate features).
* ``their_win155``: the XGBoost detector's five cleaned signals, every 4th offset of
  its 121-frame window (31 offsets x 5 signals). Stride 4 keeps training tractable;
  the full 605-column window is the XGBoost detector's own input.

Rows match ``model_frames.parquet`` (frames with a valid pair), so any learner can be
trained on shared features, existing features, or both under identical conditions.

    uv run python -m experiments.shared_features.baselines --data-dir ~/Downloads
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from models.logistic_regression import features as lrf
from models.rule_based import circling_cleaning as cc
from models.xgboost import features as xgf

from .common import EXPERIMENT_DIR, MODEL_FRAMES, VIDEO_FRAMES, VIDEOS

BASELINE_FRAMES = EXPERIMENT_DIR / "baseline_frames.parquet"
WINDOW_STRIDE = 4
WINDOW_OFFSETS = list(range(-xgf.HALF_WINDOW, xgf.HALF_WINDOW + 1, WINDOW_STRIDE))
WIN_COLUMNS = [f"win_{signal}_offset_{offset:+d}" for signal in xgf.SERIES_COLUMNS for offset in WINDOW_OFFSETS]


def lr_features(data_dir: Path, video: str) -> pd.DataFrame:
    video_id = video.removesuffix("_vid")
    count = VIDEO_FRAMES[video]
    coordinates = [f"{k}_{axis}" for k in lrf.KEYPOINTS for axis in ("x", "y")]
    poses = pl.read_parquet(cc.fish_path(data_dir, video_id), columns=[*lrf.TRACK_KEYS, "FrameNum", *coordinates])
    pairs = pl.read_parquet(
        cc.pairs_path(data_dir, video_id),
        columns=["project_id", "day_label", "FrameNum", "TrackID_1", "TrackID_2",
                 *lrf.PAIR_DISTANCES, *lrf.PAIR_ANGLES, *lrf.PAIR_RATES],
    )
    table = lrf.frame_features(poses, pairs, 0, count - 1).to_pandas()
    table.columns = ["FrameNum"] + [f"lr_{c}" for c in table.columns[1:]]
    return table


def window_features(data_dir: Path, video: str, frames: np.ndarray) -> pd.DataFrame:
    series = xgf.load_series(data_dir, video, VIDEO_FRAMES[video])
    full = xgf.window_features(series, frames)  # (frames, signals * 121)
    picked = [s * xgf.WINDOW + (o + xgf.HALF_WINDOW) for s in range(len(xgf.SERIES_COLUMNS)) for o in WINDOW_OFFSETS]
    out = pd.DataFrame(full[:, picked], columns=WIN_COLUMNS)
    out.insert(0, "FrameNum", frames)
    return out


def build(data_dir: Path, model_frames: Path = MODEL_FRAMES) -> pd.DataFrame:
    keys = pd.read_parquet(model_frames, columns=["VideoID", "FrameNum"])
    parts = []
    for video in VIDEOS:
        print(f"Baseline features: {video}", flush=True)
        frames = keys.loc[keys["VideoID"] == video, "FrameNum"].to_numpy()
        lr = lr_features(data_dir, video).set_index("FrameNum").loc[frames].reset_index()
        win = window_features(data_dir, video, frames)
        merged = lr.merge(win, on="FrameNum", validate="1:1")
        merged.insert(0, "VideoID", video)
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=cc.default_data_dir(cc.ROOT))
    parser.add_argument("--out", type=Path, default=BASELINE_FRAMES)
    args = parser.parse_args()
    table = build(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    lr_cols = [c for c in table.columns if c.startswith("lr_")]
    print(f"Wrote {args.out}: {len(table):,} rows, {len(lr_cols)} lr columns, {len(WIN_COLUMNS)} window columns")


if __name__ == "__main__":
    main()
