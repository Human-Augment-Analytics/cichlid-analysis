"""
Cleaning rules for the lab pipeline's pair table, shared by every notebook.

Three known tracking artifacts are flagged (see models/rule_based/data_cleaning.py for pictures and numbers):

  reflection     a fish whose center is off the gravel floor, i.e. its mirror image in the glass
  duplicate      two "fish" less than DUP_PX apart, i.e. one fish tracked twice
  keypoint_jump  a fish whose skeleton sits more than one body length from its own bounding box,
                 i.e. keypoints placed on a different fish for a frame

A pair row is dropped if either fish is flagged, or the pair is a duplicate.

Known flaw: "reflection" really means "off the floor outline". A real fish swimming high near a wall can
project past the outline and is dropped too (example: 0031 frame 363523, track 20244). A stricter test would
require a fish on the floor at the mirror position moving with it.

Data location: set CIRCLING_DATA_DIR, or put the parquet files in <repo>/data (git-ignored) or ~/Downloads.

Command line, from the repository root:  uv run python -m models.rule_based.circling_cleaning [DATA_DIR]
writes <video>_vid_pairs_clean.parquet next to the inputs.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as Polygon

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ["0031", "0028"]
DUP_PX = 40

# Gravel floor outline per video (pixels, 1296x972 frame), traced by hand on a gridded frame.
# The camera does not move within a video, so one outline covers all 10 hours. Re-trace for a new tank.
FLOOR = {
    "0031": [(12, 305), (90, 300), (300, 272), (600, 254), (870, 236), (905, 400),
             (935, 560), (962, 720), (988, 972), (42, 972)],
    "0028": [(105, 40), (300, 55), (500, 70), (1000, 80), (1005, 300), (1003, 600),
             (1012, 875), (600, 870), (165, 872), (135, 500)],
}

FISH_COLS = ["FrameNum", "TrackID", "X_center", "Y_center",
             "midline_centroid_x", "midline_centroid_y", "body_length_px"]


def default_data_dir(repo_dir="."):
    """First folder holding the tracking parquet files: $CIRCLING_DATA_DIR, <repo>/data, ~/Downloads, cwd."""
    candidates = [os.environ.get("CIRCLING_DATA_DIR"), Path(repo_dir) / "data", Path.home() / "Downloads", Path(".")]
    for c in candidates:
        if c and (Path(c) / "0031_vid_pairs.parquet").exists():
            return Path(c)
    return Path(repo_dir) / "data"


def find_file(data_dir, *candidates):
    for name in candidates:
        if (Path(data_dir) / name).exists():
            return Path(data_dir) / name
    raise FileNotFoundError(f"none of {candidates} found in {Path(data_dir).resolve()}")


def fish_path(data_dir, video):
    return find_file(data_dir, f"{video}_vid.parquet", f"{video}_vid (2).parquet", f"{video}_vid (1).parquet")


def pairs_path(data_dir, video):
    return find_file(data_dir, f"{video}_vid_pairs.parquet")


def flag_fish(fish, video):
    """Add per-detection flags: reflection (off the floor) and keypoint_jump (skeleton far from its box)."""
    fish = fish.copy()
    fish["reflection"] = ~Polygon(FLOOR[video]).contains_points(fish[["X_center", "Y_center"]].to_numpy())
    offset = np.hypot(fish.X_center - fish.midline_centroid_x, fish.Y_center - fish.midline_centroid_y)
    fish["keypoint_jump"] = offset > fish.body_length_px.median()
    return fish


def flag_pairs(pairs, flagged_fish):
    """Add per-pair flags. A pair inherits a fish flag if either fish has it."""
    pairs = pairs.copy()
    key = flagged_fish.set_index(["FrameNum", "TrackID"])[["reflection", "keypoint_jump"]]
    for col in ("reflection", "keypoint_jump"):
        hit = np.zeros(len(pairs), dtype=bool)
        for s in ("1", "2"):
            idx = pd.MultiIndex.from_arrays([pairs.FrameNum, pairs[f"TrackID_{s}"]])
            hit |= key[col].reindex(idx).fillna(False).to_numpy(bool)
        pairs[col] = hit
    pairs["duplicate"] = (pairs.centroid_distance_px < DUP_PX).to_numpy()
    pairs["keep"] = ~(pairs.reflection | pairs.duplicate | pairs.keypoint_jump)
    return pairs


def load_flagged(data_dir, video, pair_cols=None):
    fish = flag_fish(pd.read_parquet(fish_path(data_dir, video), columns=FISH_COLS), video)
    pairs = pd.read_parquet(pairs_path(data_dir, video), columns=pair_cols)
    return fish, flag_pairs(pairs, fish)


def clean_pairs(data_dir, video, pair_cols=None):
    """Pair table with artifact rows removed (flag columns dropped)."""
    _, pairs = load_flagged(data_dir, video, pair_cols)
    return pairs[pairs.keep].drop(columns=["reflection", "keypoint_jump", "duplicate", "keep"])


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_data_dir(ROOT)
    for v in VIDEOS:
        out = clean_pairs(data_dir, v)
        path = data_dir / f"{v}_vid_pairs_clean.parquet"
        out.to_parquet(path, index=False)
        print(f"{v}: wrote {len(out):,} clean pair rows to {path}")
