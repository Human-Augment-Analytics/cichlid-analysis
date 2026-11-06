import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as Polygon

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ["0031", "0028"]
DUP_PX = 40

FLOOR = {
    "0031": [(12, 305), (90, 300), (300, 272), (600, 254), (870, 236), (905, 400),
             (935, 560), (962, 720), (988, 972), (42, 972)],
    "0028": [(105, 40), (300, 55), (500, 70), (1000, 80), (1005, 300), (1003, 600),
             (1012, 875), (600, 870), (165, 872), (135, 500)],
}

FISH_COLS = ["FrameNum", "TrackID", "X_center", "Y_center",
             "midline_centroid_x", "midline_centroid_y", "body_length_px"]

def default_data_dir(repo_dir="."):
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
    fish = fish.copy()
    fish["reflection"] = ~Polygon(FLOOR[video]).contains_points(fish[["X_center", "Y_center"]].to_numpy())
    offset = np.hypot(fish.X_center - fish.midline_centroid_x, fish.Y_center - fish.midline_centroid_y)
    fish["keypoint_jump"] = offset > fish.body_length_px.median()
    return fish

def flag_pairs(pairs, flagged_fish):
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
    _, pairs = load_flagged(data_dir, video, pair_cols)
    return pairs[pairs.keep].drop(columns=["reflection", "keypoint_jump", "duplicate", "keep"])

if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_data_dir(ROOT)
    for v in VIDEOS:
        out = clean_pairs(data_dir, v)
        path = data_dir / f"{v}_vid_pairs_clean.parquet"
        out.to_parquet(path, index=False)
        print(f"{v}: wrote {len(out):,} clean pair rows to {path}")
