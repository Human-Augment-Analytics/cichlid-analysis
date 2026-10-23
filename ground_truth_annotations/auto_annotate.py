"""Run the annotation notebook's circling detector and export frame intervals.

From the repository root:
    uv run python ground_truth_annotations/auto_annotate.py data/0018_vid.mp4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.circling import add_circling_features, add_rule_hit, detect_circling_events


def build_pairs(pose, fps):
    """Rebuild the detector inputs using the existing pair-table contract.

    Pairs are ordered by TrackUID, episodes break at gaps over 0.2 seconds,
    and episodes with fewer than five observations are discarded.
    """
    keys = ["project_id", "day_index", "day_label", "FrameNum"]
    columns = keys + ["TrackUID", "midline_centroid_x", "midline_centroid_y",
                      "heading_rad", "d_heading_rad_s", "Sex", "qc_n_missing_kp"]
    if pose.duplicated(keys + ["TrackUID"]).any():
        raise ValueError("Duplicate pose observations for the same track and frame")
    pairs = pose[columns].merge(pose[columns], on=keys, suffixes=("_1", "_2"))
    pairs = pairs.loc[pairs.TrackUID_1 < pairs.TrackUID_2].copy()
    pairs = pairs.sort_values(["TrackUID_1", "TrackUID_2", "FrameNum"]).reset_index(drop=True)
    pair_keys = ["TrackUID_1", "TrackUID_2"]
    gaps = pairs.groupby(pair_keys, sort=False).FrameNum.diff()
    starts = gaps.isna() | (gaps > round(0.2 * fps))
    pairs["pair_episode_id"] = starts.cumsum() - 1 + int(pose.day_index.iloc[0]) * 1_000_000
    sizes = pairs.groupby("pair_episode_id").FrameNum.transform("size")
    pairs = pairs.loc[sizes >= 5].copy()
    dx = pairs.midline_centroid_x_2 - pairs.midline_centroid_x_1
    dy = pairs.midline_centroid_y_2 - pairs.midline_centroid_y_1
    pairs["centroid_distance_px"] = np.hypot(dx, dy)
    pairs["orbital_rad"] = np.arctan2(dy, dx)
    wrap = lambda angle: (angle + np.pi) % (2 * np.pi) - np.pi
    pairs["rel_heading_rad"] = wrap(pairs.heading_rad_2 - pairs.heading_rad_1)
    grouped = pairs.groupby("pair_episode_id", sort=False)
    pairs["dt_frames"] = grouped.FrameNum.diff()
    pairs["d_orbital_rad_s"] = wrap(grouped.orbital_rad.diff()) / (pairs.dt_frames / fps)
    pairs = pairs.rename(columns={"Sex_1": "sex_1", "Sex_2": "sex_2",
                                  "d_heading_rad_s_1": "d_heading_1_rad_s",
                                  "d_heading_rad_s_2": "d_heading_2_rad_s"})
    sex_codes = {"female": "F", "male": "M"}
    pairs["pair_sex"] = ["".join(sorted((sex_codes.get(a, "U"), sex_codes.get(b, "U"))))
                         for a, b in zip(pairs.sex_1, pairs.sex_2)]
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    pose_path = args.video.with_suffix(".parquet")
    metadata = pq.ParquetFile(pose_path).schema_arrow.metadata or {}
    fps = float(metadata[b"fps"])
    video = cv2.VideoCapture(str(args.video))
    try:
        if not video.isOpened():
            raise ValueError(f"Cannot open {args.video}")
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isclose(video.get(cv2.CAP_PROP_FPS), fps):
            raise ValueError("Video and pose frame rates differ")
    finally:
        video.release()
    pose = pd.read_parquet(pose_path)
    if not pose.FrameNum.between(0, n_frames - 1).all():
        raise ValueError("Pose frame numbers exceed the video bounds")
    pairs_path = pose_path.with_stem(pose_path.stem + "_pairs")
    pairs = pd.read_parquet(pairs_path) if pairs_path.exists() else build_pairs(pose, fps)
    pairs = add_rule_hit(add_circling_features(pose, pairs))
    events = detect_circling_events(pairs, video_id=args.video.stem, fps=fps)
    columns = {"video_id": "VideoID", "event_id": "EventID", "start_frame": "StartFrame",
               "stop_frame": "EndFrame", "behavior": "Behavior", "fish_ids": "FishIDs",
               "confidence": "Confidence/Notes"}
    exported = events[list(columns)].rename(columns=columns)
    for column in ("StartFrame", "EndFrame"):
        exported[column] = exported[column].astype("int64")
    assert (exported.StartFrame <= exported.EndFrame).all()
    assert exported.EndFrame.between(0, n_frames - 1).all()
    output = args.out or Path("outputs/auto_annotations") / f"{args.video.stem}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output, index=False)
    print(f"Video: {n_frames:,} frames at {fps:g} fps; pose: {len(pose):,} rows")
    print(f"Pair observations: {len(pairs):,}; rule hits: {pairs.rule_hit.sum():,}")
    print(f"Wrote {len(exported)} intervals to {output} (inclusive source frame numbers)")


if __name__ == "__main__":
    main()
