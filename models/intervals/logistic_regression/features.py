"""Per-frame pose and pair features for the circling logistic regression."""

import numpy as np
import polars as pl

WINDOW = 30
FEATURE_SOURCE = "pose_and_pairs_v1"
KEYPOINTS = (
    "Nose", "LeftEye", "RightEye", "Head", "Spine1", "Spine2",
    "Spine3", "Spine4", "Peduncle", "TailTip",
)
MIDLINE = ("Nose", "Head", "Spine1", "Spine2", "Spine3", "Spine4", "Peduncle", "TailTip")
TRACK_KEYS = ["project_id", "day_label", "TrackID"]
PAIR_DISTANCES = (
    "centroid_distance_px", "dist_nose1_nose2_px", "dist_nose1_tailtip2_px",
    "dist_nose2_tailtip1_px", "dist_nose1_spine4of2_px", "dist_nose2_spine4of1_px",
)
PAIR_ANGLES = ("orbital_rad", "rel_heading_rad", "bearing_1_rad", "bearing_2_rad")
PAIR_RATES = ("d_heading_1_rad_s", "d_heading_2_rad_s", "d_orbital_rad_s")


def keypoint_displacements(poses):
    """Per-keypoint movement from (video, TrackID, t-1) to (video, TrackID, t).

    Missing/nonfinite endpoints and track starts/gaps produce null distances.
    """
    row_keys = [*TRACK_KEYS, "FrameNum"]
    coordinates = [f"{name}_{axis}" for name in KEYPOINTS for axis in ("x", "y")]
    ordered = poses.select(*row_keys, *coordinates).sort(row_keys)
    if ordered.select(pl.any_horizontal(pl.col(name).is_null() for name in row_keys).any()).item():
        raise ValueError("Video, TrackID, and frame identifiers must not be missing.")
    if ordered.select(pl.struct(row_keys).is_duplicated().any()).item():
        raise ValueError("Expected one pose per video, track, and frame.")

    adjacent = pl.col("FrameNum").diff().over(TRACK_KEYS) == 1
    distances = []
    for name in KEYPOINTS:
        dx = pl.col(f"{name}_x").cast(pl.Float64).diff().over(TRACK_KEYS)
        dy = pl.col(f"{name}_y").cast(pl.Float64).diff().over(TRACK_KEYS)
        distance = (dx.pow(2) + dy.pow(2)).sqrt()
        distances.append(
            pl.when(adjacent & distance.is_finite()).then(distance).otherwise(None).alias(name)
        )
    return ordered.select(*row_keys, *distances)


def pose_frame_features(poses, start, end):
    """50 causal pose features per frame: motion, bend, turn, track count and trailing means."""
    if start < 0 or end < start:
        raise ValueError("Invalid frame range")
    if poses.select("project_id", "day_label").unique().height > 1:
        raise ValueError("Provide poses for exactly one video")
    coords = [f"{k}_{axis}" for k in KEYPOINTS for axis in ("x", "y")]
    poses = poses.select(*TRACK_KEYS, "FrameNum", *coords).with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c).cast(pl.Float64)).otherwise(None).alias(c)
        for c in coords
    ]).sort([*TRACK_KEYS, "FrameNum"])
    lengths = [((pl.col(f"{a}_x") - pl.col(f"{b}_x")).pow(2)
                + (pl.col(f"{a}_y") - pl.col(f"{b}_y")).pow(2)).sqrt()
               for a, b in zip(MIDLINE[:-1], MIDLINE[1:])]
    length = sum(lengths)
    chord = ((pl.col("Nose_x") - pl.col("TailTip_x")).pow(2)
             + (pl.col("Nose_y") - pl.col("TailTip_y")).pow(2)).sqrt()
    poses = poses.with_columns(length.alias("_length"),
        pl.arctan2(pl.col("Nose_y") - pl.col("Spine1_y"),
                   pl.col("Nose_x") - pl.col("Spine1_x")).alias("_heading"))
    delta = pl.col("_heading").diff().over(TRACK_KEYS)
    adjacent = pl.col("FrameNum").diff().over(TRACK_KEYS) == 1
    poses = poses.with_columns(
        pl.when(pl.col("_length") > 1).then(1 - chord / pl.col("_length")).alias("bend"),
        pl.when(adjacent).then(pl.arctan2(delta.sin(), delta.cos()).abs()).alias("turn"))
    movement = keypoint_displacements(poses)
    rows = movement.join(poses.select(*TRACK_KEYS, "FrameNum", "_length", "bend", "turn"),
                         on=[*TRACK_KEYS, "FrameNum"], how="left")
    rows = rows.with_columns([
        pl.when(pl.col("_length") > 1)
        .then((pl.col(k) / pl.col("_length")).log1p()).otherwise(None).alias(k)
        for k in KEYPOINTS
    ])
    signals = [*KEYPOINTS, "bend", "turn"]
    aggregate = rows.group_by("FrameNum").agg(
        pl.len().cast(pl.Float64).alias("track_count"),
        *[expr for k in signals for expr in
          (pl.col(k).mean().alias(f"{k}_mean"), pl.col(k).max().alias(f"{k}_max"))])
    first = max(0, start - WINDOW + 1)
    frames = pl.DataFrame({"FrameNum": np.arange(first, end + 1, dtype=np.int64)})
    frames = frames.join(aggregate, on="FrameNum", how="left").sort("FrameNum")
    frames = frames.with_columns(pl.col("track_count").fill_null(0))
    base = [c for c in frames.columns if c != "FrameNum"]
    frames = frames.with_columns([
        pl.col(c).rolling_mean(WINDOW, min_samples=1).alias(f"{c}_trailing30") for c in base
    ])
    return frames.filter(pl.col("FrameNum") >= start).with_columns(pl.exclude("FrameNum").cast(pl.Float32))


def pair_frame_features(pairs, start, end):
    """Aggregate supplied pair measurements into one row per video frame.

    Six distances, four angles encoded as sine/cosine, and three supplied
    angular rates give 17 signals. Mean/min/max across pairs plus pair count
    give 52 predictors.
    """
    if start < 0 or end < start:
        raise ValueError("Invalid frame range")
    if pairs.select("project_id", "day_label").unique().height > 1:
        raise ValueError("Provide pairs for exactly one video")
    keys = ["FrameNum", "TrackID_1", "TrackID_2"]
    if pairs.select(pl.any_horizontal(pl.col(k).is_null() for k in keys).any()).item():
        raise ValueError("Pair frame and track identifiers cannot be missing")
    if pairs.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError("Duplicate pair within a frame")
    signals = [*PAIR_DISTANCES, *PAIR_RATES]
    rows = pairs.select("FrameNum", *PAIR_DISTANCES, *PAIR_ANGLES, *PAIR_RATES).with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c).cast(pl.Float64)).otherwise(None).alias(c)
        for c in (*PAIR_DISTANCES, *PAIR_ANGLES, *PAIR_RATES)
    ])
    rows = rows.with_columns([
        expr for c in PAIR_ANGLES for expr in
        (pl.col(c).sin().alias(f"{c}_sin"), pl.col(c).cos().alias(f"{c}_cos"))
    ])
    signals += [f"{c}_{part}" for c in PAIR_ANGLES for part in ("sin", "cos")]
    aggregate = rows.group_by("FrameNum").agg(
        pl.len().cast(pl.Float32).alias("pair_count"),
        *[expr for c in signals for expr in (
            pl.col(c).mean().alias(f"{c}_mean"), pl.col(c).min().alias(f"{c}_min"),
            pl.col(c).max().alias(f"{c}_max"))])
    frames = pl.DataFrame({"FrameNum": np.arange(start, end + 1, dtype=np.int64)})
    return (frames.join(aggregate, on="FrameNum", how="left").sort("FrameNum")
            .with_columns(pl.col("pair_count").fill_null(0), pl.exclude("FrameNum", "pair_count").cast(pl.Float32)))


def frame_features(poses, pairs, start, end):
    """Combine the 50 pose features with the 52 pair features by frame."""
    videos = pl.concat([poses.select("project_id", "day_label"),
                        pairs.select("project_id", "day_label")]).unique()
    if videos.height > 1:
        raise ValueError("Pose and pair rows must belong to the same video")
    return pose_frame_features(poses, start, end).join(
        pair_frame_features(pairs, start, end), on="FrameNum", how="left", validate="1:1")
