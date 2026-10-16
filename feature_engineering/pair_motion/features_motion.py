"""Frame-level distance, orientation, and speed features.

Angles are radians. Distances in the output are body lengths unless the
column name ends in ``_px``. Positive ``approach_rate_bl_s`` means the
fish are moving apart. ``closing_speed_bl_s`` is the negation of that rate.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

FPS = 30.0
GAP_TOLERANCE_S = 0.2
PROXIMITY_BL = 1.5
FACING_MAX_RAD = math.pi / 4.0
SMOOTH_WINDOW_FRAMES = 15

PARAMETERS = {
    "fps": FPS,
    "gap_tolerance_s": GAP_TOLERANCE_S,
    "proximity_bl": PROXIMITY_BL,
    "proximity_rule": "centroid_distance_bl <= proximity_bl",
    "facing_max_rad": FACING_MAX_RAD,
    "facing_rule": "abs(wrapped bearing) < facing_max_rad",
    "smooth_window_frames": SMOOTH_WINDOW_FRAMES,
    "smooth_window_s": SMOOTH_WINDOW_FRAMES / FPS,
    "smooth_method": "centered rolling median, then centered rolling mean",
}

GROUP_KEYS = [
    "project_id",
    "day_label",
    "TrackUID_1",
    "TrackUID_2",
    "pair_episode_id",
]
SMOOTH_KEYS = GROUP_KEYS + ["smooth_segment"]

PX_DISTANCE_COLUMNS = [
    "centroid_distance_px",
    "dist_nose1_nose2_px",
    "dist_nose1_tailtip2_px",
    "dist_nose2_tailtip1_px",
    "dist_nose1_spine4of2_px",
    "dist_nose2_spine4of1_px",
]
BL_DISTANCE_COLUMNS = [name.replace("_px", "_bl") for name in PX_DISTANCE_COLUMNS]
BEARING_COLUMNS = ["bearing_1_rad", "bearing_2_rad"]
SPEED_COLUMNS = [
    "speed_bl_s_1",
    "speed_bl_s_2",
    "speed_mean_bl_s",
    "speed_diff_bl_s",
]

def wrap_rad(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def mean_body_length(length_1: np.ndarray, length_2: np.ndarray) -> np.ndarray:
    """Mean of the positive body lengths available on each row."""
    stacked = np.vstack([np.asarray(length_1, dtype=float), np.asarray(length_2, dtype=float)])
    stacked = np.where(stacked > 0, stacked, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(stacked, axis=0)


def _fish_frame(tracks: pd.DataFrame, suffix: str) -> pd.DataFrame:
    renamed = {
        "TrackUID": f"TrackUID_{suffix}",
        "body_length_px": f"body_length_px_{suffix}",
        "speed_bl_s": f"speed_bl_s_{suffix}",
        "Nose_x": f"Nose_x_{suffix}",
        "Nose_y": f"Nose_y_{suffix}",
        "TailTip_x": f"TailTip_x_{suffix}",
        "TailTip_y": f"TailTip_y_{suffix}",
        "is_track_start": f"is_track_start_{suffix}",
        "is_track_end": f"is_track_end_{suffix}",
    }
    keep = ["project_id", "day_label", "FrameNum", *renamed]
    return tracks.loc[:, keep].rename(columns=renamed)


def join_tracks(pairs: pd.DataFrame, tracks: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keys = ["project_id", "day_label", "FrameNum"]
    fish_1 = _fish_frame(tracks, "1")
    fish_2 = _fish_frame(tracks, "2")
    joined = pairs.merge(
        fish_1,
        on=keys + ["TrackUID_1"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched_1 = joined["_merge"] != "both"
    joined = joined.drop(columns="_merge")
    joined = joined.merge(
        fish_2,
        on=keys + ["TrackUID_2"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched_2 = joined["_merge"] != "both"
    joined = joined.drop(columns="_merge")
    stats = {
        "pair_rows": int(len(joined)),
        "unmatched_fish_1": int(unmatched_1.sum()),
        "unmatched_fish_2": int(unmatched_2.sum()),
        "unmatched_fish_1_frac": float(unmatched_1.mean()) if len(joined) else float("nan"),
        "unmatched_fish_2_frac": float(unmatched_2.mean()) if len(joined) else float("nan"),
    }
    return joined, stats


def add_distance_features(frame: pd.DataFrame) -> None:
    scale = mean_body_length(
        frame["body_length_px_1"].to_numpy(),
        frame["body_length_px_2"].to_numpy(),
    )
    safe_scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)
    for source, target in zip(PX_DISTANCE_COLUMNS, BL_DISTANCE_COLUMNS):
        frame[target] = frame[source].to_numpy(dtype=float) / safe_scale


def add_speed_and_orientation(frame: pd.DataFrame) -> None:
    speed_1 = frame["speed_bl_s_1"]
    speed_2 = frame["speed_bl_s_2"]
    frame["speed_mean_bl_s"] = (speed_1 + speed_2) / 2.0
    frame["speed_diff_bl_s"] = speed_1 - speed_2

    bearing_1 = wrap_rad(frame["bearing_1_rad"].to_numpy(dtype=float))
    bearing_2 = wrap_rad(frame["bearing_2_rad"].to_numpy(dtype=float))
    facing_1 = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    facing_2 = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    ok_1 = np.isfinite(bearing_1)
    ok_2 = np.isfinite(bearing_2)
    facing_1.loc[ok_1] = np.abs(bearing_1[ok_1]) < FACING_MAX_RAD
    facing_2.loc[ok_2] = np.abs(bearing_2[ok_2]) < FACING_MAX_RAD
    frame["facing_1"] = facing_1
    frame["facing_2"] = facing_2
    frame["mutual_facing"] = facing_1 & facing_2

    distance = frame["centroid_distance_bl"]
    proximity = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    ok_distance = distance.notna()
    proximity.loc[ok_distance] = distance.loc[ok_distance] <= PROXIMITY_BL
    frame["proximity"] = proximity


def add_approach_rate(frame: pd.DataFrame) -> None:
    delta = frame.groupby(GROUP_KEYS, sort=False)["centroid_distance_bl"].diff()
    dt_s = frame["dt_frames"].to_numpy(dtype=float) / FPS
    rate = delta.to_numpy(dtype=float) / dt_s
    invalid = ~np.isfinite(dt_s) | (dt_s <= 0) | ~np.isfinite(rate)
    rate = rate.astype(float)
    rate[invalid] = np.nan
    frame["approach_rate_bl_s"] = rate
    frame["closing_speed_bl_s"] = -frame["approach_rate_bl_s"]


def assign_smooth_segments(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(GROUP_KEYS + ["FrameNum"], kind="mergesort").reset_index(drop=True)
    gap = ordered["dt_frames"] / FPS > GAP_TOLERANCE_S
    ordered["smooth_segment"] = (
        gap.groupby([ordered[key] for key in GROUP_KEYS], sort=False).cumsum().astype("int32")
    )
    return ordered


def _unwrap_blocks(values: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    unwrapped = values.copy()
    if len(values) == 0:
        return unwrapped
    boundaries = np.flatnonzero(np.diff(segment_ids)) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(values)]
    for start, end in zip(starts, ends):
        block = unwrapped[start:end]
        finite = np.isfinite(block)
        if finite.sum() < 2:
            continue
        if finite.all():
            unwrapped[start:end] = np.unwrap(block)
            continue
        indices = np.flatnonzero(finite)
        splits = np.flatnonzero(np.diff(indices) > 1) + 1
        for part in np.split(indices, splits):
            block[part] = np.unwrap(block[part])
    return unwrapped


def _rolling_median_mean(frame: pd.DataFrame, columns: list[str], segment_ids: pd.Series) -> pd.DataFrame:
    """Centered rolling median, then centered rolling mean, inside each segment."""
    work = frame.loc[:, columns].astype(float).copy()
    work["_seg"] = segment_ids.to_numpy()
    window = SMOOTH_WINDOW_FRAMES
    median = work.groupby("_seg", sort=False)[columns].rolling(window, center=True, min_periods=1).median()
    median.index = median.index.droplevel(0)
    if not median.index.equals(frame.index):
        median = median.reindex(frame.index)
    median["_seg"] = segment_ids.to_numpy()
    mean = median.groupby("_seg", sort=False)[columns].rolling(window, center=True, min_periods=1).mean()
    mean.index = mean.index.droplevel(0)
    if not mean.index.equals(frame.index):
        mean = mean.reindex(frame.index)
    return mean.loc[:, columns]


def add_smoothed_columns(frame: pd.DataFrame) -> None:
    segment_ids = frame.groupby(SMOOTH_KEYS, sort=False).ngroup()
    # ngroup follows first-appearance order. Rows are already sorted by SMOOTH_KEYS,
    # so each segment is one contiguous block.
    if len(segment_ids) > 1 and np.any(np.diff(segment_ids.to_numpy()) < 0):
        raise RuntimeError("Smooth segments are not contiguous; refuse to smooth")

    linear_columns = BL_DISTANCE_COLUMNS + SPEED_COLUMNS
    bearing_work = [f"{column}__unwrap" for column in BEARING_COLUMNS]
    work = frame.loc[:, linear_columns].copy()
    raw_ids = segment_ids.to_numpy()
    for column, work_column in zip(BEARING_COLUMNS, bearing_work):
        work[work_column] = _unwrap_blocks(frame[column].to_numpy(dtype=float), raw_ids)
    smoothed = _rolling_median_mean(work, linear_columns + bearing_work, segment_ids)
    for column in linear_columns:
        frame[f"{column}_smooth"] = smoothed[column].to_numpy()
    for column, work_column in zip(BEARING_COLUMNS, bearing_work):
        raw_smooth = smoothed[work_column].to_numpy(dtype=float)
        wrapped = wrap_rad(raw_smooth)
        wrapped[~np.isfinite(raw_smooth)] = np.nan
        frame[f"{column}_smooth"] = wrapped


def add_qc_flags(frame: pd.DataFrame) -> None:
    missing = False
    for suffix in ("1", "2"):
        for point in ("Nose_x", "Nose_y", "TailTip_x", "TailTip_y"):
            missing = missing | frame[f"{point}_{suffix}"].isna()
    frame["qc_missing_nose_or_tail"] = missing.to_numpy()
    drop_points = [
        f"{point}_{suffix}"
        for suffix in ("1", "2")
        for point in ("Nose_x", "Nose_y", "TailTip_x", "TailTip_y")
    ]
    frame.drop(columns=drop_points, inplace=True)


OUTPUT_COLUMNS = (
    [
        "project_id",
        "day_index",
        "day_label",
        "Time_s",
        "FrameNum",
        "TrackID_1",
        "TrackID_2",
        "TrackUID_1",
        "TrackUID_2",
        "pair_episode_id",
        "dt_frames",
        "smooth_segment",
        "sex_1",
        "sex_2",
        "pair_sex",
    ]
    + PX_DISTANCE_COLUMNS
    + [
        "orbital_rad",
        "rel_heading_rad",
        "bearing_1_rad",
        "bearing_2_rad",
        "body_length_px_1",
        "body_length_px_2",
        "speed_bl_s_1",
        "speed_bl_s_2",
    ]
    + BL_DISTANCE_COLUMNS
    + [
        "approach_rate_bl_s",
        "proximity",
        "facing_1",
        "facing_2",
        "mutual_facing",
        "speed_mean_bl_s",
        "speed_diff_bl_s",
        "closing_speed_bl_s",
    ]
    + [f"{column}_smooth" for column in BL_DISTANCE_COLUMNS]
    + [f"{column}_smooth" for column in BEARING_COLUMNS]
    + [f"{column}_smooth" for column in SPEED_COLUMNS]
    + [
        "qc_n_missing_kp_1",
        "qc_n_missing_kp_2",
        "qc_missing_nose_or_tail",
        "is_track_start_1",
        "is_track_end_1",
        "is_track_start_2",
        "is_track_end_2",
    ]
)


def build_motion_features(pairs: pd.DataFrame, tracks: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    joined, join_stats = join_tracks(pairs, tracks)
    add_distance_features(joined)
    ordered = assign_smooth_segments(joined)
    add_speed_and_orientation(ordered)
    add_approach_rate(ordered)
    add_smoothed_columns(ordered)
    add_qc_flags(ordered)
    missing = [column for column in OUTPUT_COLUMNS if column not in ordered.columns]
    if missing:
        raise RuntimeError(f"Feature table is missing columns: {missing}")
    return ordered.loc[:, OUTPUT_COLUMNS].reset_index(drop=True), join_stats


def run_synthetic_checks() -> None:
    """Check definitions on a tiny pair that never touches the pose files."""
    rows = []
    # Episode 1: closing by 2 body lengths each frame, bearings near +/- pi.
    for frame_number in range(6):
        rows.append(
            {
                "project_id": "toy",
                "day_index": 1,
                "day_label": "toy",
                "Time_s": frame_number / FPS,
                "FrameNum": frame_number,
                "TrackID_1": 1,
                "TrackID_2": 2,
                "TrackUID_1": 1,
                "TrackUID_2": 2,
                "centroid_distance_px": 200.0 - 20.0 * frame_number,
                "dist_nose1_nose2_px": 100.0,
                "dist_nose1_tailtip2_px": 100.0,
                "dist_nose2_tailtip1_px": 100.0,
                "dist_nose1_spine4of2_px": 100.0,
                "dist_nose2_spine4of1_px": 100.0,
                "orbital_rad": 0.0,
                "rel_heading_rad": 0.0,
                "bearing_1_rad": math.pi - 0.05,
                "bearing_2_rad": -math.pi + 0.05,
                "sex_1": "male",
                "sex_2": "female",
                "pair_sex": "FM",
                "qc_n_missing_kp_1": 0,
                "qc_n_missing_kp_2": 0,
                "pair_episode_id": 1,
                "dt_frames": np.nan if frame_number == 0 else 1.0,
            }
        )
    # Same pair, new episode after a long gap. Distance is far, so smoothing
    # must not mix these rows into episode 1.
    for offset, frame_number in enumerate(range(100, 104)):
        rows.append(
            {
                "project_id": "toy",
                "day_index": 1,
                "day_label": "toy",
                "Time_s": frame_number / FPS,
                "FrameNum": frame_number,
                "TrackID_1": 1,
                "TrackID_2": 2,
                "TrackUID_1": 1,
                "TrackUID_2": 2,
                "centroid_distance_px": 800.0,
                "dist_nose1_nose2_px": 800.0,
                "dist_nose1_tailtip2_px": 800.0,
                "dist_nose2_tailtip1_px": 800.0,
                "dist_nose1_spine4of2_px": 800.0,
                "dist_nose2_spine4of1_px": 800.0,
                "orbital_rad": 1.0,
                "rel_heading_rad": 1.0,
                "bearing_1_rad": 0.0,
                "bearing_2_rad": 0.0,
                "sex_1": "male",
                "sex_2": "female",
                "pair_sex": "FM",
                "qc_n_missing_kp_1": 0,
                "qc_n_missing_kp_2": 0,
                "pair_episode_id": 2,
                "dt_frames": np.nan if offset == 0 else 1.0,
            }
        )
    pairs = pd.DataFrame(rows)
    tracks = []
    for frame_number in list(range(6)) + list(range(100, 104)):
        for track_uid, speed in ((1, 1.0), (2, 3.0)):
            tracks.append(
                {
                    "project_id": "toy",
                    "day_label": "toy",
                    "FrameNum": frame_number,
                    "TrackUID": track_uid,
                    "body_length_px": 100.0,
                    "speed_bl_s": speed,
                    "Nose_x": 0.0,
                    "Nose_y": 0.0,
                    "TailTip_x": 100.0,
                    "TailTip_y": 0.0,
                    "is_track_start": frame_number in (0, 100),
                    "is_track_end": False,
                }
            )
    # One row with a missing tail so the QC flag has something to catch.
    tracks.append(
        {
            "project_id": "toy",
            "day_label": "toy",
            "FrameNum": 5,
            "TrackUID": 2,
            "body_length_px": 100.0,
            "speed_bl_s": 3.0,
            "Nose_x": 0.0,
            "Nose_y": 0.0,
            "TailTip_x": np.nan,
            "TailTip_y": np.nan,
            "is_track_start": False,
            "is_track_end": True,
        }
    )
    track_df = pd.DataFrame(tracks).drop_duplicates(
        ["project_id", "day_label", "FrameNum", "TrackUID"], keep="last"
    )
    features, join_stats = build_motion_features(pairs, track_df)
    episode_1 = features.loc[features["pair_episode_id"] == 1].sort_values("FrameNum")
    # Body length is 100 px for both fish, so px / 100 = body lengths.
    expected_bl = episode_1["centroid_distance_px"].to_numpy() / 100.0
    if not np.allclose(episode_1["centroid_distance_bl"], expected_bl):
        raise AssertionError("Body-length distance does not match px / mean length")
    # From frame 1 onward, distance falls by 0.2 BL per frame (20 px / 100).
    later = episode_1.iloc[1:]
    if not np.allclose(later["approach_rate_bl_s"], -0.2 * FPS):
        raise AssertionError(f"Approach rate mismatch: {later['approach_rate_bl_s'].tolist()}")
    if not np.allclose(later["closing_speed_bl_s"], 0.2 * FPS):
        raise AssertionError("Closing speed is not the negation of approach rate")
    if not np.allclose(episode_1["speed_mean_bl_s"], 2.0):
        raise AssertionError("Pair mean speed mismatch")
    if not np.allclose(episode_1["speed_diff_bl_s"], -2.0):
        raise AssertionError("Pair speed difference mismatch")
    # Bearings are about pi apart from 0, so neither fish is facing.
    if episode_1["facing_1"].any() or episode_1["facing_2"].any():
        raise AssertionError("Wrapped bearings near pi were treated as facing")
    episode_2 = features.loc[features["pair_episode_id"] == 2]
    if not episode_2["facing_1"].all() or not episode_2["mutual_facing"].all():
        raise AssertionError("Zero bearings were not treated as mutual facing")
    if episode_1["centroid_distance_bl_smooth"].max() > 3:
        raise AssertionError("Smoothing crossed from the far episode into the near one")
    if not np.allclose(episode_2["centroid_distance_bl_smooth"], 8.0):
        raise AssertionError("Constant far-episode distance was not preserved by smoothing")
    wrapped_error = np.abs(wrap_rad(episode_1["bearing_1_rad_smooth"] - (math.pi - 0.05)))
    if np.nanmax(wrapped_error) > 0.1:
        raise AssertionError("Circular smoothing collapsed bearings that sit next to +/- pi")
    if not features.loc[features["FrameNum"] == 5, "qc_missing_nose_or_tail"].all():
        raise AssertionError("Missing tail did not raise the QC flag")
    if features.loc[features["FrameNum"] != 5, "qc_missing_nose_or_tail"].any():
        raise AssertionError("QC flag fired on a complete pose")
    if join_stats["unmatched_fish_1"] or join_stats["unmatched_fish_2"]:
        raise AssertionError("Synthetic join dropped a track")

    # A dt larger than 0.2 s inside one episode must split the smooth window.
    gap_rows = []
    for frame_number, distance, dt in (
        (0, 100.0, np.nan),
        (1, 100.0, 1.0),
        (2, 100.0, 1.0),
        (20, 900.0, 18.0),
        (21, 900.0, 1.0),
    ):
        gap_rows.append(
            {
                "project_id": "toy",
                "day_index": 1,
                "day_label": "toy",
                "Time_s": frame_number / FPS,
                "FrameNum": frame_number,
                "TrackID_1": 1,
                "TrackID_2": 2,
                "TrackUID_1": 1,
                "TrackUID_2": 2,
                "centroid_distance_px": distance,
                "dist_nose1_nose2_px": distance,
                "dist_nose1_tailtip2_px": distance,
                "dist_nose2_tailtip1_px": distance,
                "dist_nose1_spine4of2_px": distance,
                "dist_nose2_spine4of1_px": distance,
                "orbital_rad": 0.0,
                "rel_heading_rad": 0.0,
                "bearing_1_rad": 0.1,
                "bearing_2_rad": 0.1,
                "sex_1": "male",
                "sex_2": "male",
                "pair_sex": "MM",
                "qc_n_missing_kp_1": 0,
                "qc_n_missing_kp_2": 0,
                "pair_episode_id": 9,
                "dt_frames": dt,
            }
        )
    gap_pairs = pd.DataFrame(gap_rows)
    gap_tracks = []
    for frame_number in (0, 1, 2, 20, 21):
        for track_uid in (1, 2):
            gap_tracks.append(
                {
                    "project_id": "toy",
                    "day_label": "toy",
                    "FrameNum": frame_number,
                    "TrackUID": track_uid,
                    "body_length_px": 100.0,
                    "speed_bl_s": 1.0,
                    "Nose_x": 1.0,
                    "Nose_y": 1.0,
                    "TailTip_x": 2.0,
                    "TailTip_y": 2.0,
                    "is_track_start": False,
                    "is_track_end": False,
                }
            )
    gap_features, _ = build_motion_features(gap_pairs, pd.DataFrame(gap_tracks))
    near = gap_features.loc[gap_features["FrameNum"] <= 2, "centroid_distance_bl_smooth"]
    if near.max() > 2:
        raise AssertionError("Smoothing crossed a gap longer than 0.2 s")
    if gap_features["smooth_segment"].nunique() != 2:
        raise AssertionError("A 0.6 s gap did not open a new smooth segment")


def spot_check_features(features: pd.DataFrame) -> None:
    """Invariants on the real feature table."""
    scale = mean_body_length(
        features["body_length_px_1"].to_numpy(),
        features["body_length_px_2"].to_numpy(),
    )
    expected = features["centroid_distance_px"].to_numpy(dtype=float) / scale
    actual = features["centroid_distance_bl"].to_numpy(dtype=float)
    ok = np.isfinite(expected) & np.isfinite(actual)
    if not np.allclose(actual[ok], expected[ok], rtol=1e-5, atol=1e-5):
        raise AssertionError("centroid_distance_bl drifted from px / mean body length")

    both_speeds = features["speed_bl_s_1"].notna() & features["speed_bl_s_2"].notna()
    mean_ok = np.allclose(
        features.loc[both_speeds, "speed_mean_bl_s"],
        (
            features.loc[both_speeds, "speed_bl_s_1"]
            + features.loc[both_speeds, "speed_bl_s_2"]
        )
        / 2.0,
    )
    if not mean_ok:
        raise AssertionError("speed_mean_bl_s is not the mean of the two fish")

    rate_ok = features["approach_rate_bl_s"].notna()
    if not np.allclose(
        features.loc[rate_ok, "closing_speed_bl_s"],
        -features.loc[rate_ok, "approach_rate_bl_s"],
    ):
        raise AssertionError("closing_speed_bl_s is not -approach_rate_bl_s")

    bearing = wrap_rad(features["bearing_1_rad"].to_numpy(dtype=float))
    known = np.isfinite(bearing)
    facing = features["facing_1"].to_numpy(dtype=object)
    expected_facing = np.abs(bearing) < FACING_MAX_RAD
    if not np.array_equal(facing[known].astype(bool), expected_facing[known]):
        raise AssertionError("facing_1 does not match the bearing cutoff")

    distance = features["centroid_distance_bl"]
    known_distance = distance.notna()
    proximity = features.loc[known_distance, "proximity"].astype(bool)
    expected_proximity = distance.loc[known_distance] <= PROXIMITY_BL
    if not np.array_equal(proximity.to_numpy(), expected_proximity.to_numpy()):
        raise AssertionError("proximity does not match the 1.5 body-length cutoff")

    overlap = features["centroid_distance_px"] <= 5
    if overlap.any():
        median_bl = float(features.loc[overlap, "centroid_distance_bl"].median())
        if median_bl > 0.15:
            raise AssertionError(
                f"Near-zero pixel distances are not near zero in body lengths (median {median_bl})"
            )

    raw = features["centroid_distance_bl"]
    smooth = features["centroid_distance_bl_smooth"]
    both = raw.notna() & smooth.notna()
    median_abs = float((raw.loc[both] - smooth.loc[both]).abs().median())
    # Pose jumps pull a few frames far from the local median. The typical
    # residual should still be a small fraction of a body length.
    if median_abs > 0.1:
        raise AssertionError(f"Smoothed distance median absolute residual is {median_abs}")

    keys = ["day_label", "pair_episode_id", "TrackUID_1", "TrackUID_2"]
    sizes = features.groupby(keys, sort=False).size()
    for day_label in features["day_label"].drop_duplicates():
        day_sizes = sizes.loc[sizes.index.get_level_values("day_label") == day_label]
        episode_key = day_sizes.sort_values(ascending=False).index[0]
        mask = np.ones(len(features), dtype=bool)
        for column, value in zip(keys, episode_key):
            mask &= features[column].to_numpy() == value
        episode = features.loc[mask, ["centroid_distance_bl", "smooth_segment"]].copy()
        segment_ids = episode.groupby("smooth_segment", sort=False).ngroup()
        alone = _rolling_median_mean(episode, ["centroid_distance_bl"], segment_ids)
        table_smooth = features.loc[mask, "centroid_distance_bl_smooth"].to_numpy()
        if not np.allclose(alone["centroid_distance_bl"].to_numpy(), table_smooth, equal_nan=True):
            raise AssertionError(f"Smoothing for {episode_key} changed when the episode was isolated")

    if not (features["TrackUID_1"] < features["TrackUID_2"]).all():
        raise AssertionError("Pair ordering TrackUID_1 < TrackUID_2 was lost")
