"""Cleaned pair signals and per-frame motion features for XGBoost."""

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.evaluate import GROUND_TRUTH
from models.rule_based import circling_cleaning as cc

WINDOW = 121
HALF_WINDOW = WINDOW // 2
FPS = 30
SIGNALS = ("centroid_distance_px", "d_orbital_rad_s")
RATE_SIGNALS = ("centroid_distance_rate_px_s", "d_orbital_rate_rad_s2", "intersection_midpoint_speed_px_s")
SERIES_COLUMNS = (*SIGNALS, *RATE_SIGNALS)
WINDOW_FEATURES = [f"{signal}_offset_{offset:+d}"
                   for signal in SERIES_COLUMNS for offset in range(-HALF_WINDOW, HALF_WINDOW + 1)]
FEATURES = WINDOW_FEATURES
FEATURE_DESCRIPTION = (
    f"Rule-based cleaning; closest retained pair's centroid_distance_px and d_orbital_rad_s, "
    f"offsets {-HALF_WINDOW}..+{HALF_WINDOW}. Missing or abs>15 orbital rates become 0 in raw windows. "
    "Three additional per-frame signals: signed distance change (px/s), signed orbital-rate change "
    "(rad/s^2), and head-to-head/tail-to-tail segment-intersection speed (px/s). "
    "All five signals are flattened at every window offset, with no averaging. "
    "Each derivative at frame t uses consecutive same-pair frames t-1 and t at 30 fps, "
    "including a preceding context frame for the window's leftmost offset. "
    "Missing measurements, pair switches, and episode changes invalidate derivatives. "
    "No unique finite segment intersection or invalid consecutive step yields NaN."
)
VIDEO_FRAMES = {"0028_vid": 1_080_175, "0031_vid": 1_080_398}
ROOT = Path(__file__).resolve().parents[2]


def read_spans(paths=GROUND_TRUTH):
    """Read usable positive annotations from one or more CSV files, merging overlaps.

    Header case, spaces and punctuation are ignored ("Start Frame" == "StartFrame");
    fully blank rows are dropped.
    """
    from evaluation.metrics import merge

    if isinstance(paths, (str, Path)):
        paths = [paths]
    table = pd.concat([pd.read_csv(path).rename(columns=lambda name: "".join(filter(str.isalpha, name.lower())))
                       for path in paths], ignore_index=True).dropna(how="all")
    for flag in ("circling", "shoulduse"):
        if flag in table:
            table = table[table[flag].astype(str).str.lower().isin(("true", "1"))]
    spans = {}
    for video, rows in table.groupby("videoid"):
        intervals = []
        for start, end in rows[["startframe", "endframe"]].itertuples(index=False, name=None):
            if not np.isfinite(start) or not np.isfinite(end) or int(start) != start or int(end) != end:
                raise ValueError(f"Non-integer annotation for {video}: {start}, {end}")
            if start < 0 or end < start:
                raise ValueError(f"Invalid annotation for {video}: {start}, {end}")
            intervals.append((int(start), int(end)))
        spans[video] = merge(intervals)
    return spans


def frame_labels(count, spans):
    labels = np.zeros(count, dtype=np.int8)
    for start, end in spans:
        if start < 0 or end < start or end >= count:
            raise ValueError(f"Annotation {start, end} exceeds {count} frames")
        labels[start:end + 1] = 1
    return labels


def sample_frames(count, spans):
    """All positives plus half a segment's length on each side, once per frame.

    Inclusive length is used; half lengths round up. Padding is clipped to the
    video, and every frame in any positive span remains positive. Overlapping
    buffers are unioned, so negatives are never drawn twice or through positives.
    """
    from evaluation.metrics import merge

    spans = merge(spans)
    labels = frame_labels(count, spans)
    selected = np.zeros(count, dtype=bool)
    for start, end in spans:
        half_length = (end - start + 2) // 2
        selected[max(0, start - half_length):min(count, end + half_length + 1)] = True
    frames = np.flatnonzero(selected)
    return frames, labels[frames]


def segment_intersections(head_1, head_2, tail_1, tail_2):
    """Unique intersection of finite head-to-head and tail-to-tail segments.

    Parallel, collinear, degenerate, missing or non-intersecting segments yield
    NaN coordinates. Shared endpoints count as intersections. Calculations use
    float64 and a relative parallelism tolerance of 1e-9.
    """
    h1, h2, t1, t2 = [np.asarray(points, dtype=np.float64)
                      for points in (head_1, head_2, tail_1, tail_2)]
    if h1.ndim != 2 or h1.shape[1] != 2 or any(p.shape != h1.shape for p in (h2, t1, t2)):
        raise ValueError("Intersection inputs must have matching (rows, 2) shapes")
    cross = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    head_direction, tail_direction, offset = h2 - h1, t2 - t1, t1 - h1
    denominator = cross(head_direction, tail_direction)
    scale = np.linalg.norm(head_direction, axis=1) * np.linalg.norm(tail_direction, axis=1)
    valid = np.isfinite(np.column_stack((h1, h2, t1, t2))).all(axis=1)
    valid &= (scale > 0) & (np.abs(denominator) > 1e-9 * scale)
    head_fraction = np.full(len(h1), np.nan)
    tail_fraction = np.full(len(h1), np.nan)
    np.divide(cross(offset, tail_direction), denominator, out=head_fraction, where=valid)
    np.divide(cross(offset, head_direction), denominator, out=tail_fraction, where=valid)
    valid &= ((head_fraction >= -1e-9) & (head_fraction <= 1 + 1e-9)
              & (tail_fraction >= -1e-9) & (tail_fraction <= 1 + 1e-9))
    intersection = np.full_like(h1, np.nan)
    intersection[valid] = h1[valid] + np.clip(head_fraction[valid], 0, 1)[:, None] * head_direction[valid]
    return intersection


def consecutive_same_pair(closest):
    """Do not derive motion across missing frames, pair switches or episodes."""
    ids = closest[["TrackID_1", "TrackID_2"]].to_numpy(dtype=np.float64, copy=True)
    ids.sort(axis=1)  # Reversing the stored fish order does not change the pair.
    valid = np.isfinite(ids).all(axis=1)
    same = np.zeros(len(closest), dtype=bool)
    same[1:] = valid[1:] & valid[:-1] & (ids[1:] == ids[:-1]).all(axis=1)
    if "pair_episode_id" in closest:
        episodes = closest.pair_episode_id
        same &= episodes.eq(episodes.shift()).fillna(False).to_numpy(dtype=bool)
    return same


def step_rates(values, same_pair):
    """Signed first differences per second; NaN when either endpoint is invalid."""
    rates = np.full(len(values), np.nan, dtype=np.float64)
    valid = same_pair[1:] & np.isfinite(values[1:]) & np.isfinite(values[:-1])
    rates[1:] = np.where(valid, np.diff(values) * FPS, np.nan)
    return rates


def closest_pair_series(pairs, count, fish=None):
    """Match detector.py: closest retained pair per frame; reject rates >15 rad/s.

    Raw signals retain baseline missing-value behavior. Three extra columns hold
    valid same-pair frame-to-frame derivatives, retained separately at each offset.
    """
    closest = (pairs[pairs.keep].sort_values(["FrameNum", "centroid_distance_px"])
               .drop_duplicates("FrameNum").set_index("FrameNum")
               .reindex(pd.RangeIndex(count)))
    distance = closest.centroid_distance_px.to_numpy(dtype=np.float32, copy=True)
    distance[~np.isfinite(distance)] = np.nan
    measured_rate = closest.d_orbital_rad_s.where(closest.d_orbital_rad_s.abs() <= 15).to_numpy(dtype=np.float64)
    rate = np.nan_to_num(measured_rate, nan=0).astype(np.float32)
    same_pair = consecutive_same_pair(closest)
    midpoint_speed = np.full(count, np.nan)
    if fish is not None:
        poses = fish.set_index(["FrameNum", "TrackID"])
        coordinates = []
        for side in ("1", "2"):
            key = pd.MultiIndex.from_arrays([closest.index, closest[f"TrackID_{side}"]])
            coordinates.append(poses.reindex(key)[["Head_x", "Head_y", "TailTip_x", "TailTip_y"]].to_numpy(dtype=float))
        midpoint = segment_intersections(coordinates[0][:, :2], coordinates[1][:, :2],
                                         coordinates[0][:, 2:], coordinates[1][:, 2:])
        valid = same_pair[1:] & np.isfinite(midpoint[1:]).all(axis=1) & np.isfinite(midpoint[:-1]).all(axis=1)
        midpoint_speed[1:] = np.where(valid, np.linalg.norm(np.diff(midpoint, axis=0), axis=1) * FPS, np.nan)
    return np.column_stack((distance, rate, step_rates(distance, same_pair),
                            step_rates(measured_rate, same_pair), midpoint_speed)).astype(np.float32)


def load_series(data_dir, video, count):
    video_id = video.removesuffix("_vid")
    fish = cc.flag_fish(pd.read_parquet(cc.fish_path(data_dir, video_id),
                                      columns=[*cc.FISH_COLS, "Head_x", "Head_y", "TailTip_x", "TailTip_y"]), video_id)
    pairs = cc.flag_pairs(pd.read_parquet(cc.pairs_path(data_dir, video_id),
                                         columns=["FrameNum", "TrackID_1", "TrackID_2", "pair_episode_id", *SIGNALS]), fish)
    if len(pairs) and (pairs.FrameNum.min() < 0 or pairs.FrameNum.max() >= count):
        raise ValueError(f"Pair frames exceed video length: {video}")
    return closest_pair_series(pairs, count, fish)


def window_features(series, frames):
    """Flatten all five signal windows in signal-first, then offset order.

    Missing context at the video boundaries is padded with NaN. XGBoost learns
    missing-value branches from training rows; there is no fitted imputation.
    """
    series = np.asarray(series, dtype=np.float32)
    frames = np.asarray(frames, dtype=np.int64)
    if series.ndim != 2 or series.shape[1] != len(SERIES_COLUMNS):
        raise ValueError(f"Expected a (frames, {len(SERIES_COLUMNS)}) signal array")
    if frames.ndim != 1 or np.any(frames < 0) or np.any(frames >= len(series)):
        raise ValueError("Window centers must be valid video frames")
    padded = np.pad(series, ((HALF_WINDOW, HALF_WINDOW), (0, 0)), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, WINDOW, axis=0)
    return windows[frames].reshape(len(frames), len(FEATURES))
