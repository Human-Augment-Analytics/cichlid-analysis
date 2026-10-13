"""The rule-based detector's cleaned signals, flattened over 121 frames."""

from pathlib import Path

import numpy as np
import pandas as pd

from models.rule_based import circling_cleaning as cc

WINDOW = 61
HALF_WINDOW = WINDOW // 2
SIGNALS = ("centroid_distance_px", "d_orbital_rad_s")
FEATURES = [f"{signal}_offset_{offset:+d}"
            for signal in SIGNALS for offset in range(-HALF_WINDOW, HALF_WINDOW + 1)]
VIDEO_FRAMES = {"0028_vid": 1_080_175, "0031_vid": 1_080_398}
ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH = ROOT / "ground_truth_annotations/finalized_annotations/merged_spans.csv"


def read_spans(path=GROUND_TRUTH):
    """Read usable positive annotations, merging overlaps before sampling."""
    from evaluation.metrics import merge

    table = pd.read_csv(path)
    for flag in ("Circling", "ShouldUse"):
        if flag in table:
            table = table[table[flag].astype(str).str.lower().isin(("true", "1"))]
    spans = {}
    for video, rows in table.groupby("VideoID"):
        intervals = []
        for start, end in rows[["StartFrame", "EndFrame"]].itertuples(index=False, name=None):
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


def closest_pair_series(pairs, count):
    """Match detector.py: closest retained pair per frame; reject rates >15 rad/s.

    Missing frames retain NaN distance and zero rate, as in the rule baseline.
    Frame indices remain contiguous even when no pair survives cleaning.
    """
    closest = (pairs[pairs.keep].sort_values(["FrameNum", "centroid_distance_px"])
               .drop_duplicates("FrameNum").set_index("FrameNum")
               .reindex(pd.RangeIndex(count)))
    distance = closest.centroid_distance_px.to_numpy(dtype=np.float32, copy=True)
    distance[~np.isfinite(distance)] = np.nan
    rate = closest.d_orbital_rad_s.where(closest.d_orbital_rad_s.abs() <= 15).fillna(0)
    return np.column_stack((distance, rate.to_numpy(dtype=np.float32)))


def load_series(data_dir, video, count):
    _, pairs = cc.load_flagged(data_dir, video.removesuffix("_vid"),
                              ["FrameNum", "TrackID_1", "TrackID_2", *SIGNALS])
    if len(pairs) and (pairs.FrameNum.min() < 0 or pairs.FrameNum.max() >= count):
        raise ValueError(f"Pair frames exceed video length: {video}")
    return closest_pair_series(pairs, count)


def window_features(series, frames):
    """242 columns: 121 distances then 121 rates, ordered from t-60 to t+60.

    Missing context at the video boundaries is padded with NaN. XGBoost learns
    missing-value branches from training rows; there is no fitted imputation.
    """
    series = np.asarray(series, dtype=np.float32)
    frames = np.asarray(frames, dtype=np.int64)
    if series.ndim != 2 or series.shape[1] != len(SIGNALS):
        raise ValueError("Expected a (frames, 2) signal array")
    if frames.ndim != 1 or np.any(frames < 0) or np.any(frames >= len(series)):
        raise ValueError("Window centers must be valid video frames")
    padded = np.pad(series, ((HALF_WINDOW, HALF_WINDOW), (0, 0)), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, WINDOW, axis=0)
    return windows[frames].reshape(len(frames), len(FEATURES))
