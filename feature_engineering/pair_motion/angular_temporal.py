"""Angular-movement and time-windowed features for pair rows.

Inputs are the rows of the pair table after the distance, orientation, speed and
smoothing steps, sorted by pair, episode, smooth segment and ``FrameNum``.
Angles are radians and rates are per second.

Windows are measured in frames, not rows, so a gap of a few frames inside a
smooth segment does not stretch a window. A window never reaches into another
smooth segment: segments are separated by pair, episode, or a gap longer than
0.2 s. Windows are ``centered`` (offline analysis) or ``trailing`` (causal).
Rows missing from a window count as "not close" in the fractions and are left
out of the means; ``w*_coverage`` reports how much of the window had data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

FPS = 30.0
ANGULAR_RATE_MAX_RAD_S = 15.0
BODY_LENGTH_RATIO_RANGE = (0.5, 2.0)
WINDOW_SECONDS = (0.5, 1.0, 2.0, 4.0)
WINDOW_DIRECTION = "centered"
_FRAME_STRIDE = 1 << 24  # larger than any frame number, keeps segments apart

PARAMETERS = {
    "angular_rate_max_rad_s": ANGULAR_RATE_MAX_RAD_S,
    "angular_rate_rule": "abs(rate) > max is flagged and set to NaN in *_clean columns",
    "body_length_ratio_range": list(BODY_LENGTH_RATIO_RANGE),
    "body_length_rule": "either body length outside range x the video median (or not positive) is flagged",
    "window_seconds": list(WINDOW_SECONDS),
    "window_direction": WINDOW_DIRECTION,
    "window_unit": "frames; windows stop at smooth-segment boundaries",
}

RAW_ANGULAR_COLUMNS = [
    "d_orbital_rad_s",
    "d_heading_1_rad_s",
    "d_heading_2_rad_s",
    "bend_angle_rad_1",
    "bend_angle_rad_2",
]

ANGULAR_COLUMNS = [
    "qc_bad_body_length",
    "qc_angular_rate_clipped",
    "d_orbital_rad_s_clean",
    "d_heading_1_rad_s_clean",
    "d_heading_2_rad_s_clean",
    "abs_d_orbital_rad_s",
    "orbital_step_rad",
    "orbital_rate_derived_rad_s",
    "orbital_accel_rad_s2",
    "corotation",
    "heading_rate_abs_mean_rad_s",
    "bend_abs_mean_rad",
    "rel_heading_sin",
    "rel_heading_cos",
    "bearing_1_sin",
    "bearing_1_cos",
    "bearing_2_sin",
    "bearing_2_cos",
]

# name -> description, one entry per windowed statistic
WINDOW_STATS = {
    "close_frac": "fraction of window frames with the pair within the proximity cutoff",
    "mutual_facing_frac": "fraction of window frames with both fish facing each other",
    "coverage": "fraction of window frames that have a pair row",
    "distance_bl_mean": "mean centroid distance in body lengths",
    "speed_mean_bl_s_mean": "mean pair speed",
    "closing_speed_bl_s_mean": "mean closing speed (positive = approaching)",
    "abs_d_orbital_mean": "mean absolute orbital rate on close frames",
    "d_orbital_std": "standard deviation of the orbital rate on close frames",
    "orbit_turns_net": "signed orbital rotation on close frames, in turns",
    "orbit_turns_abs": "total absolute orbital rotation on close frames, in turns",
    "corotation_mean": "mean corotation on close frames",
    "bend_abs_mean": "mean absolute bend angle",
}


def window_label(seconds: float) -> str:
    text = f"{seconds:g}".replace(".", "p")
    return f"w{text}s"


def window_columns(window_seconds=WINDOW_SECONDS) -> list[str]:
    return [f"{window_label(s)}_{stat}" for s in window_seconds for stat in WINDOW_STATS]


def wrap_rad(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _nanmean_rows(stacked: np.ndarray) -> np.ndarray:
    """Column-wise mean ignoring NaN; all-NaN columns give NaN without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stacked, axis=0).astype(np.float32)


def _group_diff(values: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) > 1:
        same = group_ids[1:] == group_ids[:-1]
        out[1:] = np.where(same, values[1:] - values[:-1], np.nan)
    return out


def add_angular_features(frame: pd.DataFrame, segment_ids: np.ndarray) -> None:
    """Add cleaned angular rates, orbital movement, corotation and flags in place.

    Frame-to-frame differences never cross a smooth segment, matching the windows.
    """
    # Body-length sanity, relative to the video's own median.
    lengths = pd.concat([frame["body_length_px_1"], frame["body_length_px_2"]])
    video = pd.concat([frame["day_label"], frame["day_label"]])
    median = lengths.where(lengths > 0).groupby(video.to_numpy()).transform("median")
    n = len(frame)
    median_1, median_2 = median.iloc[:n].to_numpy(), median.iloc[n:].to_numpy()
    low, high = BODY_LENGTH_RATIO_RANGE
    bad = np.zeros(n, dtype=bool)
    for lengths_i, median_i in ((frame["body_length_px_1"], median_1), (frame["body_length_px_2"], median_2)):
        values = lengths_i.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            bad |= ~(values > 0) | (values < low * median_i) | (values > high * median_i)
    frame["qc_bad_body_length"] = bad

    clipped = np.zeros(n, dtype=bool)
    clean: dict[str, np.ndarray] = {}
    for column in ("d_orbital_rad_s", "d_heading_1_rad_s", "d_heading_2_rad_s"):
        values = frame[column].to_numpy(dtype=float).copy()
        with np.errstate(invalid="ignore"):
            too_big = np.abs(values) > ANGULAR_RATE_MAX_RAD_S
        clipped |= too_big
        values[too_big] = np.nan
        clean[column] = values
        frame[f"{column}_clean"] = values.astype(np.float32)
    frame["qc_angular_rate_clipped"] = clipped

    orbit = clean["d_orbital_rad_s"]
    frame["abs_d_orbital_rad_s"] = np.abs(orbit).astype(np.float32)

    # Orbital movement derived from the position angle itself: wrapped step
    # between consecutive rows of one segment, so it does not depend on how the
    # converter defined d_orbital_rad_s.
    dt_s = frame["dt_frames"].to_numpy(dtype=float) / FPS
    step = wrap_rad(_group_diff(frame["orbital_rad"].to_numpy(dtype=float), segment_ids))
    step_ok = np.isfinite(dt_s) & (dt_s > 0) & np.isfinite(step)
    step = np.where(step_ok, step, np.nan)
    derived_rate = np.where(step_ok, step / np.where(step_ok, dt_s, 1.0), np.nan)
    derived_rate[np.abs(derived_rate) > ANGULAR_RATE_MAX_RAD_S] = np.nan
    step[~np.isfinite(derived_rate)] = np.nan
    frame["orbital_step_rad"] = step.astype(np.float32)
    frame["orbital_rate_derived_rad_s"] = derived_rate.astype(np.float32)

    accel = _group_diff(orbit, segment_ids) / np.where(step_ok, dt_s, np.nan)
    accel[~np.isfinite(accel)] = np.nan
    frame["orbital_accel_rad_s2"] = accel.astype(np.float32)

    # +1 when a fish turns the same way the pair rotates, -1 when opposite.
    signs = [np.sign(clean[f"d_heading_{i}_rad_s"]) * np.sign(orbit) for i in (1, 2)]
    frame["corotation"] = _nanmean_rows(np.vstack(signs))
    heading_abs = np.vstack([np.abs(clean["d_heading_1_rad_s"]), np.abs(clean["d_heading_2_rad_s"])])
    frame["heading_rate_abs_mean_rad_s"] = _nanmean_rows(heading_abs)
    bend_abs = np.vstack(
        [np.abs(frame["bend_angle_rad_1"].to_numpy(dtype=float)), np.abs(frame["bend_angle_rad_2"].to_numpy(dtype=float))]
    )
    frame["bend_abs_mean_rad"] = _nanmean_rows(bend_abs)

    for name, source in (
        ("rel_heading", "rel_heading_rad"),
        ("bearing_1", "bearing_1_rad"),
        ("bearing_2", "bearing_2_rad"),
    ):
        angle = frame[source].to_numpy(dtype=float)
        frame[f"{name}_sin"] = np.sin(angle).astype(np.float32)
        frame[f"{name}_cos"] = np.cos(angle).astype(np.float32)


def _window_bounds(segment_ids: np.ndarray, frames: np.ndarray, before: int, after: int):
    position = segment_ids.astype(np.int64) * _FRAME_STRIDE + frames.astype(np.int64)
    if len(position) > 1 and np.any(np.diff(position) < 0):
        raise RuntimeError("Rows must be sorted by segment and frame before windowing")
    low = np.searchsorted(position, position - before, side="left")
    high = np.searchsorted(position, position + after, side="right")
    return low, high


def _window_sum(values: np.ndarray, mask: np.ndarray, low: np.ndarray, high: np.ndarray):
    """Sum and count of the valid entries inside each [low, high) range."""
    valid = mask & np.isfinite(values)
    cumulative = np.r_[0.0, np.cumsum(np.where(valid, values, 0.0))]
    counts = np.r_[0, np.cumsum(valid, dtype=np.int64)]
    return cumulative[high] - cumulative[low], counts[high] - counts[low]


def add_window_features(
    frame: pd.DataFrame,
    segment_ids: np.ndarray,
    window_seconds=WINDOW_SECONDS,
    direction: str = WINDOW_DIRECTION,
) -> None:
    """Add per-window statistics (see ``WINDOW_STATS``) to ``frame`` in place."""
    if direction not in ("centered", "trailing"):
        raise ValueError("direction must be 'centered' or 'trailing'")
    frames = frame["FrameNum"].to_numpy()
    n = len(frame)
    everything = np.ones(n, dtype=bool)

    close = frame["proximity"].fillna(False).to_numpy(dtype=bool)
    mutual = frame["mutual_facing"].fillna(False).to_numpy(dtype=bool)
    orbit_abs = frame["abs_d_orbital_rad_s"].to_numpy(dtype=float)
    orbit_rate = frame["d_orbital_rad_s_clean"].to_numpy(dtype=float)
    step = frame["orbital_step_rad"].to_numpy(dtype=float)
    corotation = frame["corotation"].to_numpy(dtype=float)

    series = {
        "distance": frame["centroid_distance_bl"].to_numpy(dtype=float),
        "speed": frame["speed_mean_bl_s"].to_numpy(dtype=float),
        "closing": frame["closing_speed_bl_s"].to_numpy(dtype=float),
        "bend": frame["bend_abs_mean_rad"].to_numpy(dtype=float),
    }
    new: dict[str, np.ndarray] = {}
    for seconds in window_seconds:
        label = window_label(seconds)
        length = int(round(seconds * FPS))
        before, after = (length // 2, length - length // 2) if direction == "centered" else (length, 0)
        low, high = _window_bounds(segment_ids, frames, before, after)
        span = before + after + 1.0

        def col(stat: str, values: np.ndarray) -> None:
            new[f"{label}_{stat}"] = values.astype(np.float32)

        close_sum, _ = _window_sum(close.astype(float), everything, low, high)
        col("close_frac", close_sum / span)
        mutual_sum, _ = _window_sum(mutual.astype(float), everything, low, high)
        col("mutual_facing_frac", mutual_sum / span)
        count = (high - low).astype(float)
        col("coverage", np.minimum(count / span, 1.0))

        for stat, key in (
            ("distance_bl_mean", "distance"),
            ("speed_mean_bl_s_mean", "speed"),
            ("closing_speed_bl_s_mean", "closing"),
            ("bend_abs_mean", "bend"),
        ):
            total, valid = _window_sum(series[key], everything, low, high)
            col(stat, np.where(valid > 0, total / np.maximum(valid, 1), np.nan))

        total, valid = _window_sum(orbit_abs, close, low, high)
        col("abs_d_orbital_mean", np.where(valid > 0, total / np.maximum(valid, 1), np.nan))
        total_r, valid_r = _window_sum(orbit_rate, close, low, high)
        total_sq, _ = _window_sum(orbit_rate**2, close, low, high)
        mean = total_r / np.maximum(valid_r, 1)
        variance = np.maximum(total_sq / np.maximum(valid_r, 1) - mean**2, 0.0)
        col("d_orbital_std", np.where(valid_r > 1, np.sqrt(variance), np.nan))
        net, valid_s = _window_sum(step, close, low, high)
        col("orbit_turns_net", np.where(valid_s > 0, net / (2.0 * np.pi), np.nan))
        total_abs, _ = _window_sum(np.abs(step), close, low, high)
        col("orbit_turns_abs", np.where(valid_s > 0, total_abs / (2.0 * np.pi), np.nan))
        total_c, valid_c = _window_sum(corotation, close, low, high)
        col("corotation_mean", np.where(valid_c > 0, total_c / np.maximum(valid_c, 1), np.nan))

    for name, values in new.items():
        frame[name] = values
