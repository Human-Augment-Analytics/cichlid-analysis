"""Rule-based circling direction and cycle count for any circling intervals.

Labels (start, end) frame intervals, from any detector, with a rotation direction
("CW"/"CCW") and a number of circles, from the rotation of the line between the two
closest fish. Use it to pair any interval detector with these rule-based labels.

Rules:

  rotation   per frame, d_orbital_rad_s / FPS while the pair is within DIST_MAX px,
             else 0 (positive is clockwise on screen, checked against the annotations)
  direction  each frame takes the sign of the rotation over the surrounding WIN_S
             seconds; an interval takes the direction covering most of its frames
  cycles     the net rotation of each same-direction stretch, summed, in turns, so
             jitter cancels out and an interval that reverses counts circles both ways

Label a model's predictions from the repository root, writing a copy of each CSV
with Direction and Cycles columns (existing ones are replaced):

    uv run python -m models.rule_based.episode_labels outputs/predictions/xgboost

writes outputs/predictions/xgboost_rule_labels/, which the evaluation picks up as a
separate model. Data location as in circling_cleaning.py.

Passing one multi-video interval CSV instead creates oracle-span labels, by
default in outputs/predictions/rule_based_oracle_labels/.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import circling_cleaning as cc
except ImportError:  # imported as a sibling module, e.g. by the marimo notebooks
    import circling_cleaning as cc

FPS = 30
DIST_MAX = 160      # px: about one body length
WIN_S = 4           # seconds: window for each frame's local rotation direction
RATE_MAX = 15       # rad/s: faster orbital rates are tracking glitches and count as 0


def video_key(video):
    """'0031' from '0031', '0031_vid' or '0031_vid.csv'."""
    match = re.match(r"\d+", Path(str(video)).name)
    if not match:
        raise ValueError(f"cannot read a video number from {video!r}")
    return match.group()


def build_series(data_dir, video):
    """Per-frame closest-pair distance and orbital rate, before ("raw") and after ("clean") cleaning."""
    _, pairs = cc.load_flagged(data_dir, video, ["FrameNum", "TrackID_1", "TrackID_2", "centroid_distance_px",
                                                 "d_orbital_rad_s"])
    n = int(pairs.FrameNum.max()) + 1
    out = {}
    for tag, q in (("raw", pairs), ("clean", pairs[pairs.keep])):
        q = q.sort_values(["FrameNum", "centroid_distance_px"]).drop_duplicates("FrameNum").set_index("FrameNum")
        q = q.reindex(pd.RangeIndex(0, n))
        out[f"dist_{tag}"] = q.centroid_distance_px.to_numpy()
        out[f"rate_{tag}"] = q.d_orbital_rad_s.where(q.d_orbital_rad_s.abs() <= RATE_MAX).fillna(0).to_numpy()
    return out


def load_series(data_dir, video, cache_dir=None):
    """`build_series`, cached as <data_dir>/rule_detector_cache/series_<video>.npz.

    The first run reads the full tracking tables (~1 min); later runs take seconds.
    """
    video = video_key(video)
    path = Path(cache_dir or Path(data_dir) / "rule_detector_cache") / f"series_{video}.npz"
    if path.exists():
        return dict(np.load(path))
    series = build_series(data_dir, video)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **series)
    return series


def rotation(dist, rate, dist_max=DIST_MAX, win_s=WIN_S):
    """Per-frame rotation (radians) and its centered `win_s` rolling sum, which gives the local direction."""
    step = np.asarray(rate, dtype=float) * (np.nan_to_num(dist, nan=1e9) < dist_max) / FPS
    win = int(win_s * FPS)
    local = pd.Series(step).rolling(win, center=True, min_periods=win // 2).sum().to_numpy()
    return step, local


def label_interval(step, local, start, end):
    """(direction, cycles) of one inclusive frame interval, from `rotation` outputs.

    Frames past the end of the tracking data count as no rotation.
    """
    step, local = step[start:end + 1], local[start:end + 1]
    if not len(step):
        return None, 0.0
    cw = np.nan_to_num(local) >= 0
    runs = np.split(np.arange(len(cw)), np.flatnonzero(np.diff(cw)) + 1)
    cycles = sum(abs(step[run].sum()) for run in runs) / (2 * np.pi)
    return ("CW" if cw.mean() >= 0.5 else "CCW"), float(cycles)


def label_intervals(dist, rate, intervals, dist_max=DIST_MAX, win_s=WIN_S):
    """(direction, cycles) for each inclusive (start, end) frame interval of one video."""
    step, local = rotation(dist, rate, dist_max, win_s)
    return [label_interval(step, local, int(start), int(end)) for start, end in intervals]


def label_video(data_dir, video, intervals, cleaned=True, **kwargs):
    """`label_intervals` on a video's tracking data, loaded with `load_series`."""
    series = load_series(data_dir, video)
    tag = "clean" if cleaned else "raw"
    return label_intervals(series[f"dist_{tag}"], series[f"rate_{tag}"], intervals, **kwargs)


def frame_columns(table):
    """Names of the start and end frame columns, matched as in evaluation.evaluate.read_intervals."""
    columns = {re.sub(r"[^a-z]", "", column.lower()): column for column in table.columns}
    starts = [columns[key] for key in ("startframe", "start") if key in columns]
    ends = [columns[key] for key in ("endframe", "end") if key in columns]
    if not starts or not ends:
        raise ValueError(f"missing start/end frame columns in {list(table.columns)}")
    return starts[0], ends[0]


def label_table(table, data_dir, video, **kwargs):
    """Copy of a prediction table with Direction and Cycles columns from the rule-based labels."""
    table = table.drop(columns=[c for c in table.columns
                                if re.sub(r"[^a-z]", "", c.lower()) in ("direction", "cycles")])
    start, end = frame_columns(table)
    labels = label_video(data_dir, video, zip(table[start], table[end]), **kwargs) if len(table) else []
    return table.assign(Direction=[d for d, _ in labels], Cycles=[round(c, 2) for _, c in labels])


def label_predictions(input_dir, output_dir=None, data_dir=None, **kwargs):
    """Label every <video>.csv in a prediction folder; returns the output folder.

    The output defaults to <input_dir>_rule_labels next to the input.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir or input_dir.with_name(f"{input_dir.name}_rule_labels"))
    data_dir = Path(data_dir or cc.default_data_dir(cc.ROOT))
    paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"{input_dir}: no prediction CSV files")
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("output folder must differ from the input folder")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        label_table(pd.read_csv(path), data_dir, path.stem, **kwargs).to_csv(output_dir / path.name, index=False)
    return output_dir


def label_interval_file(input_path, output_dir=None, data_dir=None, **kwargs):
    """Label one multi-video interval CSV, writing one file per video.

    This is the oracle-span workflow: pass a canonical multi-video interval CSV
    to measure direction and count independently of detector boundaries.
    """
    input_path = Path(input_path)
    table = pd.read_csv(input_path)
    columns = {re.sub(r"[^a-z]", "", column.lower()): column for column in table.columns}
    if "videoid" not in columns:
        raise ValueError(f"{input_path}: missing VideoID column")
    video_column = columns["videoid"]
    output_dir = Path(output_dir or cc.ROOT / "outputs/predictions/rule_based_oracle_labels")
    output_dir.mkdir(parents=True, exist_ok=True)
    for video, rows in table.groupby(video_column, sort=True):
        start_column, end_column = frame_columns(rows)
        merged = []
        for start, end in sorted(zip(rows[start_column], rows[end_column])):
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([int(start), int(end)])
        canonical = pd.DataFrame({"VideoID": video,
                                  "StartFrame": [span[0] for span in merged],
                                  "EndFrame": [span[1] for span in merged]})
        label_table(canonical, data_dir or cc.default_data_dir(cc.ROOT), video, **kwargs).to_csv(
            output_dir / f"{video}.csv", index=False)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", help="prediction folder, or one multi-video interval CSV")
    parser.add_argument("--output-dir", help="default: <input_dir>_rule_labels")
    parser.add_argument("--data-dir", help="tracking parquet folder (default as in circling_cleaning.py)")
    parser.add_argument("--raw", action="store_true", help="use the tracking data before cleaning")
    args = parser.parse_args()
    source = Path(args.input_dir)
    if source.is_file():
        out = label_interval_file(source, args.output_dir, args.data_dir, cleaned=not args.raw)
    else:
        out = label_predictions(source, args.output_dir, args.data_dir, cleaned=not args.raw)
    print(f"wrote {out}")
