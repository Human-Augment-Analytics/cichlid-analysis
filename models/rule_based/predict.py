"""Run the refined rule-based detector (`rule_v2` in detector.py) on any annotated video.

The rule has no training step: the thresholds are fixed (close: < 160 px; start: >= 0.75 turns in
4 s; keep going: >= 0.4 turns; pauses < 3 s joined; minimum length 2 s). It reads
`<video>.parquet` and `<video>_pairs.parquet` (or the `0031_vid` file names), drops the pair rows
that circling_cleaning.py flags, and writes `<output dir>/<video>.csv` with Direction and Cycles
from episode_labels.py. Videos without a traced floor outline get no reflection flag, as in
models/xgboost/features.py.

    uv run python -m models.rule_based.predict --data-dir outputs/generalizability/data \
        --videos 0007_vid,0028_vid --output-dir outputs/generalizability/rule_based/predictions
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models.rule_based import circling_cleaning as cc
from models.rule_based import episode_labels as el
from models.xgboost import features as xgf

FPS = 30
PAIR_COLUMNS = ["FrameNum", "TrackID_1", "TrackID_2", "centroid_distance_px", "d_orbital_rad_s"]


def closest_pair_series(data_dir, video, frames):
    """Distance and orbital rate of the closest cleaned pair in each frame (NaN distance: no pair)."""
    fish_path, pairs_path = xgf.input_paths(data_dir, video)
    fish = xgf.flag_fish(pd.read_parquet(fish_path, columns=cc.FISH_COLS), video)
    pairs = cc.flag_pairs(pd.read_parquet(pairs_path, columns=PAIR_COLUMNS), fish)
    closest = (pairs[pairs.keep].sort_values(["FrameNum", "centroid_distance_px"])
               .drop_duplicates("FrameNum").set_index("FrameNum").reindex(pd.RangeIndex(0, frames)))
    rate = closest.d_orbital_rad_s.where(closest.d_orbital_rad_s.abs() <= el.RATE_MAX).fillna(0)
    return closest.centroid_distance_px.to_numpy(), rate.to_numpy()


def signals(dist, rate, dist_max=160, win_s=4, frac_close_min=0.6):
    """Turns of the pair line in a centered window, and whether the pair is close for most of it."""
    win = int(win_s * FPS)
    close = np.nan_to_num(dist, nan=1e9) < dist_max
    turns = (pd.Series(rate * close / FPS).rolling(win, center=True, min_periods=win // 2).sum().abs()
             .to_numpy() / (2 * np.pi))
    frac = pd.Series(close.astype(float)).rolling(win, center=True, min_periods=win // 2).mean().to_numpy()
    return turns, frac >= frac_close_min


def detect(turns, mostly_close, min_turns=0.75, keep_turns=0.4, merge_s=3, gap_s=1, min_dur_s=2):
    """Inclusive (start, end) frame spans; the defaults are the refined rule."""
    flag = (turns >= min_turns) & mostly_close
    if keep_turns is not None:
        weak = (turns >= keep_turns) & mostly_close
        run = np.cumsum(np.r_[1, np.diff(weak.astype(int)) != 0]) * weak
        started = np.unique(run[flag & weak])
        flag = np.isin(run, started[started > 0])
    idx = np.flatnonzero(flag)
    spans = []
    if len(idx):
        for group in np.split(idx, np.flatnonzero(np.diff(idx) > gap_s * FPS) + 1):
            if group[-1] - group[0] >= min_dur_s * FPS:
                spans.append([int(group[0]), int(group[-1])])
    if merge_s and spans:
        merged = [spans[0]]
        for start, end in spans[1:]:
            if start - merged[-1][1] <= merge_s * FPS:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        spans = merged
    return [tuple(span) for span in spans]


def predict(videos, data_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = xgf.frame_counts(videos, data_dir)
    for video in videos:
        dist, rate = closest_pair_series(data_dir, video, frames[video])
        spans = detect(*signals(dist, rate))
        labels = el.label_intervals(dist, rate, spans)
        table = pd.DataFrame(
            [{"VideoID": video, "EventID": f"{video}-R{index:03d}", "StartFrame": start, "EndFrame": end,
              "Direction": direction, "Cycles": round(cycles, 2), "detector_name": "rule_v2"}
             for index, ((start, end), (direction, cycles)) in enumerate(zip(spans, labels), start=1)],
            columns=["VideoID", "EventID", "StartFrame", "EndFrame", "Direction", "Cycles", "detector_name"])
        table.to_csv(output_dir / f"{video}.csv", index=False)
        print(f"{video}: {len(table)} spans", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=lambda text: text.split(","), default=xgf.selected_videos())
    parser.add_argument("--data-dir", type=Path, default=cc.default_data_dir(cc.ROOT))
    parser.add_argument("--output-dir", type=Path, default=cc.ROOT / "outputs" / "predictions" / "rule_based")
    args = parser.parse_args()
    predict(args.videos, args.data_dir, args.output_dir)
