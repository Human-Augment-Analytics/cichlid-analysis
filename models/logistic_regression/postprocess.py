"""Turn per-frame circling scores into sustained predicted spans."""

import math

import numpy as np
import polars as pl


def find_positive_spans(predictions, fps=30., threshold=.5, fraction=.9, seconds=5.):
    """Greedily merge qualifying windows, retaining >=fraction over each union.

    Start from the shortest window strictly longer than `seconds`. Merge
    overlapping or touching windows only when the combined interval qualifies.
    Skip overlapping windows that would invalidate the previous interval.
    Results are disjoint and every returned interval satisfies both conditions.
    """
    if not np.isfinite(fps) or fps <= 0 or seconds < 0 or not 0 < fraction <= 1 or not 0 <= threshold <= 1:
        raise ValueError("Invalid span detection settings")
    frames = predictions["FrameNum"].to_numpy()
    scores = predictions["circling_score"].to_numpy()
    if not np.isfinite(scores).all() or (len(frames) > 1 and not np.all(np.diff(frames) == 1)):
        raise ValueError("Need finite scores for consecutive frames")
    length = math.floor(seconds * fps) + 1
    # Source FPS is approximately 29.999994; use the nominal 30 FPS so that
    # 300 frames (10 seconds) do not pass the strict limit.
    prefix = np.r_[0, np.cumsum(scores >= threshold)]
    candidates = np.flatnonzero(prefix[length:] - prefix[:-length] >= math.ceil(fraction * length))
    intervals = []
    for start in candidates:
        end = int(start + length - 1)
        start = int(start)
        if intervals and start <= intervals[-1][1] + 1:
            previous_start, previous_end = intervals[-1]
            count = int(prefix[end + 1] - prefix[previous_start])
            if count >= math.ceil(fraction * (end - previous_start + 1)):
                intervals[-1] = (previous_start, end)
            elif start > previous_end:
                intervals.append((start, end))
        else:
            intervals.append((start, end))
    rows = []
    for index, (start, end) in enumerate(intervals, 1):
        n = end - start + 1
        count = int(prefix[end + 1] - prefix[start])
        rows.append(dict(span=index, start_frame=int(frames[start]), end_frame=int(frames[end]),
                         duration_seconds=n / fps, positive_frames=count, frame_count=n,
                         positive_fraction=count / n))
    return pl.DataFrame(rows, schema={"span": pl.Int64, "start_frame": pl.Int64, "end_frame": pl.Int64,
        "duration_seconds": pl.Float64, "positive_frames": pl.Int64, "frame_count": pl.Int64,
        "positive_fraction": pl.Float64})
