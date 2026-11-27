"""Native ActionFormer proposal conversion; no frame-span detector is used."""
import numpy as np
import polars as pl

PREDICTION_SCHEMA = {
    'VideoID': pl.String, 'StartFrame': pl.Int64, 'EndFrame': pl.Int64,
    'duration_seconds': pl.Float64, 'positive_fraction': pl.Float64,
    'Confidence': pl.Float64,
}


def inclusive_intervals(proposals, frame_count, fps=30.0, min_seconds=0.0, merge_seconds=0.0):
    """Convert continuous [start, end) proposals to clipped inclusive frames."""
    converted = []
    for (start, end), confidence in proposals:
        first = max(0, int(np.floor(start)))
        last = min(frame_count - 1, int(np.ceil(end)) - 1)
        if last < first:
            continue
        if (last - first + 1) / fps < min_seconds:
            continue
        converted.append([first, last, float(confidence)])
    converted.sort(key=lambda row: (row[0], row[1], -row[2]))
    if merge_seconds > 0:
        merged = []
        gap = merge_seconds * fps
        for first, last, confidence in converted:
            if merged and first - merged[-1][1] - 1 < gap:
                merged[-1][1] = max(merged[-1][1], last)
                merged[-1][2] = max(merged[-1][2], confidence)
            else:
                merged.append([first, last, confidence])
        converted = merged
    rows = [{
        'StartFrame': first, 'EndFrame': last,
        'duration_seconds': (last - first + 1) / fps,
        'positive_fraction': 1.0, 'Confidence': confidence,
    } for first, last, confidence in converted]
    return pl.DataFrame(rows, schema={k: v for k, v in PREDICTION_SCHEMA.items() if k != 'VideoID'})


def prediction_table(video, proposals, frame_count, fps=30.0, min_seconds=0.0, merge_seconds=0.0):
    intervals = inclusive_intervals(proposals, frame_count, fps, min_seconds, merge_seconds)
    return intervals.with_columns(pl.lit(video).alias('VideoID')).select(list(PREDICTION_SCHEMA))


def interval_scores(intervals, frame_count):
    """Maximum native proposal confidence covering each frame, for held-out AP."""
    scores = np.zeros(frame_count, dtype=np.float64)
    for start, end, confidence in intervals.select('StartFrame', 'EndFrame', 'Confidence').iter_rows():
        scores[start:end + 1] = np.maximum(scores[start:end + 1], confidence)
    return scores
