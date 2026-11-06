import argparse
from pathlib import Path

import numpy as np
import polars as pl

from models.common.config import OUTPUTS, model_name, video_frames
from models.logistic_regression.postprocess import find_positive_spans
from models.logistic_regression.predict import FPS

def merge_spans(spans, scores, threshold, merge_seconds, fps=FPS):
    if merge_seconds <= 0 or spans.height < 2:
        return spans
    gap = merge_seconds * fps
    merged = []
    for start, end in spans.select('start_frame', 'end_frame').iter_rows():
        if merged and start - merged[-1][1] - 1 < gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    positive = np.r_[0, np.cumsum(scores['circling_score'].to_numpy() >= threshold)]
    first = int(scores['FrameNum'][0])
    rows = []
    for index, (start, end) in enumerate(merged, 1):
        n = end - start + 1
        count = int(positive[end - first + 1] - positive[start - first])
        rows.append(dict(span=index, start_frame=start, end_frame=end, duration_seconds=n / fps,
                         positive_frames=count, frame_count=n, positive_fraction=count / n))
    return pl.DataFrame(rows, schema=spans.schema)

def predict(scores_dir, output_dir, threshold=.5, seconds=7.5, fraction=.9, train_command='train',
            merge_seconds=0.0):
    scores_dir, output_dir = Path(scores_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = video_frames()
    for stale in output_dir.glob('*.csv'):
        if stale.stem not in frames:
            stale.unlink()
    paths = []
    for video_id, frame_count in frames.items():
        path = scores_dir / f'{video_id}_scores.parquet'
        if not path.exists():
            raise SystemExit(f'{path} not found: run `python -m {train_command}` first')
        scores = pl.read_parquet(path)
        if scores.height != frame_count:
            raise ValueError(f'{path}: {scores.height} frames, expected {frame_count}')
        spans = find_positive_spans(scores, fps=FPS, threshold=threshold, fraction=fraction, seconds=seconds)
        before = spans.height
        spans = merge_spans(spans, scores, threshold, merge_seconds)
        path = output_dir / f'{video_id}.csv'
        spans.select(pl.lit(video_id).alias('VideoID'), pl.col('start_frame').alias('StartFrame'),
                     pl.col('end_frame').alias('EndFrame'), 'duration_seconds', 'positive_fraction',
                     ).write_csv(path)
        merged = f' ({before} before joining gaps < {merge_seconds:g} s)' if merge_seconds > 0 else ''
        print(f'{video_id}: {spans.height} spans{merged} -> {path}', flush=True)
        paths.append(path)
    return paths

def main(names, default, train_command, description=None):
    parser = argparse.ArgumentParser(description=description)
    if len(names) > 1:
        parser.add_argument('--features', choices=sorted(names), default=default,
                            help=', '.join(f'{k} -> {v}' for k, v in names.items()))
    parser.add_argument('--scores-dir', type=Path, default=None,
                        help='Per-frame scores written by train (default outputs/<name>/scores)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Default outputs/predictions/<name>')
    parser.add_argument('--threshold', type=float, default=.5, help='Per-frame score threshold')
    parser.add_argument('--min-seconds', type=float, default=7.5, help='Spans must be strictly longer than this')
    parser.add_argument('--min-fraction', type=float, default=.9, help='Minimum fraction of positive frames per span')
    parser.add_argument('--merge-seconds', type=float, default=0.0,
                        help='Join spans separated by a gap shorter than this (0 = off, as the logistic regression)')
    args = parser.parse_args()
    name = model_name(names[getattr(args, 'features', default)])
    predict(args.scores_dir or OUTPUTS / name / 'scores', args.output_dir or OUTPUTS / 'predictions' / name,
            args.threshold, args.min_seconds, args.min_fraction, train_command, args.merge_seconds)
