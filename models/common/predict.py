import argparse
from pathlib import Path

import numpy as np
import polars as pl

from models.common.config import OUTPUTS, model_name, video_frames
from models.intervals.logistic_regression.postprocess import find_positive_spans
from models.intervals.logistic_regression.predict import FPS


def smooth_scores(scores, seconds, fps=FPS):
    if not seconds:
        return scores
    window = max(1, int(round(seconds * fps)))
    values = scores['circling_score'].to_numpy()
    padded = np.pad(values, window // 2, mode='edge')
    smoothed = np.convolve(padded, np.ones(window) / window, mode='valid')[:len(values)]
    return scores.with_columns(pl.Series('circling_score', smoothed))


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


def postprocess_scores(scores, threshold=.5, min_seconds=7.5, min_fraction=.9,
                       merge_seconds=0.0, smooth_seconds=0.0):
    """Convert one continuous full-video score sequence to inclusive spans."""
    smoothed = smooth_scores(scores, smooth_seconds)
    spans = find_positive_spans(smoothed, fps=FPS, threshold=threshold,
                                fraction=min_fraction, seconds=min_seconds)
    return merge_spans(spans, smoothed, threshold, merge_seconds)


def predict(scores_dir, output_dir, threshold=None, seconds=None, fraction=None, train_command='train',
            merge_seconds=None, smooth_seconds=None):
    from models.common.postprocess_tuning import load_settings

    scores_dir, output_dir = Path(scores_dir), Path(output_dir)
    settings = load_settings(scores_dir.parent, {
        'threshold': threshold, 'min_seconds': seconds, 'min_fraction': fraction,
        'merge_seconds': merge_seconds, 'smooth_seconds': smooth_seconds,
    })
    threshold, seconds, fraction = settings['threshold'], settings['min_seconds'], settings['min_fraction']
    merge_seconds, smooth_seconds = settings['merge_seconds'], settings['smooth_seconds']
    print('Postprocessing: ' + ', '.join(f'{key}={value}' for key, value in settings.items()), flush=True)
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
        before_spans = find_positive_spans(smooth_scores(scores, smooth_seconds), fps=FPS,
                                          threshold=threshold, fraction=fraction, seconds=seconds)
        before = before_spans.height
        spans = postprocess_scores(scores, threshold=threshold, min_seconds=seconds, min_fraction=fraction,
                                   merge_seconds=merge_seconds, smooth_seconds=smooth_seconds)
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
    parser.add_argument('--threshold', type=float, default=None,
                        help='Per-frame score threshold (default: saved OOF choice, otherwise 0.5)')
    parser.add_argument('--min-seconds', type=float, default=None,
                        help='Spans must be strictly longer than this (default: saved OOF choice, otherwise 7.5)')
    parser.add_argument('--min-fraction', type=float, default=None,
                        help='Minimum positive fraction (default: saved OOF choice, otherwise 0.9)')
    parser.add_argument('--merge-seconds', type=float, default=None,
                        help='Gap joining (default: saved OOF choice, otherwise 0)')
    parser.add_argument('--smooth-seconds', type=float, default=None,
                        help='Score moving average in seconds (default: saved OOF choice, otherwise 0)')
    args = parser.parse_args()
    name = model_name(names[getattr(args, 'features', default)])
    predict(args.scores_dir or OUTPUTS / name / 'scores', args.output_dir or OUTPUTS / 'predictions' / name,
            args.threshold, args.min_seconds, args.min_fraction, train_command, args.merge_seconds,
            args.smooth_seconds)
