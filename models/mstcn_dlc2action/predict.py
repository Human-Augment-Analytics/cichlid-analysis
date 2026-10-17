

import argparse
from pathlib import Path

import polars as pl

from models.logistic_regression.postprocess import find_positive_spans
from models.logistic_regression.predict import FPS
from models.logistic_regression.train import ROOT, VIDEO_FRAMES

NAME = 'mstcn_dinov2'


def predict(scores_dir, output_dir, threshold=.5, seconds=7.5, fraction=.9):
    """Write `<video_id>.csv` with one row per sustained predicted span."""
    scores_dir, output_dir = Path(scores_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for video_id, frame_count in VIDEO_FRAMES.items():
        path = scores_dir / f'{video_id}_scores.parquet'
        if not path.exists():
            raise SystemExit(f'{path} not found: run `python -m models.mstcn_dlc2action.train` first')
        scores = pl.read_parquet(path)
        if scores.height != frame_count:
            raise ValueError(f'{path}: {scores.height} frames, expected {frame_count}')
        spans = find_positive_spans(scores, fps=FPS, threshold=threshold, fraction=fraction, seconds=seconds)
        path = output_dir / f'{video_id}.csv'
        spans.select(pl.lit(video_id).alias('VideoID'), pl.col('start_frame').alias('StartFrame'),
                     pl.col('end_frame').alias('EndFrame'), 'duration_seconds', 'positive_fraction',
                     ).write_csv(path)
        print(f'{video_id}: {spans.height} spans -> {path}', flush=True)
        paths.append(path)
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scores-dir', type=Path, default=ROOT / 'outputs' / NAME / 'scores',
                        help='Per-frame scores written by train')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'predictions' / NAME)
    parser.add_argument('--threshold', type=float, default=.5, help='Per-frame score threshold')
    parser.add_argument('--min-seconds', type=float, default=7.5, help='Spans must be strictly longer than this')
    parser.add_argument('--min-fraction', type=float, default=.9, help='Minimum fraction of positive frames per span')
    args = parser.parse_args()
    predict(args.scores_dir, args.output_dir, args.threshold, args.min_seconds, args.min_fraction)
