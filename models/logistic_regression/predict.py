"""Score every frame with the trained model and write predicted circling spans.

Writes one CSV per video to outputs/predictions/logistic_regression/. Run from
the repository root:

    uv run python -m models.logistic_regression.predict
"""

import argparse
from pathlib import Path

import joblib
import polars as pl

from .features import WINDOW, frame_features
from .postprocess import find_positive_spans
from .train import ROOT, VIDEO_FRAMES

FPS = 30


def score_window(bundle, data_dir, video_id, start, end):
    """Circling score for every frame in an inclusive range."""
    data_dir = Path(data_dir)
    poses = (pl.scan_parquet(data_dir / f"{video_id}.parquet")
             .filter(pl.col("FrameNum").is_between(max(0, start - WINDOW), end)).collect())
    pairs = (pl.scan_parquet(data_dir / f"{video_id}_pairs.parquet")
             .filter(pl.col("FrameNum").is_between(start, end)).collect())
    features = frame_features(poses, pairs, start, end)
    scores = bundle["model"].predict_proba(features.select(bundle["features"]).to_numpy())[:, 1]
    return features.select("FrameNum").with_columns(pl.Series("circling_score", scores))


def score_video(bundle, data_dir, video_id, frame_count):
    """Score every frame in bounded chunks."""
    return pl.concat([score_window(bundle, data_dir, video_id, start, min(frame_count - 1, start + 59_999))
                      for start in range(0, frame_count, 60_000)])


def predict(model_path, data_dir, output_dir, threshold=.5, seconds=7.5, fraction=.9):
    """Write `<video_id>.csv` with one row per sustained predicted span."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(model_path)
    paths = []
    for video_id, frame_count in VIDEO_FRAMES.items():
        print(f"Scoring {video_id}", flush=True)
        scores = score_video(bundle, data_dir, video_id, frame_count)
        spans = find_positive_spans(scores, fps=FPS, threshold=threshold, fraction=fraction, seconds=seconds)
        path = output_dir / f"{video_id}.csv"
        spans.select(pl.lit(video_id).alias("VideoID"), pl.col("start_frame").alias("StartFrame"),
                     pl.col("end_frame").alias("EndFrame"), "duration_seconds", "positive_fraction",
                     ).write_csv(path)
        print(f"{video_id}: {spans.height} spans -> {path}", flush=True)
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs" / "logistic_regression" / "model.joblib")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "predictions" / "logistic_regression")
    parser.add_argument("--threshold", type=float, default=.5, help="Per-frame score threshold")
    parser.add_argument("--min-seconds", type=float, default=7.5, help="Spans must be strictly longer than this")
    parser.add_argument("--min-fraction", type=float, default=.9, help="Minimum fraction of positive frames per span")
    args = parser.parse_args()
    predict(args.model, args.data_dir, args.output_dir,
            args.threshold, args.min_seconds, args.min_fraction)
