"""Score every frame with the trained model and write predicted circling spans.

Writes one CSV per video to outputs/predictions/logistic_regression/. Run from
the repository root:

    uv run python -m models.logistic_regression.predict
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import polars as pl

from .features import FEATURE_SOURCE, WINDOW, frame_features
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


def score_video(model_path, data_dir, video_id, frame_count, output_dir):
    """Score in bounded chunks; cache by model hash and input file sizes."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    inputs = [data_dir / f"{video_id}.parquet", data_dir / f"{video_id}_pairs.parquet"]
    signature = {
        "frames": frame_count, "feature_source": FEATURE_SOURCE,
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "inputs": {p.name: p.stat().st_size for p in inputs},
    }
    cache = output_dir / f"{video_id}_scores.parquet"
    meta = cache.with_suffix(".json")
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == signature:
        return pl.read_parquet(cache)
    bundle = joblib.load(model_path)
    chunks = [score_window(bundle, data_dir, video_id, start, min(frame_count - 1, start + 59_999))
              for start in range(0, frame_count, 60_000)]
    result = pl.concat(chunks)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.write_parquet(cache)
    meta.write_text(json.dumps(signature, indent=2) + "\n")
    return result


def predict(model_path, data_dir, scores_dir, output_dir, threshold=.5, seconds=7.5, fraction=.9):
    """Write `<video_id>.csv` with one row per sustained predicted span."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for video_id, frame_count in VIDEO_FRAMES.items():
        print(f"Scoring {video_id}", flush=True)
        scores = score_video(model_path, data_dir, video_id, frame_count, scores_dir)
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
    parser.add_argument("--model", type=Path, default=ROOT / "outputs" / "circling_baseline" / "model.joblib")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--scores-dir", type=Path, default=ROOT / "outputs" / "circling_baseline" / "scores",
                        help="Cache of per-frame scores")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "predictions" / "logistic_regression")
    parser.add_argument("--threshold", type=float, default=.5, help="Per-frame score threshold")
    parser.add_argument("--min-seconds", type=float, default=7.5, help="Spans must be strictly longer than this")
    parser.add_argument("--min-fraction", type=float, default=.9, help="Minimum fraction of positive frames per span")
    args = parser.parse_args()
    predict(args.model, args.data_dir, args.scores_dir, args.output_dir,
            args.threshold, args.min_seconds, args.min_fraction)
