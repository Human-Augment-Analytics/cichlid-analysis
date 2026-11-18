"""Paths, labels and postprocessing shared by the shared-feature experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.metrics import evaluate_video, merge, summarize
from models.xgboost.features import GROUND_TRUTH, VIDEO_FRAMES, frame_labels, read_spans
from models.xgboost.postprocess import VOTE_FRACTION, majority_vote_fraction

ROOT = Path(__file__).resolve().parents[2]
FEATURE_TABLE = ROOT / "outputs/pair_motion/pair_features_frame.parquet"
EXPERIMENT_DIR = ROOT / "outputs/experiments"
MODEL_FRAMES = EXPERIMENT_DIR / "model_frames.parquet"
PREDICTION_ROOT = ROOT / "outputs/predictions"
VIDEOS = list(VIDEO_FRAMES)  # ["0028_vid", "0031_vid"]
SEED = 42

__all__ = [
    "EXPERIMENT_DIR", "FEATURE_TABLE", "GROUND_TRUTH", "MODEL_FRAMES", "PREDICTION_ROOT", "ROOT",
    "SEED", "VIDEOS", "VIDEO_FRAMES", "VOTE_FRACTION", "evaluate_video", "frame_labels",
    "majority_vote_fraction", "merge", "positive_spans", "read_spans", "summarize",
    "scores_to_spans",
]


def positive_spans(mask) -> list[tuple[int, int]]:
    """Contiguous True runs as inclusive (start, end) frame ranges."""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def scores_to_spans(scores, threshold: float, min_length: int = 0, vote: bool = True):
    """Threshold per-frame scores, smooth by a centered majority vote, drop short spans."""
    scores = np.asarray(scores, dtype=float)
    positive = np.where(np.isfinite(scores), scores, 0.0) >= threshold
    if vote:
        positive = majority_vote_fraction(positive.astype(float), threshold=0.5) >= VOTE_FRACTION
    spans = positive_spans(positive)
    return [(s, e) for s, e in spans if e - s + 1 >= min_length]
