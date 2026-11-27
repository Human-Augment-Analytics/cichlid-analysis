"""Paths, labels and postprocessing shared by the shared-feature experiments."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from evaluation.metrics import evaluate_video, merge, summarize
from models.rule_based.circling_cleaning import default_data_dir
from models.xgboost.features import GROUND_TRUTH, frame_counts, frame_labels, read_spans, selected_videos
from models.xgboost.postprocess import VOTE_FRACTION, majority_vote_fraction

ROOT = Path(__file__).resolve().parents[2]
# CIRCLING_OUTPUTS keeps a run with other videos (CIRCLING_VIDEOS) apart from the default two-video run.
OUTPUTS = Path(os.environ.get("CIRCLING_OUTPUTS") or ROOT / "outputs").resolve()
FEATURE_TABLE = OUTPUTS / "pair_motion/pair_features_frame.parquet"
EXPERIMENT_DIR = OUTPUTS / "experiments"
MODEL_FRAMES = EXPERIMENT_DIR / "model_frames.parquet"
PREDICTION_ROOT = OUTPUTS / "predictions"
DATA_DIR = Path(os.environ.get("CIRCLING_DATA_DIR") or default_data_dir(ROOT))
VIDEOS = selected_videos()  # default ["0028_vid", "0031_vid"]
VIDEO_FRAMES = frame_counts(VIDEOS, DATA_DIR)
SEED = 42

__all__ = [
    "DATA_DIR", "EXPERIMENT_DIR", "FEATURE_TABLE", "GROUND_TRUTH", "MODEL_FRAMES", "PREDICTION_ROOT", "ROOT",
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
