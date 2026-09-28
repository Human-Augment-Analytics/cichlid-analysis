"""Aligned binary label bars for the notebook's circling model review."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter

from circling_model import WINDOW, frame_features, frame_labels


def score_window(model_path, poses_path, start, end, pairs_path=None):
    """Cache predictions; invalidate when model, poses or pairs change."""
    model_path, poses_path = Path(model_path), Path(poses_path)
    pairs_path = Path(pairs_path) if pairs_path else poses_path.with_name(f'{poses_path.stem}_pairs.parquet')
    return _score_window(str(model_path.resolve()), model_path.stat().st_mtime_ns,
                         str(poses_path.resolve()), poses_path.stat().st_mtime_ns,
                         str(pairs_path.resolve()), pairs_path.stat().st_mtime_ns,
                         int(start), int(end))


@lru_cache(maxsize=2)
def _score_window(model_path, model_mtime, poses_path, poses_mtime, pairs_path, pairs_mtime, start, end):
    bundle = joblib.load(model_path)
    poses = (pl.scan_parquet(poses_path)
             .filter(pl.col('FrameNum').is_between(max(0, start - WINDOW), end)).collect())
    pairs = (pl.scan_parquet(pairs_path)
             .filter(pl.col('FrameNum').is_between(start, end)).collect())
    features = frame_features(poses, pairs, start, end)
    scores = bundle['model'].predict_proba(features.select(bundle['features']).to_numpy())[:, 1]
    return features.select('FrameNum').with_columns(pl.Series('circling_score', scores))


def positive_runs(frames, labels):
    """Return (start, width) intervals, preserving single frames and gaps."""
    frames, labels = np.asarray(frames), np.asarray(labels, dtype=bool)
    selected = frames[labels]
    if not len(selected):
        return []
    breaks = np.flatnonzero(np.diff(selected) != 1)
    starts = np.r_[selected[0], selected[breaks + 1]]
    ends = np.r_[selected[breaks], selected[-1]]
    return list(zip(starts, ends - starts + 1))


def plot_timeline(predictions, spans, *, threshold=0.5, video_name=''):
    """Draw two aligned bars, with one color per positive label source."""
    if predictions.is_empty() or not 0 <= threshold <= 1:
        raise ValueError('Provide predictions and a threshold between zero and one')
    frames = predictions['FrameNum'].to_numpy()
    if len(frames) > 1 and not np.all(np.diff(frames) == 1):
        raise ValueError('Timeline requires sorted predictions for every consecutive frame')
    scores = predictions['circling_score'].to_numpy()
    if not np.isfinite(scores).all():
        raise ValueError('Predictions must be finite')
    truth, predicted = frame_labels(frames, spans), scores >= threshold
    figure = Figure(figsize=(15, 3.4), layout='constrained')
    axis = figure.subplots()
    start, stop = int(frames[0]), int(frames[-1]) + 1
    for y, labels, color in [(1, truth, '#159b8e'), (0, predicted, '#7954c7')]:
        axis.broken_barh([(start, stop - start)], (y - 0.30, 0.60), facecolors='#edf0f3')
        axis.broken_barh(positive_runs(frames, labels), (y - 0.30, 0.60), facecolors=color)
    for left, right in spans:
        for boundary in (left, right + 1):
            if start <= boundary <= stop:
                axis.axvline(boundary, color='#54616e', lw=0.9, ls=(0, (4, 4)), alpha=0.6)
    axis.set(yticks=[1, 0], yticklabels=['Ground truth', 'Logistic regression'],
             xlim=(start, stop), ylim=(-0.55, 1.55), xlabel='Frame number (zero-based)',
             title=f'{video_name}  ·  Circling labels  ·  threshold {threshold:.2f}')
    axis.tick_params(axis='y', length=0, pad=12, labelsize=12)
    axis.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axis.spines[['top', 'right', 'left']].set_visible(False)
    axis.legend(handles=[Patch(color='#159b8e', label='Ground truth: circling'),
                         Patch(color='#7954c7', label='Model: circling'),
                         Patch(color='#edf0f3', label='Non-circling')],
                loc='upper center', bbox_to_anchor=(0.5, -0.32), ncols=3, frameon=False)
    return figure
