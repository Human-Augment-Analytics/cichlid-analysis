"""Leakage-resistant postprocessing selection from out-of-fold video scores.

Each score must come from a model that did not train on any sampled segment from
that video. Selection therefore uses realistic full-video score sequences without
using the final fitted model's predictions. Ground-truth labels still select the
settings, so this is cross-validation, not an untouched external test.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl

from evaluation.metrics import evaluate_video
from models.common.predict import postprocess_scores

DEFAULT_SETTINGS = {
    'threshold': 0.5,
    'min_seconds': 7.5,
    'min_fraction': 0.9,
    'merge_seconds': 0.0,
    'smooth_seconds': 0.0,
}
GRID = {
    'threshold': (0.25, 0.35, 0.5, 0.65, 0.8),
    'smooth_seconds': (0.0, 1.0, 2.0),
    'min_seconds': (2.0, 4.0, 7.5),
    'merge_seconds': (0.0, 1.0, 2.0),
}


def load_settings(model_dir, overrides=None):
    """Use explicit CLI values first, then saved OOF settings, then defaults."""
    model_dir = Path(model_dir)
    path = model_dir / 'postprocess.json'
    saved = {}
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get('status') == 'selected':
            saved = payload.get('settings', {})
    settings = {**DEFAULT_SETTINGS, **saved}
    settings.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return settings


def _candidate_settings():
    keys = list(GRID)
    for values in itertools.product(*(GRID[key] for key in keys)):
        yield {**DEFAULT_SETTINGS, **dict(zip(keys, values))}


def select_postprocess(oof_scores, ground_truth, frame_counts, output_dir, folds=None):
    """Select one common span rule by mean OOF frame-F1 and segmental-F1@50."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(oof_scores)
    if not videos or set(videos) != set(frame_counts):
        raise ValueError('OOF postprocessing requires scores for every selected video')
    rows = []
    for settings in _candidate_settings():
        metrics = []
        for video in videos:
            intervals = postprocess_scores(oof_scores[video], **settings).select(
                'start_frame', 'end_frame').rows()
            metrics.append(evaluate_video(ground_truth[video], intervals, frame_counts[video]))
        row = {
            **settings,
            'selection_score': float(np.mean([
                (item['frame_f1'] + item['f1@50']) / 2 for item in metrics])),
            'frame_f1': float(np.mean([item['frame_f1'] for item in metrics])),
            'segmental_f1_50': float(np.mean([item['f1@50'] for item in metrics])),
            'frame_precision': float(np.mean([item['frame_precision'] for item in metrics])),
            'frame_recall': float(np.mean([item['frame_recall'] for item in metrics])),
            'predicted_segments': int(sum(item['pred_segments'] for item in metrics)),
        }
        rows.append(row)
    candidates = pl.DataFrame(rows).sort(
        ['selection_score', 'segmental_f1_50', 'frame_f1', 'threshold'],
        descending=[True, True, True, False])
    candidates.write_csv(output_dir / 'postprocess_candidates.csv')
    best = candidates.row(0, named=True)
    settings = {key: best[key] for key in DEFAULT_SETTINGS}
    payload = {
        'status': 'selected',
        'method': 'video-level out-of-fold full-video predictions; no fold model trains on its validation video',
        'objective': 'mean over videos of (frame_f1 + segmental_f1_50) / 2',
        'selection_score': best['selection_score'],
        'frame_f1': best['frame_f1'],
        'segmental_f1_50': best['segmental_f1_50'],
        'settings': settings,
        'folds': folds or [],
        'grid': {key: list(value) for key, value in GRID.items()},
        'fixed_min_fraction': DEFAULT_SETTINGS['min_fraction'],
        'note': 'Cross-validation diagnostics are not an untouched external test estimate.',
    }
    (output_dir / 'postprocess.json').write_text(json.dumps(payload, indent=2) + '\n')
    print('Selected OOF postprocessing: ' + ', '.join(f'{key}={value}' for key, value in settings.items())
          + f"; mean frame-F1={best['frame_f1']:.3f}, F1@50={best['segmental_f1_50']:.3f}", flush=True)
    return settings
