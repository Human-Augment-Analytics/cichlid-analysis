"""Training-only selection and diagnostics for native ActionFormer proposals."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl

from evaluation.metrics import evaluate_video
from models.actionformer.model import infer_windows, suppress_cross_window_duplicates
from models.actionformer.postprocess import inclusive_intervals

DECODE_MIN_SCORE = 0.0001
DEFAULT_SETTINGS = {'min_score': 0.001, 'min_seconds': 0.0, 'merge_seconds': 0.0}
SCORE_GRID = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8)
DURATION_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 7.5)
MERGE_GRID = (0.0, 1.0, 2.0, 4.0)


def load_inference_settings(model_dir, min_score=None, min_seconds=None, merge_seconds=None):
    """Load saved settings; explicit prediction arguments take precedence."""
    path = Path(model_dir) / 'postprocess.json'
    saved = {}
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get('status') == 'selected':
            saved = payload.get('settings', {})
    settings = {**DEFAULT_SETTINGS, **saved}
    for key, value in {'min_score': min_score, 'min_seconds': min_seconds,
                       'merge_seconds': merge_seconds}.items():
        if value is not None:
            settings[key] = float(value)
    return settings


def local_ground_truth(sample, length):
    return [(max(0, int(np.floor(start))), min(length - 1, int(np.ceil(end)) - 1))
            for start, end in sample['segments'].numpy() if end > start]


def _select_settings(decoded, output_dir, method, folds=None):
    """Select interval rules on complete decoded sequences, never shared test segments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for min_score, min_seconds, merge_seconds in itertools.product(
            SCORE_GRID, DURATION_GRID, MERGE_GRID):
        metrics = []
        for item in decoded:
            filtered = [proposal for proposal in item['proposals'] if proposal[1] >= min_score]
            intervals = inclusive_intervals(filtered, item['length'], min_seconds=min_seconds,
                                            merge_seconds=merge_seconds)
            predicted = intervals.select('StartFrame', 'EndFrame').rows()
            metrics.append(evaluate_video(item['truth'], predicted, item['length']))
        rows.append({
            'min_score': min_score, 'min_seconds': min_seconds, 'merge_seconds': merge_seconds,
            'selection_score': float(np.mean([
                (m['frame_f1'] + m['f1@50'] + m['edit_score'] / 100) / 3 for m in metrics])),
            'frame_f1': float(np.mean([m['frame_f1'] for m in metrics])),
            'segmental_f1_50': float(np.mean([m['f1@50'] for m in metrics])),
            'edit_score': float(np.mean([m['edit_score'] for m in metrics])),
            'frame_precision': float(np.mean([m['frame_precision'] for m in metrics])),
            'frame_recall': float(np.mean([m['frame_recall'] for m in metrics])),
            'raw_proposals': int(sum(len(item['proposals']) for item in decoded)),
            'retained_intervals': int(sum(m['pred_segments'] for m in metrics)),
        })
    candidates = pl.DataFrame(rows).sort(
        ['selection_score', 'segmental_f1_50', 'frame_f1', 'edit_score',
         'min_score', 'min_seconds', 'merge_seconds'],
        descending=[True, True, True, True, True, True, False])
    candidates.write_csv(output_dir / 'postprocess_candidates.csv')
    best = candidates.row(0, named=True)
    settings = {key: best[key] for key in DEFAULT_SETTINGS}
    payload = {
        'status': 'selected', 'settings': settings, 'method': method,
        'objective': 'mean validation (frame_f1 + segmental_f1_50 + edit_score/100) / 3',
        'selection_score': best['selection_score'], 'frame_f1': best['frame_f1'],
        'segmental_f1_50': best['segmental_f1_50'], 'edit_score': best['edit_score'],
        'score_grid': list(SCORE_GRID), 'duration_grid': list(DURATION_GRID),
        'merge_grid': list(MERGE_GRID), 'decode_min_score': DECODE_MIN_SCORE,
        'folds': folds or [],
        'note': 'Cross-validation diagnostics are not an untouched external test estimate.',
    }
    (output_dir / 'postprocess.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(f"ActionFormer validation postprocessing: min_score={settings['min_score']}, "
          f"min_seconds={settings['min_seconds']}, merge_seconds={settings['merge_seconds']}, "
          f"frame-F1={best['frame_f1']:.3f}, F1@50={best['segmental_f1_50']:.3f}, "
          f"edit={best['edit_score']:.3f}", flush=True)
    return settings


def select_inference_settings(model, validation_samples, device, output_dir):
    """Fallback selection on short validation segments when video OOF is unavailable."""
    output_dir = Path(output_dir)
    if not validation_samples:
        (output_dir / 'postprocess.json').write_text(json.dumps({
            'status': 'default', 'settings': DEFAULT_SETTINGS,
            'reason': 'No training-only validation segments were available.',
        }, indent=2) + '\n')
        return dict(DEFAULT_SETTINGS)
    decoded = []
    for sample in validation_samples:
        length = sample['feats'].shape[1]
        proposals = infer_windows(model, [(0, sample['feats'].numpy().T)], device,
                                  min_score=DECODE_MIN_SCORE)
        decoded.append({'length': length, 'proposals': proposals,
                        'truth': local_ground_truth(sample, length)})
    return _select_settings(
        decoded, output_dir,
        'Native proposals on training-only validation segments; shared test segments excluded.')


def select_oof_inference_settings(oof_proposals, ground_truth, frame_counts, output_dir, folds=None):
    """Select settings from full videos decoded by models that excluded each video."""
    videos = sorted(oof_proposals)
    if not videos or set(videos) != set(frame_counts) or set(videos) != set(ground_truth):
        raise ValueError('ActionFormer OOF tuning requires proposals and labels for every selected video')
    decoded = []
    for video in videos:
        proposals = suppress_cross_window_duplicates(oof_proposals[video])
        decoded.append({'length': frame_counts[video], 'proposals': proposals,
                        'truth': ground_truth[video]})
    return _select_settings(
        decoded, output_dir,
        'Video-level out-of-fold full-video native proposals; each fold excludes its validation video '
        'from training and normalization.', folds=folds)


def proposal_summary(raw, deduplicated, retained):
    scores = np.asarray([score for _, score in raw], dtype=float)
    durations = np.asarray([end - start for (start, end), _ in raw], dtype=float) / 30.0
    return {
        'raw_proposals': len(raw), 'post_nms_proposals': len(deduplicated),
        'retained_intervals': len(retained),
        'confidence_max': float(scores.max()) if len(scores) else None,
        'confidence_p50': float(np.median(scores)) if len(scores) else None,
        'confidence_p90': float(np.quantile(scores, 0.9)) if len(scores) else None,
        'duration_seconds_max': float(durations.max()) if len(durations) else None,
        'duration_seconds_p50': float(np.median(durations)) if len(durations) else None,
    }
