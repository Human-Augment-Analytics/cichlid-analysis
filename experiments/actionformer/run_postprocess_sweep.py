"""Inference-only postprocessing diagnostic on seven 1024-frame CV videos.

Uses frozen *raw* proposals from run_proposal_cap.py, and never retrains,
decodes a model, or evaluates final holdout 0037_vid. Parameters are compared
on the seven development recordings; the global best is exploratory, not an
unbiased generalization estimate.

From the repository root:
  uv run python -m experiments.actionformer.run_study postprocess --max-folds 1
  uv run python -m experiments.actionformer.run_study postprocess
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _rows(path: Path) -> list[dict]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def _save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    tmp = path.with_suffix('.tmp.csv')
    with tmp.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def _as_interval_settings(row):
    return {'min_score': float(row['post_min_score']),
            'min_seconds': float(row['post_min_seconds']),
            'merge_seconds': float(row['post_merge_seconds'])}


def _eval(proposals, settings, video, count, gt, predict, evaluate):
    positive = [p for p in proposals if p[1] >= settings['min_score']]
    intervals = predict(video, positive, count, min_seconds=settings['min_seconds'],
                        merge_seconds=settings['merge_seconds'])
    spans = [(int(start), int(end)) for start, end in intervals.select('StartFrame', 'EndFrame').iter_rows()]
    return evaluate(gt, spans, count)


def _result_row(video: str, fold: int, settings: dict, result: dict) -> dict:
    return {'fold': fold, 'validation_video': video,
            'min_score': settings['min_score'],
            'min_seconds': settings['min_seconds'],
            'merge_seconds': settings['merge_seconds'],
            'f1@50': result['f1@50'], 'frame_precision': result['frame_precision'],
            'frame_recall': result['frame_recall'], 'frame_f1': result['frame_f1'],
            'gt_segments': result['gt_segments'], 'pred_segments': result['pred_segments'],
            'tp@50': result['tp@50']}


def _summarize(rows, videos):
    grouped = {}
    for row in rows:
        key = (float(row['min_score']), float(row['min_seconds']), float(row['merge_seconds']))
        grouped.setdefault(key, []).append(row)
    result = []
    for (score, duration, merge), group in grouped.items():
        if len({x['validation_video'] for x in group}) != len(group):
            raise ValueError('Duplicate video in postprocessing results')
        if len(group) != videos:
            continue
        record = {'min_score': score, 'min_seconds': duration, 'merge_seconds': merge,
                  'folds': len(group)}
        for metric in ('f1@50', 'frame_precision', 'frame_recall', 'frame_f1'):
            record['mean_' + metric] = statistics.mean(float(r[metric]) for r in group)
        for metric in ('gt_segments', 'pred_segments', 'tp@50'):
            record['total_' + metric] = sum(int(r[metric]) for r in group)
        result.append(record)
    return sorted(result, key=lambda r: (-r['mean_f1@50'], -r['mean_frame_precision'],
                                          r['total_pred_segments'], r['min_score'],
                                          r['min_seconds'], r['merge_seconds']))


def _crossfit(rows, videos):
    """Select settings using *other* videos' OOF predictions, not target labels.

    This is a diagnostic, not a fully nested model-training evaluation: the OOF
    models fitted for other videos may have trained on the target video's labels.
    """
    by_cfg = {}
    for row in rows:
        key = (float(row['min_score']), float(row['min_seconds']), float(row['merge_seconds']))
        by_cfg.setdefault(key, {})[row['validation_video']] = row
    if any(set(records) != set(videos) for records in by_cfg.values()):
        raise ValueError('Crossfit requires complete grid for all seven videos')
    chosen = []
    for video in videos:
        candidates = []
        for key, records in by_cfg.items():
            other = [records[v] for v in videos if v != video]
            # primary objective event F1@50; tie-break on frame precision.
            metric = statistics.mean(float(r['f1@50']) for r in other)
            prec = statistics.mean(float(r['frame_precision']) for r in other)
            candidates.append((metric, prec, key))
        _, _, key = max(candidates)
        chosen.append(dict(by_cfg[key][video], tuning_videos=len(videos) - 1))
    return chosen


def _format_mean(rows, metric):
    return statistics.mean(float(row[metric]) for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cv-dir', type=Path, default=Path('outputs/actionformer_cv_window1024'))
    parser.add_argument('--cap-dir', type=Path, default=Path('outputs/actionformer_cap_sweep_1024'))
    parser.add_argument('--output', type=Path, default=Path('outputs/actionformer_postprocess_1024'))
    parser.add_argument('--scores', nargs='+', type=float, default=[0.05, 0.1, 0.2, 0.35, 0.5])
    parser.add_argument('--durations', nargs='+', type=float, default=[0, 2, 4])
    parser.add_argument('--merges', nargs='+', type=float, default=[0, 1, 2, 4])
    parser.add_argument('--max-folds', type=int, default=None,
                        help='Maximum number of new video folds to process before exiting')
    args = parser.parse_args()
    if args.max_folds is not None and args.max_folds < 1:
        parser.error('--max-folds must be positive')
    if not args.scores or not args.durations or not args.merges:
        parser.error('All grids must contain values')
    if any(not math.isfinite(x) or x <= 0 or x > 1 for x in args.scores):
        parser.error('Scores must be finite in (0,1]')
    if any(not math.isfinite(x) or x < 0 for x in [*args.durations, *args.merges]):
        parser.error('Durations and merge gaps must be finite and nonnegative')
    cv, cap, out = (p.resolve() for p in (args.cv_dir, args.cap_dir, args.output))
    if (out == cv or out == cap or out in cv.parents or out in cap.parents
            or cv in out.parents or cap in out.parents):
        parser.error('--output must be separate from source experiment directories')
    cv_manifest = cv / 'manifest.json'
    cap_manifest = cap / 'manifest.json'
    gt_file = cv / 'normalized_annotations.csv'
    for p in (cv_manifest, cap_manifest, gt_file):
        if not p.is_file():
            parser.error(f'Missing experiment source file: {p}')
    cv_cfg = json.loads(cv_manifest.read_text())
    cap_cfg = json.loads(cap_manifest.read_text())
    if int(cv_cfg.get('window', -1)) != 1024 or int(cv_cfg.get('max_proposals', -1)) != 1000:
        parser.error('Require a complete 1024-frame source CV with cap=1000')
    if _sha256(gt_file) != cv_cfg['GT_SHA256']:
        parser.error('Source ground truth changed; refusing to proceed')
    if cap_cfg.get('source_cv_manifest_sha256') != _sha256(cv_manifest):
        parser.error('Proposal-cap run does not belong to this CV source')
    if 1000 not in cap_cfg['caps']:
        parser.error('Proposal-cap sweep must have a cap=1000 baseline')
    videos = [v for v in cv_cfg['videos'] if v != cv_cfg['holdout']]
    if len(videos) != 7 or cv_cfg['holdout'] in videos:
        parser.error('Require seven development videos excluding final holdout')
    source_rows = {r['validation_video']: r for r in _rows(cv / 'cv_results.csv')}
    if set(source_rows) != set(videos):
        parser.error('Source CV results incomplete or mismatched')
    configs = list(itertools.product(sorted(set(args.scores)), sorted(set(args.durations)),
                                     sorted(set(args.merges))))
    manifest_data = {'source_cv_manifest_sha256': _sha256(cv_manifest),
                     'cap_manifest_sha256': _sha256(cap_manifest),
                     'holdout': cv_cfg['holdout'], 'videos': videos, 'cap': 1000,
                     'scores': sorted(set(args.scores)),
                     'durations': sorted(set(args.durations)),
                     'merges': sorted(set(args.merges)),
                     'purpose': 'exploratory postprocessing sweep over frozen 1024-frame OOF proposals'}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest_data:
        parser.error('Output manifest incompatible: use a new --output folder')
    out.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest_data, indent=2) + '\n')
    from experiments.actionformer.run_proposal_cap import load_normalized_annotations, read_raw, save_raw
    import polars as pl
    from evaluation.metrics import evaluate_video
    from models.actionformer.model import suppress_cross_window_duplicates
    from models.actionformer.postprocess import prediction_table
    ground_truth = load_normalized_annotations(gt_file, videos)
    print(f'Frozen 1024-frame checkpoints; using cached raw proposals; cap=1000; '
          f'{len(configs)} postprocessing combinations; NO training; NO {cv_cfg["holdout"]} test.')
    all_rows = []
    baseline_rows = []
    processed = 0
    for fold, video in enumerate(videos, 1):
        done = out / 'fold_results' / f'fold_{fold:02d}_{video}.json'
        previous = source_rows[video]
        baseline = {'fold': fold, 'validation_video': video,
                    **_as_interval_settings(previous),
                    **{k: previous[k] for k in ('f1@50', 'frame_precision', 'frame_recall',
                                                'frame_f1', 'gt_segments', 'pred_segments', 'tp@50')}}
        baseline_rows.append(baseline)
        if done.exists():
            saved = json.loads(done.read_text())
            if len(saved) != len(configs):
                raise ValueError(f'Incomplete cached fold results: {done}')
            all_rows.extend(saved)
            print(f'SKIP {video}: cached postprocessing results', flush=True)
            continue
        nms_file = out / 'nms_proposals' / f'fold_{fold:02d}_{video}.parquet'
        if nms_file.exists():
            proposals = read_raw(nms_file, pl)
        else:
            raw_file = cap / 'raw_proposals' / f'fold_{fold:02d}_{video}.parquet'
            if not raw_file.is_file():
                raise FileNotFoundError(f'Missing cached raw proposals: {raw_file}')
            raw = read_raw(raw_file, pl)
            print(f'{video}: hard NMS of {len(raw):,} cached proposals to cap 1,000', flush=True)
            proposals = suppress_cross_window_duplicates(raw, max_segments=1000)
            save_raw(nms_file, proposals, pl)
            del raw
        count = int(previous['num_frames'])
        truth = ground_truth[video]
        base_metrics = _eval(proposals, _as_interval_settings(previous), video,
                             count, truth, prediction_table, evaluate_video)
        for key in ('f1@50', 'frame_precision', 'frame_recall', 'frame_f1',
                    'gt_segments', 'pred_segments', 'tp@50'):
            if not math.isclose(float(base_metrics[key]), float(previous[key]),
                                rel_tol=1e-5, abs_tol=1e-5):
                raise RuntimeError(f'{video}: baseline mismatch for {key}: '
                                   f'{base_metrics[key]} vs {previous[key]}. '
                                   'Do not interpret new results until reconciled.')
        current = []
        for score, duration, merge in configs:
            settings = {'min_score': score, 'min_seconds': duration, 'merge_seconds': merge}
            measures = _eval(proposals, settings, video, count, truth,
                             prediction_table, evaluate_video)
            current.append(_result_row(video, fold, settings, measures))
        done.parent.mkdir(parents=True, exist_ok=True)
        tmp = done.with_suffix('.tmp.json')
        tmp.write_text(json.dumps(current, indent=2) + '\n')
        tmp.replace(done)
        all_rows.extend(current)
        processed += 1
        print(f'{video}: baseline F1@50={float(previous["f1@50"]):.4f}, '
              f'grid={len(configs)} configurations complete', flush=True)
        if args.max_folds is not None and processed >= args.max_folds:
            print('Stopped after requested new folds; rerun without --max-folds to finish.')
            break
    _save_csv(out / 'results.csv', all_rows)
    _save_csv(out / 'baseline.csv', baseline_rows)
    full_results = _summarize(all_rows, len(videos))
    _save_csv(out / 'grid_summary.csv', full_results)
    if len({r['validation_video'] for r in all_rows}) != len(videos):
        print(f'Partial study: {len({r["validation_video"] for r in all_rows})}/7 videos. '
              'No cross-video selection until all seven complete.', flush=True)
        return
    crossfit = _crossfit(all_rows, videos)
    _save_csv(out / 'crossfit.csv', crossfit)
    baseline_f1 = _format_mean(baseline_rows, 'f1@50')
    cross_f1 = _format_mean(crossfit, 'f1@50')
    baseline_precision = _format_mean(baseline_rows, 'frame_precision')
    cross_precision = _format_mean(crossfit, 'frame_precision')
    best = full_results[0]
    report = [
        '# ActionFormer 1024-frame postprocessing study', '',
        'Seven development videos only. Frozen models, cap=1000, no retraining.',
        'The full-grid best is **exploratory** because the same validation videos',
        'were used to evaluate and select it. Do not report this as unbiased test performance.', '',
        '## Original fold-specific validation settings', '',
        f'- Mean event F1@50: {baseline_f1:.4f}',
        f'- Mean frame precision: {baseline_precision:.4f}',
        f'- Predicted events (total): {sum(int(r["pred_segments"]) for r in baseline_rows):,}', '',
        '## Leave-one-video-out postprocessing calibration (diagnostic)', '',
        'For each target video, postprocessing settings were selected using only',
        'the other six videos\' out-of-fold predictions and labels.',
        'The target video\'s annotations were excluded from that selection.',
        '**Caveat:** Other folds\' models may have trained on the target video;',
        'therefore this is not a fully nested end-to-end evaluation.', '',
        f'- Mean event F1@50: {cross_f1:.4f}',
        f'- Mean frame precision: {cross_precision:.4f}',
        f'- Predicted events (total): {sum(int(r["pred_segments"]) for r in crossfit):,}', '',
        '## Best fixed setting over all 7 videos (exploratory, optimistic)', '',
        f'- min_score={best["min_score"]}, min_seconds={best["min_seconds"]}, '
        f'merge_seconds={best["merge_seconds"]}',
        f'- Mean event F1@50: {best["mean_f1@50"]:.4f}',
        f'- Mean frame precision: {best["mean_frame_precision"]:.4f}',
        f'- Predicted events (total): {best["total_pred_segments"]:,}', '',
        'See `results.csv`, `grid_summary.csv`, and `crossfit.csv` for details.',
    ]
    (out / 'summary.md').write_text('\n'.join(report) + '\n')
    print('\n'.join(report), flush=True)
    print(f'\nDONE: {len(configs)} settings x 7 videos. Saved {out / "summary.md"}', flush=True)


if __name__ == '__main__':
    main()
