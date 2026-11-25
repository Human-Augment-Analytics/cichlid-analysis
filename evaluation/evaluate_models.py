import argparse
import json
from pathlib import Path

import polars as pl

from evaluation.evaluate import GROUND_TRUTH, evaluate, prediction_models, read_intervals, read_predictions
from evaluation.families import (family_evaluations, family_metric_table, family_summary_table,
                                 selected_predictions)
from evaluation.windows import window_metrics
from models.common.config import BASELINE, BASELINE_SUFFIX, LABELS, OUTPUTS, VIDEOS, video_frames

METRICS_DIRS = {'logistic_regression': 'circling_baseline', 'rule_based': None}
BASELINE_MODELS = ('logistic_regression', 'rule_based')
SUMMARY_COLUMNS = {
    'pred_segments': 'Pred segments', 'gt_segments': 'GT segments', 'mof': 'MoF', 'edit_score': 'Edit score',
    'frame_precision': 'Frame P', 'frame_recall': 'Frame R', 'frame_f1': 'Frame F1',
    'f1@10': 'F1@10', 'f1@25': 'F1@25', 'f1@50': 'F1@50', 'tp@10': 'Matched@10',
}


def comparable_models(models=None):
    models = prediction_models() if models is None else models
    if BASELINE:
        return {k: v for k, v in models.items() if k in BASELINE_MODELS or BASELINE_SUFFIX in k}
    return {k: v for k, v in models.items() if BASELINE_SUFFIX not in k}


def model_videos(folder):
    return sorted(p.stem for p in Path(folder).glob('*.csv'))


def evaluate_models(gt=None, models=None, videos=None):
    gt = read_intervals(GROUND_TRUTH) if gt is None else gt
    videos = sorted(videos or VIDEOS)
    frames = video_frames(videos)
    evaluations, predictions = {}, {}
    for name, folder in (comparable_models() if models is None else models).items():
        covered = [v for v in model_videos(folder) if v in frames]
        if not covered:
            continue
        truth = gt.filter(pl.col('video_id').is_in(covered))
        predicted = read_predictions(folder).filter(pl.col('video_id').is_in(covered))
        results, warnings = evaluate(truth, predicted, num_frames={v: frames[v] for v in covered})
        missing = sorted(set(videos) - set(covered))
        if missing:
            warnings = [*warnings, f'No predictions for {missing}: evaluated on {covered} only']
        evaluations[name], predictions[name] = (results, warnings), predicted
    return evaluations, predictions


def evaluated_videos(results):
    return [v for v in results['video_id'] if v != 'overall']


def summary_table(evaluations, video='overall'):
    rows = []
    for name, (results, _) in evaluations.items():
        row = results.filter(pl.col('video_id') == video)
        if row.height:
            row = row.row(0, named=True)
            rows.append({'Model': name, 'Videos': len(evaluated_videos(results)),
                         **{label: row[key] for key, label in SUMMARY_COLUMNS.items()}})
    return pl.DataFrame(rows)


def per_video_table(evaluations):
    return pl.concat([results.with_columns(pl.lit(name).alias('model')).select('model', pl.exclude('model'))
                      for name, (results, _) in evaluations.items()], how='diagonal_relaxed')


def metrics_path(name):
    folder = METRICS_DIRS.get(name, name)
    return OUTPUTS / folder / 'metrics.json' if folder else None


def heldout_metrics(names):
    rows = []
    for name in names:
        path = metrics_path(name)
        if path is None or not path.exists():
            continue
        report = json.loads(path.read_text())
        test = report.get('test')
        if not test:
            continue
        rows.append({'model': name, 'test_unit': report.get('test_unit', 'frames'),
                     'test_items': test['frames'], 'circling_items': test['positive_frames'],
                     'prevalence': test['prevalence'], 'average_precision': test['average_precision'],
                     'roc_auc': test['roc_auc'], 'precision': test['precision'], 'recall': test['recall'],
                     'f1': test['f1']})
    return pl.DataFrame(rows)


def window_report(gt, evaluations, predictions):
    frames = video_frames()
    tables = []
    for name, (results, _) in evaluations.items():
        covered = evaluated_videos(results)
        table = window_metrics(gt.filter(pl.col('video_id').is_in(covered)), predictions[name],
                               {v: frames[v] for v in covered})
        tables.append(table.with_columns(pl.lit(name).alias('model')).select('model', pl.exclude('model')))
    return pl.concat(tables) if tables else pl.DataFrame()


def write_metric_tables(gt, raw_evaluations, raw_predictions, figures=None):
    """Write family-first report CSVs and separate raw variant audit CSVs."""
    figures = Path(figures or OUTPUTS / 'figures')
    details = figures / 'details'
    details.mkdir(parents=True, exist_ok=True)

    evaluations = family_evaluations(raw_evaluations)
    predictions = selected_predictions(raw_predictions)
    selected_raw = {name: value for name, value in raw_evaluations.items() if name in predictions}
    tables = {
        'overall': family_summary_table(evaluations),
        'per_video': per_video_table(evaluations) if evaluations else pl.DataFrame(),
        'heldout': family_metric_table(heldout_metrics(list(selected_raw))),
        'windows': family_metric_table(window_report(gt, selected_raw, predictions)),
    }
    for key, filename in (
        ('overall', 'family_metrics_overall.csv'),
        ('per_video', 'family_metrics_per_video.csv'),
        ('heldout', 'family_metrics_heldout.csv'),
        ('windows', 'family_metrics_60s_windows.csv'),
    ):
        tables[key].write_csv(figures / filename)

    raw_per_video = per_video_table(raw_evaluations) if raw_evaluations else pl.DataFrame()
    raw_windows = window_report(gt, raw_evaluations, raw_predictions)
    for table, filename in (
        (summary_table(raw_evaluations), 'model_metrics_overall.csv'),
        (raw_per_video, 'model_metrics_per_video.csv'),
        (heldout_metrics(list(raw_evaluations)), 'model_metrics_heldout.csv'),
        (raw_windows, 'model_metrics_60s_windows.csv'),
    ):
        table.write_csv(details / filename)
    for stale in figures.glob('model_metrics_*.csv'):
        stale.unlink()
    return tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Family-level circling metrics from outputs/predictions')
    parser.add_argument('models', nargs='*', help='model variants to include (default: all)')
    parser.add_argument('--details', action='store_true',
                        help='also print raw per-model overall and held-out tables')
    args = parser.parse_args()
    models = comparable_models()
    if args.models:
        models = prediction_models()
        models = {k: v for k, v in models.items() if k in args.models}
    gt = read_intervals(GROUND_TRUTH)
    raw_evaluations, raw_predictions = evaluate_models(gt, models)
    evaluations = family_evaluations(raw_evaluations)
    tables = write_metric_tables(gt, raw_evaluations, raw_predictions)
    overall, per_video = tables['overall'], tables['per_video']
    heldout, windows = tables['heldout'], tables['windows']
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=3):
        print(f'Ground truth: {GROUND_TRUTH}')
        print(f'Videos: {VIDEOS}')
        print(f'Labels: {LABELS} (baseline = logistic_regression, rule_based and the *{BASELINE_SUFFIX} models '
              f'trained on the 15 spans)')
        print('Family values are equal-weight means over available variants; they are not ensemble predictions.\n')
        for name, (_, warnings) in evaluations.items():
            for warning in warnings:
                print(f'[{name}] {warning}')
        print('\nFamily whole-video metrics (overall)')
        print(overall)
        print('\nFamily metrics per video')
        if per_video.height:
            print(per_video.select('model', 'video_id', 'variant_count',
                                   *[k for k in SUMMARY_COLUMNS if k != 'tp@10']))
        print('\nFamily held-out test metrics (metrics.json)')
        print(heldout)
        print('\nFamily 60 s windows every 30 s (overall)')
        if windows.height:
            print(windows.filter(pl.col('video_id') == 'overall').drop('video_id'))
        if args.details:
            print('\nDETAILS: raw whole-video variant metrics')
            print(summary_table(raw_evaluations))
            print('\nDETAILS: raw held-out variant metrics')
            print(heldout_metrics(list(raw_evaluations)))

    figures = OUTPUTS / 'figures'
    print(f'\nFamily metric tables saved to {figures}')
    print(f'Per-model audit tables saved to {figures / "details"}')
