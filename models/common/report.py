import json
from pathlib import Path

import numpy as np
import polars as pl

from models.common.config import LABELS, label_description
from models.logistic_regression.train import THRESHOLD, metrics

def write_report(output_dir, title, label, test, report, notes=(), unit='frames'):
    from matplotlib.figure import Figure
    from sklearn.metrics import precision_recall_curve

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y = test['circling'].to_numpy().astype(np.int8)
    score = test['circling_score'].to_numpy()
    result = metrics(y, score)
    report = {**report, 'threshold': THRESHOLD, 'test_unit': unit, 'test': result,
              'all_negative_baseline': {'accuracy': float((y == 0).mean()), 'recall': 0.0, 'f1': 0.0},
              'constant_score_average_precision': float(y.mean())}
    (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    test.with_columns(pl.Series('predicted_circling', score >= THRESHOLD)).write_parquet(
        output_dir / 'test_predictions.parquet')

    fig = Figure(figsize=(8, 6), layout='constrained')
    axis = fig.subplots()
    precision, recall, _ = precision_recall_curve(y, score)
    axis.plot(recall, precision, label=f'{label}: AP={result["average_precision"]:.3f}')
    axis.axhline(y.mean(), linestyle=':', label='Constant-score baseline')
    axis.set(xlabel='Recall', ylabel='Precision', title=f'{title}: sampled test {unit}',
             xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    fig.savefig(output_dir / 'precision_recall.png', dpi=160)

    lines = [f'# {title}', '']
    for note in notes:
        lines += [note, '']
    lines += [f'Test: {len(y):,} {unit} ({int(y.sum()):,} circling).', '',
              '| Test metric | Value |', '|---|---:|']
    for key in ('average_precision', 'roc_auc', 'precision', 'recall', 'f1'):
        lines.append(f'| {key} | {result[key]:.4f} |')
    lines += ['', f'Confusion matrix: TN={result["tn"]}, FP={result["fp"]}, FN={result["fn"]}, '
              f'TP={result["tp"]}.', '']
    for key in ('limitation', 'score_warning'):
        if key in report:
            lines += [report[key], '']
    (output_dir / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2), flush=True)
    return result

def protocol(annotations, model, **extra):
    return {
        'label_assumption': f'All frames outside the intervals in {label_description(annotations)} are non-circling.',
        'labels': LABELS,
        'annotations': str(Path(annotations).resolve()),
        'sampling': 'Every 128-row segment (stride 2, 256 frames) containing circling plus an equal number '
                    'of segments without circling, sampled without replacement within each source video, seed 42.',
        'split': 'Stratified random 80/20 train/test split of the sampled segments after sampling, '
                 'random_state=42. Segments do not overlap.',
        'limitation': 'Neighboring segments and spans can occur in both splits; this is not an '
                      'independent-event or new-video evaluation.',
        'score_warning': 'Scores describe the balanced sampled dataset, not calibrated probabilities '
                         'at full-video prevalence.',
        'model': model, **extra,
    }

def segment_counts(segments):
    return {s: {'with_circling': int(g['circling'].sum()), 'without': int((~g['circling']).sum())}
            for s, g in ((s, segments.filter(pl.col('split') == s)) for s in ('train', 'test'))}
