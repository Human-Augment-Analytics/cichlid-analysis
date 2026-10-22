"""Train the frame-level circling logistic regression.

Uses pose and pair-Parquet features, balanced sampling and a stratified
train/test split. Run from the repository root:

    uv run python -m models.logistic_regression.train
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.evaluate import read_ground_truth
from evaluation.metrics import merge

from .features import (FEATURE_SOURCE, PAIR_ANGLES, PAIR_DISTANCES, PAIR_RATES, WINDOW,
                       frame_features)

ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = 0.5
# Frame counts of the source videos, so the 31 GB files are not needed.
VIDEO_FRAMES = {'0028_vid': 1_080_175, '0031_vid': 1_080_398}


def circling_spans_by_video():
    """Merged inclusive ground-truth spans per video, from `evaluation.evaluate.GROUND_TRUTH`."""
    gt = read_ground_truth()
    return {video: merge(gt.filter(pl.col('video_id') == video).select('start', 'end').iter_rows())
            for video in VIDEO_FRAMES}


def frame_labels(frames, spans):
    """Inclusive annotated spans are positive; other frames are assumed negative."""
    labels = np.zeros(len(frames), dtype=np.int8)
    for start, end in spans:
        if start < 0 or end < start:
            raise ValueError('Invalid circling interval')
        labels[(frames >= start) & (frames <= end)] = 1
    return labels


def new_model():
    # Fit every preprocessing step on training frames only.
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                         StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs'))


def fit_model(x, y):
    if len(np.unique(y)) != 2:
        raise ValueError('Training data must contain both classes')
    model = new_model()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y)
    return model


def metrics(y, score):
    predicted = score >= THRESHOLD
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(frames=len(y), positive_frames=int(y.sum()), prevalence=float(y.mean()),
                average_precision=float(average_precision_score(y, score)),
                roc_auc=float(roc_auc_score(y, score)),
                precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)),
                f1=float(f1_score(y, predicted, zero_division=0)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def train(data_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    sampled = []
    source_counts = {}
    spans_by_video = circling_spans_by_video()
    span_count = sum(map(len, spans_by_video.values()))
    for video, spans in spans_by_video.items():
        print(f'Building features: {video}', flush=True)
        count = VIDEO_FRAMES[video]
        if any(end >= count for _, end in spans):
            raise ValueError(f'Annotation exceeds video length: {video}')
        labels = frame_labels(np.arange(count), spans)
        positive = np.flatnonzero(labels)
        negative = rng.choice(np.flatnonzero(labels == 0), size=len(positive), replace=False)
        selected = np.sort(np.concatenate([positive, negative]))
        # Aggregate all pairs within each frame before selecting training examples.
        table = frame_features(
            pl.read_parquet(data_dir / f'{video}.parquet'),
            pl.read_parquet(data_dir / f'{video}_pairs.parquet'), 0, count - 1)
        table = table.filter(pl.col('FrameNum').is_in(selected)).with_columns(
            pl.lit(video).alias('video'), pl.Series('circling', labels[selected]))
        sampled.append(table)
        source_counts[video] = {'total_frames': count, 'positive_frames': len(positive),
                                'sampled_negative_frames': len(negative)}
    table = pl.concat(sampled)
    features = [c for c in table.columns if c not in ('FrameNum', 'video', 'circling')]
    x, y = table.select(features).to_numpy(), table['circling'].to_numpy()
    train_ids, test_ids = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    print(f'Fitting {len(train_ids):,} train frames; evaluating {len(test_ids):,} test frames', flush=True)
    model = fit_model(x[train_ids], y[train_ids])
    scores = model.predict_proba(x[test_ids])[:, 1]
    result = metrics(y[test_ids], scores)
    report = {
        'label_assumption': f'All frames outside the {span_count} annotated inclusive spans are non-circling.',
        'sampling': 'All positives plus an equal number of negatives sampled without replacement within each source video, seed 42.',
        'split': 'Stratified random 80/20 train/test split after sampling, random_state=42.',
        'limitation': 'Neighboring frames and spans can occur in both splits; this is not an independent-event or new-video evaluation.',
        'score_warning': 'Scores describe the balanced sampled dataset, not calibrated probabilities at full-video prevalence.',
        'threshold': THRESHOLD, 'window_frames': WINDOW, 'features': features,
        'feature_source': FEATURE_SOURCE,
        'raw_columns': [*PAIR_DISTANCES, *PAIR_ANGLES, *PAIR_RATES],
        'source_counts': source_counts,
        'train_frames': len(train_ids), 'train_positive_frames': int(y[train_ids].sum()),
        'test': result,
        'all_negative_baseline': {'accuracy': float((y[test_ids] == 0).mean()), 'recall': 0.0, 'f1': 0.0},
        'constant_score_average_precision': float(y[test_ids].mean()),
    }
    joblib.dump({'model': model, 'features': features, 'threshold': THRESHOLD,
                 'training_split': 'train only; test frames excluded', 'seed': 42,
                 'feature_source': FEATURE_SOURCE,
                 'label_assumption': report['label_assumption']}, output_dir / 'model.joblib')
    names = model[0].get_feature_names_out(features)
    pl.DataFrame({'feature': names, 'standardized_coefficient': model[-1].coef_[0]}).sort(
        'standardized_coefficient', descending=True).write_csv(output_dir / 'coefficients.csv')
    (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    fig = Figure(figsize=(8, 6), layout='constrained')
    axis = fig.subplots()
    precision, recall, _ = precision_recall_curve(y[test_ids], scores)
    axis.plot(recall, precision, label=f'Logistic regression: AP={result["average_precision"]:.3f}')
    axis.axhline(y[test_ids].mean(), linestyle=':', label='Constant-score baseline')
    axis.set(xlabel='Recall', ylabel='Precision', title='Circling baseline: sampled-frame test set', xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    fig.savefig(output_dir / 'precision_recall.png', dpi=160)
    lines = ['# Circling logistic regression baseline', '', report['label_assumption'], '',
             report['sampling'], '', report['split'], '',
             f'Train: {len(train_ids):,} frames. Test: {len(test_ids):,} frames.', '',
             '102 features: the original 50 pose features (motion, bend, turn, track count and trailing 30-frame means), plus 52 pair features (mean/min/max of six distances, sine/cosine of four angles, and three angular rates, plus pair count). Identifiers, time, sex, QC flags and episode metadata are excluded.', '',
             'Pose features retain the original causal 30-frame history. Pair features aggregate current-frame measurements; angular rates are used as supplied in the pair Parquet file. Median imputation with missingness indicators and standard scaling are fitted on training frames only. L2 logistic regression uses C=1, no class weighting, and a fixed threshold of 0.5. No tuning uses test labels.', '',
             '| Test metric | Value |', '|---|---:|']
    for name in ('average_precision', 'roc_auc', 'precision', 'recall', 'f1'):
        lines.append(f'| {name} | {result[name]:.4f} |')
    lines += ['', f'Confusion matrix: TN={result["tn"]}, FP={result["fp"]}, FN={result["fn"]}, TP={result["tp"]}.', '',
              report['limitation'], '', report['score_warning'], '',
              f'Only two videos and {span_count} spans are available. Frames with no pair rows remain in the dataset with pair_count=0 and missing measurements. Missing measurements are imputed; tracking and measurement errors can affect predictions.', '',
              '`model.joblib` is the exact evaluated model, trained only on the training split.', '',
              'Implementation reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html', '']
    (output_dir / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'logistic_regression')
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)
