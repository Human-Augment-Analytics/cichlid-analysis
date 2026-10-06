"""Frame-level logistic regression using pose and pair-Parquet features, balanced sampling and a stratified train/test split."""

import argparse
import json
import warnings
from pathlib import Path

import cv2
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

from circling_clips import CIRCLING_SPANS_BY_VIDEO
from pose_features import pose_frame_features

ROOT = Path(__file__).resolve().parent
WINDOW = 30
THRESHOLD = 0.5
PAIR_DISTANCES = (
    'centroid_distance_px', 'dist_nose1_nose2_px', 'dist_nose1_tailtip2_px',
    'dist_nose2_tailtip1_px', 'dist_nose1_spine4of2_px', 'dist_nose2_spine4of1_px',
)
PAIR_ANGLES = ('orbital_rad', 'rel_heading_rad', 'bearing_1_rad', 'bearing_2_rad')
PAIR_RATES = ('d_heading_1_rad_s', 'd_heading_2_rad_s', 'd_orbital_rad_s')
FEATURE_SOURCE = 'pose_and_pairs_v1'


def frame_labels(frames, spans):
    """Inclusive annotated spans are positive; other frames are assumed negative."""
    labels = np.zeros(len(frames), dtype=np.int8)
    for start, end in spans:
        if start < 0 or end < start:
            raise ValueError('Invalid circling interval')
        labels[(frames >= start) & (frames <= end)] = 1
    return labels


def pair_frame_features(pairs, start, end):
    """Aggregate supplied pair measurements into one row per video frame.

    Six distances, four angles encoded as sine/cosine, and three supplied
    angular rates give 17 signals. Mean/min/max across pairs plus pair count
    give 52 predictors. No identifiers, sex, quality flags, or time predictors.
    No additional temporal windows or pose-derived features are used.
    """
    if start < 0 or end < start:
        raise ValueError('Invalid frame range')
    if pairs.select('project_id', 'day_label').unique().height > 1:
        raise ValueError('Provide pairs for exactly one video')
    keys = ['FrameNum', 'TrackID_1', 'TrackID_2']
    if pairs.select(pl.any_horizontal(pl.col(k).is_null() for k in keys).any()).item():
        raise ValueError('Pair frame and track identifiers cannot be missing')
    if pairs.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError('Duplicate pair within a frame')
    signals = [*PAIR_DISTANCES, *PAIR_RATES]
    rows = pairs.select('FrameNum', *PAIR_DISTANCES, *PAIR_ANGLES, *PAIR_RATES).with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c).cast(pl.Float64)).otherwise(None).alias(c)
        for c in (*PAIR_DISTANCES, *PAIR_ANGLES, *PAIR_RATES)
    ])
    rows = rows.with_columns([
        expr for c in PAIR_ANGLES for expr in
        (pl.col(c).sin().alias(f'{c}_sin'), pl.col(c).cos().alias(f'{c}_cos'))
    ])
    signals += [f'{c}_{part}' for c in PAIR_ANGLES for part in ('sin', 'cos')]
    aggregate = rows.group_by('FrameNum').agg(
        pl.len().cast(pl.Float32).alias('pair_count'),
        *[expr for c in signals for expr in (
            pl.col(c).mean().alias(f'{c}_mean'), pl.col(c).min().alias(f'{c}_min'),
            pl.col(c).max().alias(f'{c}_max'))])
    frames = pl.DataFrame({'FrameNum': np.arange(start, end + 1, dtype=np.int64)})
    return (frames.join(aggregate, on='FrameNum', how='left').sort('FrameNum')
            .with_columns(pl.col('pair_count').fill_null(0), pl.exclude('FrameNum', 'pair_count').cast(pl.Float32)))


def frame_features(poses, pairs, start, end):
    """Combine the original 50 pose features with 52 pair features by frame."""
    videos = pl.concat([poses.select('project_id', 'day_label'),
                        pairs.select('project_id', 'day_label')]).unique()
    if videos.height > 1:
        raise ValueError('Pose and pair rows must belong to the same video')
    return pose_frame_features(poses, start, end).join(
        pair_frame_features(pairs, start, end), on='FrameNum', how='left', validate='1:1')


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
    for video, spans in CIRCLING_SPANS_BY_VIDEO.items():
        print(f'Building features: {video}', flush=True)
        capture = cv2.VideoCapture(str(data_dir / video))
        try:
            if not capture.isOpened():
                raise ValueError(f'Cannot open {video}')
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if any(end >= count for _, end in spans):
            raise ValueError(f'Annotation exceeds video length: {video}')
        labels = frame_labels(np.arange(count), spans)
        positive = np.flatnonzero(labels)
        negative = rng.choice(np.flatnonzero(labels == 0), size=len(positive), replace=False)
        selected = np.sort(np.concatenate([positive, negative]))
        # Aggregate all pairs within each frame before selecting training examples.
        table = frame_features(
            pl.read_parquet((data_dir / video).with_suffix('.parquet')),
            pl.read_parquet(data_dir / f'{Path(video).stem}_pairs.parquet'), 0, count - 1)
        table = table.filter(pl.col('FrameNum').is_in(selected)).with_columns(
            pl.lit(video).alias('video'), pl.Series('circling', labels[selected]))
        sampled.append(table)
        source_counts[video] = {'total_frames': count, 'positive_frames': len(positive),
                                'sampled_negative_frames': len(negative)}
    table = pl.concat(sampled)
    features = [c for c in table.columns if c not in ('FrameNum', 'video', 'circling')]
    x, y = table.select(features).to_numpy(), table['circling'].to_numpy()
    train_ids, test_ids = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    split = np.full(len(y), 'train', dtype='<U5')
    split[test_ids] = 'test'
    table = table.with_columns(pl.Series('split', split))
    table.write_parquet(output_dir / 'sampled_frames.parquet')
    print(f'Fitting {len(train_ids):,} train frames; evaluating {len(test_ids):,} test frames', flush=True)
    model = fit_model(x[train_ids], y[train_ids])
    scores = model.predict_proba(x[test_ids])[:, 1]
    result = metrics(y[test_ids], scores)
    report = {
        'label_assumption': 'All frames outside the 15 annotated inclusive spans are non-circling.',
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
    metadata = table[test_ids].select('video', 'FrameNum', 'circling')
    metadata.with_columns(pl.Series('circling_score', scores),
                          pl.Series('predicted_circling', scores >= THRESHOLD)).write_parquet(output_dir / 'test_predictions.parquet')
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
              'Only two videos and 15 spans are available. Frames with no pair rows remain in the dataset with pair_count=0 and missing measurements. Missing measurements are imputed; tracking and measurement errors can affect predictions.', '',
              '`sampled_frames.parquet` records every selected frame, its label, features and split. `test_predictions.parquet` contains test-only predictions. `model.joblib` is the exact evaluated model, trained only on the training split.', '',
              'Implementation reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html', '']
    (output_dir / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2), flush=True)


def predict_frame(model_path, poses_path, frame, pairs_path=None):
    if frame < 0:
        raise ValueError('Frame must be nonnegative')
    bundle = joblib.load(model_path)
    if pairs_path is None:
        pairs_path = Path(poses_path).with_name(f'{Path(poses_path).stem}_pairs.parquet')
    poses = pl.scan_parquet(poses_path).filter(pl.col('FrameNum').is_between(max(0, frame - WINDOW), frame)).collect()
    pairs = pl.scan_parquet(pairs_path).filter(pl.col('FrameNum').is_between(frame, frame)).collect()
    table = frame_features(poses, pairs, frame, frame)
    score = float(bundle['model'].predict_proba(table.select(bundle['features']).to_numpy())[0, 1])
    return {'frame': frame, 'circling_score': score, 'predicted_circling': score >= bundle['threshold'],
            'threshold': bundle['threshold'], 'detected_pairs': int(table['pair_count'][0]),
            'detected_tracks': int(table['track_count'][0]),
            'score_note': 'Score for the balanced sampled population; not a full-video probability.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    training.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'circling_baseline')
    prediction = commands.add_parser('predict')
    prediction.add_argument('--model', type=Path, default=ROOT / 'outputs' / 'circling_baseline' / 'model.joblib')
    prediction.add_argument('--poses', type=Path, required=True)
    prediction.add_argument('--pairs', type=Path, help='Defaults to the matching *_pairs.parquet')
    prediction.add_argument('--frame', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'train':
        train(args.data_dir, args.output_dir)
    else:
        print(json.dumps(predict_frame(args.model, args.poses, args.frame, args.pairs), indent=2))
