"""Reproducible DLC2Action MS-TCN smoke test on reviewed merged circling spans.

Uses DLC2Action's public model API with a small PyTorch training loop, allowing
our existing frame features and partially observed labels to be used directly.
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
from dlc2action.model.ms_tcn import MS_TCN3
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from circling_model import frame_features, metrics
from temporal_metrics import temporal_metrics

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'outputs' / 'dlc2action_test'
IGNORE = -100
LABEL_POLICY = (
    'Verified=true rows are circling (1), with inclusive endpoints. '
    'Verified=false or blank rows are ignored (-100), not negative examples. '
    'Frames within the context buffer but outside every CSV span are assumed '
    'background (0); these frames have not been exhaustively annotated. '
    'Predictions are video-level, not assignments to individual fish.'
)


def read_spans(span_dir):
    paths = sorted(Path(span_dir).glob('*_merged_spans.csv'))
    if not paths:
        raise ValueError(f'No merged span CSVs in {span_dir}')
    rows = pl.concat([pl.read_csv(p, schema_overrides={'Verified': pl.Boolean}) for p in paths])
    if rows.select(pl.col('Merge ID').is_duplicated().any()).item():
        raise ValueError('Merge IDs must be unique')
    for r in rows.iter_rows(named=True):
        start, end = r['Start Frame'], r['End Frame']
        if start is None or end is None or not 0 <= start <= end:
            raise ValueError(f"Invalid inclusive frame range: {r['Merge ID']}")
    return rows.sort('Video ID', 'Start Frame')


def source_hash(span_dir):
    digest = hashlib.sha256()
    for path in sorted(Path(span_dir).glob('*_merged_spans.csv')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def context_intervals(rows, context, last_frame):
    """Union overlapping context windows so no original frame occurs twice."""
    if context < 0:
        raise ValueError('Context must be nonnegative')
    result = []
    for row in sorted(rows, key=lambda r: r['Start Frame']):
        if row['End Frame'] > last_frame:
            raise ValueError(f"Span exceeds pose data: {row['Merge ID']}")
        start = max(0, row['Start Frame'] - context)
        end = min(last_frame, row['End Frame'] + context)
        if result and start <= result[-1][1] + 1:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def span_labels(frames, rows):
    labels = np.zeros(len(frames), dtype=np.int64)
    # Mask rejected/unknown spans last: ambiguity never becomes supervision.
    for accepted in (True, False):
        for row in rows:
            if (row['Verified'] is True) == accepted:
                mask = (frames >= row['Start Frame']) & (frames <= row['End Frame'])
                labels[mask] = 1 if accepted else IGNORE
    return labels


def build_sequences(data_dir, spans, context):
    sequences, feature_names = [], None
    for video in spans['Video ID'].unique().sort():
        print(f'Extracting buffered pose/pair features: {video}', flush=True)
        rows = spans.filter(pl.col('Video ID') == video).to_dicts()
        pose_path = Path(data_dir) / f'{video}.parquet'
        pair_path = Path(data_dir) / f'{video}_pairs.parquet'
        pose_scan = pl.scan_parquet(pose_path)
        last = pose_scan.select(pl.col('FrameNum').max()).collect().item()
        intervals = context_intervals(rows, context, last)
        # Include history for causal pose features; do not load hours of unused data.
        predicate = pl.any_horizontal([
            pl.col('FrameNum').is_between(max(0, a - 30), b) for a, b in intervals
        ])
        poses = pose_scan.filter(predicate).collect()
        pairs = pl.scan_parquet(pair_path).filter(predicate).collect()
        for index, (start, end) in enumerate(intervals):
            table = frame_features(
                poses.filter(pl.col('FrameNum').is_between(max(0, start - 30), end)),
                pairs.filter(pl.col('FrameNum').is_between(start, end)), start, end,
            )
            names = [c for c in table.columns if c != 'FrameNum']
            if feature_names is not None and feature_names != names:
                raise ValueError('Inconsistent feature columns')
            feature_names = names
            frames = table['FrameNum'].to_numpy()
            sequences.append(dict(video=video, sequence=f'{video}-{index:03d}',
                                  frames=frames, x=table.select(names).to_numpy(),
                                  y=span_labels(frames, rows)))
    return sequences, feature_names


def make_windows(sequences, length):
    """Pad each contiguous sequence independently; mask padding in the loss."""
    windows, labels, locations = [], [], []
    for sequence in sequences:
        for start in range(0, len(sequence['y']), length):
            count = min(length, len(sequence['y']) - start)
            x = np.zeros((sequence['x'].shape[1], length), dtype=np.float32)
            y = np.full(length, IGNORE, dtype=np.int64)
            x[:, :count] = sequence['x'][start:start + count].T
            y[:count] = sequence['y'][start:start + count]
            windows.append(x)
            labels.append(y)
            locations.append((sequence, start, count))
    return torch.from_numpy(np.stack(windows)), torch.from_numpy(np.stack(labels)), locations


def new_model(feature_count):
    return MS_TCN3(num_f_maps=32, num_classes=2, exclusive=True,
                   dims={'loaded': torch.Size([feature_count])}, num_layers_R=3,
                   num_R=1, num_layers_PG=4, dropout_rate=0.2)


def predicted_segments(predictions, threshold=0.5):
    """Export positive runs without bridging absent frames or sequence gaps."""
    runs = []
    for group in predictions.partition_by(['video', 'sequence'], maintain_order=True):
        frame = group['FrameNum'].to_numpy()
        score = group['circling_score'].to_numpy()
        positive = score >= threshold
        starts = np.flatnonzero(positive & np.r_[True, (~positive[:-1]) | (np.diff(frame) != 1)])
        ends = np.flatnonzero(positive & np.r_[(~positive[1:]) | (np.diff(frame) != 1), True])
        for start, end in zip(starts, ends):
            runs.append(dict(video=group['video'][0], sequence=group['sequence'][0],
                             start_frame=int(frame[start]), end_frame=int(frame[end]),
                             frames=int(end - start + 1), mean_score=float(score[start:end + 1].mean())))
    return pl.DataFrame(runs, schema={'video': pl.String, 'sequence': pl.String,
                                    'start_frame': pl.Int64, 'end_frame': pl.Int64,
                                    'frames': pl.Int64, 'mean_score': pl.Float64})


def run_test(data_dir=ROOT / 'data', span_dir=ROOT / 'outputs' / 'merged_spans',
             output_dir=DEFAULT_OUTPUT, epochs=5, context=150, window=512, seed=42,
             feature_sequences=None, feature_names=None, feature_source=None):
    if epochs < 1 or window < 32:
        raise ValueError('Use at least one epoch and a window of at least 32 frames')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    spans = read_spans(span_dir)
    if set(spans['Video ID']) != {'0028_vid', '0031_vid'}:
        raise ValueError('This smoke test expects videos 0028_vid and 0031_vid')
    if feature_sequences is None:
        sequences, names = build_sequences(data_dir, spans, context)
    else:
        sequences, names = feature_sequences, feature_names
        if not sequences or not names:
            raise ValueError('Provide nonempty feature sequences and feature names')
    train = [s for s in sequences if s['video'] == '0028_vid']
    test = [s for s in sequences if s['video'] == '0031_vid']
    preprocessing = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                                  StandardScaler())
    # Fit on supervised training frames only; held-out video never contributes.
    preprocessing.fit(np.concatenate([s['x'][s['y'] != IGNORE] for s in train]))
    for sequence in sequences:
        sequence['x'] = preprocessing.transform(sequence['x']).astype(np.float32)
        if not np.isfinite(sequence['x']).all():
            raise ValueError('Non-finite model inputs after preprocessing')
    x_train, y_train, _ = make_windows(train, window)
    x_test, _, locations = make_windows(test, window)
    valid_windows = (y_train != IGNORE).any(dim=1)
    loader = DataLoader(TensorDataset(x_train[valid_windows], y_train[valid_windows]),
                        batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed))
    counts = torch.bincount(y_train[y_train != IGNORE], minlength=2)
    if torch.any(counts == 0):
        raise ValueError('Training requires both circling and background frames')
    weights = counts.sum() / (2 * counts.float())
    model = new_model(len(names))
    model.ssl_off()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad()
            stages, _ = model(x, [])
            # Supervise every MS-TCN stage; no loss on unreviewed/rejected or padded frames.
            loss = torch.stack([torch.nn.functional.cross_entropy(
                stage, y, weight=weights, ignore_index=IGNORE) for stage in stages]).mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        history.append({'epoch': epoch, 'train_loss': total / batches})
        print(f'Epoch {epoch}/{epochs}: loss={total / batches:.4f}', flush=True)
    model.eval()
    scores = []
    with torch.inference_mode():
        for x in x_test.split(16):
            stages, _ = model(x, [])
            scores.append(stages[-1].softmax(dim=1)[:, 1].numpy())
    scores = np.concatenate(scores)
    tables = []
    for score, (sequence, start, count) in zip(scores, locations):
        tables.append(pl.DataFrame({
            'video': sequence['video'], 'sequence': sequence['sequence'],
            'FrameNum': sequence['frames'][start:start + count],
            'label': sequence['y'][start:start + count], 'circling_score': score[:count],
            'predicted_circling': score[:count] >= 0.5,
        }))
    predictions = pl.concat(tables).sort('video', 'FrameNum')
    evaluated = predictions.filter(pl.col('label') != IGNORE)
    test_metrics = metrics(evaluated['label'].to_numpy(), evaluated['circling_score'].to_numpy())
    test_metrics.update(temporal_metrics(predictions))
    segments = predicted_segments(predictions)
    manifest = spans.with_columns(
        pl.when(pl.col('Video ID') == '0028_vid').then(pl.lit('train')).otherwise(pl.lit('test')).alias('split'),
        pl.when(pl.col('Verified') == True).then(pl.lit('circling')).otherwise(pl.lit('ignored')).alias('label_policy'),
    )
    report = {
        'model': 'dlc2action.model.ms_tcn.MS_TCN3', 'dlc2action_version': version('dlc2action'),
        'feature_source': feature_source or {'type': 'pose_and_pairs'},
        'temporal_metric_policy': 'MoF, Edit and segmental F1 are percentages. MoF includes background. '
            'Edit is normalized Levenshtein on run-collapsed action labels, averaged over contiguous '
            'evaluation runs. Segmental F1@10/25/50 uses one-to-one same-class IoU matches pooled '
            'over runs, excluding background. Ignored frames and excerpt gaps split runs. '
            'Edit with background is also reported; foreground-only Edit has limited ordering '
            'information for a single action class.',
        'api': 'DLC2Action model API with a custom PyTorch supervised training loop',
        'label_policy': LABEL_POLICY, 'source_sha256': source_hash(span_dir),
        'source_rows': spans.height, 'accepted_rows': spans.filter(pl.col('Verified') == True).height,
        'ignored_rows': spans.filter(pl.col('Verified').fill_null(False).not_()).height,
        'train_video': '0028_vid', 'test_video': '0031_vid', 'seed': seed, 'device': 'cpu',
        'epochs': epochs, 'context_frames': context, 'window_frames': window, 'threshold': 0.5,
        'features': names, 'train_frames': int((y_train != IGNORE).sum()),
        'train_class_counts': counts.tolist(), 'class_weights': weights.tolist(),
        'test_predicted_frames': predictions.height, 'test_ignored_frames': predictions.height - evaluated.height,
        'predicted_segments': segments.height, 'history': history, 'test': test_metrics,
        'constant_score_average_precision': test_metrics['prevalence'],
        'all_background_accuracy': 1 - test_metrics['prevalence'],
        'limitations': 'Smoke test on candidate-centered excerpts of one held-out video, not full-video validation. '
                       'Background labels are assumptions. No tuning or checkpoint selection uses test labels. '
                       'Non-overlapping 512-frame windows (or the configured length) can create boundary artifacts; '
                       'no smoothing or minimum-duration filter is applied. No identity-specific actions are inferred.',
        'input_files': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
                        {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                        for p in sorted(Path(data_dir).glob('*.parquet'))},
    }
    predictions.write_parquet(output_dir / 'test_predictions.parquet')
    segments.write_csv(output_dir / 'predicted_segments.csv')
    manifest.write_csv(output_dir / 'span_manifest.csv')
    joblib.dump(preprocessing, output_dir / 'preprocessing.joblib')
    torch.save({'state_dict': model.state_dict(), 'features': names, 'seed': seed,
                'window_frames': window, 'classes': ['background', 'circling']}, output_dir / 'model.pt')
    (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    (output_dir / 'report.md').write_text(
        '# DLC2Action segmentation smoke test\n\n' + LABEL_POLICY + '\n\n'
        f"Trained MS-TCN on 0028_vid; tested on 0031_vid. {spans.height} input rows, "
        f"{report['accepted_rows']} accepted, {report['ignored_rows']} ignored. "
        f"{epochs} CPU epochs, seed {seed}, {len(names)} input features.\n\n"
        + '\n'.join(f'- {k}: {test_metrics[k]:.4f}' for k in ('precision', 'recall', 'f1', 'average_precision', 'roc_auc',
            'mof', 'edit_score', 'edit_score_with_background', 'segmental_f1_10', 'segmental_f1_25', 'segmental_f1_50'))
        + f"\n\n{predictions.height:,} test frames predicted; {evaluated.height:,} evaluated; "
        f"{segments.height} positive segments at threshold 0.5.\n\n" + report['limitations'] + '\n\n'
        + report['temporal_metric_policy'] + '\n')
    print(json.dumps(test_metrics, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--context', type=int, default=150)
    parser.add_argument('--window', type=int, default=512)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_test(epochs=args.epochs, context=args.context, window=args.window, output_dir=args.output_dir)
