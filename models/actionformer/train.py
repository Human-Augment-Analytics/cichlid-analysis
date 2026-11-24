"""Train a compact one-class ActionFormer on the repository's shared segments."""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from models.actionformer.model import build_model, infer_windows
from models.actionformer.postprocess import inclusive_intervals, interval_scores
from models.common.config import (FINALIZED_ANNOTATIONS, OUTPUTS, model_name, mstcn_workdir,
                                  read_annotations, video_frames)
from models.common.frame_model import DESCRIPTIONS, video_features
from models.common.report import protocol, segment_counts, write_report
from models.common.segments import rows_to_frames, shared_segments
from models.logistic_regression.train import frame_labels

NAMES = {
    'pairmotion': 'actionformer_pairmotion',
    'pairmotionemb': 'actionformer_pairmotion_dinov2',
    'pairmotionvmae': 'actionformer_pairmotion_videomae',
}


def normalization_stats(arrays):
    """Finite-feature mean/std computed exclusively from training frames."""
    count = None
    total = None
    total_sq = None
    for array in arrays:
        finite = np.isfinite(array)
        values = np.where(finite, array, 0.0).astype(np.float64)
        n = finite.sum(axis=0)
        count = n if count is None else count + n
        total = values.sum(axis=0) if total is None else total + values.sum(axis=0)
        total_sq = (values * values).sum(axis=0) if total_sq is None else total_sq + (values * values).sum(axis=0)
    if count is None or not count.any():
        raise ValueError('ActionFormer has no finite training features')
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    variance = np.divide(total_sq, count, out=np.zeros_like(total_sq), where=count > 0) - mean * mean
    std = np.sqrt(np.maximum(variance, 0.0))
    std = np.where((std > 1e-6) & np.isfinite(std), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(array, mean, std):
    values = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    # Replace missing/non-finite values with the training mean (zero after scaling).
    values = np.where(np.isfinite(array), values, mean)
    return ((values - mean) / std).astype(np.float32)


def local_targets(spans, first, length):
    targets = []
    for start, end in spans:
        start, end = max(float(start - first), 0.0), min(float(end - first + 1), float(length))
        if end > start:
            targets.append((start, end))
    return torch.as_tensor(targets, dtype=torch.float32).reshape(-1, 2)


def prepare_samples(segments, meta, spans, counts, workdir, feature):
    """Load only sampled frames, never an entire expanded video at once."""
    indexed = segments.with_row_index('sample_id')
    rows = []
    for sample_id, video, start_row, end_row, split in indexed.select(
            'sample_id', 'video', 'start_row', 'end_row', 'split').iter_rows():
        frames = rows_to_frames(start_row, end_row, meta[video], counts[video])
        rows.append({'sample_id': int(sample_id), 'video': video, 'split': split, 'frames': frames})
    cache = {}
    columns = None
    for video in sorted(meta):
        selected = [row for row in rows if row['video'] == video]
        if not selected:
            continue
        frames = np.unique(np.concatenate([row['frames'] for row in selected]))
        table = video_features(workdir, video, feature, meta[video], counts[video], frames)
        current = [column for column in table.columns if column != 'FrameNum']
        if columns is None:
            columns = current
        elif current != columns:
            raise ValueError(f'{video}: inconsistent ActionFormer feature columns')
        cache[video] = ({int(frame): i for i, frame in enumerate(table['FrameNum'].to_numpy())},
                        table.select(columns).to_numpy().astype(np.float32))
    train_arrays = []
    for row in rows:
        if row['split'] == 'train':
            positions, array = cache[row['video']]
            train_arrays.append(array[[positions[int(f)] for f in row['frames']]])
    mean, std = normalization_stats(train_arrays)
    samples = []
    for row in rows:
        positions, array = cache[row['video']]
        values = normalize(array[[positions[int(f)] for f in row['frames']]], mean, std)
        first = int(row['frames'][0])
        length = len(row['frames'])
        targets = local_targets(spans[row['video']], first, length)
        samples.append({
            **row, 'feats': torch.from_numpy(values.T.copy()), 'segments': targets,
            'labels': torch.zeros(len(targets), dtype=torch.long),
        })
    if not samples:
        raise ValueError('ActionFormer received no shared segments')
    return samples, columns, mean, std


def training_validation_split(samples, fraction=0.2, seed=42):
    """Stratified segment split inside the training partition only."""
    if not 0 < fraction < 1:
        raise ValueError('validation fraction must be between 0 and 1')
    if len(samples) < 4:
        return list(samples), []
    from sklearn.model_selection import train_test_split

    labels = np.asarray([int(sample['segments'].shape[0] > 0) for sample in samples])
    indices = np.arange(len(samples))
    stratify = labels if min(np.bincount(labels, minlength=2)) >= 2 else None
    fit_ids, validation_ids = train_test_split(
        indices, test_size=fraction, random_state=seed, stratify=stratify)
    return [samples[index] for index in fit_ids], [samples[index] for index in validation_ids]


def validation_loss(model, samples, batch_size):
    """Loss with deterministic stochastic layers while the root stays in loss mode."""
    stochastic = []
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) or module.__class__.__name__ in {'DropPath', 'AffineDropPath'}:
            stochastic.append((module, module.training))
            module.eval()
    total, batches = 0.0, 0
    try:
        with torch.no_grad():
            for first in range(0, len(samples), batch_size):
                loss = model(samples[first:first + batch_size])['final_loss']
                total += float(loss.detach().cpu())
                batches += 1
    finally:
        for module, training in stochastic:
            module.train(training)
    return total / max(1, batches)


def train_model(model, samples, device, epochs=30, batch_size=4, learning_rate=1e-4,
                weight_decay=1e-4, seed=42, validation_samples=None, patience=6, min_epochs=10,
                checkpoint_start=5):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def lr_factor(epoch):
        warmup = min(2, max(0, epochs - 1))
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    history = []
    validation_samples = list(validation_samples or [])
    best_loss = float('inf')
    best_state = None
    stale_epochs = 0
    for epoch in range(epochs):
        started = time.time()
        shuffled = list(samples)
        random.shuffle(shuffled)
        totals = {'final_loss': 0.0, 'cls_loss': 0.0, 'reg_loss': 0.0, 'batches': 0}
        for first in range(0, len(shuffled), batch_size):
            batch = shuffled[first:first + batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = model(batch)
            losses['final_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key in ('final_loss', 'cls_loss', 'reg_loss'):
                totals[key] += float(losses[key].detach().cpu())
            totals['batches'] += 1
        scheduler.step()
        batches = max(1, totals['batches'])
        row = {'epoch': epoch + 1, 'learning_rate': optimizer.param_groups[0]['lr'],
               'seconds': time.time() - started,
               **{key: totals[key] / batches for key in ('final_loss', 'cls_loss', 'reg_loss')}}
        if validation_samples:
            # The official architecture returns losses only when the root module is
            # training; dropout and drop-path are disabled for deterministic validation.
            row['validation_loss'] = validation_loss(model, validation_samples, batch_size)
            eligible = epoch + 1 >= checkpoint_start
            if eligible and row['validation_loss'] < best_loss - 1e-6:
                best_loss = row['validation_loss']
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
                row['is_best'] = True
            else:
                stale_epochs = stale_epochs + 1 if eligible else 0
                row['is_best'] = False
        history.append(row)
        validation_text = f" val={row['validation_loss']:.4f}" if 'validation_loss' in row else ''
        print(f"ActionFormer epoch {epoch + 1}/{epochs}: loss={row['final_loss']:.4f} "
              f"cls={row['cls_loss']:.4f} reg={row['reg_loss']:.4f}{validation_text}", flush=True)
        if validation_samples and patience > 0 and stale_epochs >= patience and epoch + 1 >= min_epochs:
            print(f'ActionFormer early stopping after {patience} stale validation epochs', flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def tune_video_postprocess(segments, meta, spans, counts, workdir, features, columns, device,
                           epochs, batch_size, learning_rate, validation_fraction, patience, output_dir):
    """Train video-disjoint folds and tune native intervals on complete held-out videos."""
    videos = sorted(counts)
    if len(videos) < 2:
        return None
    output_dir = Path(output_dir)
    proposal_dir = output_dir / 'oof_proposals'
    proposal_dir.mkdir(parents=True, exist_ok=True)
    shuffled = np.asarray(videos)
    np.random.default_rng(42).shuffle(shuffled)
    fold_count = min(3, len(videos))
    oof = {}
    folds = []
    from models.actionformer.predict import decode_video
    from models.actionformer.tuning import select_oof_inference_settings

    for index in range(fold_count):
        validation_videos = sorted(shuffled[index::fold_count].tolist())
        fold_segments = segments.filter(
            (pl.col('split') == 'train') & ~pl.col('video').is_in(validation_videos))
        fit_videos = sorted(fold_segments['video'].unique().to_list())
        if not fit_videos:
            raise ValueError(f'ActionFormer OOF fold {index + 1} has no training videos')
        print(f'ActionFormer OOF fold {index + 1}/{fold_count}: fit videos {fit_videos}; '
              f'full-video validation {validation_videos}', flush=True)
        fold_samples, fold_columns, mean, std = prepare_samples(
            fold_segments, {video: meta[video] for video in fit_videos}, spans, counts, workdir, features)
        if fold_columns != columns:
            raise ValueError(f'ActionFormer OOF fold {index + 1} has inconsistent feature columns')
        fitting, validation = (training_validation_split(fold_samples, validation_fraction)
                               if validation_fraction > 0 else (fold_samples, []))
        fold_model, _ = build_model(len(columns))
        history = train_model(fold_model, fitting, device, epochs, batch_size, learning_rate,
                              validation_samples=validation, patience=patience)
        pl.DataFrame(history).write_csv(output_dir / f'oof_fold_{index + 1}_history.csv')
        for video in validation_videos:
            proposals = decode_video(fold_model, video, counts[video], columns, mean, std,
                                     workdir, features, meta, device)
            oof[video] = proposals
            pl.DataFrame({
                'start': [float(item[0][0]) for item in proposals],
                'end': [float(item[0][1]) for item in proposals],
                'confidence': [item[1] for item in proposals],
            }, schema={'start': pl.Float64, 'end': pl.Float64, 'confidence': pl.Float64}).write_parquet(
                proposal_dir / f'{video}.parquet')
        folds.append({'fold': index + 1, 'fit_videos': fit_videos,
                      'validation_videos': validation_videos})
        del fold_model
    return select_oof_inference_settings(oof, spans, counts, output_dir, folds=folds)


def heldout_test(model, samples, spans, device):
    tables = []
    model.eval()
    for sample in samples:
        if sample['split'] != 'test':
            continue
        proposals = infer_windows(model, [(0, sample['feats'].numpy().T)], device, min_score=0.001)
        length = sample['feats'].shape[1]
        intervals = inclusive_intervals(proposals, length)
        score = interval_scores(intervals, length)
        frames = sample['frames']
        tables.append(pl.DataFrame({
            'video': [sample['video']] * length, 'FrameNum': frames,
            'circling': frame_labels(frames, spans[sample['video']]),
            'circling_score': score,
        }))
    if not tables:
        raise ValueError('ActionFormer shared split contains no test segments')
    return pl.concat(tables)


def train(features='pairmotion', epochs=60, batch_size=4, learning_rate=1e-4, device=None,
          validation_fraction=0.2, patience=6, tune_postprocess=True):
    name = model_name(NAMES[features])
    output_dir = OUTPUTS / name
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir = mstcn_workdir()
    spans = read_annotations(FINALIZED_ANNOTATIONS)
    segments, meta = shared_segments(workdir, spans, output_dir)
    counts = video_frames(sorted(meta))
    samples, columns, mean, std = prepare_samples(segments, meta, spans, counts, workdir, features)
    training = [sample for sample in samples if sample['split'] == 'train']
    fitting, validation = (training_validation_split(training, validation_fraction)
                           if validation_fraction > 0 else (training, []))
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, config = build_model(len(columns))
    print(f'Training {name} on {device}: {len(fitting)} fit + {len(validation)} validation segments, '
          f'{len(columns)} features, {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
    history = train_model(model, fitting, device, epochs, batch_size, learning_rate,
                          validation_samples=validation, patience=patience)
    inference_settings = None
    if tune_postprocess:
        inference_settings = tune_video_postprocess(
            segments, meta, spans, counts, workdir, features, columns, device,
            epochs, batch_size, learning_rate, validation_fraction, patience, output_dir)
    if inference_settings is None:
        from models.actionformer.tuning import select_inference_settings

        inference_settings = select_inference_settings(model, validation, device, output_dir)
    best = next((row for row in reversed(history) if row.get('is_best')), None)
    checkpoint = {
        'model_state_dict': model.state_dict(), 'config': config, 'feature_columns': columns,
        'feature_set': features, 'normalization_mean': mean, 'normalization_std': std,
        'window': 256, 'seed': 42, 'best_epoch': best['epoch'] if best else len(history),
        'best_validation_loss': best['validation_loss'] if best else None,
        'inference_settings': inference_settings,
    }
    torch.save(checkpoint, output_dir / 'model.pt')
    pl.DataFrame(history).write_csv(output_dir / 'training_history.csv')
    (output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    np.savez(output_dir / 'normalization.npz', columns=np.asarray(columns), mean=mean, std=std)
    test = heldout_test(model, samples, spans, device)
    report = protocol(
        FINALIZED_ANNOTATIONS, 'ActionFormer', features=features, feature_columns=columns,
        segments=segment_counts(segments), train_segments=len(training),
        fit_segments=len(fitting), validation_segments=len(validation),
        test_segments=sum(sample['split'] == 'test' for sample in samples),
        architecture='Official ActionFormer LocPointTransformer: convolutional projection, local multihead '
                     'self-attention, temporal feature pyramid, classification and boundary-distance regression.',
        hyperparameters={'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate,
                         'weight_decay': 1e-4, 'window_frames': 256, 'embedding_dim': 128,
                         'attention_window': 16, 'pyramid_levels': 4,
                         'validation_fraction_of_train': validation_fraction, 'patience': patience,
                         'minimum_epochs_before_stopping': 10, 'first_eligible_checkpoint': 5,
                         'video_level_oof_postprocess_tuning': bool(tune_postprocess and len(counts) >= 2),
                         'inference_settings': inference_settings},
        model_selection=('Best validation-loss checkpoint with early stopping; validation is a stratified subset '
                         'of shared training segments only.' if validation else
                         'Fixed epochs because the training partition was too small for a validation subset.') +
                        ' Shared test segments are never used for early stopping, normalization or model selection.',
        selected_epoch=best['epoch'] if best else len(history),
        validation_loss=best['validation_loss'] if best else None,
        native_output='Confidence-scored temporal intervals decoded by ActionFormer with portable Gaussian Soft-NMS.',
        parameters=sum(p.numel() for p in model.parameters()), device=str(device),
    )
    notes = [report['label_assumption'], report['sampling'], report['split'], report['model_selection'],
             f'Input: {DESCRIPTIONS[features]}. Features are standardized from training frames only.',
             'Held-out frame scores are the maximum native interval confidence covering each test frame; '
             'the intervals themselves, not a sustained-span rule, define the predictions.']
    result = write_report(output_dir, f'Circling ActionFormer, {features}', 'ActionFormer', test, report, notes)
    print(f'Saved {output_dir / "model.pt"}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ActionFormer temporal localization on shared 256-frame segments')
    parser.add_argument('--features', choices=sorted(NAMES), default='pairmotion')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--device', default=None, help='Default cuda when available, otherwise cpu')
    parser.add_argument('--validation-fraction', type=float, default=0.2,
                        help='Fraction of shared training segments used for checkpoint selection; 0 disables')
    parser.add_argument('--patience', type=int, default=6, help='Stale validation epochs before stopping; 0 disables')
    parser.add_argument('--skip-oof-tuning', action='store_true',
                        help='Use short validation segments instead of video-disjoint full-video OOF tuning')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('--epochs and --batch-size must be positive')
    if not 0 <= args.validation_fraction < 1 or args.patience < 0:
        parser.error('--validation-fraction must be in [0, 1) and --patience nonnegative')
    train(args.features, args.epochs, args.batch_size, args.learning_rate, args.device,
          args.validation_fraction, args.patience, tune_postprocess=not args.skip_oof_tuning)
