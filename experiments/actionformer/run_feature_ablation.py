"""Frozen-split ActionFormer ablation: remove angular-movement features.
Run from repository root: uv run python -m experiments.actionformer.run_feature_ablation
"""
import importlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from feature_engineering.pair_motion.feature_groups import ANGULAR_MOVEMENT
from models.actionformer.model import build_model
from models.common.config import FINALIZED_ANNOTATIONS, mstcn_workdir, read_annotations, video_frames
from models.common.pair_motion import ROLE_NAMES
from models.common.report import write_report
from models.common.segments import load_meta

trainer = importlib.import_module('models.actionformer.train')
BASELINE = Path('outputs/experiment_backups/baseline_actionformer/sampled_segments.parquet')
OUT = Path('outputs/actionformer_pairmotion_no_angular_ablation')


def main():
    if not BASELINE.is_file():
        raise SystemExit(f'Missing frozen baseline split: {BASELINE}')
    if (OUT / 'model.pt').exists():
        raise SystemExit(f'Refusing to overwrite existing checkpoint: {OUT / "model.pt"}')
    segments = pl.read_parquet(BASELINE)
    videos = sorted(segments['video'].unique().to_list())
    if videos != ['0028_vid', '0031_vid']:
        raise SystemExit(f'Unexpected baseline videos: {videos}')
    workdir = mstcn_workdir()
    meta = load_meta(workdir, videos)
    spans = read_annotations(FINALIZED_ANNOTATIONS, videos=videos)
    counts = video_frames(videos)
    samples, columns, mean, std = trainer.prepare_samples(
        segments, meta, spans, counts, workdir, 'pairmotion')
    excluded = {'pm_' + ROLE_NAMES.get(name, name) for name in ANGULAR_MOVEMENT}
    missing = excluded - set(columns)
    if missing:
        raise RuntimeError(f'Missing angular feature columns: {sorted(missing)}')
    indices = [i for i, col in enumerate(columns) if col not in excluded]
    selected = [columns[i] for i in indices]
    for sample in samples:
        sample['feats'] = sample['feats'][indices, :].contiguous()
    mean, std = mean[indices], std[indices]
    training = [s for s in samples if s['split'] == 'train']
    fitting, validation = trainer.training_validation_split(training, 0.2, 42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = build_model(len(selected))
    OUT.mkdir(parents=True, exist_ok=True)
    segments.write_parquet(OUT / 'sampled_segments.parquet')
    (OUT / 'ablation.json').write_text(json.dumps({
        'baseline_split': str(BASELINE), 'feature_source': 'pairmotion',
        'excluded_group': 'angular_movement', 'excluded_columns': sorted(excluded),
        'retained_columns': selected, 'seed': 42,
        'note': 'No full-video OOF tuning; compare heldout frame metrics only.'
    }, indent=2) + '\n')
    print(f'Ablation: {len(columns)} -> {len(selected)} features; '
          f'{len(fitting)} fit + {len(validation)} validation; device={device}', flush=True)
    history = trainer.train_model(model, fitting, device, epochs=60, batch_size=4,
                                  learning_rate=1e-4, validation_samples=validation, patience=6)
    pl.DataFrame(history).write_csv(OUT / 'training_history.csv')
    best = next((row for row in reversed(history) if row.get('is_best')), None)
    torch.save({
        'model_state_dict': model.state_dict(), 'config': config,
        'feature_columns': selected, 'feature_set': 'pairmotion',
        'normalization_mean': mean, 'normalization_std': std,
        'window': 256, 'seed': 42,
        'best_epoch': best['epoch'] if best else len(history),
        'best_validation_loss': best['validation_loss'] if best else None,
    }, OUT / 'model.pt')
    np.savez(OUT / 'normalization.npz', columns=np.asarray(selected), mean=mean, std=std)
    test = trainer.heldout_test(model, samples, spans, device)
    result = write_report(OUT, 'ActionFormer without angular movement', 'ActionFormer',
        test, {'model': 'ActionFormer', 'features': 'pairmotion minus angular_movement',
               'feature_columns': selected, 'excluded_columns': sorted(excluded),
               'baseline_split': str(BASELINE),
               'limitation': 'Same-video segment-level holdout; not independent-video evaluation.',
               'score_warning': 'No OOF tuning; compare heldout frame AP and ROC-AUC.'},
        notes=['Frozen baseline segments; only angular_movement features removed.',
               'Retained features normalized using training data only.'])
    print(f'Done: {OUT} | AP={result["average_precision"]:.4f} '
          f'ROC-AUC={result["roc_auc"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
