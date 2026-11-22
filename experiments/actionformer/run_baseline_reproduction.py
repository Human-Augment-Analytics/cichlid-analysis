"""Frozen-split full-feature ActionFormer baseline reproduction.
Run from repository root: uv run python -m experiments.actionformer.run_baseline_reproduction

Matches the previous ablation scripts' training/evaluation flow, but keeps all
pairmotion features. Does not perform full-video OOF postprocessing tuning.
"""
import importlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from models.actionformer.model import build_model
from models.common.config import FINALIZED_ANNOTATIONS, mstcn_workdir, read_annotations, video_frames
from models.common.report import write_report
from models.common.segments import load_meta

trainer = importlib.import_module('models.actionformer.train')
BASELINE = Path('outputs/experiment_backups/baseline_actionformer/sampled_segments.parquet')
OUT = Path('outputs/actionformer_pairmotion_baseline_reproduction')


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
    if len(columns) != 132:
        raise RuntimeError(f'Expected baseline 132 features, found {len(columns)}; check branch and inputs')

    training = [s for s in samples if s['split'] == 'train']
    fitting, validation = trainer.training_validation_split(training, 0.2, 42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # The previous ablation scripts also call build_model before train_model,
    # whose seed=42 initialization controls training but not model creation.
    # Preserve their behavior for the closest protocol match.
    model, config = build_model(len(columns))
    OUT.mkdir(parents=True, exist_ok=True)
    segments.write_parquet(OUT / 'sampled_segments.parquet')
    (OUT / 'experiment.json').write_text(json.dumps({
        'baseline_split': str(BASELINE),
        'feature_source': 'pairmotion',
        'excluded_columns': [],
        'feature_columns': columns,
        'train_validation_split_seed': 42,
        'trainer_seed': 42,
        'epochs_max': 60,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'validation_fraction': 0.2,
        'patience': 6,
        'note': 'Full 132-feature baseline; no full-video OOF tuning. '
                'Model initialization was not seeded before build_model in the previous ablations.'
    }, indent=2) + '\n')

    print(f'Baseline reproduction: {len(columns)} features; '
          f'{len(fitting)} fit + {len(validation)} validation; device={device}', flush=True)
    history = trainer.train_model(
        model, fitting, device, epochs=60, batch_size=4,
        learning_rate=1e-4, validation_samples=validation, patience=6)
    pl.DataFrame(history).write_csv(OUT / 'training_history.csv')
    best = next((row for row in reversed(history) if row.get('is_best')), None)

    torch.save({
        'model_state_dict': model.state_dict(), 'config': config,
        'feature_columns': columns, 'feature_set': 'pairmotion',
        'normalization_mean': mean, 'normalization_std': std,
        'window': 256, 'seed': 42,
        'best_epoch': best['epoch'] if best else len(history),
        'best_validation_loss': best['validation_loss'] if best else None,
    }, OUT / 'model.pt')
    np.savez(OUT / 'normalization.npz', columns=np.asarray(columns), mean=mean, std=std)

    test = trainer.heldout_test(model, samples, spans, device)
    result = write_report(
        OUT, 'ActionFormer full-feature baseline reproduction', 'ActionFormer',
        test,
        {'model': 'ActionFormer', 'features': 'pairmotion',
         'feature_columns': columns, 'excluded_columns': [],
         'baseline_split': str(BASELINE),
         'limitation': 'Same-video segment-level holdout; not independent-video evaluation.',
         'score_warning': 'No OOF tuning; compare heldout frame AP and ROC-AUC.'},
        notes=['Frozen baseline segments; all original 132 pairmotion features retained.',
               'Features normalized using training data only.'])
    print(f'Done: {OUT} | AP={result["average_precision"]:.4f} '
          f'ROC-AUC={result["roc_auc"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
