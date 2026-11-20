# ActionFormer Hard-Negative Experiments

Experiments investigating whether different negative-sample
selection strategies improve circling detection.

## Experiments

### Baseline
Original ActionFormer model using pair-motion features.

### Proximity negatives
Replaces 82 ordinary training negatives with 82 proximity-based
hard negatives.

Script: `run_proximity_negatives.py`

### Reviewed annotated negatives
Replaces 42 ordinary training negatives with 42 human-reviewed
negative segments.

Script: `run_annotated_negatives.py`

## Held-out Evaluation

| Experiment | Frame AP | Frame ROC-AUC |
|---|---:|---:|
| Baseline | 0.8910 | 0.9440 |
| Proximity negatives | 0.8566 | 0.9317 |
| Reviewed annotated negatives | 0.8668 | 0.9358 |

All experiments used the same held-out set of 145 sampled segments
from videos 0028 and 0031.

Neither hard-negative strategy improved frame AP over baseline.

## Reproducibility

Run experiments from the repository root.

Both scripts currently depend on a local baseline sample file:

`outputs/experiment_backups/baseline_actionformer/sampled_segments.parquet`

The annotated-negative experiment also requires:

`experiments/actionformer/data/reviewed_hard_negatives.xlsx`

These local data files are excluded from Git.

Training outputs are written under `outputs/`.

## Limitations

- The held-out split is segment-based, not leave-one-video-out.
- Only one training run per experiment was evaluated.
- Hard negatives replaced ordinary negatives rather than being added.
- Default-threshold precision, recall, and F1 were all zero.
- Validation event metrics should not be confused with held-out test metrics.
