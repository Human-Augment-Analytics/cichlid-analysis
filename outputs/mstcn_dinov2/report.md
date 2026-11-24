# Circling MS-TCN (DLC2Action), pairemb

All frames outside the intervals in merged_spans.csv are non-circling.

Every 128-row segment (stride 2, 256 frames) containing circling plus an equal number of segments without circling, sampled without replacement within each source video, seed 42.

Stratified random 80/20 train/test split of the sampled segments after sampling, random_state=42. Segments do not overlap.

DLC2Action validation subset = test segments, used for logging only (no early stopping or model selection).

Input features: pose + 17 pair features + 32 PCs of DINOv2 ViT-S/14 fish-masked male crops. Segment length 128 rows, 100 epochs.

Test: 38,990 frames (13,360 circling).

| Test metric | Value |
|---|---:|
| average_precision | 0.9767 |
| roc_auc | 0.9889 |
| precision | 0.9173 |
| recall | 0.9517 |
| f1 | 0.9342 |

Confusion matrix: TN=24483, FP=1147, FN=645, TP=12715.

Neighboring segments and spans can occur in both splits; this is not an independent-event or new-video evaluation.

Scores describe the balanced sampled dataset, not calibrated probabilities at full-video prevalence.
