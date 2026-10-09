# DLC2Action segmentation smoke test

Verified=true rows are circling (1), with inclusive endpoints. Verified=false or blank rows are ignored (-100), not negative examples. Frames within the context buffer but outside every CSV span are assumed background (0); these frames have not been exhaustively annotated. Predictions are video-level, not assignments to individual fish.

Trained MS-TCN on 0028_vid; tested on 0031_vid. 153 input rows, 140 accepted, 13 ignored. 30 CPU epochs, seed 42, 768 input features.

- precision: 0.8706
- recall: 0.3211
- f1: 0.4691
- average_precision: 0.8057
- roc_auc: 0.7180
- mof: 54.1774
- edit_score: 40.1582
- edit_score_with_background: 50.6554
- segmental_f1_10: 18.9189
- segmental_f1_25: 8.7838
- segmental_f1_50: 4.0541

57,106 test frames predicted; 54,484 evaluated; 246 positive segments at threshold 0.5.

Smoke test on candidate-centered excerpts of one held-out video, not full-video validation. Background labels are assumptions. No tuning or checkpoint selection uses test labels. Non-overlapping 512-frame windows (or the configured length) can create boundary artifacts; no smoothing or minimum-duration filter is applied. No identity-specific actions are inferred.

MoF, Edit and segmental F1 are percentages. MoF includes background. Edit is normalized Levenshtein on run-collapsed action labels, averaged over contiguous evaluation runs. Segmental F1@10/25/50 uses one-to-one same-class IoU matches pooled over runs, excluding background. Ignored frames and excerpt gaps split runs. Edit with background is also reported; foreground-only Edit has limited ordering information for a single action class.
