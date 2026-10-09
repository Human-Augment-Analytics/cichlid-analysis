# DLC2Action segmentation smoke test

Verified=true rows are circling (1), with inclusive endpoints. Verified=false or blank rows are ignored (-100), not negative examples. Frames within the context buffer but outside every CSV span are assumed background (0); these frames have not been exhaustively annotated. Predictions are video-level, not assignments to individual fish.

Trained MS-TCN on 0028_vid; tested on 0031_vid. 153 input rows, 140 accepted, 13 ignored. 50 CPU epochs, seed 42, 102 pose/pair features.

- precision: 0.9582
- recall: 0.9081
- f1: 0.9325
- average_precision: 0.9851
- roc_auc: 0.9448

45,379 test frames predicted; 42,757 evaluated; 154 positive segments at threshold 0.5.

Smoke test on candidate-centered excerpts of one held-out video, not full-video validation. Background labels are assumptions. No tuning or checkpoint selection uses test labels. Non-overlapping 512-frame windows (or the configured length) can create boundary artifacts; no smoothing or minimum-duration filter is applied. No identity-specific actions are inferred.
