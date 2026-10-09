# VideoMAE → MS-TCN held-out evaluation

Trained both stages using 0028_vid only; evaluated buffered merged-span excerpts from 0031_vid.
57,106 predicted frames; 54,484 evaluated; 2,622 ignored; 246 predicted positive segments.

Comparison below uses only 45,379 shared frames with identical labels (42,757 evaluated). Full visual-run metrics remain in metrics.json and report.md.

| Metric | Units | Pose/pair baseline | VideoMAE → MS-TCN |
|---|---|---:|---:|
| mof | percent | 89.4356 | 43.3824 |
| edit_score | percent | 64.6296 | 48.9370 |
| edit_score_with_background | percent | 72.2559 | 59.1729 |
| segmental_f1_10 | percent | 56.4356 | 20.5882 |
| segmental_f1_25 | percent | 54.4554 | 9.5588 |
| segmental_f1_50 | percent | 45.5446 | 4.4118 |
| precision | fraction | 0.9582 | 0.9260 |
| recall | fraction | 0.9081 | 0.3211 |
| f1 | fraction | 0.9325 | 0.4768 |
| average_precision | fraction | 0.9851 | 0.8956 |
| roc_auc | fraction | 0.9448 | 0.6950 |

The visual model underperforms the pose/pair baseline in this run, with many missed circling frames and fragmented/poorly aligned segments. This is not a controlled feature ablation: the saved runs use different training schedules.

Possible contributors (not established causes): small subjects in full-scene 224×224 inputs, only 256 clips for encoder adaptation, domain shift between the two videos, and sparse embedding anchors interpolated to frames. Test results were not used to tune the model or threshold.

MoF, Edit and segmental F1 are percentages. MoF includes background. Edit is normalized Levenshtein on run-collapsed action labels, averaged over contiguous evaluation runs. Segmental F1@10/25/50 uses one-to-one same-class IoU matches pooled over runs, excluding background. Ignored frames and excerpt gaps split runs. Edit with background is also reported; foreground-only Edit has limited ordering information for a single action class.

Smoke test on candidate-centered excerpts of one held-out video, not full-video validation. Background labels are assumptions. No tuning or checkpoint selection uses test labels. Non-overlapping 512-frame windows (or the configured length) can create boundary artifacts; no smoothing or minimum-duration filter is applied. No identity-specific actions are inferred.

Validation: saved labels, metrics, segment export and training-video membership checked. MS-TCN checkpoint predictions reproduced from cached visual embeddings.
