# Circling logistic regression baseline

All frames outside the 15 annotated inclusive spans are non-circling.

All positives plus an equal number of negatives sampled without replacement within each source video, seed 42.

Stratified random 80/20 train/test split after sampling, random_state=42.

Train: 12,508 frames. Test: 3,128 frames.

50 causal pose features: mean/max normalized keypoint motion, body bend, absolute heading change, track count, and their trailing 30-frame means. No frame number, absolute position, video ID, or TrackID is a predictor.

Features are computed on the original continuous timeline before sampling. Motion uses consecutive frames within the same track. No future frames are used. Median imputation with missingness indicators and standard scaling are fitted on training frames only. L2 logistic regression uses C=1, no class weighting, and a fixed threshold of 0.5. No tuning uses test labels.

| Test metric | Value |
|---|---:|
| average_precision | 0.8530 |
| roc_auc | 0.9223 |
| precision | 0.8481 |
| recall | 0.9457 |
| f1 | 0.8942 |

Confusion matrix: TN=1299, FP=265, FN=85, TP=1479.

Neighboring frames and spans can occur in both splits; this is not an independent-event or new-video evaluation.

Scores describe the balanced sampled dataset, not calibrated probabilities at full-video prevalence.

Only two videos and 15 spans are available. A frame with no detected fish is still scored using missingness and available past observations. Track fragmentation and pose errors may affect motion features.

`sampled_frames.parquet` records every selected frame, its label, features and split. `test_predictions.parquet` contains test-only predictions. `model.joblib` is the exact evaluated model, trained only on the training split.

Implementation reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
