# Circling logistic regression baseline

All frames outside the 15 annotated inclusive spans are non-circling.

All positives plus an equal number of negatives sampled without replacement within each source video, seed 42.

Stratified random 80/20 train/test split after sampling, random_state=42.

Train: 12,508 frames. Test: 3,128 frames.

102 features: the original 50 pose features (motion, bend, turn, track count and trailing 30-frame means), plus 52 pair features (mean/min/max of six distances, sine/cosine of four angles, and three angular rates, plus pair count). Identifiers, time, sex, QC flags and episode metadata are excluded.

Pose features retain the original causal 30-frame history. Pair features aggregate current-frame measurements; angular rates are used as supplied in the pair Parquet file. Median imputation with missingness indicators and standard scaling are fitted on training frames only. L2 logistic regression uses C=1, no class weighting, and a fixed threshold of 0.5. No tuning uses test labels.

| Test metric | Value |
|---|---:|
| average_precision | 0.9552 |
| roc_auc | 0.9777 |
| precision | 0.9384 |
| recall | 0.9642 |
| f1 | 0.9511 |

Confusion matrix: TN=1465, FP=99, FN=56, TP=1508.

Neighboring frames and spans can occur in both splits; this is not an independent-event or new-video evaluation.

Scores describe the balanced sampled dataset, not calibrated probabilities at full-video prevalence.

Only two videos and 15 spans are available. Frames with no pair rows remain in the dataset with pair_count=0 and missing measurements. Missing measurements are imputed; tracking and measurement errors can affect predictions.

`sampled_frames.parquet` records every selected frame, its label, features and split. `test_predictions.parquet` contains test-only predictions. `model.joblib` is the exact evaluated model, trained only on the training split.

Implementation reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
