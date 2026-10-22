# Circling logistic regression baseline

All frames outside the 149 annotated inclusive spans are non-circling.

All positives plus an equal number of negatives sampled without replacement within each source video, seed 42.

Stratified random 80/20 train/test split after sampling, random_state=42.

Train: 95,038 frames. Test: 23,760 frames.

102 features: the original 50 pose features (motion, bend, turn, track count and trailing 30-frame means), plus 52 pair features (mean/min/max of six distances, sine/cosine of four angles, and three angular rates, plus pair count). Identifiers, time, sex, QC flags and episode metadata are excluded.

Pose features retain the original causal 30-frame history. Pair features aggregate current-frame measurements; angular rates are used as supplied in the pair Parquet file. Median imputation with missingness indicators and standard scaling are fitted on training frames only. L2 logistic regression uses C=1, no class weighting, and a fixed threshold of 0.5. No tuning uses test labels.

| Test metric | Value |
|---|---:|
| average_precision | 0.9832 |
| roc_auc | 0.9847 |
| precision | 0.9550 |
| recall | 0.9556 |
| f1 | 0.9553 |

Confusion matrix: TN=11345, FP=535, FN=528, TP=11352.

Neighboring frames and spans can occur in both splits; this is not an independent-event or new-video evaluation.

Scores describe the balanced sampled dataset, not calibrated probabilities at full-video prevalence.

Only two videos and 149 spans are available. Frames with no pair rows remain in the dataset with pair_count=0 and missing measurements. Missing measurements are imputed; tracking and measurement errors can affect predictions.

`model.joblib` is the exact evaluated model, trained only on the training split.

Implementation reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
