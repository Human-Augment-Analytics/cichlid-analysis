# Circling classification: pose-only and pose-plus-pair logistic regression

Model 1 uses 50 pose features. Model 2 retains those same 50 features and adds 52 features from the pair Parquet files, for 102 features before missingness indicators. Both predict whether a **video frame** contains circling; neither assigns the circling label to a particular fish or pair.

## Data, labels and evaluation

The 15 annotated spans use zero-based, inclusive start and end frames. Every frame inside a span is positive, and every other frame is treated as negative.

| Video | Annotated spans | Positive frames retained | Negative frames sampled |
|---|---:|---:|---:|
| `0028_vid.mp4` | 7 | 1,224 | 1,224 |
| `0031_vid.mp4` | 8 | 6,594 | 6,594 |
| Total | 15 | 7,818 | 7,818 |

Negative frames are sampled uniformly without replacement within each video using seed 42. After sampling, a stratified random 80/20 split (`random_state=42`) produces 12,508 training frames (6,254 per class) and 3,128 test frames (1,564 per class).

The saved datasets were checked: **both models use exactly the same sampled frames, labels and split**, and Model 2 preserves every original pose-feature value. Only the added pair features distinguish the feature sets.

## Model 1: 50 pose features

Input files: `data/0028_vid.parquet` and `data/0031_vid.parquet`. Calculations are implemented in [pose_features.py](../../pose_features.py), with consecutive-frame displacement checks in [pose_change.py](../../pose_change.py).

The 10 keypoints are Nose, LeftEye, RightEye, Head, Spine1, Spine2, Spine3, Spine4, Peduncle and TailTip. Each detected fish contributes one pose row per frame. Temporal differences are grouped by `(project_id, day_label, TrackID)` and require consecutive frame numbers; track starts and gaps do not produce motion measurements.

| Feature group | Derivation per fish or frame | Aggregation across fish | Count |
|---|---|---|---:|
| Keypoint movement | For each keypoint, Euclidean displacement from frame t−1 to t, normalized by current body length and transformed with `log(1 + displacement / body_length)` | Mean and maximum for each of 10 keypoints | 20 |
| Body bend | `1 − nose_to_tail_distance / body_length` at frame t | Mean and maximum | 2 |
| Absolute heading change | Absolute wrapped angular change from t−1 to t | Mean and maximum | 2 |
| Track count | Number of detected pose rows in frame t | One count per frame | 1 |
| Trailing means | Mean of each of the preceding 25 frame-level features over frames t−29 through t | Temporal aggregation after aggregation across fish | 25 |
| **Total** | | | **50** |

**Body length.** Sum the Euclidean lengths of the segments along Nose → Head → Spine1 → Spine2 → Spine3 → Spine4 → Peduncle → TailTip. A missing segment makes the body length missing. Normalized movement and bend require body length greater than one pixel. Eye points contribute movement features but are not part of this length calculation.

**Movement.** For keypoint k, displacement is `sqrt((x[k,t] − x[k,t−1])² + (y[k,t] − y[k,t−1])²)`. Missing or nonfinite endpoints produce a missing measurement. Dividing by body length reduces size dependence; the natural-log transform compresses large values. These are normalized displacement features, not pixel-per-second speeds.

**Bend.** Nose-to-tail distance is the straight-line distance from Nose to TailTip. A straight body has a ratio near one and bend near zero; a curved body has a larger bend value.

**Heading change.** Heading is `atan2(Nose_y − Spine1_y, Nose_x − Spine1_x)`. If Δθ is the difference between consecutive headings, the feature is `abs(atan2(sin(Δθ), cos(Δθ)))`, in radians per frame. This handles the wrap at −π/π.

**Trailing means and missing frames.** All video frames are retained before the 30-frame windows are computed. A frame without detections has track count zero and missing pose measurements. Rolling means use available nonmissing values within the current and previous 29 frames, require at least one valid value, and otherwise remain missing. Thus the first 29 frames and windows with missing observations can use fewer than 30 measurements. No future pose frames are used.

Feature names use `<keypoint>_mean`, `<keypoint>_max`, `bend_mean`, `bend_max`, `turn_mean`, `turn_max` and `track_count`; trailing features append `_trailing30`.

## Model 2: original pose features plus 52 pair features

Additional input files: `data/0028_vid_pairs.parquet` and `data/0031_vid_pairs.parquet`. Pair transformations and the join with pose features are implemented in [circling_model.py](../../circling_model.py).

Pair measurements are **read from the supplied Parquet columns**, rather than recalculated from pose coordinates. The descriptions below identify the measurements; the model does not independently establish the upstream centroid definition or angle-sign conventions.

| Pair source column(s) | Measurement / transformation | Frame-level features |
|---|---|---:|
| `centroid_distance_px` | Supplied distance between the two fish centroids, in pixels | Mean, min, max: 3 |
| `dist_nose1_nose2_px` | Nose-to-nose distance, in pixels | Mean, min, max: 3 |
| `dist_nose1_tailtip2_px`, `dist_nose2_tailtip1_px` | Both directed nose-to-other-fish-tail-tip distances | Mean, min, max of each: 6 |
| `dist_nose1_spine4of2_px`, `dist_nose2_spine4of1_px` | Both directed nose-to-other-fish-Spine4 distances | Mean, min, max of each: 6 |
| `orbital_rad`, `rel_heading_rad`, `bearing_1_rad`, `bearing_2_rad` | Supplied orbital angle, relative heading and two bearings; transform each angle θ into `sin(θ)` and `cos(θ)` | Mean, min, max of all 8 encoded signals: 24 |
| `d_heading_1_rad_s`, `d_heading_2_rad_s`, `d_orbital_rad_s` | Supplied signed heading-change rates for both fish and orbital angular velocity, in radians/second | Mean, min, max of each: 9 |
| Pair-row count | Number of pair rows available in the frame | `pair_count`: 1 |
| **Added pair features** | **17 measurement signals × 3 statistics + 1 count** | **52** |

Statistics are computed across **all available pairs in the same frame**. Angle encoding happens before aggregation, avoiding the discontinuity in raw angles at −π/π. Pair distances remain in pixels without body-length normalization or log transformation. Angular rates retain their supplied signs and are not converted to absolute values or recomputed; their upstream temporal derivation is not validated by this training pipeline. No additional rolling features are added for pairs.

Frames without pair rows are kept with `pair_count=0` and missing pair measurements. Nonfinite pair values become missing; aggregates use available valid values. Pair features are joined to pose features by frame number within one video. Feature names append `_mean`, `_min` and `_max` to the source signal, with `_sin` or `_cos` inserted for angles.

Neither model uses frame number, timestamps, video identity or TrackIDs as predictors. These fields are used only for grouping, alignment, labels and bookkeeping. Pair sex, quality flags, episode IDs and `dt_frames` are also excluded.

## Shared preprocessing and classifier

1. Fit a median imputer on training frames only. Add a binary missingness indicator for each feature that has missing values in training. Entirely missing training features are retained with zero fill.
2. Standardize the imputed features and indicators using training-set means and standard deviations.
3. Fit L2-regularized logistic regression with `C=1.0`, `solver='lbfgs'`, `max_iter=1000`, an intercept and no class weighting.
4. Classify a frame as positive when its logistic score is at least **0.5**. Neither hyperparameters nor the decision threshold are tuned on test labels.

The fitted imputers add 48 missingness indicators to Model 1 and 99 to Model 2, giving 98 and 201 classifier inputs respectively. The 50/102 feature counts above refer to engineered measurements before these indicators.

## Test performance

The positive class is circling. Both columns below evaluate the same 3,128 test frames.

| Metric | Model 1: pose only | Model 2: pose + pairs |
|---|---:|---:|
| Precision | 0.8481 | 0.9378 |
| Recall | 0.9457 | 0.9636 |
| F1 | 0.8942 | 0.9505 |
| Average precision (AP) | 0.8530 | 0.9551 |
| ROC AUC | 0.9223 | 0.9776 |
| Accuracy | 0.8881 | 0.9498 |

| Confusion-matrix count | Model 1 | Model 2 |
|---|---:|---:|
| True positives | 1,479 | 1,507 |
| False positives | 265 | 100 |
| True negatives | 1,299 | 1,464 |
| False negatives | 85 | 57 |

Adding pair features improves F1 by 5.63 percentage points, from 89.42% to 95.05%. False positives decrease from 265 to 100, and false negatives decrease from 85 to 57.

Precision is the fraction of positive predictions that are correct; recall is the fraction of true circling frames recovered. F1 is their harmonic mean. AP summarizes precision over recall thresholds; ROC AUC measures score ranking across the two classes. A constant-score classifier has AP=0.5 on this balanced test set; always predicting non-circling has accuracy=0.5 and circling recall/F1=0.

## Interpretation and limitations

Neighboring frames from the same circling event can occur in both splits, and their trailing pose windows can overlap. This random frame split is the requested baseline, **not an independent-event or unseen-video evaluation**. With only two videos and 15 annotated spans, these measurements do not establish generalization to other recordings.

The sampled data are 50% positive, unlike the full videos. Logistic scores and test precision therefore describe the balanced sample and should not be interpreted as calibrated full-video probabilities or full-video precision. All unannotated frames are assumed negative. Missing detections, fragmented tracks and errors in supplied measurements can affect both models.

## Saved artifacts

- **Model 1:** [metrics](../circling_baseline_pose/metrics.json), [trained pipeline](../circling_baseline_pose/model.joblib), [sampled frames and split](../circling_baseline_pose/sampled_frames.parquet), [test predictions](../circling_baseline_pose/test_predictions.parquet).
- **Model 2:** [metrics](metrics.json), [trained pipeline](model.joblib), [sampled frames and split](sampled_frames.parquet), [test predictions](test_predictions.parquet), [standardized coefficients](coefficients.csv), [precision–recall plot](precision_recall.png).

Both saved pipelines were fitted only on their training split. The notebook timeline and sustained-positive-span review use Model 2, including predictions on training frames; those displays are not additional test evaluations.
