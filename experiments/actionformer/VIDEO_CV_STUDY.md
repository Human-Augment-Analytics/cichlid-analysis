# ActionFormer: seven-video circling localization study

> **Status (November 2026):** Development-stage results; selected model not yet evaluated under the frozen final protocol. The values in this document are not independent final-test performance estimates.
>
> This study is **separate from** the older sampled-segment hard-negative experiments described in [`README.md`](README.md). Their frame-level AP and ROC-AUC numbers use a different evaluation protocol and should not be compared directly with these full-video temporal-localization results.

## Research objective

Detect when cichlids engage in circling behavior and localize the start and end frames of each episode. ActionFormer predicts confidence-scored temporal intervals from **132 engineered pose/pair-motion features**, rather than from raw RGB video. Features include relative positions, proximity, orientation, movement, angular motion, and temporal summaries. The video frame rate used to convert frames to durations is 30 FPS.

## Evaluation protocol

- **Seven development recordings:** `0002_vid`, `0007a_vid`, `0007b_vid`, `0013_vid`, `0018_vid`, `0028_vid`, `0031_vid`. A separate ActionFormer model is trained for each leave-one-recording-out (LOVO) fold, and evaluated on the **entire** excluded recording.
- **Reserved recording:** `0037_vid` is excluded from these seven development folds and from postprocessing selection. **Important:** the older 256-frame baseline was previously evaluated on `0037_vid` once. Consequently `0037_vid` must not be described as entirely untouched; do not use its results to tune configurations. A future frozen-configuration evaluation must disclose that earlier exposure.
- **Temporal mAP:** Mean of confidence-ranked average precision at tIoU thresholds 10%, 30%, 50%, and 70%, averaged equally across the seven recordings. The mAP calculation uses native proposals before applying a hard confidence decision threshold.
- **Event F1@50:** F1 of the final, merged predicted intervals against annotated events with a 50% temporal intersection-over-union matching requirement, averaged equally across the seven recordings.
- **Frame precision/recall:** Calculated from predicted versus annotated frames and then averaged equally across recordings unless specified otherwise.
- Reported postprocessing-grid performance is **exploratory**: settings were chosen using development recordings. The auxiliary leave-one-video-out calibration excludes the target recording from its setting-selection scores, but the other fold models may have trained on that recording; therefore it is **not a fully nested end-to-end generalization estimate**.

## Experiments and results

### 1. Temporal context

| Window | Duration at 30 FPS | Mean temporal mAP | Mean event F1@50 |
|---:|---:|---:|---:|
| 256 frames | 8.53 s | 0.2211 | 0.2062 |
| 512 frames | 17.07 s | 0.2520 | 0.2138 |
| **1024 frames** | **34.13 s** | **0.2783** | **0.3559** |

Longer windows improved both aggregate scores. This comparison is **not a single-variable ablation**: sampled training segments and selected per-fold postprocessing also changed, and the 1024-frame runs used batch size 2 versus batch size 4 for the smaller windows. The 1024-frame configuration is the current leading model for event F1, before the separate postprocessing comparison below.

### 2. Global cross-window proposal cap (frozen 1024-frame checkpoints)

| Maximum proposals per recording | Mean temporal mAP | Mean event F1@50 | Predicted events (seven recordings) |
|---:|---:|---:|---:|
| 1,000 | 0.2783 | 0.3559 | 1,618 |
| 2,000 | 0.2786 | 0.3559 | 1,618 |
| 3,000 | 0.2786 | 0.3559 | 1,618 |

Increasing the cap did not change event F1 or the predicted-event count with the fold-specific postprocessing settings. It had negligible effect on mAP. Thus the cap is unlikely to be the leading limitation in the current inference pipeline.

### 3. Confidence, minimum duration, and merging (frozen original 1024-frame checkpoints)

Sixty configurations were tested: scores `{0.05, 0.1, 0.2, 0.35, 0.5}` × minimum durations `{0, 2, 4}` seconds × merge gaps `{0, 1, 2, 4}` seconds.

| Decision rule | Mean event F1@50 | Mean frame precision | Mean frame recall | Predicted events |
|---|---:|---:|---:|---:|
| Original fold-specific settings | 0.3559 | 0.5091 | 0.8988 | 1,618 |
| **Fixed 0.20 confidence / 4 s minimum / 4 s gap** | **0.5627** | **0.7679** | **0.8126** | **657** |

The selected fixed rule reduces predicted events by approximately 59%, while reducing mean frame recall by approximately 8.6 percentage points. The 0.5627 event F1 is an **optimistically selected development score, not an independent test result**. All seven diagnostic six-video setting-selection runs happened to select the same rule, but that is not independent validation of the full pipeline.

### 4. Strict-proximity hard-negative sampling (1024 frames)

A second 1024-frame training run requested a 25% hard-negative fraction, using windows in which a valid male/partner pair was within 160 pixels for at least half of the segment. Only **18 of 693 negative segments (2.6%)** satisfied the criterion and were selected; this **did not realize a 25% hard-negative training mix**. The comparison is therefore a test of *strict proximity-biased sampling*, not a substantive 25% hard-negative ablation.

| Model | Mean temporal mAP | Event F1@50 with own fold-selected settings | Event F1@50 with fixed 0.20 / 4 s / 4 s settings | Fixed-setting frame precision | Fixed-setting predicted events |
|---|---:|---:|---:|---:|---:|
| Original 1024-frame | 0.2783 | **0.3559** | **0.5627** | 0.7679 | 657 |
| Strict-proximity hard negatives | **0.2986** | 0.3241 | 0.5278 | **0.7728** | 558 |

The hard-negative model improved temporal mAP, but did **not** improve event F1 under either postprocessing comparison. The fixed settings were originally chosen on development predictions from the baseline model and then applied unchanged to the hard-negative model; this is a development-side comparison.

### 5. Qualitative review of development recordings

Human inspection suggests several distinct error modes; these interpretations are **tentative** and are not ground-truth revisions:

- `0002_vid` near 00:26:19: an incomplete turn in one direction appeared before a subsequent circling event in the opposite direction, potentially explaining an overly long prediction.
- `0013_vid` near 05:09:43: circling seemed to begin before the model became confident, possibly related to changing fish separation.
- `0028_vid` near 04:16:55: the apparent prediction start was within a few seconds of the behavior onset, despite failing the required event-level IoU threshold.
- Isolated predictions: `0002_vid` near 06:20:35 showed close, largely parallel/antiparallel interactions without clear circling; `0013_vid` near 07:03:26 appeared to show real circling absent from annotations; `0028_vid` near 04:22:46 may have involved separation or reflection/tracking effects.

These examples motivate clearer annotation definitions (particularly incomplete rotations), expert adjudication of possible missing annotations, and review of tracking errors before changing labels.

## Leading development configuration

| Setting | Value |
|---|---|
| Architecture | ActionFormer (temporal localization) |
| Input | 132 engineered pair-motion/pose features |
| Temporal input window | 1024 frames (34.13 s at 30 FPS) |
| Inference stride | 512 frames |
| Cross-window hard NMS | IoU threshold 0.3; maximum 1,000 proposals |
| Confidence threshold | 0.20 |
| Minimum native interval duration | 4 seconds |
| Gap for joining retained predictions | 4 seconds |
| Status | **Frozen candidate for future evaluation; not an independently measured final-test score** |

## Reproduction

Run from the repository root with the project's locked Python environment (`uv sync --locked`). Pose/pair parquet files and finalized annotation CSVs must be available locally. Generated outputs, checkpoints and raw proposals are local and ignored by Git; **do not delete them**.

```bash
# Window-size cross-validation; output folders are resumable when manifests match.
uv run python -m experiments.actionformer.run_study window --window 512
uv run python -m experiments.actionformer.run_study window --window 1024

# Reuse existing 1024-frame checkpoints and cached full-video proposals.
uv run python -m experiments.actionformer.run_study proposal-caps --caps 1000 2000 3000
uv run python -m experiments.actionformer.run_study postprocess

# Refit with strict proximity-based negative sampling.
uv run python -m experiments.actionformer.run_study hard-negatives --fraction 0.25

# Fixed-setting diagnostic for the hard-negative model (requires its proposal-cap cache).
uv run python -m experiments.actionformer.run_study proposal-caps \
  --cv-dir outputs/actionformer_cv_window1024_hn25 \
  --output outputs/actionformer_cap_sweep_1024_hn25 --caps 1000
uv run python -m experiments.actionformer.run_study postprocess \
  --cv-dir outputs/actionformer_cv_window1024_hn25 \
  --cap-dir outputs/actionformer_cap_sweep_1024_hn25 \
  --output outputs/actionformer_postprocess_1024_hn25 \
  --scores 0.2 --durations 4 --merges 4

# Inspection and reproducible aggregate summary, no retraining.
uv run python -m experiments.actionformer.run_study review
uv run python -m experiments.actionformer.run_study summary
```

The original 256-frame development CV is stored locally in `outputs/actionformer_video_cv`. The unified runner uses `outputs/actionformer_cv_window256_reproduction` for fresh 256-frame runs to avoid changing older artifacts. Do **not** combine folds produced under different manifests, batch sizes or sampling rules.

Useful local result paths:

- `outputs/actionformer_video_cv/cv_results.csv` (256-frame baseline)
- `outputs/actionformer_cv_window512/cv_results.csv`
- `outputs/actionformer_cv_window1024/cv_results.csv`
- `outputs/actionformer_cap_sweep_1024/cap_results.csv`
- `outputs/actionformer_postprocess_1024/{grid_summary.csv,crossfit.csv,summary.md}`
- `outputs/actionformer_cv_window1024_hn25/cv_results.csv`
- `outputs/actionformer_postprocess_1024_hn25/summary.md`
- `outputs/actionformer_visual_review_1024/{review.html,cases.csv}`

A small, version-controllable aggregate of the completed results is kept in [`results/video_cv_summary.csv`](results/video_cv_summary.csv). Metrics in that file are per-video means unless the column is marked as a total; blank cells mean the metric was not available from the retained summary, **not zero**.

## Remaining limitations and next steps

1. **Annotation ambiguity:** Some predictions may correspond to genuine but unannotated behavior; incomplete turns and behavior onset need a shared team definition. No development labels were changed based on model output.
2. **Tracking quality:** Valid male/partner pair coverage was low in some recordings (`0007a_vid`: 23.9%; `0007b_vid`: 3.7%), limiting the information available to the engineered features.
3. **Selection bias:** The 0.5627 result was selected using development annotations; no fully nested tuning evaluation was performed.
4. **Window experiment confounding:** Batch size, training segments and postprocessing differed with window length, so the gain cannot be assigned to context length alone.
5. **Final-test procedure:** Before any new scoring of `0037_vid`, commit the model and fixed postprocessing settings, document the prior baseline exposure, and ensure the final-test code uses the fixed `0.20 / 4 s / 4 s` rule rather than medians of older fold-specific settings. No holdout score for this selected 1024-frame configuration is reported here.
