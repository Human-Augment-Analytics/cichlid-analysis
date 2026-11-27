# Shared-feature experiments

Compares the shared pair-motion features (`feature_engineering/pair_motion`) across
detectors with leave-one-video-out (LOVO) validation: each video is held out once
and the model is fitted on the others (today: fit `0028`, test `0031`, and the
reverse). Only the combined held-out predictions are scored, with the shared
`evaluation/` metrics. Videos and frame counts come from `VIDEO_FRAMES` in
`models/xgboost/features.py`; a new video there becomes a new fold. Generated
files go to `outputs/experiments/` and `outputs/predictions/lovo_<variant>/` (both
ignored).

Color and sex are not model inputs: the parquets have no color and sex flips
inside about a third of tracks.

## Run

```sh
# 1. features (angular + windowed), from the four pose parquets
uv run python -m feature_engineering.pair_motion.build_features --data-dir ~/Downloads
# 2. label-joined frame table: closest valid pair per frame + all-pair aggregates
uv run python -m experiments.shared_features.dataset --data-dir ~/Downloads
# 3. the existing detectors' features on the same frames (for the their_* variants)
uv run python -m experiments.shared_features.baselines --data-dir ~/Downloads
# 4. circling vs background feature report
uv run python -m experiments.shared_features.report
# 5. LOVO experiments (see --list for variants; xgb_* needs `brew install libomp`)
uv run python -m experiments.shared_features.lovo --variants all
# 6. shipped detectors (in-sample) next to the LOVO results
uv run python -m experiments.shared_features.compare
```

Other videos: set `CIRCLING_VIDEOS` (annotation IDs), `CIRCLING_DATA_DIR` (their
`<video>.parquet` and `<video>_pairs.parquet`) and `CIRCLING_OUTPUTS` (a separate output
root). Build the pair-motion features with `build_features --videos …`, then steps 2, 3
and 5 as above. `lovo --holdout 0028_vid` keeps that video out of every fold and tests it
last on a model of all other videos (the issue #19 protocol).
`lovo_folds.sbatch` runs the twelve `xgb_*` variants this way on PACE, with and without
reviewed negatives; `generalizability_report.py` shows the results.

Results: `outputs/experiments/lovo/summary.md` and `results.csv`; per-variant fold
settings, permutation importance and logistic coefficients are in
`outputs/experiments/lovo/folds/`. The side-by-side table is
`outputs/experiments/comparison.md`.

## What is fitted inside a fold

Imputer, scaler, model weights, and the threshold / minimum-span-length
postprocessing. Postprocessing is chosen on blocked out-of-fold scores inside the
training video (4 contiguous blocks with a 300-frame buffer), maximizing event
F1@50. The rule variant grid-searches `w4s_close_frac` and `w4s_orbit_turns_abs`
cutoffs on the training video only. Frames without a valid pair are scored as not
circling. A valid pair is one the rule-based cleaning keeps.

XGBoost variants give reviewed hard-negative frames a training weight of 2.
Those frames are already in the experiment's frame table and have weight 1 in a
baseline run. Finalized positives keep their original weight. The weights are
applied only to the fit rows of each outer and inner fold. To compare with the
unweighted baseline, run with `--no-annotated-hard-negatives` and a separate
`--out-dir` for its results.

## Variants

Learners: `logreg_*`, `hgb_*` (scikit-learn gradient boosting, no extra install),
`xgb_*` (XGBoost; needs `brew install libomp` on macOS) and `rule_shared` (a
two-threshold rule). For each learner:

- `all`, `all_agg` (adds all-pair aggregates), `v1_features` (first-half features only)
- `no_<group>`: drop one feature group; `logreg_only_<group>`: train on one group
- `their_lr102`, `their_win155`, `their_both`: the feature sets used by the shipped
  logistic regression and XGBoost models, recomputed on the same rows
- `mine_plus_their`: shared features plus the shipped models' features

Groups: `distance_proximity`, `orientation_facing`, `speed_closing`,
`angular_movement`, `temporal_consistency`. Run `lovo --list` for the full list.

## Limitations

- Labels came from a rule-based proposal that was then hand-verified, so distance and
  orbit features may be favored. Treat feature rankings as evidence, not proof.
- Two videos mean two folds, 139 events and no confidence intervals; differences under
  about 0.04 F1 are noise. More videos should become extra folds.
- The shipped detectors in `models/` are unchanged and are not run under LOVO. Their
  feature sets are evaluated here with this folder's learners instead. Their published
  scores are in-sample, so they are not comparable with the held-out numbers.
- `rule_shared` has no hysteresis or gap merging, so it scores below the shipped rule-based
  detector; that is a postprocessing gap, not a feature gap.
