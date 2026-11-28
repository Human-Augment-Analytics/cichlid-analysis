# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "matplotlib>=3.11.2",
#     "polars",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    from io import BytesIO

    import marimo as mo
    import polars as pl
    from matplotlib.figure import Figure

    from evaluation.generalizability import FOLDS_DIR, MODELS, SEED, fold_dir, fold_results, split_videos

    METRICS = ("mof", "edit_score", "f1@10", "f1@25", "f1@50", "frame_precision", "frame_recall",
               "frame_f1", "frame_iou", "direction_f1", "direction_accuracy", "cycle_mae")


@app.cell(hide_code=True)
def _():
    holdout, cv_videos = split_videos()
    mo.md(f"""
    # Generalizability report

    How well each model does on a video it never trained on.

    - **Holdout.** `random.Random({SEED}).choice(...)` over the sorted annotated videos picks
      **`{holdout}`**. No fold trains or is evaluated on it. It is scored only once, in the
      `{holdout}` columns of the results table.
    - **Leave one video out.** Each of the other {len(cv_videos)} videos is the test video of one
      fold. Models train on the remaining {len(cv_videos) - 1} only (segment sampling, early stopping and
      postprocessing selection included), then predict the whole test video.
    - **Scores.** Each fold is scored on its test video with the metrics of
      `evaluation/evaluate.py`; the tables below average the folds, so every video counts equally.
      Std is across folds.

    Run folds with `evaluation/generalizability.py`; each fold is independent, so they can run as a
    Slurm array:

    ```
    python -m evaluation.generalizability --list
    python -m evaluation.generalizability --fold-index $SLURM_ARRAY_TASK_ID --data-dir <pose data>
    ```

    Pose inputs `<video>.parquet` and `<video>_pairs.parquet` are needed for every fold video;
    `--alias VIDEO=STEM` maps an annotation id to a differently named file. Results land in
    `{FOLDS_DIR.relative_to(FOLDS_DIR.parents[2])}/<test video>/`.
    """)
    return cv_videos, holdout


@app.cell(hide_code=True)
def _(cv_videos):
    import json

    _rows = []
    for _index, _video in enumerate(cv_videos):
        _path = fold_dir(_video) / "fold.json"
        _status = json.loads(_path.read_text()) if _path.exists() else {"models": {}}
        _rows.append({
            "fold": _index, "test_video": _video,
            "train_videos": ", ".join(_status.get("train_videos", [])) or "—",
            **{_name: ("✓" if (fold_dir(_video) / _entry.get("predictions", "")).is_file() else "…")
               if (_entry := _status["models"].get(_name)) else "" for _name in MODELS},
        })
    mo.vstack([
        mo.md("## Fold status\n\n✓ predictions written · … started, not finished · blank not run"),
        mo.ui.table(pl.DataFrame(_rows), selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(cv_videos, holdout):
    folds = fold_results()
    mo.stop(folds.is_empty(), mo.md("No finished folds yet in `outputs/generalizability/folds/`."))
    _outside = sorted(set(folds["test_video"]) - set(cv_videos))
    mo.stop(holdout in _outside, mo.md(f"**`{holdout}` is the holdout video but has fold results.**"))
    folds = folds.filter(pl.col("test_video").is_in(cv_videos))
    _leaky = folds.filter(~pl.col("standard_train_set"))["test_video"].unique().to_list()
    folds = folds.filter(pl.col("standard_train_set"))
    mo.stop(folds.is_empty(), mo.md(
        f"Only debugging folds found (not trained on the other {len(cv_videos) - 1} videos): {_leaky}."))
    metrics = [_m for _m in METRICS if _m in folds.columns and folds[_m].fill_nan(None).drop_nulls().len()]
    return folds, metrics


@app.cell(hide_code=True)
def _(cv_videos, folds, holdout, metrics):
    from evaluation.generalizability import GENERALIZABILITY_DIR

    _scores = list(dict.fromkeys(("f1@50", "frame_f1", "frame_precision", "frame_recall", *metrics)))
    _columns = ["model", "source", "reviewed_negatives", "test_video", "pred_segments", "gt_segments",
                "wall_seconds", *_scores]
    _parts = [folds.with_columns(source=pl.lit("fold run"))]
    _plain = GENERALIZABILITY_DIR / "xgboost_plain_fold_scores.csv"
    if _plain.exists():
        _parts.append(pl.read_csv(_plain, infer_schema_length=None).with_columns(
            source=pl.lit("plain xgboost"),
            reviewed_negatives=pl.when(pl.col("reviewed_negatives").cast(pl.String).str.to_lowercase() == "true")
            .then(pl.lit("with")).otherwise(pl.lit("without"))))
    _rule = GENERALIZABILITY_DIR / "rule_based_scores.csv"
    if _rule.exists():
        _parts.append(pl.read_csv(_rule, infer_schema_length=None)
                      .with_columns(source=pl.lit("rule-based"), reviewed_negatives=pl.lit("n/a")))
    for _setting in ("with", "without"):
        for _path in sorted((GENERALIZABILITY_DIR / "shared_features" / f"run_{_setting}"
                             / "experiments" / "lovo").glob("*/results.csv")):
            _parts.append(pl.read_csv(_path).rename({"variant": "model", "video": "test_video"})
                          .with_columns(source=pl.lit("shared features"), reviewed_negatives=pl.lit(_setting)))
    _all = pl.concat([_part.select([pl.col(_c) if _c in _part.columns else pl.lit(None).alias(_c)
                                    for _c in _columns]) for _part in _parts], how="vertical_relaxed")
    _all = _all.with_columns(pl.col(_scores).cast(pl.Float64).fill_nan(None))
    _keys = ["model", "source", "reviewed_negatives"]
    results = _all.filter(pl.col("test_video").is_in(cv_videos)).sort(*_keys, "test_video")
    _holdout = _all.filter(pl.col("test_video") == holdout).select(
        *_keys, pl.col("f1@50").alias(f"f1@50_{holdout}"), pl.col("frame_f1").alias(f"frame_f1_{holdout}"))
    combined = (
        results.group_by(_keys)
        .agg(pl.len().alias("folds"),
             *[pl.col(_m).mean() for _m in _scores],
             pl.col("f1@50").std().alias("f1@50_std"), pl.col("frame_f1").std().alias("frame_f1_std"),
             pl.col("pred_segments").sum(), pl.col("gt_segments").sum(),
             pl.col("wall_seconds").mean().round(0).alias("mean_wall_s"))
        .join(_holdout, on=_keys, how="left")
        .with_columns(complete=pl.col("folds") == len(cv_videos))
        .select(*_keys, "folds", "f1@50", "f1@50_std", "frame_f1", "frame_f1_std", "frame_precision",
                "frame_recall", f"f1@50_{holdout}", f"frame_f1_{holdout}", "pred_segments", "gt_segments",
                *[_m for _m in _scores if _m not in ("f1@50", "frame_f1", "frame_precision", "frame_recall")],
                "mean_wall_s", "complete")
        .sort(["complete", "f1@50"], descending=True)
    )
    mo.vstack([
        mo.md(f"""
        ## Results on unseen test videos

        One row for each model and reviewed-negatives setting, highest mean F1@50 first. Each score
        is the mean over the {len(cv_videos)} folds; `_std` is the standard deviation across folds.
        A model is `complete` when all {len(cv_videos)} folds have finished. The rows come from
        four runs on the same folds:

        - **fold run.** `evaluation/generalizability.py` as a Slurm array on PACE.
          `reviewed_negatives` is the setting `fold.json` records for each model (`unknown` when
          the record predates it). The current `xgboost_*` train CLIs add the reviewed negatives by
          default; `lrcv_*` and ActionFormer have no such option. `mean_wall_s` is the mean seconds
          for one fold: training on the fold's training videos plus scoring the test video, without
          shared feature preparation.
        - **rule-based.** `models/rule_based/predict.py`: the refined rule of
          `models/rule_based/detector.py` (pair closer than 160 px; start at 0.75 turns in 4 s, continue
          above 0.4 turns; pauses under 3 s joined; spans of 2 s or more). It has no training step, so
          the score of a fold is its score on the test video, and `reviewed_negatives` is `n/a`. Its
          thresholds were set on `0031_vid`. Annotators could start from candidate spans of a
          different rule that uses similar signals (pair distance and orbital rotation; see
          `ground_truth_annotations/README.md`). Both can make its scores higher.
        - **plain xgboost.** `models/xgboost` (`train.py --skip-full-video`, then `predict.py`).
          `with`: the frames in `hard_negatives_annotated.csv` are added as training rows with label
          0 and weight 1.
        - **shared features.** The twelve `xgb_*` variants of `experiments/shared_features/lovo.py`
          (`lovo_folds.sbatch`). The training rows are all frames where the closest pair survives the
          rule-based cleaning, with no segment sampling. Frames without a valid pair count as not
          circling. XGBoost has 300 trees, depth 4, learning rate 0.08 and no early stopping. The
          threshold and minimum span length are chosen on out-of-fold scores from 4 contiguous
          blocks in each training video (300-frame buffer), then a centered 30-frame majority vote
          is applied. `with`: the reviewed-negative frames are already rows; they get weight 2.

        The `with` setting is not the same in every run that has it (fold run and plain xgboost add
        unsampled reviewed-negative frames as training rows; shared features reweights rows it already
        has), so compare `with` and `without` only inside one source.

        **`{holdout}` columns** are the final test: one model trained on all {len(cv_videos)} fold
        videos. The fold run did not score `{holdout}`, so those cells are empty for it.
        """),
        mo.ui.table(combined, selection=None, pagination=False),
    ])
    return combined, results


@app.cell(hide_code=True)
def _(combined, results):
    from matplotlib.lines import Line2D

    _colors = {"with": "#7954c7", "without": "#2a9d8f", "unknown": "#9ca3af", "n/a": "#d97706"}
    _markers = {"fold run": "o", "rule-based": "D", "plain xgboost": "s", "shared features": "^"}
    _rows = combined.sort("f1@50", descending=False, nulls_last=False)
    _figure = Figure(figsize=(10, 0.32 * _rows.height + 1.6), layout="constrained")
    _axis = _figure.subplots()
    for _y, (_model, _source, _setting, _mean) in enumerate(
            _rows.select("model", "source", "reviewed_negatives", "f1@50").iter_rows()):
        _scores = results.filter((pl.col("model") == _model) & (pl.col("source") == _source)
                                 & (pl.col("reviewed_negatives") == _setting))["f1@50"].to_list()
        _axis.scatter(_scores, [_y] * len(_scores), color=_colors[_setting], marker=_markers[_source],
                      alpha=0.55, s=24, zorder=3)
        _axis.plot([_mean, _mean], [_y - 0.3, _y + 0.3], color="#1f2937", linewidth=2.5, zorder=4)
    _labels = [f"{_m} ({_s})" for _m, _s in _rows.select("model", "reviewed_negatives").iter_rows()]
    _axis.set(yticks=range(len(_labels)), yticklabels=_labels, xlim=(0, 1), ylim=(-0.7, len(_labels) - 0.3),
              xlabel="F1@50 on the unseen test video (dots: folds · bar: mean)")
    _axis.grid(axis="x", color="#e5e7eb")
    _axis.spines[["top", "right", "left"]].set_visible(False)
    _axis.tick_params(axis="y", length=0)
    _handles = ([Line2D([], [], color=_c, marker="o", linestyle="",
                        label="no training" if _s == "n/a" else f"{_s} reviewed negatives")
                 for _s, _c in _colors.items()]
                + [Line2D([], [], color="#6b7280", marker=_m, linestyle="", label=_s) for _s, _m in _markers.items()])
    _figure.legend(handles=_handles, loc="outside lower center", ncols=4, frameon=False)
    _buffer = BytesIO()
    _figure.savefig(_buffer, format="png", dpi=150)
    mo.vstack([
        mo.md("### Spread across test videos\n\nOrdered by mean F1@50, highest at the top. "
              "Color: reviewed-negatives setting. Marker: source run."),
        mo.image(_buffer.getvalue(), width="80%", alt="Per-fold F1@50 for each model, ordered by mean"),
        mo.accordion({"Per-fold scores (one row for each model, setting and test video)":
                      mo.ui.table(results.drop("wall_seconds"), selection=None, pagination=False)}),
    ])
    return


if __name__ == "__main__":
    app.run()
