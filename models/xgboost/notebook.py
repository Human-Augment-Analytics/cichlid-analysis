"""Train and evaluate XGBoost: uv run marimo edit models/xgboost/notebook.py."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="XGBoost circling classifier")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

    # Support both marimo and direct execution from any working directory.
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from models.rule_based.circling_cleaning import default_data_dir
    from models.xgboost.features import FEATURES, read_spans
    from models.xgboost.train import (OUTPUT_DIR, PREDICTION_DIR, evaluate_full_videos,
                                     fit_model, frame_metrics, prepare_dataset,
                                     save_artifacts, split_dataset)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # XGBoost circling classifier

    Each row is a **121-frame sliding window**, from center −60 to center +60.
    The label is whether the **center (median) frame** belongs to a usable positive
    interval in `ground_truth_annotations/finalized_annotations/merged_spans.csv`.

    The inputs match `models/rule_based/detector.py`: remove reflections, duplicate
    fish and keypoint jumps, then pick the closest retained pair in each frame.
    Use its **centroid distance (px)** and **orbital rotation rate (rad/s)**. Flatten
    the window into 242 features: 121 distances followed by 121 rates. Track IDs,
    video IDs, frame numbers and annotations are excluded from model inputs.

    Missing distance stays NaN; missing or implausible rotation rate (|rate| >15)
    becomes zero, matching the rule-based baseline. Context outside either video
    is padded with NaN, so every labeled center remains eligible. XGBoost handles
    NaN directly ([API documentation](https://xgboost.readthedocs.io/en/stable/python/python_api.html)).
    """)
    return


@app.cell
def _():
    data_dir = default_data_dir(ROOT)
    spans = read_spans()
    mo.md(f"**Data:** `{data_dir}` · **Artifacts:** `{OUTPUT_DIR}`\n\n"
          f"Usable circling segments: {sum(map(len, spans.values()))} across both videos.")
    return data_dir, spans


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Build the training rows

    Keep all positive frames. For each positive segment of inclusive length L,
    include ceil(L/2) frames immediately before and after it, clipped to the video.
    Overlapping buffers are unioned; any frame in a positive segment stays positive.
    Each `(video, center frame)` appears once. This gives roughly equal classes;
    overlapping buffers can reduce the negative count. All 121 context frames come
    from the original signal series, even when their centers are not sampled.

    Loading and cleaning the pose/pair Parquets can take a minute on the first run.
    """)
    return


@app.cell
def _(data_dir, spans):
    series, sampled, x = prepare_dataset(data_dir, spans)
    sampled.groupby(["VideoID", "circling"]).size().rename("frames").reset_index()
    return sampled, series, x


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Split 70% training / 30% holdout and fit

    Stratify by the center-frame label with seed 42. Fit 300 histogram-based trees
    (depth 4, learning rate 0.05) on **training rows only**, with a fixed threshold
    of 0.5. No tuning or early stopping uses the holdout labels.

    Adjacent windows can share 120 of 121 frames across this random split. The
    holdout measures sampled-frame performance; it does not establish performance
    on independent events or unseen videos.
    """)
    return


@app.cell
def _(sampled):
    dataset, train_ids, test_ids = split_dataset(sampled)
    dataset.groupby(["split", "circling"]).size().rename("frames").reset_index()
    return dataset, test_ids, train_ids


@app.cell
def _(dataset, train_ids, x):
    model = fit_model(x[train_ids], dataset.circling.to_numpy()[train_ids])
    return (model,)


@app.cell
def _(dataset, model, test_ids, x):
    test_labels = dataset.circling.to_numpy()[test_ids]
    test_scores = model.predict_proba(x[test_ids])[:, 1]
    holdout = frame_metrics(test_labels, test_scores)
    mo.vstack([mo.md("## Holdout evaluation"), pd.DataFrame([holdout])])
    return holdout, test_labels, test_scores


@app.cell(hide_code=True)
def _(holdout, test_labels, test_scores):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(test_labels, test_scores >= 0.5,
                                           display_labels=["background", "circling"],
                                           ax=_axes[0], colorbar=False)
    _precision, _recall, _ = precision_recall_curve(test_labels, test_scores)
    _axes[1].plot(_recall, _precision, label=f"AP = {holdout['average_precision']:.3f}")
    _axes[1].axhline(test_labels.mean(), linestyle=":", label="Constant-score baseline")
    _axes[1].set(xlabel="Recall", ylabel="Precision", title="Holdout precision–recall",
                 xlim=(0, 1), ylim=(0, 1.02))
    _axes[1].legend()
    _fig
    return


@app.cell(hide_code=True)
def _(model):
    importance = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    _top = importance.nlargest(15, "importance").sort_values("importance")
    _fig, _ax = plt.subplots(figsize=(10, 5), layout="constrained")
    _ax.barh(_top.feature, _top.importance)
    _ax.set(xlabel="Normalized gain importance", title="Top 15 window features")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Evaluate every frame of both full videos

    Keep the exact model evaluated on the holdout. Score all 1,080,175 frames of
    0028 and 1,080,398 frames of 0031 in bounded batches, including boundary frames
    and frames without retained pairs.

    **Post-processing:** threshold scores at 0.5, then take a centered **30-frame
    rolling majority vote**. A frame is circling when at least 50% of the window
    predictions are positive (15 of 30; ties included). The even window uses
    offsets −15 through +14; video boundaries use only available frames.
    Apply the vote across the full video after joining scoring batches, then
    export contiguous positive spans. This bridges short gaps and removes brief
    isolated bursts; it does not guarantee merging every gap of 30 frames.

    Compare raw and processed metrics below. Processed AP and ROC AUC use the
    rolling vote fraction; it is not a calibrated probability. Holdout metrics
    remain on raw predictions because sampled holdout rows are not contiguous.

    This evaluation **includes training frames**. It assumes all unannotated frames
    are background; incomplete annotations can make real circling detections count
    as false positives. The balanced training scores are not calibrated to the
    much lower full-video prevalence. Segment metrics reuse `evaluation/metrics.py`.
    """)
    return


@app.cell
def _(model, series, spans):
    full = evaluate_full_videos(model, series, spans)
    _variants = {"raw": full, "30-frame majority": full["postprocessed"]}
    mo.vstack([mo.md("### Full-video frame metrics"),
               pd.concat({name: pd.DataFrame.from_dict(result["frame_metrics"], orient="index")
                          for name, result in _variants.items()}, names=["prediction", "video"]),
               mo.md("### Full-video segment metrics"),
               pd.concat({name: pd.DataFrame.from_dict({**result["segment_metrics"],
                                                       "overall": result["overall_segments"]}, orient="index")
                          for name, result in _variants.items()}, names=["prediction", "video"])])
    return (full,)


@app.cell
def _(data_dir, dataset, full, holdout, model, spans, test_ids, test_scores, x):
    report = save_artifacts(model, dataset, x, test_ids, test_scores, holdout,
                            full, spans, data_dir=data_dir)
    mo.md(f"""
    ## Saved artifacts

    - `{OUTPUT_DIR}/model.ubj`: the model fitted on the 70% training split.
    - `sampled_frames.parquet`: center frames, labels, split, and all 242 features.
    - `test_predictions.parquet`: holdout scores and predictions.
    - `feature_importance.csv` and `metrics.json`: importance, configuration and evaluations.
    - `{OUTPUT_DIR}/scores/<video>_scores.parquet`: labels, raw scores/predictions, vote fractions and processed predictions.
    - `{PREDICTION_DIR}/<video>.csv`: processed spans for the shared evaluation notebook.
    """)
    return (report,)


if __name__ == "__main__":
    app.run()
