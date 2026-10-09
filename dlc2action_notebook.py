"""Train and inspect a small DLC2Action circling segmentation experiment."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")

with app.setup:
    import json
    from pathlib import Path

    import marimo as mo
    import polars as pl
    from matplotlib.figure import Figure
    from temporal_metrics import temporal_metrics

    from dlc2action_segmentation import (
        DEFAULT_OUTPUT, LABEL_POLICY, read_spans, run_test, source_hash,
    )

    ROOT = Path(__file__).resolve().parent
    SPAN_DIR = ROOT / "outputs" / "merged_spans"


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    # DLC2Action: circling action segmentation

    A small **MS-TCN temporal segmentation** experiment using all rows in
    `outputs/merged_spans`, with the existing 102 pose and pair features.
    **Train: 0028_vid. Held-out test: 0031_vid.** Whole videos stay in separate
    splits; preprocessing uses training frames only.

    {LABEL_POLICY}

    The model sees contiguous excerpts extending 150 frames around each candidate
    span by default. Overlapping excerpts are united, then split into 512-frame
    windows. This is a smoke test on candidate-centered excerpts, not full-video
    validation. The notebook uses DLC2Action's
    [model API](https://github.com/amathislab/DLC2action) with a custom PyTorch
    training loop, two classes, and cross-entropy at each refinement stage.
    """)
    return


@app.cell(hide_code=True)
def _():
    for _path in sorted(SPAN_DIR.glob("*_merged_spans.csv")):
        mo.watch.file(_path)
    source_spans = read_spans(SPAN_DIR)
    current_hash = source_hash(SPAN_DIR)
    mo.vstack([
        mo.md("## Source rows\n`Verified=false` and blank rows are masked, not used as negative examples."),
        mo.ui.table(source_spans, selection=None, page_size=10),
    ])
    return current_hash, source_spans


@app.cell(hide_code=True)
def _():
    experiment = mo.ui.dictionary({
        "epochs": mo.ui.number(start=1, stop=100, value=5, label="Training epochs"),
        "context": mo.ui.number(start=30, stop=900, step=30, value=150,
                                label="Context on each side (frames)"),
    }).form(submit_button_label="Run segmentation test", show_clear_button=True)
    mo.vstack([
        mo.md("## Run or review\nSaved results load automatically. Submit to train again on CPU "
              "and replace the artifacts in `outputs/dlc2action_test/`. The test video is never "
              "used for model selection. Defaults are five epochs and seed 42."),
        experiment,
    ])
    return (experiment,)


@app.cell
def _(experiment):
    if experiment.value is not None:
        with mo.status.spinner(title="Extracting features, training DLC2Action and predicting the held-out video…"):
            run_test(epochs=int(experiment.value["epochs"]), context=int(experiment.value["context"]))
    _report_file = DEFAULT_OUTPUT / "metrics.json"
    mo.stop(not _report_file.exists(), mo.md("Submit the form above to run the first experiment."))
    report = json.loads(_report_file.read_text())
    predictions = pl.read_parquet(DEFAULT_OUTPUT / "test_predictions.parquet")
    report["test"].update(temporal_metrics(predictions))
    segments = pl.read_csv(DEFAULT_OUTPUT / "predicted_segments.csv")
    # Use the exact annotation snapshot used for this experiment in the plots.
    experiment_spans = pl.read_csv(DEFAULT_OUTPUT / "span_manifest.csv")
    return experiment_spans, predictions, report, segments


@app.cell(hide_code=True)
def _(current_hash, report):
    _metric_names = ["precision", "recall", "f1", "average_precision", "roc_auc"]
    _summary = pl.DataFrame({"metric": _metric_names, "value": [report["test"][k] for k in _metric_names]})
    mo.vstack([
        mo.callout("Source CSVs changed since this run. Submit the form to retrain. "
                   "Plots below use the saved annotation snapshot.", kind="warn")
        if current_hash != report["source_sha256"] else mo.md("Results match the current merged span CSVs."),
        mo.md(f"## Held-out results\n**{report['train_frames']:,} training frames** · "
              f"**{report['test']['frames']:,} evaluated test frames** · "
              f"{report['test_ignored_frames']:,} ignored test frames · "
              f"{report['epochs']} epochs · DLC2Action {report['dlc2action_version']}\n\n"
              f"Constant-score AP baseline: **{report['constant_score_average_precision']:.3f}**. "
              f"Predicted positive segments at 0.5: **{report['predicted_segments']}**."),
        mo.ui.table(_summary, selection=None),
        mo.callout(report["limitations"], kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(report):
    _names = ["mof", "edit_score", "edit_score_with_background",
              "segmental_f1_10", "segmental_f1_25", "segmental_f1_50"]
    mo.vstack([
        mo.md("### Temporal segmentation metrics (percent)\n"
              "**MoF** is frame accuracy including background. **Edit** compares collapsed "
              "action sequences using normalized Levenshtein distance. **F1@10/25/50** "
              "matches predicted and labeled segments one-to-one at each IoU threshold. "
              "Edit and segmental F1 exclude background; Edit including background is also shown. "
              "Ignored frames and excerpt gaps split evaluation runs. Edit is averaged over "
              "runs; segment counts are pooled. With only one action class, foreground-only "
              "Edit mostly measures fragmentation, not action ordering."),
        mo.ui.table(pl.DataFrame({"metric": _names,
                                  "percent": [report["test"][k] for k in _names]}), selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(report):
    _fig = Figure(figsize=(9, 2.5), layout="constrained")
    _axis = _fig.subplots()
    _axis.plot([r["epoch"] for r in report["history"]],
               [r["train_loss"] for r in report["history"]], marker="o")
    _axis.set(xlabel="Epoch", ylabel="Training loss", title="Mean weighted cross-entropy across stages")
    _fig
    return


@app.cell(hide_code=True)
def _(experiment_spans):
    _test_rows = experiment_spans.filter(pl.col("split") == "test")
    _options = {
        f"{r['Merge ID']} · {r['Start Frame']:,}–{r['End Frame']:,} · {r['label_policy']}": r["Merge ID"]
        for r in _test_rows.iter_rows(named=True)
    }
    selected_span = mo.ui.dropdown(_options, value=next(iter(_options)),
                                    allow_select_none=False, label="Held-out span")
    mo.vstack([mo.md("## Inspect temporal segmentation"), selected_span])
    return (selected_span,)


@app.cell(hide_code=True)
def _(experiment_spans, predictions, report, selected_span):
    _row = experiment_spans.filter(pl.col("Merge ID") == selected_span.value).row(0, named=True)
    _start = max(0, _row["Start Frame"] - report["context_frames"])
    _end = _row["End Frame"] + report["context_frames"]
    selected_predictions = predictions.filter(pl.col("FrameNum").is_between(_start, _end))
    _fig = Figure(figsize=(14, 5), layout="constrained")
    _axes = _fig.subplots(3, 1, sharex=True, height_ratios=[1, 1, 2])
    _frames = selected_predictions["FrameNum"].to_numpy()
    _labels = selected_predictions["label"].to_numpy()
    _score = selected_predictions["circling_score"].to_numpy()
    for _axis, _values, _title, _color in (
        (_axes[0], _labels == 1, "Reviewed circling / assumed background", "#0d9488"),
        (_axes[1], _score >= 0.5, "MS-TCN prediction at 0.5", "#7c3aed"),
    ):
        _axis.fill_between(_frames, 0, _values, step="mid", color=_color)
        _axis.fill_between(_frames, 0, 1, where=_labels == -100, step="mid", color="#9ca3af", alpha=0.5)
        _axis.set(ylim=(0, 1.1), yticks=[0, 1], title=_title)
    _axes[2].plot(_frames, _score, color="#7c3aed")
    _axes[2].axhline(0.5, color="#6b7280", linestyle=":")
    _axes[2].set(ylim=(0, 1), ylabel="Circling score", xlabel="Original video frame (zero-based)")
    for _axis in _axes:
        _axis.axvline(_row["Start Frame"], linestyle="--", color="#e11d48", alpha=0.6)
        _axis.axvline(_row["End Frame"] + 1, linestyle="--", color="#e11d48", alpha=0.6)
        _axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    mo.vstack([
        mo.md("Gray shading marks ignored labels; dashed lines mark the selected CSV span. "
              "Scores are not calibrated probabilities. Frame endpoints are inclusive."),
        _fig,
    ])
    return (selected_predictions,)


@app.cell(hide_code=True)
def _(segments, selected_predictions):
    mo.stop(selected_predictions.is_empty(), mo.md("No predictions to preview in this excerpt."))
    preview_segments = segments.filter(
        pl.col("video").is_in(selected_predictions["video"].unique().to_list()),
        pl.col("start_frame") <= selected_predictions["FrameNum"].max(),
        pl.col("end_frame") >= selected_predictions["FrameNum"].min(),
    ).sort("start_frame")
    mo.stop(preview_segments.is_empty(), mo.md("No predicted circling segments overlap this excerpt."))
    _options = {
        f"{r['start_frame']:,}–{r['end_frame']:,} · {r['frames']} frames · "
        f"mean score {r['mean_score']:.3f}": i
        for i, r in enumerate(preview_segments.iter_rows(named=True))
    }
    prediction_segment = mo.ui.dropdown(
        _options, value=next(iter(_options)), allow_select_none=False,
        label="Predicted segment",
    )
    prediction_buffer = mo.ui.slider(
        start=0, stop=300, step=30, value=30, show_value=True,
        label="Preview buffer on each side (frames)",
    )
    mo.vstack([
        mo.md("### Preview model prediction segments\nChoose a predicted positive run "
              "overlapping the displayed excerpt. The preview includes the complete run, "
              "plus the selected buffer, with pose points and TrackIDs."),
        mo.hstack([prediction_segment, prediction_buffer], justify="start", gap=2),
    ])
    return prediction_buffer, prediction_segment, preview_segments


@app.cell(hide_code=True)
def _(prediction_buffer, prediction_segment, preview_segments):
    from merged_span_preview import render_merged_preview

    _segment = preview_segments.row(prediction_segment.value, named=True)
    _video = ROOT / "data" / f"{_segment['video']}.mp4"
    mo.stop(not _video.exists() or not _video.with_suffix(".parquet").exists(),
            mo.md(f"Put `{_video.name}` and its pose Parquet file in `data/` to preview this segment."))
    with mo.status.spinner(title="Rendering predicted segment preview…"):
        _preview, _start, _end = render_merged_preview(
            _video, _segment["start_frame"], _segment["end_frame"],
            prediction_buffer.value, DEFAULT_OUTPUT / "previews",
        )
    mo.vstack([
        mo.md(f"**{_segment['video']}** · predicted circling: "
              f"**{_segment['start_frame']:,}–{_segment['end_frame']:,}** · "
              f"mean score **{_segment['mean_score']:.3f}**  \n"
              f"Preview frames {_start:,}–{_end:,} (inclusive). "
              "The video marks prediction frames as `span` and surrounding frames as `buffer`."),
        mo.video(str(_preview), width="80%"),
    ])
    return


@app.cell(hide_code=True)
def _(segments, selected_predictions):
    mo.vstack([
        mo.md("## Predicted segments\nPositive runs within the held-out excerpts. "
              "Gaps between excerpts are never joined. No duration filtering is applied."),
        mo.ui.table(segments, selection=None, page_size=10),
        mo.accordion({"Selected span: per-frame predictions": mo.ui.table(
            selected_predictions, selection=None, page_size=10)}),
        mo.md("Artifacts: `metrics.json`, `report.md`, `span_manifest.csv`, "
              "`test_predictions.parquet`, `predicted_segments.csv`, `model.pt` and "
              "`preprocessing.joblib` in `outputs/dlc2action_test/`."),
    ])
    return


if __name__ == "__main__":
    app.run()
