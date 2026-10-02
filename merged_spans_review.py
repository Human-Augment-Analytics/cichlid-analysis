# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "imageio-ffmpeg>=0.6.0",
#     "marimo>=0.24.0",
#     "matplotlib>=3.11.2",
#     "opencv-python-headless",
#     "polars",
# ]
# ///

"""Review merged circling spans against their source videos."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")

with app.setup:
    from io import BytesIO
    from pathlib import Path

    import cv2
    import marimo as mo
    import polars as pl

    ROOT = Path(__file__).resolve().parent


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Review merged circling spans

    Choose a merged span file and its source video. Select a span to see its
    position in the full video and play a buffered preview with keypoints and
    a TrackID legend. Frame endpoints are zero-based and inclusive; End Time is
    the boundary after the final frame.
    """)
    return


@app.cell(hide_code=True)
def _():
    _span_files = sorted((ROOT / "outputs" / "merged_spans").glob("*_merged_spans.csv"))
    _video_files = sorted((ROOT / "data").glob("*_vid.mp4"))
    mo.stop(not _span_files, mo.md("Run `uv run python merge_intervals.py` to create the merged span files."))
    mo.stop(not _video_files, mo.md("Put the source videos in `data/` to preview the spans."))

    spans_file = mo.ui.dropdown(
        options={path.name: str(path) for path in _span_files},
        value=_span_files[0].name, label="Merged spans CSV", allow_select_none=False,
    )
    video_file = mo.ui.dropdown(
        options={path.name: str(path) for path in _video_files},
        value=_video_files[0].name, label="Source video", allow_select_none=False,
    )
    mo.hstack([spans_file, video_file], justify="start", gap=2)
    return spans_file, video_file


@app.cell(hide_code=True)
def _(spans_file, video_file):
    _csv_path = mo.watch.file(Path(spans_file.value))
    video_path = Path(video_file.value)
    _all_spans = pl.read_csv(str(_csv_path), schema_overrides={
        "Video ID": pl.String, "Merge ID": pl.String,
        "Start Frame": pl.Int64, "End Frame": pl.Int64,
    })
    spans = _all_spans.filter(pl.col("Video ID") == video_path.stem).sort("Start Frame")
    mo.stop(spans.is_empty(), mo.md(
        f"`{_csv_path.name}` has no spans for `{video_path.name}`. Select the matching video."
    ))
    mo.vstack([
        mo.md(f"**{spans.height} merged spans** for `{video_path.name}`"),
        mo.ui.table(spans, selection=None, pagination=True, page_size=10),
    ])
    return spans, video_path


@app.cell(hide_code=True)
def _(spans):
    _options = {
        f"{row['Merge ID']} · {row['Start Frame']:,}–{row['End Frame']:,} · {row['Source']}":
            row["Merge ID"]
        for row in spans.iter_rows(named=True)
    }
    span_choice = mo.ui.dropdown(
        options=_options, value=next(iter(_options)),
        label="Preview span", allow_select_none=False,
    )
    span_choice
    return (span_choice,)


@app.cell(hide_code=True)
def _():
    buffer_frames = mo.ui.slider(
        start=0, stop=300, step=1, value=30, show_value=True,
        label="Buffer before and after (frames)",
    )
    buffer_frames
    return (buffer_frames,)


@app.cell(hide_code=True)
def _(span_choice, spans):
    selected = spans.filter(pl.col("Merge ID") == span_choice.value).row(0, named=True)
    mo.md(
        f"**{selected['Merge ID']}** · frames {selected['Start Frame']:,}–"
        f"{selected['End Frame']:,} · {selected['Start Time']}–{selected['End Time']} "
        f"· source: {selected['Source']}  \n"
        f"Rule IDs: {selected['Rule IDs'] or 'none'}  \n"
        f"Predicted event IDs: {selected['Predicted Event IDs'] or 'none'}"
    )
    return (selected,)


@app.cell(hide_code=True)
def _(video_path):
    _capture = cv2.VideoCapture(str(video_path))
    try:
        if not _capture.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        _fps = float(_capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        _capture.release()
    if _fps <= 0 or frame_count <= 0:
        raise ValueError(f"Invalid video metadata: {video_path}")
    return (frame_count,)


@app.cell(hide_code=True)
def _(buffer_frames, frame_count, selected, spans):
    from matplotlib.figure import Figure as _Figure
    from matplotlib.patches import Patch as _Patch
    from matplotlib.ticker import StrMethodFormatter as _StrMethodFormatter

    _colors = {"both": "#2563eb", "rule-based": "#16a34a", "predicted": "#d97706"}
    _start, _end = selected["Start Frame"], selected["End Frame"]
    _preview_start = max(0, _start - buffer_frames.value)
    _preview_end = min(frame_count - 1, _end + buffer_frames.value)
    _radius = max(3000, (_end - _start + 1) * 3)
    _figure = _Figure(figsize=(14, 3.5), layout="constrained")
    _axes = _figure.subplots(2, 1)
    for _axis, _limits, _title in zip(
        _axes,
        ((0, frame_count), (max(0, _start - _radius), min(frame_count, _end + 1 + _radius))),
        ("Full video", "Around selected span"),
    ):
        _axis.broken_barh([(0, frame_count)], (0, 0.65), facecolors="#edf0f3")
        for _row in spans.iter_rows(named=True):
            _axis.broken_barh(
                [(_row["Start Frame"], _row["End Frame"] - _row["Start Frame"] + 1)],
                (0, 0.65), facecolors=_colors[_row["Source"]],
            )
        _axis.axvspan(_preview_start, _preview_end + 1, color="#f59e0b", alpha=0.15)
        _axis.axvspan(_start, _end + 1, color="#e11d48", alpha=0.2, linewidth=2)
        _axis.set(xlim=_limits, ylim=(-0.08, 0.75), yticks=[], title=_title,
                  xlabel="Frame number (zero-based)")
        _axis.xaxis.set_major_formatter(_StrMethodFormatter("{x:,.0f}"))
        _axis.spines[["top", "right", "left"]].set_visible(False)
    _figure.legend(
        handles=[_Patch(color=_color, label=_source) for _source, _color in _colors.items()]
                + [_Patch(color="#f59e0b", alpha=0.4, label="preview buffer"),
                   _Patch(color="#e11d48", alpha=0.4, label="selected")],
        loc="upper right", ncol=5,
    )
    _buffer = BytesIO()
    _figure.savefig(_buffer, format="png", dpi=150)
    _figure.clear()
    mo.image(_buffer.getvalue(), width="100%", alt="Merged span timeline with selected span highlighted")
    return


@app.cell(hide_code=True)
def _(buffer_frames, frame_count, selected, video_path):
    import importlib as _importlib
    import merged_span_preview as _merged_span_preview

    _importlib.reload(_merged_span_preview)

    _start, _end = selected["Start Frame"], selected["End Frame"]
    if not 0 <= _start <= _end < frame_count:
        raise ValueError(f"Span {_start}–{_end} is outside {video_path.name} ({frame_count} frames)")

    _preview_dir = ROOT / "outputs" / "merged_spans" / "previews"
    with mo.status.spinner(title=f"Rendering buffered frames around {_start:,}–{_end:,}..."):
        _preview, _preview_start, _preview_end = _merged_span_preview.render_merged_preview(
            video_path, _start, _end, buffer_frames.value, _preview_dir,
        )
    mo.vstack([
        mo.md(f"Preview frames {selected['Merge ID']} **{_preview_start:,}–{_preview_end:,}** · "
              "colored keypoints and a per-frame TrackID legend"),
        mo.video(str(_preview), width="80%"),
    ])
    return


if __name__ == "__main__":
    app.run()
