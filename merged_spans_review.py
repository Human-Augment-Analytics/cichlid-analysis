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

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup:
    from io import BytesIO
    from pathlib import Path

    import cv2
    import imageio_ffmpeg
    import marimo as mo
    import polars as pl

    from model_performance_report import overlay_legend, overlay_poses

    ROOT = Path(__file__).resolve().parent
    PREVIEW_DIR = ROOT / "outputs" / "clips" / "merged_spans"


@app.function
def buffered_bounds(start, end, buffer_frames, frame_count):
    """Return an inclusive span extended within the source video's bounds."""
    if not 0 <= start <= end < frame_count or buffer_frames < 0:
        raise ValueError("Invalid span or buffer for video")
    return max(0, start - buffer_frames), min(frame_count - 1, end + buffer_frames)


@app.function
def render_merged_preview(video_path, start, end, buffer_frames, output_dir, max_width=640):
    """Cache a small video with pose points and a per-frame TrackID legend."""
    video_path, output_dir = Path(video_path), Path(output_dir)
    pose_path = video_path.with_suffix(".parquet")
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose data not found: {pose_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0 or max_width < 2:
            raise ValueError(f"Invalid video metadata: {video_path}")
        preview_start, preview_end = buffered_bounds(start, end, buffer_frames, frame_count)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Keep the notebook payload small; older 960 px previews can exceed
        # marimo's output limit when VS Code serializes the video.
        preview = output_dir / (
            f"{video_path.stem}_{start}-{end}_buffer_{buffer_frames}"
            f"_w{max_width}_crf30_overlay.mp4"
        )
        sources = (video_path, pose_path, Path(__file__), ROOT / "model_performance_report.py")
        if preview.exists() and preview.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in sources):
            return preview, preview_start, preview_end

        poses = (pl.scan_parquet(pose_path)
                 .filter(pl.col("FrameNum").is_between(preview_start, preview_end))
                 .collect())
        if (not poses.is_empty()
                and poses.select(pl.struct("project_id", "day_label").n_unique()).item() > 1):
            raise ValueError(f"Pose table contains multiple videos: {pose_path}")
        groups = poses.partition_by("FrameNum", as_dict=True)
        empty = poses.head(0)

        scale = min(1., max_width / width)
        output_width = max(2, round(width * scale / 2) * 2)
        output_height = max(2, round(height * scale / 2) * 2)
        partial = preview.with_suffix(".part.mp4")
        writer = imageio_ffmpeg.write_frames(
            str(partial), (output_width, output_height), fps=fps,
            pix_fmt_in="bgr24", pix_fmt_out="yuv420p", codec="libx264",
            macro_block_size=1, ffmpeg_log_level="error",
            output_params=["-crf", "30", "-preset", "veryfast", "-movflags", "+faststart"],
        )
        try:
            writer.send(None)
            capture.set(cv2.CAP_PROP_POS_FRAMES, preview_start)
            for frame_number in range(preview_start, preview_end + 1):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode frame {frame_number} from {video_path}")
                frame_poses = groups.get((frame_number,), empty)
                annotated = cv2.resize(frame, (output_width, output_height),
                                       interpolation=cv2.INTER_AREA)
                overlay_poses(annotated, frame_poses, scale)
                track_ids = frame_poses["TrackID"].drop_nulls().unique().sort().to_list()
                label = f"Frame {frame_number:,} | {'span' if start <= frame_number <= end else 'buffer'}"
                writer.send(overlay_legend(annotated, track_ids, [(label, (40, 40, 40))]))
            writer.close()
            check = cv2.VideoCapture(str(partial))
            try:
                if (not check.isOpened()
                        or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != preview_end - preview_start + 1):
                    raise RuntimeError(f"Encoded preview has incorrect frame count: {partial}")
            finally:
                check.release()
            partial.replace(preview)
        finally:
            writer.close()
            partial.unlink(missing_ok=True)
        return preview, preview_start, preview_end
    finally:
        capture.release()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Review circling annotations

    Choose a circling annotation file and its source video. Select an event to
    see its position in the full video and play a buffered preview with
    keypoints and a TrackID legend. Frame endpoints are zero-based and inclusive.
    """)
    return


@app.cell(hide_code=True)
def _():
    _span_dir = ROOT / "ground_truth_annotations"
    # Search recursively, keeping only CSVs with the merged-span columns read below.
    _span_columns = {"VideoID", "EventID", "StartFrame", "EndFrame", "Source", "Circling", "ShouldUse", "Notes"}
    _span_files = [path for path in sorted(_span_dir.rglob("*.csv"))
                   if _span_columns <= set(pl.read_csv(path, n_rows=0, infer_schema=False, encoding="utf8-lossy").columns)]
    _video_files = sorted((ROOT / "data").glob("*_vid.mp4"))
    mo.stop(not _span_files, mo.md("No merged-span CSV files under `ground_truth_annotations/`."))
    mo.stop(not _video_files, mo.md("Put the source videos in `data/` to preview the spans."))

    spans_file = mo.ui.dropdown(
        options={str(path.relative_to(_span_dir)): str(path) for path in _span_files},
        value=str(_span_files[0].relative_to(_span_dir)), label="Circling annotations CSV", allow_select_none=False,
    )
    video_file = mo.ui.dropdown(
        options={path.name: str(path) for path in _video_files},
        value=_video_files[0].name, label="Source video", allow_select_none=False,
    )
    mo.hstack([spans_file, video_file], justify="start", gap=2)
    return spans_file, video_file


@app.cell(hide_code=True)
def _(spans_file, video_file):
    _csv_path = Path(spans_file.value)
    video_path = Path(video_file.value)
    _all_spans = pl.read_csv(_csv_path, schema_overrides={
        "VideoID": pl.String, "EventID": pl.String,
        "StartFrame": pl.Int64, "EndFrame": pl.Int64,
        "Source": pl.String, "Circling": pl.String,
        "ShouldUse": pl.String, "Notes": pl.String,
    })
    spans = _all_spans.filter(pl.col("VideoID") == video_path.stem).sort("StartFrame", "EndFrame")
    mo.stop(spans.is_empty(), mo.md(
        f"`{_csv_path.name}` has no events for `{video_path.name}`. Select the matching video."
    ))
    mo.vstack([
        mo.md(f"**{spans.height} circling events** for `{video_path.name}`"),
        mo.ui.table(spans, selection=None, pagination=True, page_size=10),
    ])
    return spans, video_path


@app.cell(hide_code=True)
def _(spans):
    _options = {
        f"{row['EventID']} · {row['StartFrame']:,}–{row['EndFrame']:,} · {row['Source']}"
        f" · Circling: {row['Circling'] or 'blank'}": row["EventID"]
        for row in spans.iter_rows(named=True)
    }
    span_choice = mo.ui.dropdown(
        options=_options, value=next(iter(_options)),
        label="Preview event", allow_select_none=False,
    )
    span_choice
    return (span_choice,)


@app.cell(hide_code=True)
def _():
    buffer_frames = mo.ui.slider(
        start=0, stop=300, step=1, value=0, show_value=True,
        label="Buffer before and after (frames)",
    )
    buffer_frames
    return (buffer_frames,)


@app.cell(hide_code=True)
def _(span_choice, spans):
    selected = spans.filter(pl.col("EventID") == span_choice.value).row(0, named=True)
    mo.md(
        f"**{selected['EventID']}** · frames {selected['StartFrame']:,}–"
        f"{selected['EndFrame']:,} · source: {selected['Source']} "
        f"· Circling: {selected['Circling'] or 'blank'} "
        f"· ShouldUse: {selected['ShouldUse'] or 'blank'}  \n"
        f"Notes: {selected['Notes'] or 'none'}"
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

    _colors = {"finalized": "#2563eb", "merged_spans": "#16a34a", "labeled_repo": "#d97706",
               "original": "#9333ea"}
    _lanes = {_source: _lane * 0.25 for _lane, _source in enumerate(reversed(_colors))}
    _start, _end = selected["StartFrame"], selected["EndFrame"]
    _preview_start, _preview_end = buffered_bounds(_start, _end, buffer_frames.value, frame_count)
    _radius = max(3000, (_end - _start + 1) * 3)
    _figure = _Figure(figsize=(14, 3.5), layout="constrained")
    _axes = _figure.subplots(2, 1)
    for _axis, _limits, _title in zip(
        _axes,
        ((0, frame_count), (max(0, _start - _radius), min(frame_count, _end + 1 + _radius))),
        ("Full video", "Around selected event"),
    ):
        _axis.broken_barh([(0, frame_count)], (0, 0.95), facecolors="#edf0f3")
        for _row in spans.iter_rows(named=True):
            _axis.broken_barh(
                [(_row["StartFrame"], _row["EndFrame"] - _row["StartFrame"] + 1)],
                (_lanes[_row["Source"]], 0.2), facecolors=_colors[_row["Source"]],
            )
        _axis.axvspan(_preview_start, _preview_end + 1, color="#f59e0b", alpha=0.15)
        _axis.axvspan(_start, _end + 1, color="#e11d48", alpha=0.2, linewidth=2)
        _axis.set(xlim=_limits, ylim=(-0.08, 1.0), yticks=[], title=_title,
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
    mo.image(_buffer.getvalue(), width="100%", alt="Circling event timeline with selected event highlighted")
    return


@app.cell(hide_code=True)
def _(buffer_frames, frame_count, selected, video_path):
    _start, _end = selected["StartFrame"], selected["EndFrame"]
    if not 0 <= _start <= _end < frame_count:
        raise ValueError(f"Event {_start}–{_end} is outside {video_path.name} ({frame_count} frames)")

    with mo.status.spinner(title=f"Rendering buffered frames around {_start:,}–{_end:,}..."):
        _preview, _preview_start, _preview_end = render_merged_preview(
            video_path, _start, _end, buffer_frames.value, PREVIEW_DIR,
        )
    mo.vstack([
        mo.md(f"Preview frames {selected['EventID']} **{_preview_start:,}–{_preview_end:,}** · "
              "colored keypoints and a per-frame TrackID legend"),
        mo.video(str(_preview), width="80%"),
    ])
    return


if __name__ == "__main__":
    app.run()
