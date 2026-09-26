# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "imageio-ffmpeg==0.6.0",
#     "joblib==1.6.0",
#     "scikit-learn==1.9.1",
#     "marimo==0.24.0",
#     "matplotlib==3.11.2",
#     "numpy==2.5.2",
#     "opencv-python-headless==5.0.0.93",
#     "polars==1.44.1",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")

with app.setup:
    import subprocess
    from pathlib import Path
    from urllib.request import urlretrieve

    import cv2
    import imageio_ffmpeg
    import marimo as mo
    import polars as pl

    from pose_overlay import overlay_frame

    DATA_DIR = Path(__file__).parent / "data"
    FPS = 30.0


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Preliminary exploration
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Download video and parquet files
    """)
    return


@app.cell
def _():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download the source artifacts when missing; add other URLs here as needed.
    _downloads = {
        "0028_vid.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AIzGAhjimK-vvUN1IRetNrI/0028_vid.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=zd96dl6g&dl=1",
        "0028_vid_pairs.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AL6msxoSPODIp3jfuXxWb6Q/0028_vid_pairs.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=nb7wminh&dl=1",
        "0028_vid.mp4": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AMkLWMbEcHxMGEXGPuxG1NI/0028_vid.mp4?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=m42jq5db&dl=1",
        "0031_vid.mp4": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/ADCA61tJU9SWYSnh9DNWkkk/0031_vid.mp4?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=xmrzc3hs&dl=1",
        "0031_vid_pairs.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AOLNU5Q-ydFtOxPr3Bppvss/0031_vid_pairs.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=pkqm83so&dl=1",
        "0031_vid.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/ACNtNV3xM1k06lXgc4zpB0o/0031_vid.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=nox9c3q0&dl=1",
    }

    for _filename, _url in _downloads.items():
        _destination = DATA_DIR / _filename
        if _destination.exists():
            continue

        _partial = _destination.with_suffix(_destination.suffix + ".part")
        print(f"Downloading {_filename}...")
        try:
            urlretrieve(_url, _partial)
            _partial.replace(_destination)
        except Exception:
            _partial.unlink(missing_ok=True)
            raise

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {DATA_DIR}")
    return


@app.cell
def _():
    # Change this filename to reuse the inspection and plots for another video.
    VIDEOS = ["0031_vid.mp4", "0028_vid.mp4"]
    dd = mo.ui.dropdown(
        options=VIDEOS,
        value=VIDEOS[0],
        label="Select video file"
    )
    dd
    return (dd,)


@app.cell
def _(dd):
    video_filename = dd.value
    table = pl.read_parquet((DATA_DIR / video_filename).with_suffix(".parquet"))
    # table.filter(pl.col('FrameNum') == 1)
    return table, video_filename


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Overlay pose points to video frame
    """)
    return


@app.cell
def _(table, video_filename):
    # Use this row to choose a frame, then overlay every detected fish in it.
    _row_index = 6000
    _pose_row = (
        pl.scan_parquet((DATA_DIR / video_filename).with_suffix(".parquet"))
        .slice(_row_index, 1)
        .collect()
        .row(0, named=True)
    )
    _frame_number = int(_pose_row["FrameNum"])
    _frame_poses = table.filter(
        (pl.col("project_id") == _pose_row["project_id"])
        & (pl.col("day_label") == _pose_row["day_label"])
        & (pl.col("FrameNum") == _frame_number)
    ).sort("TrackID")
    _video_path = DATA_DIR / video_filename

    if not _video_path.exists():
        raise FileNotFoundError(f"Source video not found: {_video_path}")

    _capture = cv2.VideoCapture(str(_video_path))
    if not _capture.isOpened():
        raise RuntimeError(f"Could not open source video: {_video_path}")

    try:
        # FrameNum is zero-based: Time_s == FrameNum / FPS in this dataset.
        _capture.set(cv2.CAP_PROP_POS_FRAMES, _frame_number)
        _ok, _frame = _capture.read()
    finally:
        _capture.release()

    if not _ok:
        raise RuntimeError(f"Could not decode frame {_frame_number}")

    _frame = overlay_frame(_frame, _frame_poses)

    _encoded_ok, _png = cv2.imencode(".png", _frame)
    if not _encoded_ok:
        raise RuntimeError("Could not encode the annotated video frame")

    _annotated_frame = mo.image(
        _png.tobytes(),
        alt=f"All {_frame_poses.height} TrackIDs overlaid on frame {_frame_number}",
        width="100%",
        rounded=True,
        caption=(
            f"All {_frame_poses.height} TrackIDs over frame {_frame_number} "
            f"({float(_pose_row['Time_s']):.3f} s)"
        ),
    )

    from pose_change import KEYPOINTS as _keypoint_names

    _keypoint_columns = {
        f"{_name}_{_axis}"
        for _name in _keypoint_names
        for _axis in ("x", "y")
    }
    _metadata = _frame_poses.drop(sorted(_keypoint_columns))

    mo.vstack([
    _annotated_frame,
    mo.ui.table(
        _metadata,
        pagination=False,
        selection=None,
        show_column_summaries=False,
        show_download=False,
        show_search=False,
    )
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Circling spans and displacement bands

    The clip intervals below are treated as circling annotations, with inclusive
    start and end frames. Orange shading applies to the video timeline; these
    intervals do not identify which fish is circling.
    """)
    return


@app.cell
def _():
    from circling_clips import CIRCLING_SPANS_BY_VIDEO

    circling_spans_by_video = CIRCLING_SPANS_BY_VIDEO
    return (circling_spans_by_video,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Circling ground truth vs. logistic regression

    The top bar shows annotated circling in **teal**; the bottom bar shows
    pose-and-pair model predictions in **purple**. Gray means non-circling. Dashed lines
    mark annotation boundaries. Select a video, then a span or custom range.

    Predictions cover **every frame** in the selected range, including training
    frames. This is a visual review, not a held-out evaluation. The score reflects
    the balanced training sample; the default classification threshold is 0.5.
    """)
    return


@app.cell(hide_code=True)
def _(circling_spans_by_video, video_filename):
    timeline_video = mo.ui.dropdown(
        options=list(circling_spans_by_video), value=video_filename,
        label="Video", allow_select_none=False,
    )
    timeline_video
    return (timeline_video,)


@app.cell(hide_code=True)
def _(circling_spans_by_video, timeline_video):
    _capture = cv2.VideoCapture(str(DATA_DIR / timeline_video.value))
    try:
        if not _capture.isOpened():
            raise RuntimeError(f"Cannot open {timeline_video.value}")
        timeline_last_frame = int(_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    finally:
        _capture.release()
    _spans = circling_spans_by_video[timeline_video.value]
    _options = {
        "Annotated region (+10 s padding)": (max(0, _spans[0][0] - 300), min(timeline_last_frame, _spans[-1][1] + 300)),
        "Full video": (0, timeline_last_frame),
        **{f"Span {_index}: {_start:,}–{_end:,} (+10 s)":
           (max(0, _start - 300), min(timeline_last_frame, _end + 300))
           for _index, (_start, _end) in enumerate(_spans, 1)},
    }
    timeline_preset = mo.ui.dropdown(
        options=_options, value="Annotated region (+10 s padding)",
        label="View", allow_select_none=False,
    )
    timeline_preset
    return timeline_last_frame, timeline_preset


@app.cell(hide_code=True)
def _(timeline_last_frame, timeline_preset):
    timeline_range = mo.ui.range_slider(
        start=0, stop=timeline_last_frame, step=1,
        value=timeline_preset.value, debounce=True, show_value=True,
        full_width=True, label="Frame range (inclusive)",
    )
    timeline_threshold = mo.ui.slider(
        start=0, stop=1, step=0.01, value=0.5, debounce=True,
        show_value=True, label="Prediction threshold",
    )
    mo.vstack([timeline_range, timeline_threshold])
    return timeline_range, timeline_threshold


@app.cell(hide_code=True)
def _(timeline_range, timeline_video):
    import importlib as _importlib
    import pose_features as _pose_features
    import circling_model as _circling_model
    import circling_timeline as _circling_timeline

    # A live kernel may retain pose-only helpers after the model is retrained.
    # Reload dependencies in order before loading the model's feature schema.
    _importlib.reload(_pose_features)
    _importlib.reload(_circling_model)
    _importlib.reload(_circling_timeline)
    score_window = _circling_timeline.score_window

    _model_path = DATA_DIR.parent / "outputs" / "circling_baseline" / "model.joblib"
    mo.stop(not _model_path.exists(), mo.md("Run `uv run python circling_model.py train` to create the model."))
    timeline_predictions = score_window(
        _model_path, (DATA_DIR / timeline_video.value).with_suffix(".parquet"),
        *timeline_range.value,
    )
    return (timeline_predictions,)


@app.cell(hide_code=True)
def _(
    circling_spans_by_video,
    timeline_predictions,
    timeline_threshold,
    timeline_video,
):
    from io import BytesIO as _BytesIO
    from circling_timeline import plot_timeline

    _figure = plot_timeline(
        timeline_predictions, circling_spans_by_video[timeline_video.value],
        threshold=timeline_threshold.value, video_name=timeline_video.value,
    )
    _buffer = _BytesIO()
    _figure.savefig(_buffer, format="png", dpi=160)
    _figure.clear()
    mo.image(_buffer.getvalue(), width="100%",
             alt="Aligned ground-truth and logistic-regression circling label bars")
    return


@app.cell(hide_code=True)
def _(circling_spans_by_video):
    _options = {
        f"{_video} · {_start:,}–{_end:,}": f"{Path(_video).stem}_{_start}-{_end}_overlay.mp4"
        for _video, _spans in circling_spans_by_video.items()
        for _start, _end in _spans
    }
    clip_selection = mo.ui.dropdown(
        options=_options, value=next(iter(_options)), label="Circling clip",
        allow_select_none=False,
    )
    mo.vstack([mo.md("## Circling clips — pose points and TrackID legend"), clip_selection])
    return (clip_selection,)


@app.cell(hide_code=True)
def _(clip_selection):
    _clip_dir = DATA_DIR.parent / "outputs" / "circling_clips"
    _clip = _clip_dir / clip_selection.value
    mo.stop(not _clip.exists(), mo.md("Run `python circling_clips.py` to export clips."))
    # VS Code embeds local videos in the cell output. Keep one small preview
    # per cell instead of embedding all full-resolution clips (~680 MB).
    _preview_dir = _clip_dir / "previews"
    _preview_dir.mkdir(exist_ok=True)
    _preview = _preview_dir / _clip.name
    if not _preview.exists() or _preview.stat().st_mtime_ns < _clip.stat().st_mtime_ns:
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-i", str(_clip), "-vf", "scale=480:-2", "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "32",
            "-maxrate", "250k", "-bufsize", "500k", "-movflags", "+faststart",
            str(_preview),
        ], check=True)
    mo.vstack([
        mo.video(str(_preview), width="100%"),
        mo.md(f"Preview · Full-resolution clip: `outputs/circling_clips/{_clip.name}`"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sustained positive predictions

    Scan the **whole selected video** (the Video selector above), using the current
    prediction threshold. Each returned span lasts **more than 10 seconds** and
    has **at least 90% positive frames**. This refers to the proportion of positive
    classifications, not a 0.9 probability threshold.

    We scan 301-frame windows at 30 FPS, then greedily merge overlapping/touching
    windows only if the combined span still meets 90%. The table lists all matches;
    choose a span to render its pose-overlay clip. Predictions include training frames.

    Start Time is the first frame's time; End Time is the boundary after the last
    included frame. Fish IDs lists tracks observed during the span, not confirmed
    participants. Confidence / Notes reports the fraction of frames classified
    positive; it is not a calibrated event probability.
    """)
    return


@app.cell(hide_code=True)
def _(timeline_last_frame, timeline_video):
    # timeline_predictions ensures the model helper reload has finished first.
    import importlib as _importlib
    import predicted_spans as _predicted_spans

    _importlib.reload(_predicted_spans)
    _model = DATA_DIR.parent / "outputs" / "circling_baseline" / "model.joblib"
    with mo.status.spinner(title=f"Scanning all frames of {timeline_video.value}..."):
        sustained_scores = _predicted_spans.score_video(
            _model, DATA_DIR / timeline_video.value, timeline_last_frame + 1,
            DATA_DIR.parent / "outputs" / "predicted_spans",
        )
    return (sustained_scores,)


@app.cell(hide_code=True)
def _(sustained_scores, timeline_threshold, timeline_video):
    from predicted_spans import annotate_spans, find_positive_spans

    sustained_spans = annotate_spans(
        find_positive_spans(
            sustained_scores, fps=FPS, threshold=timeline_threshold.value,
        ),
        DATA_DIR / timeline_video.value, fps=FPS,
    )
    _output = DATA_DIR.parent / "outputs" / "predicted_spans"
    sustained_spans.write_csv(_output / f"{Path(timeline_video.value).stem}_spans_threshold_{timeline_threshold.value:.2f}.csv")
    mo.vstack([
        mo.md(f"**{sustained_spans.height} qualifying spans** in {timeline_video.value} · threshold {timeline_threshold.value:.2f}"),
        mo.ui.table(sustained_spans, selection=None, pagination=True, page_size=10),
    ])
    return (sustained_spans,)


@app.cell(hide_code=True)
def _(sustained_spans):
    mo.stop(sustained_spans.is_empty(), mo.md("No spans meet these conditions."))
    _options = {
        f"Span {_r['span']}: {_r['start_frame']:,}–{_r['end_frame']:,} · {_r['duration_seconds']:.1f}s · {_r['positive_fraction']:.1%} positive":
        (_r['start_frame'], _r['end_frame'])
        for _r in sustained_spans.iter_rows(named=True)
    }
    sustained_selection = mo.ui.dropdown(options=_options, value=next(iter(_options)),
                                         label="Predicted span", allow_select_none=False)
    sustained_selection
    return (sustained_selection,)


@app.cell(hide_code=True)
def _(sustained_selection, timeline_video):
    from predicted_spans import render_span

    with mo.status.spinner(title="Rendering selected span with pose points and TrackID legend..."):
        _clip, _preview = render_span(
            DATA_DIR / timeline_video.value, *sustained_selection.value,
            DATA_DIR.parent / "outputs" / "predicted_spans" / "clips",
        )
    mo.vstack([
        mo.video(str(_preview), width="100%"),
        mo.md(f"Full-resolution clip: `{_clip.relative_to(DATA_DIR.parent)}`"),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
