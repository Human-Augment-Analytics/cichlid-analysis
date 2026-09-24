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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Inspect pose data
    """)
    return


@app.cell
def _():
    # Change this filename to reuse the inspection and plots for another video.
    video_filename = "0031_vid.mp4"
    return (video_filename,)


@app.cell
def _(video_filename):
    table = pl.read_parquet((DATA_DIR / video_filename).with_suffix(".parquet"))
    # table.filter(pl.col('FrameNum') == 1)
    table.select(pl.col('TrackID').unique())
    return (table,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Frame-to-frame keypoint movement

    Each panel plots **√(Δx² + Δy²)** in pixels for one keypoint across
    all frames in the video selected by `video_filename`. Each dot is a measurement for one fish,
    assigned to the later frame. All panels use the same scales.

    Differences compare **the same TrackID in the same video** at frames t − 1
    and t. Multiple fish in one frame produce separate measurements; rows from
    different TrackIDs are never subtracted. Track starts, gaps, and missing or nonfinite coordinates
    are excluded. No averaging, smoothing, or downsampling is applied.
    The cell outputs a PNG image.
    """)
    return


@app.cell
def _(table, video_filename):
    from io import BytesIO as _BytesIO

    from pose_change import keypoint_displacements, plot_keypoint_displacements

    keypoint_changes = keypoint_displacements(table)
    _figure = plot_keypoint_displacements(keypoint_changes)
    _image_buffer = _BytesIO()
    _figure.savefig(_image_buffer, format="png", dpi=160)
    _figure.clear()
    mo.image(
        _image_buffer.getvalue(),
        alt=f"Frame-to-frame displacement for all 10 keypoints in {video_filename}",
        width="100%",
        caption=f"{video_filename} — keypoint displacement in pixels across all frames",
    )
    return (keypoint_changes,)


@app.cell
def _():
    pair_count = 20
    return (pair_count,)


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
    ## Largest frame-to-frame movements

    Show the **20 track/frame pairs with the largest mean displacement** across
    all 10 keypoints, ordered from largest to smallest. `pair_count` controls
    the number of pairs. All 10 measurements must be present and finite.
    Ties are resolved by video, TrackID, and frame number.

    Each ranked pair belongs to one TrackID. The 10-keypoint average describes
    that fish's movement. Both frames overlay **all detected TrackIDs**, with a
    consistent color and label for each track.

    Each image shows all detected fish in frame **t − 1 on the left** and
    **t on the right**, using the same labeled keypoint overlay as above.
    """)
    return


@app.cell
def _(keypoint_changes, pair_count):
    from pose_change import top_frame_pairs

    selected_pairs = top_frame_pairs(keypoint_changes, count=pair_count)
    selected_pairs
    return (selected_pairs,)


@app.cell
def _(selected_pairs, table, video_filename):
    import numpy as _np

    from pose_overlay import render_frame_pairs

    _output_dir = DATA_DIR.parent / "outputs" / f"{Path(video_filename).stem}_top{selected_pairs.height}_pairs"
    _output_dir.mkdir(parents=True, exist_ok=True)
    (_output_dir / "pairs.json").write_text(selected_pairs.write_json())
    _images = []
    for _pair, _png in render_frame_pairs(DATA_DIR / video_filename, table, selected_pairs):
        _name = f"pair_{_pair['rank']:02d}_{_pair['PreviousFrameNum']}-{_pair['FrameNum']}"
        (_output_dir / f"{_name}.png").write_bytes(_png)
        # Full-resolution PNGs are retained above; compact previews keep the
        # combined gallery below marimo's cell-output limit.
        _frame = cv2.imdecode(_np.frombuffer(_png, dtype=_np.uint8), cv2.IMREAD_COLOR)
        _preview_width = min(1296, _frame.shape[1])
        _preview = cv2.resize(
            _frame,
            (_preview_width, round(_frame.shape[0] * _preview_width / _frame.shape[1])),
            interpolation=cv2.INTER_AREA,
        )
        _ok, _jpeg = cv2.imencode(".jpg", _preview, [cv2.IMWRITE_JPEG_QUALITY, 40])
        if not _ok:
            raise RuntimeError("Could not encode frame-pair preview")
        _preview_path = _output_dir / f"{_name}.jpg"
        _preview_path.write_bytes(_jpeg.tobytes())
        _caption = (
            f"Rank {_pair['rank']}: frames {_pair['PreviousFrameNum']:,} → "
            f"{_pair['FrameNum']:,} | Ranked TrackID {_pair['TrackID']} | all tracks overlaid | "
            f"Mean movement {_pair['mean_displacement_px']:.2f} px"
        )
        _images.append(mo.image(_preview_path, alt=_caption, caption=_caption, width="100%"))
    mo.vstack(_images)
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
    Each keypoint panel shows the **mean frame-to-frame displacement across fish**
    at each frame (gray), a trailing rolling mean (blue), and **Bollinger bands
    at mean ± 2 population standard deviations** (blue fill). These summarize
    the frame means, not the spread between individual fish. Displacements are
    computed within tracks before averaging.

    The default window is 30 frames (1 second at 30 FPS). Missing frames stay
    missing; bands require a complete window. Negative lower bands are retained
    as a statistical bound. The second view zooms to the annotated region with
    10 seconds of padding. Both views use seconds from the start of the video.
    """)
    return


@app.cell
def _():
    bollinger_window_frames = mo.ui.number(
        start=2, stop=18000, value=30, step=1, label="Bollinger window (frames)"
    )
    bollinger_window_frames
    return (bollinger_window_frames,)


@app.cell
def _(
    bollinger_window_frames,
    circling_spans_by_video,
    keypoint_changes,
    video_filename,
):
    from io import BytesIO as _BytesIO
    from pose_change import plot_displacement_bands

    _spans = circling_spans_by_video.get(video_filename, [])
    _views = [("full", None)]
    if _spans:
        _views.append(("circling", (
            max(0, min(_start for _start, _end in _spans) - int(10 * FPS)),
            max(_end for _start, _end in _spans) + int(10 * FPS),
        )))
    _output_dir = DATA_DIR.parent / "outputs"
    _output_dir.mkdir(parents=True, exist_ok=True)
    _plots = []
    for _view_name, _frame_range in _views:
        _figure = plot_displacement_bands(
            keypoint_changes, _spans, fps=FPS,
            window_frames=int(bollinger_window_frames.value), frame_range=_frame_range,
        )
        _buffer = _BytesIO()
        _figure.savefig(_buffer, format="png", dpi=160)
        _output = _output_dir / f"{Path(video_filename).stem}_displacement_bands_{_view_name}.png"
        _output.write_bytes(_buffer.getvalue())
        _figure.clear()
        _plots.append(mo.image(
            _buffer.getvalue(), width="100%",
            alt=f"{video_filename}: keypoint displacement with Bollinger bands and circling spans ({_view_name})",
            caption=f"{video_filename} — {_view_name} view; saved to {_output.name}",
        ))
    mo.vstack(_plots)
    return


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
    from predicted_spans import find_positive_spans

    sustained_spans = find_positive_spans(
        sustained_scores, fps=FPS, threshold=timeline_threshold.value,
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
