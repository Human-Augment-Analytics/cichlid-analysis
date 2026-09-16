# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "imageio-ffmpeg==0.6.0",
#     "marimo==0.24.0",
#     "matplotlib==3.11.2",
#     "numpy==2.5.2",
#     "opencv-python-headless==5.0.0.93",
#     "polars==1.44.1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
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
    Video clip spans
    """)
    return


@app.cell
def _():
    _clip_groups = [
        (
            "MC920",
            "0028_vid.mp4",
            [
                (334703, 334830),
                (334985, 335118),
                (335409, 335502),
                (336153, 336285),
                (436636, 436895),
                (437259, 437437),
                (713035, 713330),
            ],
        ),
        (
            "F1_613",
            "0031_vid.mp4",
            [
                (265510, 266227),
                (266809, 267100),
                (267242, 267837),
                (269089, 271817),
                (272241, 273018),
                (273550, 274787),
                (275040, 275190),
                (276439, 276530),
            ],
        ),
    ]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
