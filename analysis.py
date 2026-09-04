# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "marimo==0.24.0",
#     "opencv-python-headless==5.0.0.93",
#     "polars==1.44.1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path
    from urllib.request import urlretrieve

    import cv2
    import marimo as mo
    import polars as pl

    DATA_DIR = Path(__file__).parent / "data"
    FPS = 30.0


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Preliminary exploration of pair-level pose data

    This notebook inventories the Parquet inputs, checks their structure and data
    quality, and compares pair interactions with the source video.
    """)
    return


@app.cell
def _():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the analysis reproducible without automatically downloading the 31 GB
    # source video. Add other Parquet URLs here as they become available.
    _downloads = {
        "0028_vid.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AIzGAhjimK-vvUN1IRetNrI/0028_vid.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=zd96dl6g&dl=1",
        "0028_vid_pairs.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AL6msxoSPODIp3jfuXxWb6Q/0028_vid_pairs.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=nb7wminh&dl=1",
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
    table = pl.read_parquet(DATA_DIR / "0028_vid.parquet")
    table
    return


@app.cell
def _():
    # Draw the keypoints from one pose row on its corresponding video frame.
    _row_index = 636000
    _pose_row = (
        pl.scan_parquet(DATA_DIR / "0028_vid.parquet")
        .slice(_row_index, 1)
        .collect()
        .row(0, named=True)
    )
    _frame_number = int(_pose_row["FrameNum"])
    _video_path = DATA_DIR / "0028_vid.mp4"

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

    _keypoint_names = [
        "Nose",
        "LeftEye",
        "RightEye",
        "Head",
        "Spine1",
        "Spine2",
        "Spine3",
        "Spine4",
        "Peduncle",
        "TailTip",
    ]
    _height, _width = _frame.shape[:2]
    _font = cv2.FONT_HERSHEY_SIMPLEX
    _font_scale = max(0.45, min(0.7, _width / 2200))
    _points = []

    for _name in _keypoint_names:
        _x_value = _pose_row[f"{_name}_x"]
        _y_value = _pose_row[f"{_name}_y"]
        if _x_value is None or _y_value is None:
            continue

        _x = round(float(_x_value))
        _y = round(float(_y_value))
        if not (0 <= _x < _width and 0 <= _y < _height):
            continue
        _points.append((_name, _x, _y))

    # Alternate labels between the two sides of nearby points to reduce overlap.
    _min_point_x = min((_point[1] for _point in _points), default=0)
    _max_point_x = max((_point[1] for _point in _points), default=0)
    _last_label_y = {True: 0, False: 0}
    for _index, (_name, _x, _y) in enumerate(sorted(_points, key=lambda p: p[2])):
        _text_width, _text_height = cv2.getTextSize(
            _name, _font, _font_scale, 1
        )[0]
        _put_label_on_right = _index % 2 == 0
        _label_x = (
            _max_point_x + 14
            if _put_label_on_right
            else _min_point_x - _text_width - 14
        )
        _desired_label_y = (
            _y - 7 if _put_label_on_right else _y + _text_height + 7
        )
        _label_y = max(
            _desired_label_y,
            _last_label_y[_put_label_on_right] + _text_height + 5,
        )
        _label_x = max(2, min(_label_x, _width - _text_width - 2))
        _label_y = max(_text_height + 2, min(_label_y, _height - 2))
        _last_label_y[_put_label_on_right] = _label_y

        _label_anchor_x = (
            _label_x if _put_label_on_right else _label_x + _text_width
        )
        _label_anchor_y = _label_y - (_text_height // 2)
        cv2.line(
            _frame,
            (_x, _y),
            (_label_anchor_x, _label_anchor_y),
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            _frame,
            (_x, _y),
            (_label_anchor_x, _label_anchor_y),
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(_frame, (_x, _y), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(_frame, (_x, _y), 4, (0, 215, 255), -1, cv2.LINE_AA)
        cv2.putText(
            _frame,
            _name,
            (_label_x, _label_y),
            _font,
            _font_scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            _frame,
            _name,
            (_label_x, _label_y),
            _font,
            _font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    _encoded_ok, _png = cv2.imencode(".png", _frame)
    if not _encoded_ok:
        raise RuntimeError("Could not encode the annotated video frame")

    _annotated_frame = mo.image(
        _png.tobytes(),
        alt=f"Labeled pose points from row {_row_index} on frame {_frame_number}",
        width="100%",
        rounded=True,
        caption=(
            f"Pose row {_row_index} over frame {_frame_number} "
            f"({float(_pose_row['Time_s']):.3f} s)"
        ),
    )

    _keypoint_columns = {
        f"{_name}_{_axis}"
        for _name in _keypoint_names
        for _axis in ("x", "y")
    }
    _metadata = pl.DataFrame(
        [{
            _name: _value
            for _name, _value in _pose_row.items()
            if _name not in _keypoint_columns
        }]
    )

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
 
    """)
    return


if __name__ == "__main__":
    app.run()
