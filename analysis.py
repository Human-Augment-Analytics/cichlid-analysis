# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "imageio-ffmpeg==0.6.0",
#     "marimo==0.24.0",
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
    table = pl.read_parquet(DATA_DIR / "0028_vid.parquet")
    table
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Overlay pose points to video frame
    """)
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
    Extract selected video clips
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
    _clips_dir = DATA_DIR / "clips"
    _clips_dir.mkdir(parents=True, exist_ok=True)
    _ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
    _clip_results = []

    def _validate_clip(
        _path,
        _expected_frames,
        _source_capture,
        _source_fps,
        _frame_size,
        _start_frame,
        _stop_frame,
    ):
        _reader = cv2.VideoCapture(str(_path))
        if not _reader.isOpened():
            return False, "container could not be opened"

        try:
            _reported_frames = int(_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            _reported_fps = _reader.get(cv2.CAP_PROP_FPS)
            _codec_value = int(_reader.get(cv2.CAP_PROP_FOURCC))
            _reported_codec = "".join(
                chr((_codec_value >> (8 * _index)) & 0xFF)
                for _index in range(4)
            ).lower()
            _reported_size = (
                int(_reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            _decoded_frames = 0
            _first_frame = None
            _last_frame = None

            while True:
                _decoded, _decoded_frame = _reader.read()
                if not _decoded:
                    break
                if _first_frame is None:
                    _first_frame = _decoded_frame.copy()
                _last_frame = _decoded_frame
                _decoded_frames += 1
                if _decoded_frame.shape[1::-1] != _frame_size:
                    return False, f"frame {_decoded_frames - 1} has the wrong size"
                if _decoded_frames > _expected_frames:
                    return False, "contains more frames than expected"
        finally:
            _reader.release()

        if _reported_frames != _expected_frames:
            return False, f"metadata reports {_reported_frames} frames"
        if _decoded_frames != _expected_frames:
            return False, f"only {_decoded_frames} frames could be decoded"
        if _reported_size != _frame_size:
            return False, f"metadata reports size {_reported_size}"
        if abs(_reported_fps - _source_fps) > 0.01:
            return False, f"metadata reports {_reported_fps:.6f} FPS"
        if _reported_codec not in {"avc1", "h264"}:
            return False, f"codec is {_reported_codec!r}, not H.264"

        _source_capture.set(cv2.CAP_PROP_POS_FRAMES, _start_frame)
        _ok, _source_first = _source_capture.read()
        if not _ok:
            return False, f"could not decode source frame {_start_frame}"
        _source_capture.set(cv2.CAP_PROP_POS_FRAMES, _stop_frame)
        _ok, _source_last = _source_capture.read()
        if not _ok:
            return False, f"could not decode source frame {_stop_frame}"

        _first_psnr = cv2.PSNR(_source_first, _first_frame)
        _last_psnr = cv2.PSNR(_source_last, _last_frame)
        if min(_first_psnr, _last_psnr) < 25:
            return False, "first or last frame does not match the source"
        return True, f"decoded all {_decoded_frames} frames; source endpoints match"

    for _project_id, _source_filename, _frame_ranges in _clip_groups:
        _source_video = DATA_DIR / _source_filename
        if not _source_video.exists():
            raise FileNotFoundError(f"Source video not found: {_source_video}")

        _capture = cv2.VideoCapture(str(_source_video))
        if not _capture.isOpened():
            raise RuntimeError(f"Could not open source video: {_source_video}")

        _source_fps = _capture.get(cv2.CAP_PROP_FPS) or FPS
        _frame_count = int(_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        _frame_size = (
            int(_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        try:
            for _start_frame, _stop_frame in _frame_ranges:
                if not 0 <= _start_frame <= _stop_frame < _frame_count:
                    raise ValueError(
                        f"Invalid frame range {_start_frame}-{_stop_frame}; "
                        f"{_source_filename} has {_frame_count} frames"
                    )

                _clip_name = f"{_project_id}_{_start_frame}-{_stop_frame}.mp4"
                _clip_path = _clips_dir / _clip_name
                _expected_frames = _stop_frame - _start_frame + 1

                _clip_existed = _clip_path.exists()
                _is_valid, _validation = (
                    _validate_clip(
                        _clip_path,
                        _expected_frames,
                        _capture,
                        _source_fps,
                        _frame_size,
                        _start_frame,
                        _stop_frame,
                    )
                    if _clip_existed
                    else (False, "not found")
                )

                if _is_valid:
                    _status = "verified existing"
                else:
                    _partial_path = _clip_path.with_name(
                        f"{_clip_path.stem}.part.mp4"
                    )
                    _partial_path.unlink(missing_ok=True)

                    try:
                        subprocess.run(
                            [
                                _ffmpeg_executable,
                                "-y",
                                "-hide_banner",
                                "-loglevel",
                                "error",
                                "-ss",
                                f"{_start_frame / FPS:.9f}",
                                "-i",
                                str(_source_video),
                                "-frames:v",
                                str(_expected_frames),
                                "-map",
                                "0:v:0",
                                "-an",
                                "-c:v",
                                "libx264",
                                "-preset",
                                "veryfast",
                                "-crf",
                                "18",
                                "-pix_fmt",
                                "yuv420p",
                                "-movflags",
                                "+faststart",
                                str(_partial_path),
                            ],
                            check=True,
                        )

                        _is_valid, _validation = _validate_clip(
                            _partial_path,
                            _expected_frames,
                            _capture,
                            _source_fps,
                            _frame_size,
                            _start_frame,
                            _stop_frame,
                        )
                        if not _is_valid:
                            raise RuntimeError(
                                f"Created an invalid clip {_clip_path}: {_validation}"
                            )

                        _partial_path.replace(_clip_path)
                    except Exception:
                        _partial_path.unlink(missing_ok=True)
                        raise

                    _status = "rebuilt" if _clip_existed else "created"

                _clip_results.append({
                    "ProjectID": _project_id,
                    "Source": _source_filename,
                    "StartFrame": _start_frame,
                    "StopFrame": _stop_frame,
                    "Frames": _expected_frames,
                    "Duration_s": _expected_frames / _source_fps,
                    "Status": _status,
                    "Validation": _validation,
                    "Path": str(_clip_path),
                })
        finally:
            _capture.release()

    mo.ui.table(
        pl.DataFrame(_clip_results),
        pagination=False,
        selection=None,
        show_column_summaries=False,
        show_download=False,
        show_search=False,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
