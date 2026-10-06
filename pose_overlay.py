"""The notebook's labeled OpenCV keypoint overlay, shared by frame reviews."""

from math import isfinite
from colorsys import hsv_to_rgb

import cv2
import polars as pl

from pose_change import TRACK_KEYS


def overlay_pose(frame, pose_row, color=(0, 215, 255), track_label=None, *, labels=True):
    """Draw labeled keypoints on a copy of a BGR video frame."""
    _frame = frame.copy()
    _pose_row = pose_row
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
        if (
            _x_value is None or _y_value is None
            or not isfinite(_x_value) or not isfinite(_y_value)
        ):
            continue

        _x = round(float(_x_value))
        _y = round(float(_y_value))
        if not (0 <= _x < _width and 0 <= _y < _height):
            continue
        _points.append((_name, _x, _y))

    if not labels:
        for _name, _x, _y in _points:
            cv2.circle(_frame, (_x, _y), 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(_frame, (_x, _y), 4, color, -1, cv2.LINE_AA)
        return _frame

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
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.circle(_frame, (_x, _y), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(_frame, (_x, _y), 4, color, -1, cv2.LINE_AA)
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

    if track_label is not None and _points:
        _text_width, _text_height = cv2.getTextSize(track_label, _font, 0.7, 2)[0]
        _label_x = max(2, min(_min_point_x, _width - _text_width - 4))
        _label_y = max(_text_height + 4, min(p[2] for p in _points) - 24)
        cv2.putText(_frame, track_label, (_label_x, _label_y), _font, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(_frame, track_label, (_label_x, _label_y), _font, 0.7, color, 2, cv2.LINE_AA)
    return _frame


def overlay_frame(frame, frame_poses, *, labels=True):
    """Draw every detected TrackID for exactly one video frame."""
    if frame_poses.is_empty():
        return frame.copy()
    if frame_poses.select(pl.struct("project_id", "day_label", "FrameNum").n_unique()).item() != 1:
        raise ValueError("overlay_frame expects pose rows from a single video frame.")
    if frame_poses["TrackID"].null_count() or frame_poses["TrackID"].is_duplicated().any():
        raise ValueError("Expected one row per nonmissing TrackID in the frame.")
    annotated = frame.copy()
    for row in frame_poses.sort("TrackID").iter_rows(named=True):
        track_id = int(row["TrackID"])
        color = track_color(track_id)
        annotated = overlay_pose(annotated, row, color=color, track_label=f"TrackID {track_id}", labels=labels)
    return annotated


def render_frame_pairs(video_path, poses, pairs):
    """Yield pairs with every detected track overlaid in each video frame."""
    frames = set(pairs["FrameNum"]) | set(pairs["PreviousFrameNum"])
    selected_poses = poses.filter(pl.col("FrameNum").is_in(sorted(frames)))
    if selected_poses.select(pl.struct([*TRACK_KEYS, "FrameNum"]).is_duplicated().any()).item():
        raise ValueError("Expected one pose per video, TrackID, and frame for overlays.")
    frame_groups = selected_poses.partition_by(
        ["project_id", "day_label", "FrameNum"], as_dict=True,
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open source video: {video_path}")
    try:
        for pair in pairs.iter_rows(named=True):
            previous = pair["PreviousFrameNum"]
            current = pair["FrameNum"]
            if current != previous + 1:
                raise ValueError("Ranked video frames must be consecutive.")
            capture.set(cv2.CAP_PROP_POS_FRAMES, previous)
            panels = []
            # Sequential decoding keeps both images from the same adjacent pair.
            for frame_number in (previous, current):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode frame {frame_number} from {video_path}")
                key = (pair["project_id"], pair["day_label"], frame_number)
                frame_poses = frame_groups[key]
                if pair["TrackID"] not in frame_poses["TrackID"]:
                    raise ValueError("Ranked TrackID is missing from the source frame.")
                annotated = overlay_frame(frame, frame_poses)
                panel = cv2.copyMakeBorder(annotated, 44, 0, 0, 0, cv2.BORDER_CONSTANT)
                cv2.putText(
                    panel, f"Frame {frame_number:,} | {frame_poses.height} tracks | ranked TrackID {pair['TrackID']}",
                    (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )
                panels.append(panel)
            ok, png = cv2.imencode(".png", cv2.hconcat(panels))
            if not ok:
                raise RuntimeError("Could not encode frame-pair overlay")
            yield pair, png.tobytes()
    finally:
        capture.release()


def track_color(track_id):
    """Return a stable, distinct BGR color for a TrackID."""
    rgb = hsv_to_rgb((int(track_id) * 0.61803398875) % 1, 0.75, 1.0)
    return tuple(round(channel * 255) for channel in reversed(rgb))


def overlay_track_legend(frame, track_ids):
    """Draw a fixed top-right legend; leave the keypoints themselves unlabeled."""
    annotated = frame.copy()
    labels = [(int(track_id), f"TrackID {int(track_id)}") for track_id in sorted(track_ids)]
    if not labels:
        return annotated
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.65
    width = max(cv2.getTextSize(label, font, scale, 1)[0][0] for _, label in labels) + 58
    height = 32 * len(labels) + 16
    x, y = annotated.shape[1] - width - 12, 12
    cv2.rectangle(annotated, (x, y), (x + width, y + height), (24, 24, 24), -1)
    for index, (track_id, label) in enumerate(labels):
        baseline = y + 29 + index * 32
        cv2.circle(annotated, (x + 18, baseline - 6), 5, track_color(track_id), -1, cv2.LINE_AA)
        cv2.putText(annotated, label, (x + 34, baseline), font, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return annotated
