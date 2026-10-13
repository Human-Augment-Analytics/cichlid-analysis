"""Video I/O, pose overlay drawing, and clip rendering with pose baked in."""
import math
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import (KP_NAMES, KP_COLOR, SKELETON,
                        COLOR_MALE, COLOR_FEMALE, COLOR_OTHER,
                        CLIPS_DIR, FPS)


def video_info(path):
    """Return {n_frames, fps, width, height}."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    info = {
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def iter_frames(path, start=0, stop=None, step=1):
    """Yield (frame_index, BGR_frame) between start and stop."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if stop is not None and idx >= stop:
                break
            if (idx - start) % step == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def read_frame(path, index):
    """Read a single BGR frame by index."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IndexError(f"Could not read frame {index} from {path}")
    return frame


def write_video(out_path, frames, fps=30, fourcc="mp4v"):
    """Write an iterable of BGR frames to a video file."""
    out_path = str(out_path)
    frames = iter(frames)
    first = next(frames)
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    try:
        writer.write(first)
        for f in frames:
            writer.write(f)
    finally:
        writer.release()
    return out_path


def sex_color(s):
    """Map a Sex string to a BGR triple."""
    s = str(s).strip().lower()
    if s.startswith("m"): return COLOR_MALE
    if s.startswith("f"): return COLOR_FEMALE
    return COLOR_OTHER


def _draw_dot(img, pt, color, r=4):
    cv2.circle(img, pt, r + 1, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, pt, r, color, -1, lineType=cv2.LINE_AA)


def _draw_kp_legend(img, x0=10, y0=None):
    h_img = img.shape[0]
    line_h = 16
    if y0 is None:
        y0 = h_img - line_h * (len(KP_NAMES) + 1) - 8
    for i, name in enumerate(KP_NAMES):
        yc = y0 + i * line_h
        _draw_dot(img, (x0 + 8, yc), KP_COLOR[name], r=4)
        cv2.putText(img, name, (x0 + 22, yc + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def overlay_pose_on_frame(frame, fish_rows, draw_legend=True):
    """Draw per-fish skeleton, per-keypoint dots, centroid + heading arrow, and Sex/TrackID label."""
    out = frame.copy()
    for _, row in fish_rows.iterrows():
        c_sex = sex_color(row.get("Sex", "?"))
        pts = {}
        for kp in KP_NAMES:
            x, y = row.get(f"{kp}_x"), row.get(f"{kp}_y")
            if pd.notna(x) and pd.notna(y):
                pts[kp] = (int(x), int(y))
        for a, b in SKELETON:
            if a in pts and b in pts:
                cv2.line(out, pts[a], pts[b], c_sex, 1, cv2.LINE_AA)
        for name, pt in pts.items():
            _draw_dot(out, pt, KP_COLOR[name], r=4)
        cx, cy = row.get("X_center"), row.get("Y_center")
        if pd.notna(cx) and pd.notna(cy):
            cx, cy = int(cx), int(cy)
            cv2.drawMarker(out, (cx, cy), c_sex, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
            h = row.get("heading_rad")
            if pd.notna(h):
                L = 34
                tip = (int(cx + L * math.cos(h)), int(cy + L * math.sin(h)))
                cv2.arrowedLine(out, (cx, cy), tip, c_sex, 2,
                                tipLength=0.3, line_type=cv2.LINE_AA)
            label = f"{row.get('Sex','?')} #{int(row.get('TrackID', -1))}"
            cv2.putText(out, label, (cx + 7, cy - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (cx + 7, cy - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_sex, 1, cv2.LINE_AA)
    if draw_legend:
        _draw_kp_legend(out)
    return out


def render_overlay_clip(video_path, start_frame, n_frames, df_pose_video, out_path,
                        fps=None, hud=True):
    """Read n_frames from source, overlay pose, write to out_path."""
    fps = fps or FPS
    stop = start_frame + n_frames
    win = df_pose_video[df_pose_video["FrameNum"].between(start_frame, stop - 1)]
    frames_by_num = {n: g for n, g in win.groupby("FrameNum")}

    def gen():
        for idx, frame in iter_frames(video_path, start=start_frame, stop=stop):
            g = frames_by_num.get(idx)
            frame_o = overlay_pose_on_frame(frame, g) if g is not None else frame.copy()
            if hud:
                n_here = 0 if g is None else len(g)
                txt = f"frame {idx}  t={idx/fps:.2f}s  fish={n_here}"
                cv2.putText(frame_o, txt, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame_o, txt, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            yield frame_o

    return write_video(out_path, gen(), fps=fps)


def batch_render_circling_clips(circling_df, pose_by_stem, fps=None):
    """Render pose-overlaid clips for every mapped CSV-labeled event."""
    fps = fps or FPS
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    todo = circling_df[circling_df["video_stem"].notna()].reset_index(drop=True)
    print(f"Rendering {len(todo)} clips from {todo['video_stem'].nunique()} video(s)")
    print(todo.groupby("video_stem").size().rename("n_clips").to_string())

    rows = []
    for _, row in tqdm(todo.iterrows(), total=len(todo)):
        stem = row["video_stem"]
        pose = pose_by_stem.get(stem)
        start = int(row["StartFrame"])
        stop = int(row["StopFrame"])
        n = stop - start + 1
        out = CLIPS_DIR / f"{stem}__f{start}-{stop}.mp4"
        if pose is None:
            rows.append({"clip": out.name, "video_stem": stem,
                         "status": "skipped (no parquet)"})
            continue
        render_overlay_clip(row["video_path"], start, n, pose, out, fps=fps)
        rows.append({
            "clip": out.name, "video_stem": stem,
            "start_frame": start, "stop_frame": stop,
            "duration_s": round(n / fps, 3), "n_frames": n,
            "size_mb": round(out.stat().st_size / 1e6, 2) if out.exists() else None,
            "status": "ok",
        })
    return pd.DataFrame(rows)
