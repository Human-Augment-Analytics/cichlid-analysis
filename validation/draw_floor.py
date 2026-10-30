"""
Draw the gravel-floor outline for a video by clicking its corners.

The outline is used by the reflection filter: detections whose box center falls outside it are treated as
reflections in the glass. Outlines are saved to floors.json (video ID -> list of [x, y] points in full-resolution
video pixels), so new videos only need a few clicks, not a code change.

The window shows the median of several frames spread over the video, so moving fish disappear and the gravel
edge is clear. Press H to overlay where the tracker found fish (needs <video>_vid.parquet): almost everything
dense should fall inside the outline; what is left outside should be the wall bands where reflections appear.

Usage
  python draw_floor.py <folder>/0013_vid.mp4
  python draw_floor.py 0013 --videos <folder>           (finds 0013_vid.mp4 there)
  python draw_floor.py 0013 --videos ... --snapshot check_0013.png        (no window: saves the overlay image)

Mouse / keys
  left click       add a corner (click on or near an existing corner and drag to move it)
  right click / Z  remove the last corner
  C                clear all corners
  H                toggle the detection heatmap
  S or Enter       save to floors.json and quit
  Q or Esc         quit without saving
Corners are joined in the order you click them, and the last one joins back to the first: go around the edge.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
EDGE = (255, 200, 0)      # outline (BGR): sky blue
CORNER = (0, 150, 255)    # corners: orange
GRAB_PX = 12              # click this close to a corner (screen pixels) to drag it


def video_path(arg, videos_dir):
    p = Path(arg)
    if p.suffix.lower() == ".mp4" and p.exists():
        return p
    for name in (f"{arg}_vid.mp4", f"{arg}.mp4"):
        if (Path(videos_dir) / name).exists():
            return Path(videos_dir) / name
    sys.exit(f"No video found for '{arg}' (looked for {arg}_vid.mp4 in {videos_dir}).")


def video_id(path):
    return path.stem.replace("_vid", "")


def background(path, n_frames):
    """Median of n frames spread across the video: fish move, so they vanish and the floor edge shows."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for f in np.linspace(total * 0.05, total * 0.95, n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            frames.append(img)
    cap.release()
    if not frames:
        sys.exit(f"Could not read frames from {path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def heatmap(parquet, shape):
    """Log-scaled map of where detection box centers fall, as a color image (0 where no detections)."""
    import pandas as pd
    d = pd.read_parquet(parquet, columns=["X_center", "Y_center"])
    h, w = shape[:2]
    hist, _, _ = np.histogram2d(d.Y_center, d.X_center, bins=[np.arange(0, h + 4, 4), np.arange(0, w + 4, 4)])
    hist = np.log1p(hist)
    hist = (255 * hist / max(hist.max(), 1)).astype(np.uint8)
    hist = cv2.resize(hist, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(hist, cv2.COLORMAP_VIRIDIS), hist > 0, d


def outside_share(points, d):
    if len(points) < 3:
        return None
    from matplotlib.path import Path as Polygon
    inside = Polygon(points).contains_points(d[["X_center", "Y_center"]].to_numpy())
    return 1 - inside.mean()


def render(bg, points, heat, show_heat, scale, title):
    img = bg.copy()
    if show_heat and heat is not None:
        color, mask = heat
        img[mask] = (0.45 * img[mask] + 0.55 * color[mask]).astype(np.uint8)
    view = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else img
    pts = [(int(x * scale), int(y * scale)) for x, y in points]
    if len(pts) >= 2:
        cv2.polylines(view, [np.array(pts, np.int32)], len(pts) >= 3, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.polylines(view, [np.array(pts, np.int32)], len(pts) >= 3, EDGE, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(view, p, 6, (0, 0, 0), -1)
        cv2.circle(view, p, 4, CORNER, -1)
        cv2.putText(view, str(i + 1), (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, str(i + 1), (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if title:
        band = view[: 12 + 24 * len(title)]
        band[:] = (0.35 * band).astype(np.uint8)          # darkened strip behind the text
        for i, line in enumerate(title):
            cv2.putText(view, line, (10, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def load_floors(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save_floors(path, vid, points):
    floors = load_floors(path)
    floors[vid] = [[int(round(x)), int(round(y))] for x, y in points]
    lines = [f'  "{k}": {json.dumps(v)}' for k, v in sorted(floors.items())]
    path.write_text("{\n" + ",\n".join(lines) + "\n}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="path to <id>_vid.mp4, or a video ID such as 0013")
    ap.add_argument("--videos", default=".", help="folder with the videos and parquet files (when giving an ID)")
    ap.add_argument("--floors", default=str(HERE / "floors.json"), help="outline file (default: floors.json next to this script)")
    ap.add_argument("--frames", type=int, default=9, help="frames to take the median of (default 9)")
    ap.add_argument("--scale", type=float, default=None, help="window scale (default: fit about 1200 px wide)")
    ap.add_argument("--snapshot", help="no window: save the background, heatmap and current outline to this image")
    args = ap.parse_args()

    path = video_path(args.video, args.videos)
    vid = video_id(path)
    floors_path = Path(args.floors)
    points = [tuple(p) for p in load_floors(floors_path).get(vid, [])]

    print(f"{vid}: building background from {args.frames} frames ...", flush=True)
    bg = background(path, args.frames)
    h, w = bg.shape[:2]
    parquet = path.with_name(f"{path.stem}.parquet")
    heat, dets = None, None
    if parquet.exists():
        print(f"{vid}: loading detections from {parquet.name} ...", flush=True)
        color, mask, dets = heatmap(parquet, bg.shape)
        heat = (color, mask)
    scale = args.scale or min(1.0, 1200 / w)

    def title(show_heat):
        share = outside_share(points, dets) if dets is not None else None
        lines = [f"{vid}  {w}x{h}  corners: {len(points)}" + ("  (loaded from floors.json)" if loaded else "")]
        if share is not None:
            lines.append(f"detections outside outline: {share:.1%}")
        lines.append("click=add/drag  right/Z=undo  C=clear  H=heatmap  S=save  Q=quit")
        return lines

    loaded = bool(points)
    if args.snapshot:
        cv2.imwrite(args.snapshot, render(bg, points, heat, True, 1.0, title(True)[:2]))
        print(f"saved {args.snapshot}")
        return

    try:
        cv2.namedWindow("floor", cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        sys.exit("OpenCV has no window support here (opencv-python-headless?). "
                 "Run: pip uninstall opencv-python-headless && pip install opencv-python")

    state = {"drag": None, "show_heat": heat is not None}

    def on_mouse(event, x, y, flags, _):
        fx, fy = x / scale, y / scale
        if event == cv2.EVENT_LBUTTONDOWN:
            near = [i for i, (px, py) in enumerate(points) if np.hypot((px - fx) * scale, (py - fy) * scale) <= GRAB_PX]
            if near:
                state["drag"] = near[0]
            else:
                points.append((fx, fy))
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
            points[state["drag"]] = (min(max(fx, 0), w - 1), min(max(fy, 0), h - 1))
        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = None
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.setMouseCallback("floor", on_mouse)
    while True:
        cv2.imshow("floor", render(bg, points, heat, state["show_heat"], scale, title(state["show_heat"])))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("z"), ord("Z"), 8) and points:
            points.pop()
        elif key in (ord("c"), ord("C")):
            points.clear()
        elif key in (ord("h"), ord("H")) and heat is not None:
            state["show_heat"] = not state["show_heat"]
        elif key in (ord("s"), ord("S"), 13):
            if len(points) < 3:
                print("need at least 3 corners")
                continue
            save_floors(floors_path, vid, points)
            share = outside_share(points, dets) if dets is not None else None
            print(f"saved {len(points)} corners for {vid} to {floors_path}"
                  + (f"; {share:.1%} of detections fall outside" if share is not None else ""))
            break
        elif key in (ord("q"), ord("Q"), 27):
            print("quit without saving")
            break
        if cv2.getWindowProperty("floor", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
