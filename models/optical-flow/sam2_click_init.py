"""Interactive click tool: pick init points for SAM 2 on one frame.

Pops a window showing the chosen frame.  Left-click on each fish (in any
order).  Press:
    s   save clicks to JSON and exit
    r   reset clicks
    u   undo last click
    q   quit WITHOUT saving

Writes JSON of the form:
    {
      "video": "...",
      "frame": 20142,
      "image_size": [W, H],
      "points": [[x1, y1], [x2, y2], ...],
      "labels": [1, 1, ...]      # 1 = foreground (fish), 0 = background
    }

Both the window image and the saved points are in **native video
resolution** (no downsampling), so they can be fed directly to SAM 2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def _load_frame(video: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'could not read frame {frame_idx} from {video}')
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--frame', type=int,  required=True,
                    help='Frame index to click on (usually event start frame).')
    ap.add_argument('--out',   type=Path, required=True,
                    help='Path to write JSON.')
    ap.add_argument('--label', type=int, default=1,
                    help='1 = foreground point (default); 0 = background.')
    args = ap.parse_args()

    frame = _load_frame(args.video, args.frame)
    H, W = frame.shape[:2]
    disp = frame.copy()
    points: list[list[int]] = []
    labels: list[int] = []

    def _redraw():
        nonlocal disp
        disp = frame.copy()
        for i, ((x, y), lbl) in enumerate(zip(points, labels)):
            col = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(disp, (x, y), 10, col, 2)
            cv2.putText(disp, f'{i + 1}', (x + 12, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        banner = (f'frame={args.frame}  points={len(points)}  '
                  'LMB=add  s=save  r=reset  u=undo  q=quit')
        cv2.rectangle(disp, (0, 0), (W, 28), (0, 0, 0), -1)
        cv2.putText(disp, banner, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('click init (one left-click per fish)', disp)

    def on_mouse(evt, x, y, flags, _param):
        if evt == cv2.EVENT_LBUTTONDOWN:
            points.append([int(x), int(y)])
            labels.append(int(args.label))
            print(f'click #{len(points)}: ({x}, {y}) label={args.label}')
            _redraw()

    cv2.namedWindow('click init (one left-click per fish)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('click init (one left-click per fish)',
                     min(W, 1280), min(H, 960))
    cv2.setMouseCallback('click init (one left-click per fish)', on_mouse)
    _redraw()
    print(f'[click] video={args.video.name} frame={args.frame} size={W}x{H}')

    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            print('quit without saving')
            cv2.destroyAllWindows()
            return
        if key == ord('r'):
            points.clear(); labels.clear(); _redraw()
            print('reset')
        if key == ord('u') and points:
            points.pop(); labels.pop(); _redraw()
            print(f'undo -> {len(points)} points')
        if key == ord('s'):
            if not points:
                print('no points yet; refusing to save')
                continue
            payload = dict(
                video=str(args.video),
                frame=int(args.frame),
                image_size=[W, H],
                points=points,
                labels=labels,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2))
            print(f'saved {len(points)} points to {args.out}')
            cv2.destroyAllWindows()
            return


if __name__ == '__main__':
    main()
