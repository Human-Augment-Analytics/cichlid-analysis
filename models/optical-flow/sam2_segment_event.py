"""SAM 2 video propagation across one GT event.

Loads a clicks.json (from sam2_click_init.py), initialises SAM 2's video
predictor with those point prompts on the event's first frame, propagates
masks forward across the event span, and writes:

    out_dir/<video_stem>_<event_id>/
        overlay.mp4      # original frames + per-object mask + centroid
        masks/           # per-frame PNG (compressed, obj_id in label value)
        centroids.csv    # frame, obj_id, cx, cy, area
        summary.json     # init info + runtime + per-object counts

Assumes SAM 2 and its checkpoint are already installed/downloaded.  Default
checkpoint path: ~/sam2_checkpoints/sam2.1_hiera_base_plus.pt .

Usage
-----
    python sam2_segment_event.py \
        --video ../../ground_truth_annotations/data/0007_vid_day05.mp4 \
        --annotations ../../ground_truth_annotations/finalized_annotations/circling_annotations_vid0007-day5.csv \
        --event-id vid5-v07_d05_evt_0001 \
        --clicks ../../../outputs/optical-flow/sam2/vid5-v07_d05_evt_0001_clicks.json \
        --out ../../../outputs/optical-flow/sam2/
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

FPS_DEFAULT = 29
DEFAULT_CKPT = Path.home() / 'sam2_checkpoints' / 'sam2.1_hiera_base_plus.pt'
DEFAULT_CFG  = 'configs/sam2.1/sam2.1_hiera_b+.yaml'


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_event(ann_csv: Path, event_id: str) -> dict:
    df = pd.read_csv(ann_csv)
    df.columns = [c.strip() for c in df.columns]
    row = df[df['EventID'].astype(str) == event_id]
    if row.empty:
        raise ValueError(f'EventID {event_id!r} not found in {ann_csv}')
    r = row.iloc[0]
    return dict(event_id=event_id,
                start_frame=int(r['Start Frame']),
                end_frame=int(r['End Frame']),
                direction=str(r['Rotation direction']))


def dump_event_frames_as_jpegs(video: Path, start: int, end: int,
                                dst: Path, max_side: int | None = None
                                ) -> tuple[list[Path], float]:
    """SAM 2's video predictor consumes a FOLDER of JPEGs (one per frame).

    If `max_side` is given, frames are resized so max(H, W) == max_side
    (preserves aspect ratio).  Returns (paths, scale) where `scale` is the
    ratio new_side / orig_side, needed to rescale click coords.
    """
    dst.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    paths = []
    scale = 1.0
    try:
        for i, f in enumerate(range(start, end)):
            ok, frame = cap.read()
            if not ok:
                break
            if max_side is not None:
                H, W = frame.shape[:2]
                m = max(H, W)
                if m > max_side:
                    scale = max_side / m
                    frame = cv2.resize(frame,
                                       (int(round(W * scale)), int(round(H * scale))),
                                       interpolation=cv2.INTER_AREA)
            p = dst / f'{i:06d}.jpg'
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            paths.append(p)
    finally:
        cap.release()
    return paths, scale


def centroid_of_mask(mask: np.ndarray) -> tuple[float, float, int]:
    """Return (cx, cy, area) of a binary mask (True = object). (-1, -1, 0)
    if empty."""
    if mask.sum() == 0:
        return -1.0, -1.0, 0
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean()), int(mask.sum())


# -----------------------------------------------------------------------------
# Overlay rendering
# -----------------------------------------------------------------------------

_OBJ_COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 255),
               (0, 255, 255), (128, 0, 255), (255, 128, 0)]


def render_overlay(frame_bgr: np.ndarray, obj_masks: dict[int, np.ndarray],
                   obj_centroids: dict[int, tuple[float, float]],
                   info: str) -> np.ndarray:
    out = frame_bgr.copy()
    for obj_id, mask in obj_masks.items():
        col = _OBJ_COLORS[(obj_id - 1) % len(_OBJ_COLORS)]
        # Contour
        bm = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, col, 2)
        # Translucent fill
        overlay = out.copy()
        overlay[mask] = col
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
        # Centroid dot
        cx, cy = obj_centroids.get(obj_id, (-1.0, -1.0))
        if cx >= 0:
            cv2.circle(out, (int(cx), int(cy)), 6, col, -1)
            cv2.putText(out, f'#{obj_id}', (int(cx) + 8, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    # Banner
    band = np.zeros((28, out.shape[1], 3), dtype=np.uint8)
    cv2.putText(band, info, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, out])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',       type=Path, required=True)
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--event-id',    type=str, required=True)
    ap.add_argument('--clicks',      type=Path, required=True,
                    help='JSON from sam2_click_init.py.')
    ap.add_argument('--out',         type=Path, required=True)
    ap.add_argument('--ckpt',        type=Path, default=DEFAULT_CKPT)
    ap.add_argument('--model-cfg',   type=str,  default=DEFAULT_CFG)
    ap.add_argument('--fps',         type=float, default=FPS_DEFAULT)
    ap.add_argument('--device',      type=str,  default='cuda')
    ap.add_argument('--save-masks',  action='store_true',
                    help='Also dump per-frame PNG masks to <out>/masks/.')
    ap.add_argument('--max-input-side', type=int, default=720,
                    help='Resize JPEGs so max(H,W)==this.  Keeps 6GB cards '
                         'from OOMing on long events.  0 = native resolution.')
    ap.add_argument('--offload-video-to-cpu', action='store_true', default=True,
                    help='Keep JPEG tensor on CPU; move to GPU per-frame.')
    ap.add_argument('--offload-state-to-cpu', action='store_true', default=True,
                    help='Keep SAM 2 memory-bank state on CPU.  Essential '
                         'for propagation across >~100 frames on 6 GB VRAM.')
    ap.add_argument('--no-offload', action='store_true',
                    help='Disable BOTH offload flags.  Use when model + state '
                         'fit in VRAM (e.g. tiny @ 360p).  Much faster.')
    ap.add_argument('--state-to-gpu', action='store_true',
                    help='Keep SAM 2 memory-bank state on GPU while leaving '
                         'video offload-to-cpu on.  Middle ground for tiny.')
    ap.add_argument('--override-start', type=int, default=None,
                    help='Override event start frame (for extended/custom spans).')
    ap.add_argument('--override-end',   type=int, default=None,
                    help='Override event end frame.')
    ap.add_argument('--async-loading', action='store_true',
                    help='Load frames lazily (avoids 33GB preallocation for '
                         'long spans).  Slightly slower but essential for '
                         '>1500 frames.')
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f'checkpoint not found: {args.ckpt}')

    clicks = json.loads(args.clicks.read_text())
    points = np.asarray(clicks['points'], dtype=np.float32)   # (N,2)
    labels = np.asarray(clicks['labels'], dtype=np.int32)      # (N,)
    print(f'[init] {len(points)} points; labels={labels.tolist()}')

    ev = load_event(args.annotations, args.event_id)
    if args.override_start is not None:
        ev['start_frame'] = int(args.override_start)
    if args.override_end is not None:
        ev['end_frame'] = int(args.override_end)
    print(f'[event] {ev["event_id"]} frames=[{ev["start_frame"]},{ev["end_frame"]}) '
          f'dir={ev["direction"]}')
    if clicks['frame'] != ev['start_frame']:
        print(f'  NOTE: clicks are on video frame {clicks["frame"]} but '
              f"event starts at {ev['start_frame']}.  Clicks will be "
              f'applied to the FIRST frame of the event span.')

    out_dir = args.out / f'{args.video.stem}_{args.event_id}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Dump event frames to JPEG dir (SAM 2 video predictor needs this) ---
    tmp_dir = Path(tempfile.mkdtemp(prefix='sam2_frames_'))
    try:
        max_side = args.max_input_side if args.max_input_side > 0 else None
        print(f'[dump] writing {ev["end_frame"] - ev["start_frame"]} frames '
              f'to {tmp_dir} (max_side={max_side})')
        t0 = time.time()
        frame_paths, scale = dump_event_frames_as_jpegs(args.video,
                                                        ev['start_frame'],
                                                        ev['end_frame'], tmp_dir,
                                                        max_side=max_side)
        print(f'[dump] {len(frame_paths)} frames  scale={scale:.3f}  '
              f'({time.time() - t0:.1f}s)')

        # Rescale click coords to match the resized JPEGs.
        if scale != 1.0:
            points = points * scale
            print(f'[prompt] click coords rescaled by {scale:.3f}')

        # --- Build SAM 2 video predictor ---
        from sam2.build_sam import build_sam2_video_predictor
        print(f'[model] building SAM 2 video predictor on {args.device}')
        predictor = build_sam2_video_predictor(args.model_cfg, str(args.ckpt),
                                               device=args.device)

        # --- Init state + add one object per click ---
        with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
            if args.no_offload:
                offload_video = False
                offload_state = False
            else:
                offload_video = args.offload_video_to_cpu
                offload_state = args.offload_state_to_cpu
            if args.state_to_gpu:
                offload_state = False
            print(f'[init_state] offload_video={offload_video} '
                  f'offload_state={offload_state}')
            state = predictor.init_state(
                video_path=str(tmp_dir),
                offload_video_to_cpu=offload_video,
                offload_state_to_cpu=offload_state,
                async_loading_frames=args.async_loading,
            )

            # One SAM 2 object per click point (each click = one fish).
            for i, (pt, lbl) in enumerate(zip(points, labels)):
                obj_id = i + 1
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,            # first frame of the event span
                    obj_id=obj_id,
                    points=pt[None, :],     # shape (1, 2)
                    labels=np.array([lbl], dtype=np.int32),
                )
                print(f'[prompt] obj_id={obj_id} at pt={pt.tolist()} label={lbl}')

            # --- Propagate ---
            print('[propagate] running forward across event ...')
            t0 = time.time()
            # propagate_in_video yields (frame_idx, obj_ids, mask_logits)
            per_frame = {}   # frame_idx -> {obj_id: bool_mask}
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                masks = {}
                for i_o, obj_id in enumerate(obj_ids):
                    m = (mask_logits[i_o] > 0.0).cpu().numpy()
                    if m.ndim == 3:
                        m = m[0]
                    masks[int(obj_id)] = m.astype(bool)
                per_frame[int(frame_idx)] = masks
            dt = time.time() - t0
            print(f'[propagate] {len(per_frame)} frames  ({dt:.1f}s = '
                  f'{len(per_frame)/dt:.1f} fps)')

        # --- Extract centroids, render overlay ---
        # Read from the resized JPEGs so overlay/mask dimensions match.
        rows = []
        writer = None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        masks_dir = out_dir / 'masks'
        if args.save_masks:
            masks_dir.mkdir(exist_ok=True)
        inv_scale = 1.0 / scale if scale > 0 else 1.0

        try:
            for i, f in enumerate(range(ev['start_frame'], ev['end_frame'])):
                jpg_path = tmp_dir / f'{i:06d}.jpg'
                if not jpg_path.exists():
                    break
                frame_bgr = cv2.imread(str(jpg_path))
                masks = per_frame.get(i, {})
                obj_centroids = {}
                for obj_id, m in masks.items():
                    cx, cy, area = centroid_of_mask(m)
                    obj_centroids[obj_id] = (cx, cy)
                    # cx_full / cy_full are in original video pixels.
                    rows.append({'frame': f, 'obj_id': obj_id,
                                 'cx': cx, 'cy': cy, 'area': area,
                                 'cx_full': cx * inv_scale if cx >= 0 else -1,
                                 'cy_full': cy * inv_scale if cy >= 0 else -1,
                                 'area_full': int(area * inv_scale * inv_scale)})
                info = (f'frame {f}  objs={len(masks)}  dir(GT)={ev["direction"]}')
                vis = render_overlay(frame_bgr, masks, obj_centroids, info)
                if writer is None:
                    writer = cv2.VideoWriter(str(out_dir / 'overlay.mp4'),
                                             fourcc, args.fps,
                                             (vis.shape[1], vis.shape[0]))
                writer.write(vis)
                if args.save_masks:
                    # Combine per-object masks into one label image.
                    label_img = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
                    for obj_id, m in masks.items():
                        label_img[m] = obj_id
                    cv2.imwrite(str(masks_dir / f'{f:06d}.png'), label_img)
        finally:
            if writer is not None:
                writer.release()

        pd.DataFrame(rows).to_csv(out_dir / 'centroids.csv', index=False)

        summary = dict(
            video=str(args.video),
            event=ev,
            clicks=clicks,
            n_frames=len(per_frame),
            prop_time_s=dt,
            prop_fps=len(per_frame) / dt,
            obj_ids=sorted(set(r['obj_id'] for r in rows)),
        )
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2,
                                                          default=str))
        print(f'\nwrote {out_dir / "overlay.mp4"}')
        print(f'wrote {out_dir / "centroids.csv"}')
        print(f'wrote {out_dir / "summary.json"}')

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
