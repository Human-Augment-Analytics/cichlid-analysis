"""Experiment: Farneback optical flow computed over SAM 2 masks.

Given a SAM 2 segmentation output directory (centroids.csv + masks saved OR
the resized JPEG dir), compute per-frame Farneback flow between consecutive
frames and aggregate the flow INSIDE each per-object mask.  Returns a
per-frame feature series suitable for downstream span detection.

Per-frame per-object features:
    mean_u, mean_v     mean flow inside the object mask
    speed              |(mean_u, mean_v)|
    rot_orb            sum (rx*v - ry*u) over mask pixels, about mask centroid
                       Sign convention (image coords, y DOWN): + = CW
    rot_omega          rot_orb / sum(rx^2 + ry^2) (normalised angular vel)
    area               # mask pixels

Per-frame PAIR features:
    pair_rot_orb       sum (rx*v - ry*u) over union of masks, about pair midpoint
    pair_rot_omega     normalised
    pair_distance      |c1 - c2|   (centroid distance in pixels)

Aggregate over the clip:
    direction_sign_majority:  sign-majority of pair_rot_orb (+ => CW)

Reconstructs frames from the SAM 2 overlay JPEGs if they exist; otherwise
reads from the original video (slower).

Usage
-----
    python sam2_of_features.py \
        --sam2-dir ../../../outputs/optical-flow/sam2/0007_vid_day05_vid5-v07_d05_evt_0001 \
        --video ../../ground_truth_annotations/data/0007_vid_day05.mp4 \
        --event-start 20142 --event-end 21044 \
        --out ../../../outputs/optical-flow/sam2/0007_vid_day05_vid5-v07_d05_evt_0001/of_features.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Mask reconstruction from centroids.csv is not possible — we need actual
# masks.  Our sam2_segment_event.py doesn't save masks by default (only
# centroids).  We reconstruct masks by RE-RUNNING SAM 2 with --save-masks.
#
# For this experiment we take a different shortcut: read the overlay.mp4,
# which shows the colored masks; we recover per-fish masks by colour.
# -----------------------------------------------------------------------------

# Colours used in sam2_segment_event.py render_overlay() (first N objects):
_OBJ_COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 255),
               (0, 255, 255), (128, 0, 255), (255, 128, 0)]


def _mask_from_colour(frame_bgr: np.ndarray, col_bgr: tuple[int, int, int],
                      tol: int = 40) -> np.ndarray:
    """Threshold-based colour mask.  Used only as a fallback when real masks
    aren't saved; it will fail if the contour-only overlay is drawn at low
    alpha.  Prefer the --save-masks SAM 2 output.
    """
    b, g, r = cv2.split(frame_bgr)
    cb, cg, cr = col_bgr
    return ((np.abs(b.astype(int) - cb) < tol)
            & (np.abs(g.astype(int) - cg) < tol)
            & (np.abs(r.astype(int) - cr) < tol)).astype(np.uint8) * 255


# -----------------------------------------------------------------------------
# Main feature extractor
# -----------------------------------------------------------------------------

def _frame_features(flow: np.ndarray, mask: np.ndarray) -> dict:
    """flow: (H, W, 2) (u=dx, v=dy).  mask: bool."""
    if mask.sum() < 5:
        return dict(mean_u=np.nan, mean_v=np.nan, speed=np.nan,
                    rot_orb=np.nan, rot_omega=np.nan, area=int(mask.sum()))
    u = flow[..., 0]; v = flow[..., 1]
    mu = float(u[mask].mean()); mv = float(v[mask].mean())
    speed = float(np.sqrt(mu * mu + mv * mv))
    ys, xs = np.where(mask)
    cx = xs.mean(); cy = ys.mean()
    rx = xs - cx; ry = ys - cy
    # Image coords: positive (rx * v - ry * u) = CW on screen.
    rot_orb = float((rx * v[mask] - ry * u[mask]).sum())
    denom = float((rx * rx + ry * ry).sum())
    rot_omega = rot_orb / denom if denom > 0 else 0.0
    return dict(mean_u=mu, mean_v=mv, speed=speed,
                rot_orb=rot_orb, rot_omega=rot_omega, area=int(mask.sum()))


def _pair_features(flow: np.ndarray, m1: np.ndarray, m2: np.ndarray,
                   c1: tuple[float, float], c2: tuple[float, float]) -> dict:
    """Rotation of the pair about its midpoint; centroid distance."""
    pair_mask = m1 | m2
    if pair_mask.sum() < 10:
        return dict(pair_rot_orb=np.nan, pair_rot_omega=np.nan,
                    pair_distance=np.nan)
    midx = 0.5 * (c1[0] + c2[0])
    midy = 0.5 * (c1[1] + c2[1])
    u = flow[..., 0]; v = flow[..., 1]
    ys, xs = np.where(pair_mask)
    rx = xs - midx; ry = ys - midy
    rot_orb = float((rx * v[pair_mask] - ry * u[pair_mask]).sum())
    denom = float((rx * rx + ry * ry).sum())
    rot_omega = rot_orb / denom if denom > 0 else 0.0
    dist = float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
    return dict(pair_rot_orb=rot_orb, pair_rot_omega=rot_omega,
                pair_distance=dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',        type=Path, required=True)
    ap.add_argument('--masks-dir',    type=Path, required=True,
                    help='Dir of per-frame PNG masks written by '
                         'sam2_segment_event.py --save-masks '
                         '(filename=<original_frame_idx>.png, pixel value = obj_id).')
    ap.add_argument('--centroids',    type=Path, required=True,
                    help='SAM 2 centroids.csv for this event.')
    ap.add_argument('--event-start',  type=int, required=True)
    ap.add_argument('--event-end',    type=int, required=True)
    ap.add_argument('--out',          type=Path, required=True)
    ap.add_argument('--downsample',   type=float, default=2.0,
                    help='Downsample video+mask before flow (speeds up).')
    args = ap.parse_args()

    cents = pd.read_csv(args.centroids)
    # Use the full-resolution centroids (cx_full, cy_full) matching mask frame coords.

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.event_start)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError('could not read first frame')
    H, W = frame.shape[:2]
    scale = 1.0 / args.downsample

    def _ds(img):
        if args.downsample <= 1:
            return img
        return cv2.resize(img, (int(W * scale), int(H * scale)),
                          interpolation=cv2.INTER_AREA)

    def _ds_mask(m):
        if args.downsample <= 1:
            return m
        return cv2.resize(m, (int(W * scale), int(H * scale)),
                          interpolation=cv2.INTER_NEAREST)

    gray_prev = _ds(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    rows = []
    t0 = time.perf_counter()
    for f in range(args.event_start + 1, args.event_end):
        ok, frame = cap.read()
        if not ok:
            break
        gray = _ds(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # Load the mask label image for this frame.
        mask_path = args.masks_dir / f'{f:06d}.png'
        if not mask_path.exists():
            continue
        label_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        label_img = _ds_mask(label_img)
        m1 = (label_img == 1); m2 = (label_img == 2)
        if m1.sum() < 10 or m2.sum() < 10:
            continue

        # Farneback.
        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        ff1 = _frame_features(flow, m1)
        ff2 = _frame_features(flow, m2)

        # Centroids for pair features (scale to downsampled coords).
        c1_row = cents[(cents.frame == f) & (cents.obj_id == 1)].iloc[0]
        c2_row = cents[(cents.frame == f) & (cents.obj_id == 2)].iloc[0]
        c1 = (c1_row.cx_full * scale, c1_row.cy_full * scale)
        c2 = (c2_row.cx_full * scale, c2_row.cy_full * scale)
        pf = _pair_features(flow, m1, m2, c1, c2)

        rows.append({'frame': f,
                     **{f'f1_{k}': v for k, v in ff1.items()},
                     **{f'f2_{k}': v for k, v in ff2.items()},
                     **pf})
        gray_prev = gray
    cap.release()
    dt = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n = len(df)
    print(f'processed {n} frames in {dt:.1f}s = {n/dt:.1f} fps')
    print(f'wrote {args.out}')

    # Aggregate stats over clip.
    if n:
        pr = df['pair_rot_orb'].dropna()
        pos = int((pr > 0).sum()); neg = int((pr < 0).sum())
        direction = 'CW' if pos > neg else 'CCW'
        print(f'\n=== direction via Farneback-on-masks ===')
        print(f'pair_rot_orb: pos={pos} neg={neg} => {direction}')
        print(f'mean pair_distance: {df["pair_distance"].mean():.1f} px')
        print(f'mean pair_rot_omega: {df["pair_rot_omega"].mean():.4f}')


if __name__ == '__main__':
    main()
