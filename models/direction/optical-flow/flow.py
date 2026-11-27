"""Dense optical-flow features for circling detection (Step 2).

We run Farneback on the (downsampled) full frame rather than restricting to a
Step-1 ROI, because the mask is still fragmentary / misses stationary fish.
The activity mask for score aggregation comes from the flow magnitude itself,
so this stage is independent of background.py's failure modes.

Per-frame features saved per clip:
    rot_orb   : sum of (r x v) over active pixels, about the activity centroid
                > 0 => CW on screen, < 0 => CCW.    Primary rotation signal.
    rot_omega : best-fit mean angular velocity = rot_orb / sum(r^2).  Normalised.
    rot_sum   : signed sum of pixel-wise curl inside the activity mask.
                Kept for comparison -- near-zero inside rigid translating bodies,
                so it misses the pair rotation (curl lives in the gap between
                the fish, where there is no motion so no active pixels).
    trans_mag : |mean flow vector| inside activity mask          (translation)
    trans_x, trans_y : components of the mean flow vector
    mean_mag  : mean |flow| inside activity mask                 (bulk motion)
    activity  : number of active pixels
    cx, cy    : activity centroid (downsampled-image coords)

Sign convention: image coords have y pointing DOWN, so for r = (rx, ry) and
v = (u, v), the quantity (rx*v - ry*u) is positive for CW-on-screen rotation.
"""
from __future__ import annotations
 
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Small spatial-crop utility used by every stage.
# -----------------------------------------------------------------------------

def center_crop(img: np.ndarray, crop_frac: float) -> np.ndarray:
    """Crop `crop_frac` fraction off each side. crop_frac=0.1 keeps centre 80%.

    Used to exclude tank-wall fixtures / reflections at the frame boundary,
    which otherwise dominate the foreground mask.
    """
    if crop_frac <= 0:
        return img
    H, W = img.shape[:2]
    dw = int(W * crop_frac)
    dh = int(H * crop_frac)
    return img[dh:H - dh, dw:W - dw]


# -----------------------------------------------------------------------------
# Flow + derived fields
# -----------------------------------------------------------------------------

def farneback_flow(prev_gray: np.ndarray, cur_gray: np.ndarray,
                   pyr_scale: float = 0.5, levels: int = 3,
                   winsize: int = 15, iters: int = 3,
                   poly_n: int = 5, poly_sigma: float = 1.2) -> np.ndarray:
    """Dense Farneback flow. Returns (H, W, 2) with (u=dx, v=dy)."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None,
        pyr_scale, levels, winsize, iters, poly_n, poly_sigma, 0,
    )


def flow_curl(flow: np.ndarray) -> np.ndarray:
    """Signed curl field, sign-corrected so CW on screen => positive."""
    u, v = flow[..., 0], flow[..., 1]
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    raw = dv_dx - du_dy        # positive = CCW in a y-up frame
    return -raw                # flip => positive = CW on screen


def flow_divergence(flow: np.ndarray) -> np.ndarray:
    u, v = flow[..., 0], flow[..., 1]
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    return du_dx + dv_dy


# -----------------------------------------------------------------------------
# Per-frame aggregation
# -----------------------------------------------------------------------------

def frame_scores(flow: np.ndarray, mag_thr: float = 0.5) -> dict:
    """Aggregate flow -> scalars over the activity mask (|flow| > mag_thr).

    Primary rotation signal is orbital angular momentum about the (flow-mag
    weighted) centroid of the activity region:  sum_i (rx_i * v_i - ry_i * u_i).
    This is non-zero for a pair rotating around a shared center even when the
    two bodies are individually translating, which is the case for circling.
    """
    u = flow[..., 0]
    v = flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    active = mag > mag_thr
    n = int(active.sum())
    empty = dict(rot_orb=0.0, rot_omega=0.0, rot_sum=0.0, rot_abs=0.0,
                 trans_x=0.0, trans_y=0.0, trans_mag=0.0,
                 mean_mag=0.0, activity=0, cx=np.nan, cy=np.nan)
    if n == 0:
        return empty

    ys, xs = np.where(active)
    us = u[active]
    vs = v[active]
    ms = mag[active]

    # Flow-magnitude-weighted activity centroid
    w_sum = ms.sum()
    cx = float((xs * ms).sum() / w_sum)
    cy = float((ys * ms).sum() / w_sum)
    rx = xs - cx
    ry = ys - cy

    cross = rx * vs - ry * us          # z-component of r x v (CW on screen = +)
    inertia = float((rx * rx + ry * ry).sum())
    rot_orb = float(cross.sum())
    rot_omega = rot_orb / (inertia + 1e-6)

    # Keep pixel-wise curl for diagnostic comparison
    curl = flow_curl(flow)
    rot_sum = float(curl[active].sum())
    rot_abs = float(np.abs(curl[active]).sum())

    return dict(
        rot_orb=rot_orb,
        rot_omega=rot_omega,
        rot_sum=rot_sum,
        rot_abs=rot_abs,
        trans_x=float(us.mean()),
        trans_y=float(vs.mean()),
        trans_mag=float(np.hypot(us.mean(), vs.mean())),
        mean_mag=float(ms.mean()),
        activity=n,
        cx=cx, cy=cy,
    )


# -----------------------------------------------------------------------------
# Clip processing
# -----------------------------------------------------------------------------

def process_clip(clip_path: Path, out_parquet: Path,
                 downsample: float = 2.0,
                 crop_frac: float = 0.1,
                 mag_thr: float = 0.5,
                 blur_ksize: int = 3,
                 start_frame: int = 0,
                 end_frame: int | None = None,
                 frame_stride: int = 1,
                 motion_gate_thr: float = 0.0,
                 progress_every: int = 0) -> pd.DataFrame:
    """Compute per-frame flow features and save to parquet.

    By default processes the entire clip. Pass `start_frame`/`end_frame` to
    process a subrange of the source video (useful for running on the full
    10 h source mp4 without extracting an intermediate clip).

    All spatial coordinates in the returned DF are in the (downsampled, centre-
    cropped) coord system -- same convention as background.process_clip.

    `blur_ksize` applies a small Gaussian blur before computing flow to
    suppress single-pixel sand-speckle noise. 0 or 1 disables.

    `frame_stride > 1` samples every Nth frame to speed processing. The flow
    between sampled frames represents `stride` frames of real motion, so we
    rescale it by 1/stride to keep magnitudes consistent with stride=1. Frame
    indices stored in the parquet are the ORIGINAL source frame indices.

    `motion_gate_thr > 0` skips Farneback on frame pairs whose mean absolute
    frame-difference is below this threshold (in grayscale units per pixel).
    Produces zero flow features for those samples -- useful for quiet
    stretches (fish sleeping, lights off).

    `progress_every` > 0 prints a progress line every N processed frames.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f'could not open {clip_path}')
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None or end_frame > n_total:
        end_frame = n_total
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst_w = int(round(src_w / downsample))
    dst_h = int(round(src_h / downsample))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    import time as _time
    _t0 = _time.time()

    empty_feat = dict(rot_orb=0.0, rot_omega=0.0, rot_sum=0.0, rot_abs=0.0,
                      trans_x=0.0, trans_y=0.0, trans_mag=0.0,
                      mean_mag=0.0, activity=0, cx=np.nan, cy=np.nan)

    stride = max(1, int(frame_stride))

    rows: list[dict] = []
    prev = None
    n_gated = 0
    fi = start_frame
    read_counter = 0
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (read_counter % stride) != 0:
            # Skip this raw frame -- we only sample every `stride`-th one.
            read_counter += 1
            fi += 1
            continue
        read_counter += 1

        small = cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        small = center_crop(small, crop_frac)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if blur_ksize and blur_ksize > 1:
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        if prev is not None:
            # Motion gate: check whether anything moved
            if motion_gate_thr > 0:
                diff = float(np.mean(np.abs(gray.astype(np.int16) -
                                             prev.astype(np.int16))))
            else:
                diff = None
            if diff is not None and diff < motion_gate_thr:
                s = dict(empty_feat)
                n_gated += 1
            else:
                flow = farneback_flow(prev, gray)
                if stride > 1:
                    flow = flow / stride
                s = frame_scores(flow, mag_thr=mag_thr)
            s['frame'] = fi
            rows.append(s)
        prev = gray
        fi += 1
        if progress_every and (fi - start_frame) % progress_every == 0:
            elapsed = _time.time() - _t0
            done = fi - start_frame
            total = end_frame - start_frame
            rate = done / elapsed if elapsed else 0
            eta = (total - done) / rate if rate else 0
            gated_pct = 100 * n_gated / max(1, len(rows))
            print(f'  flow: {done}/{total} frames  '
                  f'({100*done/total:.1f}%)  {rate:.1f} fps  '
                  f'gated={gated_pct:.0f}%  eta {eta/60:.1f} min', flush=True)
    cap.release()

    df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)
    return df


# -----------------------------------------------------------------------------
# Visualisation helper (returns BGR, one frame)
# -----------------------------------------------------------------------------

def flow_to_hsv(flow: np.ndarray, mag_scale: float = 3.0) -> np.ndarray:
    """Standard HSV flow viz: hue = angle, value = clipped magnitude."""
    u, v = flow[..., 0], flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    ang = (np.arctan2(v, u) + np.pi) * 180 / (2 * np.pi)  # 0..180
    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ang.astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * (255.0 / mag_scale), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# -----------------------------------------------------------------------------
# Quiver (LK-style) visualisation for a single frame pair
# -----------------------------------------------------------------------------

def compute_flow_at_frame(clip_path: Path, frame_idx: int,
                          downsample: float = 2.0,
                          crop_frac: float = 0.1,
                          blur_ksize: int = 3):
    """Return (small_frame_bgr, flow_hw2) for the transition (f-1 -> f).

    No visualisation side effects; used by viz helpers and notebooks.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f'could not open {clip_path}')
    W = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  / downsample))
    H = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / downsample))
    frame_idx = max(1, int(frame_idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
    ok, f_prev = cap.read()
    if not ok: raise RuntimeError(f'read failed at f={frame_idx-1}')
    ok, f_cur = cap.read()
    if not ok: raise RuntimeError(f'read failed at f={frame_idx}')
    cap.release()

    small_prev = center_crop(cv2.resize(f_prev, (W, H), interpolation=cv2.INTER_AREA), crop_frac)
    small_cur  = center_crop(cv2.resize(f_cur,  (W, H), interpolation=cv2.INTER_AREA), crop_frac)
    g_prev = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
    g_cur  = cv2.cvtColor(small_cur,  cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize > 1:
        g_prev = cv2.GaussianBlur(g_prev, (blur_ksize, blur_ksize), 0)
        g_cur  = cv2.GaussianBlur(g_cur,  (blur_ksize, blur_ksize), 0)
    flow = farneback_flow(g_prev, g_cur)
    return small_cur, flow


def quiver_around_activity(clip_path: Path, frame_idx: int, out_png: Path,
                           downsample: float = 2.0,
                           crop_frac: float = 0.1,
                           mag_thr: float = 0.5,
                           blur_ksize: int = 3,
                           crop_half: int = 80,
                           step: int = 4,
                           arrow_scale: float = 0.3) -> dict:
    """LK-style quiver plot of the flow field around the activity centroid.

    Centroid is the flow-magnitude-weighted mean of active pixels at this
    frame; if no active pixels, falls back to frame centre.

    `arrow_scale` is the matplotlib quiver scale parameter (smaller -> longer
    arrows). With scale_units='xy' and scale=1 each arrow is drawn at native
    pixel length, which is usually too subtle for flows ~1 px/frame.
    """
    import matplotlib.pyplot as plt  # local import keeps flow.py headless-safe

    small_cur, flow = compute_flow_at_frame(clip_path, frame_idx,
                                            downsample=downsample,
                                            crop_frac=crop_frac,
                                            blur_ksize=blur_ksize)
    H, W = small_cur.shape[:2]
    u = flow[..., 0]
    v = flow[..., 1]
    mag = np.hypot(u, v)
    active = mag > mag_thr
    n = int(active.sum())
    if n > 0:
        ys, xs = np.where(active)
        w = mag[active]
        cx = int((xs * w).sum() / w.sum())
        cy = int((ys * w).sum() / w.sum())
    else:
        cx, cy = W // 2, H // 2

    x0 = max(0, cx - crop_half)
    x1 = min(W, cx + crop_half)
    y0 = max(0, cy - crop_half)
    y1 = min(H, cy + crop_half)

    bg = cv2.cvtColor(small_cur[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    uc = u[y0:y1, x0:x1]
    vc = v[y0:y1, x0:x1]
    mc = mag[y0:y1, x0:x1]

    Y, X = np.mgrid[0:bg.shape[0]:step, 0:bg.shape[1]:step]
    U = uc[::step, ::step]
    V = vc[::step, ::step]
    M = mc[::step, ::step]
    show = M > mag_thr

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(bg)
    ax.plot(cx - x0, cy - y0, marker='x', color='red', ms=10, mew=2)
    if show.any():
        q = ax.quiver(X[show], Y[show], U[show], V[show], M[show],
                      angles='xy', scale_units='xy', scale=arrow_scale,
                      cmap='plasma', width=0.004)
        plt.colorbar(q, ax=ax, label='|flow| (px/frame)', fraction=0.046, pad=0.04)
    ax.set_title(f'{clip_path.stem}  f={frame_idx}  '
                 f'centroid=({cx},{cy})  n_active={n}', fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return dict(path=out_png, cx=cx, cy=cy, activity=n)
