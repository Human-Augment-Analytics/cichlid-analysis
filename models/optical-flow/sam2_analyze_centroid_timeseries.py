"""Plot centroid-derived features over time for one SAM 2 output.

Reads a centroids.csv and plots:
    pair_distance             — fish-to-fish centroid distance
    instantaneous_rot_omega   — rate of change of pair angular position
    cumulative_turns          — unwrapped pair angle / (2 * pi)
    speed_per_fish            — centroid speed per fish
    signed_rot_sign           — sign of rot_omega per frame (CW/CCW/0)

Also overlays any GT circling events from the annotation CSV on each
subplot so the predicted features can be visually matched to GT span
boundaries.

Usage
-----
    python sam2_analyze_centroid_timeseries.py \
        --centroids ../../../outputs/.../centroids.csv \
        --annotations ../../ground_truth_annotations/.../csv \
        --event-start 19272 --event-end 21914 \
        --out ../../../outputs/.../features_timeseries.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FPS_DEFAULT = 29


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--centroids',   type=Path, required=True)
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--event-start', type=int,  required=True,
                    help='Window start frame (first frame in centroids).')
    ap.add_argument('--event-end',   type=int,  required=True)
    ap.add_argument('--out',         type=Path, required=True)
    ap.add_argument('--fps',         type=float, default=FPS_DEFAULT)
    ap.add_argument('--window',      type=int,  default=5,
                    help='Rolling window (frames) for smoothing rot_omega.')
    args = ap.parse_args()

    df = pd.read_csv(args.centroids)
    piv = (df.pivot(index='frame', columns='obj_id',
                    values=['cx_full', 'cy_full'])
           .sort_index())
    cx1 = piv[('cx_full', 1)].to_numpy()
    cy1 = piv[('cy_full', 1)].to_numpy()
    cx2 = piv[('cx_full', 2)].to_numpy()
    cy2 = piv[('cy_full', 2)].to_numpy()
    frames = piv.index.to_numpy()

    midx = 0.5 * (cx1 + cx2); midy = 0.5 * (cy1 + cy2)
    pair_dist = np.hypot(cx1 - cx2, cy1 - cy2)

    theta1 = np.arctan2(cy1 - midy, cx1 - midx)
    theta1_u = np.unwrap(theta1)
    cum_turns = (theta1_u - theta1_u[0]) / (2 * np.pi)   # fish 1 as reference
    # Instantaneous rot_omega (radians / frame), rolling-smoothed.
    rot_omega_raw = np.gradient(theta1_u)
    rot_omega = pd.Series(rot_omega_raw).rolling(args.window, center=True,
                                                  min_periods=1).mean().to_numpy()

    # Speeds.
    s1 = np.hypot(np.gradient(cx1), np.gradient(cy1))
    s2 = np.hypot(np.gradient(cx2), np.gradient(cy2))

    # ----- GT event overlay -----
    gt = pd.read_csv(args.annotations)
    gt.columns = [c.strip() for c in gt.columns]
    gt_in_win = gt[(gt['End Frame'] > args.event_start)
                   & (gt['Start Frame'] < args.event_end)]

    # ----- plot -----
    t = (frames - frames[0]) / args.fps
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax in axes:
        for _, r in gt_in_win.iterrows():
            t0 = (r['Start Frame'] - frames[0]) / args.fps
            t1 = (r['End Frame']   - frames[0]) / args.fps
            ax.axvspan(t0, t1, color='orange', alpha=0.15, label='GT circling')

    axes[0].plot(t, pair_dist, color='black')
    axes[0].set_ylabel('pair distance\n(px, full res)')
    axes[0].set_title(f'frames {frames[0]}-{frames[-1]}  '
                      f'({len(frames)} frames @ {args.fps} fps)')

    axes[1].plot(t, rot_omega, color='tab:blue')
    axes[1].axhline(0, color='grey', lw=0.5)
    axes[1].set_ylabel(f'rot_omega\n(rad/frame, smoothed w={args.window})')

    axes[2].plot(t, cum_turns, color='tab:green')
    axes[2].set_ylabel('cumulative turns\n(fish 1, unwrapped)')

    axes[3].plot(t, s1, label='fish 1', color='tab:red')
    axes[3].plot(t, s2, label='fish 2', color='tab:purple')
    axes[3].set_ylabel('centroid speed\n(px/frame)')
    axes[3].set_xlabel('time within window (s)')
    axes[3].legend(loc='upper right')

    # Dedup legend labels (axvspan adds "GT circling" per event).
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        ax.legend(uniq.values(), uniq.keys(), loc='upper left')

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print(f'wrote {args.out}')
    print(f'cumulative turns over window: {cum_turns[-1]:.2f}')
    print(f'GT events overlapping window: {len(gt_in_win)}')


if __name__ == '__main__':
    main()
