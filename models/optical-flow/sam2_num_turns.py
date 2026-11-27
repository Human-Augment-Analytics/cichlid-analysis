"""Compute num_turns and rotation direction from SAM 2 centroid tracks.

Method (classical centroid rotation):
    For each frame: compute pair-midpoint M = (c1 + c2) / 2.
    For each fish:   angle(t) = atan2(cy_i - M.y, cx_i - M.x)   [image coords]
    Unwrap both angle series.
    turns_i = (angle_i[-1] - angle_i[0]) / (2 * pi).

Sign convention (image coords, y-axis DOWN):
    POSITIVE dtheta => CW on screen
    NEGATIVE dtheta => CCW on screen

The two fish are head-to-tail during a circling bout, so their angles
around the pair midpoint stay ~180 deg apart and change at the same rate.
We report turns per fish and a joint estimate (mean of |turns|), plus a
consistency check (max |turns_i - turns_j|).

Post-processing:
    quantized turns:
        fractional part >= 0.9              -> round up          (i+1)
        0.5  <= fractional part < 0.9       -> half turn         (i + 0.5)
        fractional part < 0.5               -> round down        (i)
    is_circling:
        quantized |turns| >= 1   (i.e., at least one complete circle).
        Events below 1 full turn do not qualify as circling bouts.

Usage
-----
    python sam2_num_turns.py \
        --centroids <path/to/centroids.csv> \
        --annotations <path/to/GT_csv> \
        --event-id vid5-v07_d05_evt_0001 \
        [--out num_turns.json]

Also supports --centroids-dir to batch-process a tree of per-event folders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Quantization + circling-classification rules
# -----------------------------------------------------------------------------

def quantize_turns(raw: float) -> float:
    """Apply the 3-tier rounding rule.  Preserves sign of raw."""
    import math
    sign = 1.0 if raw >= 0 else -1.0
    a = abs(raw)
    i = math.floor(a)
    frac = a - i
    if frac >= 0.9:
        q = i + 1
    elif frac >= 0.5:
        q = i + 0.5
    else:
        q = i
    return sign * q


def is_circling(quantized_turns: float, min_turns: float = 1.0) -> bool:
    """A bout qualifies as circling iff at least one FULL circle was completed."""
    return abs(quantized_turns) >= min_turns


# -----------------------------------------------------------------------------
# Core turn-counting
# -----------------------------------------------------------------------------

def num_turns_from_centroids(df: pd.DataFrame) -> dict:
    """df: SAM 2 centroids with columns frame, obj_id, cx_full, cy_full.

    Expects exactly 2 objects.  Returns a dict with per-fish turns, joint
    estimate, direction, and a consistency metric.
    """
    assert set(df['obj_id'].unique()) >= {1, 2}, 'need exactly 2 obj_ids'
    # Pivot to wide format: one row per frame, columns cx_1, cy_1, cx_2, cy_2.
    piv = (df[['frame', 'obj_id', 'cx_full', 'cy_full']]
           .pivot(index='frame', columns='obj_id',
                  values=['cx_full', 'cy_full'])
           .sort_index())
    cx1 = piv[('cx_full', 1)].to_numpy()
    cy1 = piv[('cy_full', 1)].to_numpy()
    cx2 = piv[('cx_full', 2)].to_numpy()
    cy2 = piv[('cy_full', 2)].to_numpy()
    mask_valid = ~(np.isnan(cx1) | np.isnan(cy1) | np.isnan(cx2) | np.isnan(cy2))

    if mask_valid.sum() < 2:
        return dict(n_valid_frames=int(mask_valid.sum()),
                    error='not enough valid frames')

    # Clip to valid stretch (contiguous assumption: SAM 2 gives dense tracks).
    cx1 = cx1[mask_valid]; cy1 = cy1[mask_valid]
    cx2 = cx2[mask_valid]; cy2 = cy2[mask_valid]

    mid_x = 0.5 * (cx1 + cx2)
    mid_y = 0.5 * (cy1 + cy2)

    theta1 = np.arctan2(cy1 - mid_y, cx1 - mid_x)
    theta2 = np.arctan2(cy2 - mid_y, cx2 - mid_x)

    theta1_u = np.unwrap(theta1)
    theta2_u = np.unwrap(theta2)

    turns1 = float((theta1_u[-1] - theta1_u[0]) / (2 * np.pi))
    turns2 = float((theta2_u[-1] - theta2_u[0]) / (2 * np.pi))

    # Joint estimate: average, preserving sign (per-fish signs should agree).
    if turns1 * turns2 >= 0:
        joint = (turns1 + turns2) / 2.0
    else:
        # Signs disagree -- probably noise.  Fall back to the larger-|.|.
        joint = turns1 if abs(turns1) > abs(turns2) else turns2

    direction = 'CW' if joint > 0 else ('CCW' if joint < 0 else 'NONE')

    # Consistency check.
    consistency = float(abs(turns1 - turns2))

    q = quantize_turns(joint)
    circling = is_circling(q)

    return dict(
        n_valid_frames=int(len(cx1)),
        turns_fish1=turns1,
        turns_fish2=turns2,
        turns_joint=joint,
        turns_abs=abs(joint),
        turns_quantized=q,
        turns_quantized_abs=abs(q),
        is_circling=circling,
        direction=direction,
        consistency_abs_diff=consistency,
    )


# -----------------------------------------------------------------------------
# GT lookup
# -----------------------------------------------------------------------------

def load_gt(ann_csv: Path, event_id: str) -> dict:
    df = pd.read_csv(ann_csv)
    df.columns = [c.strip() for c in df.columns]
    row = df[df['EventID'].astype(str) == event_id]
    if row.empty:
        raise ValueError(f'EventID {event_id!r} not in {ann_csv}')
    r = row.iloc[0]
    # "No. of turns/circles" is the GT turns field.
    gt_turns = r.get('No. of turns/circles')
    try:
        gt_turns = float(gt_turns)
    except (TypeError, ValueError):
        gt_turns = None
    return dict(
        event_id=event_id,
        start_frame=int(r['Start Frame']),
        end_frame=int(r['End Frame']),
        gt_direction=str(r['Rotation direction']),
        gt_turns=gt_turns,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _process_one(centroids_csv: Path, annotations: Path, event_id: str) -> dict:
    df = pd.read_csv(centroids_csv)
    gt = load_gt(annotations, event_id)
    pred = num_turns_from_centroids(df)
    out = {**gt, **pred}
    err_turns = (pred.get('turns_abs', float('nan')) - gt['gt_turns']
                 if gt['gt_turns'] is not None else None)
    out['abs_turns_err'] = (abs(err_turns)
                            if err_turns is not None and not np.isnan(err_turns)
                            else None)
    q = pred.get('turns_quantized_abs')
    out['abs_quant_turns_err'] = (abs(q - gt['gt_turns'])
                                   if (q is not None and gt['gt_turns'] is not None)
                                   else None)
    out['dir_correct'] = (pred.get('direction') == gt['gt_direction']
                          if pred.get('direction') != 'NONE' else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--centroids',   type=Path,
                    help='Single centroids.csv (requires --event-id).')
    ap.add_argument('--centroids-dir', type=Path,
                    help='Parent dir of per-event folders (one centroids.csv each).')
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--event-id',    type=str, default=None)
    ap.add_argument('--out',         type=Path, default=None,
                    help='Write results JSON (single mode) or CSV (batch mode).')
    args = ap.parse_args()

    if args.centroids:
        if not args.event_id:
            raise SystemExit('--event-id required with --centroids')
        res = _process_one(args.centroids, args.annotations, args.event_id)
        print(json.dumps(res, indent=2, default=str))
        if args.out:
            args.out.write_text(json.dumps(res, indent=2, default=str))
            print(f'wrote {args.out}')
        return

    # Batch mode: scan centroids-dir for */centroids.csv.
    assert args.centroids_dir, 'need --centroids or --centroids-dir'
    rows = []
    for ev_dir in sorted(args.centroids_dir.iterdir()):
        csv = ev_dir / 'centroids.csv'
        if not csv.exists():
            continue
        # Parse event id from folder name '<video_stem>_<event_id>' -- the
        # stem may contain underscores, so take everything after the LAST
        # match of pattern '_vid' / 'evt_' etc.  Simplest: require the user
        # to name folders '<anything>_<event_id>' where event_id starts
        # with 'vid'.
        stem = ev_dir.name
        toks = stem.split('_')
        # Reconstruct event_id as "<first-token starting with vid>_..."
        start = next((i for i, t in enumerate(toks) if t.startswith('vid')), None)
        if start is None:
            print(f'[skip] cannot parse event id from {stem}')
            continue
        event_id = '_'.join(toks[start:])
        try:
            res = _process_one(csv, args.annotations, event_id)
        except Exception as e:
            print(f'[err] {event_id}: {e}')
            continue
        print(f"{event_id:30s}  gt={res['gt_turns']} {res['gt_direction']}  "
              f"pred={res.get('turns_abs', 0):.2f} {res.get('direction','?')}  "
              f"err={res.get('abs_turns_err', 'n/a')}")
        rows.append(res)
    if args.out and rows:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
