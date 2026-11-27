"""num_turns + direction from pre-computed pair-pose parquet.

For a given event (video + start_frame + end_frame), looks up all pose
rows whose FrameNum falls in the span, picks the (TrackID_1, TrackID_2)
pair with the most overlapping rows, unwraps `orbital_rad`, computes
turns = (orbital[-1] - orbital[0]) / (2*pi), and applies the same
quantization rule as sam2_num_turns.py.

Sign convention (confirm empirically first time): assume positive
d_orbital_rad_s = CW on screen (matches our SAM 2 convention).  Flip
if validation says otherwise.

Usage
-----
    python pose_num_turns.py \
        --pose <path/to/pairs.parquet> \
        --annotations <path/to/annotations.csv> \
        --event-id vid5-v07_d05_evt_0001
    # or batch:
    python pose_num_turns.py \
        --pose <path/to/pairs.parquet> \
        --annotations <path/to/annotations.csv> \
        --all-events
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Quantization + classification (identical rules to sam2_num_turns.py)
# -----------------------------------------------------------------------------

def quantize_turns(raw: float) -> float:
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


def is_circling(q: float, min_turns: float = 1.0) -> bool:
    return abs(q) >= min_turns


# -----------------------------------------------------------------------------
# Per-event prediction
# -----------------------------------------------------------------------------

def predict_from_pose(df_pose: pd.DataFrame, start: int, end: int,
                      sign_is_cw: bool = True) -> dict:
    """Pair-pooled unwrap of orbital_rad across the GT event span.

    Rationale on method evolution:
      v1  (dominant-pair unwrap)  : pick top (TID1, TID2), unwrap -- loses
                                    turns when track IDs swap mid-event.
      v2  (sum-of-episodes)       : group by pair_episode_id, unwrap each
                                    episode, sum -- recovers within-episode
                                    turns but still loses turns between
                                    consecutive episodes of the SAME pair
                                    (a pair_episode_id break usually comes
                                    from a brief track-flicker, not a real
                                    pause in circling).
      v3  (pair-pooled unwrap)    : group by (TrackID_1, TrackID_2), pool
                                    ALL rows across all episode_ids for that
                                    pair, sort by FrameNum, unwrap ONCE.
                                    numpy.unwrap handles the +/- pi wrap at
                                    track-flicker boundaries automatically
                                    when the gap is short and the orbital
                                    angle continued smoothly.  Sum deltas
                                    across different pairs.

    v3 is the current method.  Falls back to v2 behaviour when there is only
    one episode per pair (which is most events).
    """
    sub = df_pose[(df_pose['FrameNum'] >= start) & (df_pose['FrameNum'] < end)]
    if sub.empty:
        return dict(error='no pose rows in span')

    # Pool all rows per (TrackID_1, TrackID_2) pair across its episodes,
    # unwrap once, sum signed delta across pairs.
    total_signed_rad = 0.0
    pair_diag = []
    for (tid1, tid2), g in sub.groupby(['TrackID_1', 'TrackID_2']):
        g = g.sort_values('FrameNum')
        if len(g) < 2:
            continue
        orb_u = np.unwrap(g['orbital_rad'].to_numpy())
        delta = float(orb_u[-1] - orb_u[0])
        total_signed_rad += delta
        pair_diag.append(dict(
            track_id_1=int(tid1),
            track_id_2=int(tid2),
            n_rows=int(len(g)),
            n_episodes=int(g['pair_episode_id'].nunique()),
            frame_range=[int(g['FrameNum'].iloc[0]),
                         int(g['FrameNum'].iloc[-1])],
            delta_rad=delta,
            delta_turns=delta / (2 * np.pi),
        ))

    if not pair_diag:
        return dict(error='no pair with >=2 rows')

    turns_signed = total_signed_rad / (2 * np.pi)
    if not sign_is_cw:
        turns_signed = -turns_signed
    q = quantize_turns(turns_signed)
    direction = 'CW' if turns_signed > 0 else ('CCW' if turns_signed < 0 else 'NONE')

    # Dominant pair for metadata.
    dominant = max(pair_diag, key=lambda d: d['n_rows'])
    pair_sex = sub[(sub['TrackID_1'] == dominant['track_id_1'])
                   & (sub['TrackID_2'] == dominant['track_id_2'])]['pair_sex']
    pair_sex = pair_sex.mode()[0] if len(pair_sex) else None
    mean_dist = float(sub['centroid_distance_px'].mean())

    return dict(
        n_pairs=len(pair_diag),
        n_episodes=int(sub['pair_episode_id'].nunique()),
        dominant_pair=[dominant['track_id_1'], dominant['track_id_2']],
        n_rows=int(sum(p['n_rows'] for p in pair_diag)),
        span=[int(sub['FrameNum'].min()), int(sub['FrameNum'].max())],
        pair_sex=pair_sex,
        mean_centroid_dist_px=mean_dist,
        turns_raw=float(turns_signed),
        turns_quantized=float(q),
        is_circling=is_circling(q),
        direction=direction,
        pair_diag=pair_diag,
    )


# -----------------------------------------------------------------------------
# Helpers for CSV lookup
# -----------------------------------------------------------------------------

def load_event(ann_csv: Path, event_id: str) -> dict:
    df = pd.read_csv(ann_csv)
    df.columns = [c.strip() for c in df.columns]
    row = df[df['EventID'].astype(str) == event_id]
    if row.empty:
        raise ValueError(f'EventID {event_id!r} not found in {ann_csv}')
    r = row.iloc[0]
    try:
        gt_turns = float(r.get('No. of turns/circles'))
    except (TypeError, ValueError):
        gt_turns = None
    return dict(event_id=event_id,
                start_frame=int(r['Start Frame']),
                end_frame=int(r['End Frame']),
                gt_direction=str(r['Rotation direction']),
                gt_turns=gt_turns)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _print_one(gt: dict, pred: dict):
    if 'error' in pred:
        print(f"{gt['event_id']}: ERROR - {pred['error']}")
        return
    q = pred['turns_quantized']
    d = pred['direction']
    gt_q = gt['gt_turns']
    gt_d = gt['gt_direction']
    ok_dir = d == gt_d
    ok_q = (gt_q is not None) and (abs(q) == abs(gt_q))
    n_ep = pred.get('n_episodes', '?')
    n_pr = pred.get('n_pairs', '?')
    print(f"{gt['event_id']:28s}  GT {gt_q} {gt_d:<4}  "
          f"POSE raw={pred['turns_raw']:+.2f} q={q:+.1f} {d:<4}  "
          f"dir_ok={ok_dir} q_ok={ok_q}  "
          f"n_pair={n_pr} n_ep={n_ep} n_rows={pred['n_rows']:>4}  "
          f"mean_dist={pred['mean_centroid_dist_px']:6.0f} px  "
          f"sex={pred['pair_sex']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose',        type=Path, required=True)
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--event-id',    type=str,  default=None)
    ap.add_argument('--all-events',  action='store_true')
    ap.add_argument('--sign-is-cw',  dest='sign_is_cw', action='store_true', default=True)
    ap.add_argument('--sign-is-ccw', dest='sign_is_cw', action='store_false')
    ap.add_argument('--out',         type=Path, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    df_pose = pd.read_parquet(args.pose)
    t_load = time.perf_counter() - t0
    print(f'[load] {args.pose.name}  rows={len(df_pose)}  {t_load*1000:.0f} ms')

    gt_df = pd.read_csv(args.annotations)
    gt_df.columns = [c.strip() for c in gt_df.columns]

    event_ids = []
    if args.event_id:
        event_ids = [args.event_id]
    elif args.all_events:
        event_ids = gt_df['EventID'].astype(str).tolist()
    else:
        raise SystemExit('need --event-id or --all-events')

    rows = []
    t0 = time.perf_counter()
    for eid in event_ids:
        try:
            gt = load_event(args.annotations, eid)
        except ValueError:
            continue
        pred = predict_from_pose(df_pose, gt['start_frame'], gt['end_frame'],
                                 sign_is_cw=args.sign_is_cw)
        _print_one(gt, pred)
        rows.append({'event_id': eid, **gt, **{f'pred_{k}': v
                                               for k, v in pred.items()}})
    dt = time.perf_counter() - t0
    print(f'\n[time] {len(event_ids)} events in {dt*1000:.0f} ms  '
          f'= {dt/max(1, len(event_ids))*1000:.2f} ms/event')

    if args.out:
        # Drop list-valued fields for csv readability.
        drop_cols = {'pred_episodes', 'pred_pair_diag'}
        flat = [{k: v for k, v in r.items() if k not in drop_cols} for r in rows]
        pd.DataFrame(flat).to_csv(args.out, index=False)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
