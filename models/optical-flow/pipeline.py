"""End-to-end direction-detector pipeline for a single video.

Run modes:
  --mode gt_span
      For each GT event, predict rotation direction over the GT [start, end]
      span. Isolates the direction algorithm from span-detection errors.

  --mode predicted_span   (default)
      Run the full detector (flow -> event proposal -> locked filters),
      then predict direction per detected span. Evaluates against GT
      annotations using event-level IoU matching.

Outputs (under <out_dir>/<video_stem>/):
  flow.parquet                # per-frame flow features
  predictions.csv             # detected events (full pipeline only)
  direction_on_gt.csv         # per-GT direction accuracy
  direction_on_predicted.csv  # per-matched-pred direction accuracy
  metrics_summary.json        # mAP, F1@k, edit score, direction accuracy

Usage:
  python pipeline.py --video <path.mp4> --annotations <path.csv> \
      --out <dir> [--mode gt_span|predicted_span|both] \
      [--start-frame N --end-frame M] [--skip-flow]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flow, detect, evaluate as evl  # noqa: E402

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Defaults -- kept here so a single number-knob tweak is easy to find.
# Activity-area range is the only non-portable knob; see README for how to
# calibrate it to a new video.
# -----------------------------------------------------------------------------
FPS_DEFAULT = 30
FLOW_CFG = dict(downsample=2.0, crop_frac=0.1, mag_thr=0.5, blur_ksize=3)
# Cheaper variant for full-video sweeps -- used when --frame-stride > 1 or
# --motion-gate > 0 is passed on the CLI.
DETECTOR_CFG = dict(
    sign_win_s=2.0, sign_thr=0.6, min_dur_s=3.0, merge_gap_s=2.0,
    activity_thr=300,
    min_score=0.75,
    activity_area_range=(200, 500),
)
TIOUS = [0.1, 0.3, 0.5, 0.7, 0.9]
F1_TIOUS = [0.1, 0.5, 0.9]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_annotations(csv_path: Path, start_frame: int, end_frame: int) -> list[dict]:
    """Parse the circling annotations CSV, filtering to events in frame range."""
    ann = pd.read_csv(csv_path)
    ann.columns = [c.strip() for c in ann.columns]
    ann = ann.dropna(subset=['Start Frame', 'End Frame',
                             'Rotation direction']).reset_index(drop=True)
    ann['Start Frame'] = ann['Start Frame'].astype(int)
    ann['End Frame']   = ann['End Frame'].astype(int)
    ann = ann[(ann['End Frame'] - ann['Start Frame']) > 0]
    ann = ann[(ann['Start Frame'] >= start_frame) & (ann['Start Frame'] < end_frame)]
    gts = []
    for _, r in ann.iterrows():
        gts.append({
            'start_frame': int(r['Start Frame']),
            'end_frame':   int(r['End Frame']),
            'direction':   r['Rotation direction'],
            'event_id':    r.get('EventID', ''),
        })
    return gts


def ensure_flow(video_path: Path, flow_parquet: Path,
                start_frame: int, end_frame: int,
                fps: float,
                frame_stride: int = 1,
                motion_gate: float = 0.0,
                skip_flow: bool = False) -> pd.DataFrame:
    """Compute per-frame flow features if the parquet is missing (or forced)."""
    if flow_parquet.exists() and skip_flow:
        print(f'[flow] using cached {flow_parquet}')
        return pd.read_parquet(flow_parquet)
    print(f'[flow] computing frames [{start_frame}, {end_frame}) stride={frame_stride} '
          f'motion_gate={motion_gate} -> {flow_parquet}')
    t0 = time.time()
    df = flow.process_clip(video_path, flow_parquet,
                            start_frame=start_frame, end_frame=end_frame,
                            frame_stride=frame_stride,
                            motion_gate_thr=motion_gate,
                            progress_every=max(3000 // max(1, frame_stride), 1000),
                            **FLOW_CFG)
    dt = time.time() - t0
    print(f'[flow] done. rows={len(df)}  {dt/60:.1f} min')
    return df


# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------

def run_gt_span_mode(df_flow: pd.DataFrame, gts: list[dict], out_dir: Path) -> dict:
    """Predict direction for each GT span (isolates direction-algorithm quality)."""
    rows = []
    for g in gts:
        pred = detect.direction_on_span(df_flow, g['start_frame'], g['end_frame'],
                                        activity_thr=DETECTOR_CFG['activity_thr'])
        rows.append({'event_id': g['event_id'],
                     'start_frame': g['start_frame'],
                     'end_frame': g['end_frame'],
                     'gt_direction': g['direction'],
                     'pred_direction': pred,
                     'correct': (pred == g['direction']) if pred else None})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'direction_on_gt.csv', index=False)

    scored = df.dropna(subset=['correct'])
    acc = float(scored['correct'].mean()) if len(scored) else float('nan')
    conf = pd.crosstab(df['gt_direction'], df['pred_direction'],
                       rownames=['GT'], colnames=['Pred'], dropna=False)
    conf.to_csv(out_dir / 'direction_conf_matrix_on_gt.csv')
    return dict(
        total=len(df), scored=len(scored),
        correct=int(scored['correct'].sum()) if len(scored) else 0,
        accuracy=acc,
        confusion=conf.to_dict(),
    )


def run_predicted_span_mode(df_flow: pd.DataFrame, gts: list[dict],
                             fps: float, out_dir: Path) -> dict:
    """Full detector + event-level evaluation of spans and direction."""
    ev_list = detect.detect_events(df_flow, fps=fps, **DETECTOR_CFG)
    preds = [{'start_frame': e.start_frame, 'end_frame': e.end_frame,
              'direction':   e.direction,   'score':     e.score}
             for e in ev_list]
    pd.DataFrame(preds).to_csv(out_dir / 'predictions.csv', index=False)

    # ---- Span detection metrics ----
    aps = {f'AP@{t}': evl.ap_at_tiou(preds, gts, t) for t in TIOUS}
    mAP = float(np.mean(list(aps.values())))
    f1  = {f'F1@{int(t*100)}': evl.f1_at_k(preds, gts, t)['f1']
           for t in F1_TIOUS}
    f1_dir = {f'F1_dir@{int(t*100)}': evl.f1_at_k(preds, gts, t,
                                                   require_direction_match=True)['f1']
              for t in F1_TIOUS}
    edit_score = evl.segmental_edit_score(preds, gts, use_direction=True)

    # ---- Direction on matched preds @ tIoU=0.5 ----
    dmatched = evl.direction_accuracy_on_matched_preds(preds, gts, tiou_thr=0.5)
    pd.DataFrame([{'matched': dmatched['matched'],
                   'correct': dmatched['correct'],
                   'accuracy': dmatched['accuracy']}]
                 ).to_csv(out_dir / 'direction_on_predicted.csv', index=False)

    return dict(
        n_preds=len(preds), n_gt=len(gts),
        **aps, mAP=mAP, **f1, **f1_dir,
        edit_score=edit_score,
        direction_on_matched_acc=dmatched['accuracy'],
        direction_on_matched_correct=dmatched['correct'],
        direction_on_matched_total=dmatched['matched'],
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',       type=Path, required=True)
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--out',         type=Path, required=True,
                    help='Output directory; per-video subfolder will be created.')
    ap.add_argument('--mode', choices=('gt_span', 'predicted_span', 'both'),
                    default='both')
    ap.add_argument('--fps', type=float, default=FPS_DEFAULT)
    ap.add_argument('--start-frame', type=int, default=0)
    ap.add_argument('--end-frame',   type=int, default=None)
    ap.add_argument('--frame-stride', type=int, default=1,
                    help='Process every Nth frame (speeds up flow at cost of '
                         'temporal resolution). Effective fps = fps/stride.')
    ap.add_argument('--motion-gate', type=float, default=0.0,
                    help='Skip Farneback on frame pairs whose mean abs grayscale '
                         'difference is below this threshold. Reasonable: 1-5.')
    ap.add_argument('--skip-flow',   action='store_true',
                    help='Reuse flow parquet if it already exists.')
    args = ap.parse_args()

    video_stem = args.video.stem
    out_dir = args.out / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    end_frame = args.end_frame
    if end_frame is None:
        import cv2
        cap = cv2.VideoCapture(str(args.video))
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    print(f'video={args.video.name}  frames=[{args.start_frame},{end_frame})  '
          f'fps={args.fps}')

    gts = load_annotations(args.annotations, args.start_frame, end_frame)
    print(f'GT events in range: {len(gts)}')
    pd.DataFrame(gts).to_csv(out_dir / 'gt.csv', index=False)

    df_flow = ensure_flow(args.video, out_dir / 'flow.parquet',
                           args.start_frame, end_frame, args.fps,
                           frame_stride=args.frame_stride,
                           motion_gate=args.motion_gate,
                           skip_flow=args.skip_flow)
    effective_fps = args.fps / max(1, args.frame_stride)

    summary = dict(video=args.video.name,
                   start_frame=args.start_frame, end_frame=end_frame,
                   fps=args.fps,
                   frame_stride=args.frame_stride,
                   motion_gate=args.motion_gate,
                   effective_fps=effective_fps,
                   n_gt=len(gts))

    if args.mode in ('gt_span', 'both'):
        summary['gt_span'] = run_gt_span_mode(df_flow, gts, out_dir)
        print(f"[gt_span]  direction accuracy = "
              f"{summary['gt_span']['accuracy']:.4f}  "
              f"({summary['gt_span']['correct']}/"
              f"{summary['gt_span']['scored']} of {summary['gt_span']['total']})")

    if args.mode in ('predicted_span', 'both'):
        summary['predicted_span'] = run_predicted_span_mode(df_flow, gts,
                                                             effective_fps, out_dir)
        ps = summary['predicted_span']
        print(f"[predicted_span]  n_preds={ps['n_preds']}  mAP={ps['mAP']:.3f}  "
              f"F1@10={ps['F1@10']:.3f}  F1@50={ps['F1@50']:.3f}  "
              f"F1@90={ps['F1@90']:.3f}  "
              f"edit={ps['edit_score']:.3f}  "
              f"dir_matched={ps['direction_on_matched_acc']:.3f}")

    (out_dir / 'metrics_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'wrote: {out_dir/"metrics_summary.json"}')


if __name__ == '__main__':
    main()
