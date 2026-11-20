"""Standalone direction predictor + GT-matching helper.

Core function:
    predict_direction(video_path, start_frame, end_frame, **cfg)
        -> 'CW' | 'CCW' | None

Use directly, or via `DataFrame.apply` to score a whole CSV (GT or predicted
spans). Example in the notebook:

    df['predicted_direction'] = df.apply(
        lambda r: predict_direction(video, int(r['Start Frame']), int(r['End Frame'])),
        axis=1,
    )

Helpers:
    match_to_gt_spans(spans_df, gt_df, ...)
        Augments a predicted-spans DataFrame with the best-overlapping GT
        event's direction + IoU. Needed because a predicted span rarely
        aligns perfectly with a GT span.

    compute_direction_metrics(df, actual_col=..., pred_col=...)
        Accuracy + confusion matrix.

Self-contained -- computes Farneback flow only on frames in the given span,
no pre-existing flow parquet required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from flow import center_crop, farneback_flow, frame_scores
from detect import direction_on_span


# -----------------------------------------------------------------------------
# Defaults -- mirror pipeline.py FLOW_CFG + DETECTOR_CFG
# -----------------------------------------------------------------------------
_DEFAULTS = dict(
    downsample=2.0,
    crop_frac=0.1,
    mag_thr=0.5,
    blur_ksize=3,
    frame_stride=1,
    activity_thr=50,
)


# -----------------------------------------------------------------------------
# The one function: single-span direction predictor
# -----------------------------------------------------------------------------

def predict_direction(video_path: str | Path,
                      start_frame: int,
                      end_frame: int,
                      **cfg) -> Optional[str]:
    """Predict rotation direction for one span in the source video.

    Returns 'CW', 'CCW', or None if the span is too quiet / too short to judge.

    Kwargs (all optional, defaults mirror pipeline.py):
        downsample, crop_frac, mag_thr, blur_ksize, frame_stride, activity_thr

    Typical batch usage from a notebook:

        df['predicted_direction'] = df.apply(
            lambda r: predict_direction(VIDEO, int(r['Start Frame']), int(r['End Frame']),
                                        frame_stride=2),
            axis=1,
        )
    """
    opts = {**_DEFAULTS, **cfg}
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'could not open {video_path}')
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst_w = int(round(src_w / opts['downsample']))
    dst_h = int(round(src_h / opts['downsample']))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    stride = max(1, int(opts['frame_stride']))
    rows = []
    prev = None
    fi = int(start_frame)
    read_counter = 0
    while fi < int(end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        if read_counter % stride != 0:
            read_counter += 1
            fi += 1
            continue
        read_counter += 1

        small = cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        small = center_crop(small, opts['crop_frac'])
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if opts['blur_ksize'] and opts['blur_ksize'] > 1:
            gray = cv2.GaussianBlur(gray,
                                    (opts['blur_ksize'], opts['blur_ksize']), 0)
        if prev is not None:
            flow_arr = farneback_flow(prev, gray)
            if stride > 1:
                flow_arr = flow_arr / stride
            s = frame_scores(flow_arr, mag_thr=opts['mag_thr'])
            s['frame'] = fi
            rows.append(s)
        prev = gray
        fi += 1
    cap.release()

    if not rows:
        return None
    df_flow = pd.DataFrame(rows)
    return direction_on_span(df_flow, int(start_frame), int(end_frame),
                             activity_thr=opts['activity_thr'])


# -----------------------------------------------------------------------------
# GT-matching helper (for scoring predicted-spans CSVs)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Picklable worker: score every GT event in one video.
# Kept at module level so `ProcessPoolExecutor` can serialise it across workers.
# -----------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def score_video_gt_events(video_path: str | Path,
                          annot_path: str | Path,
                          predict_cfg: dict | None = None,
                          fps: float = 30.0) -> pd.DataFrame:
    """Run `predict_direction` on every GT event in `annot_path` for `video_path`.

    Returns a DataFrame with the consolidated schema used by the notebook:
      video, event_id, start_frame, end_frame, start_time, end_time,
      gt_direction, pred_direction, correct
    """
    predict_cfg = dict(predict_cfg or {})
    video = Path(video_path)
    annot = Path(annot_path)
    stem = video.stem

    gt = pd.read_csv(annot)
    gt.columns = [c.strip() for c in gt.columns]
    gt = gt.dropna(subset=['Start Frame', 'End Frame']).reset_index(drop=True)
    gt = gt[(gt['End Frame'] - gt['Start Frame']) > 0].reset_index(drop=True)
    gt['Start Frame'] = gt['Start Frame'].astype(int)
    gt['End Frame']   = gt['End Frame'].astype(int)

    gt['pred_direction'] = gt.apply(
        lambda r: predict_direction(video,
                                     int(r['Start Frame']),
                                     int(r['End Frame']),
                                     **predict_cfg),
        axis=1,
    )

    out = pd.DataFrame({
        'video': stem,
        'event_id': gt['EventID'] if 'EventID' in gt.columns else gt.index.astype(str),
        'start_frame': gt['Start Frame'],
        'end_frame':   gt['End Frame'],
        'start_time': (gt['Start Frame'] / fps).apply(_fmt_time),
        'end_time':   (gt['End Frame']   / fps).apply(_fmt_time),
        'gt_direction':  gt['Rotation direction'] if 'Rotation direction' in gt.columns else None,
        'pred_direction': gt['pred_direction'],
    })
    out['correct'] = (out['pred_direction'] == out['gt_direction'])
    out.loc[out['pred_direction'].isna(), 'correct'] = pd.NA
    return out


def _iou(a_s: float, a_e: float, b_s: float, b_e: float) -> float:
    inter = max(0.0, min(a_e, b_e) - max(a_s, b_s))
    union = max(a_e, b_e) - min(a_s, b_s)
    return inter / union if union > 0 else 0.0


def match_to_gt_spans(spans_df: pd.DataFrame,
                      gt_df: pd.DataFrame,
                      spans_sf: str = 'start_frame',
                      spans_ef: str = 'end_frame',
                      gt_sf: str = 'Start Frame',
                      gt_ef: str = 'End Frame',
                      gt_dir: str = 'Rotation direction',
                      gt_id:  str = 'EventID',
                      min_iou: float = 0.0) -> pd.DataFrame:
    """For each row of `spans_df`, find the best-overlapping GT span and add:
        matched_gt_id, matched_gt_iou, gt_direction

    Rows with best IoU < `min_iou` get NaN in those columns.
    Returns a NEW DataFrame (does not mutate the input).
    """
    gt_rows = gt_df.dropna(subset=[gt_sf, gt_ef]).copy()
    gt_rows[gt_sf] = gt_rows[gt_sf].astype(int)
    gt_rows[gt_ef] = gt_rows[gt_ef].astype(int)
    gt_list = gt_rows[[gt_sf, gt_ef, gt_dir, gt_id]].to_dict('records')

    matched_ids, matched_ious, matched_dirs = [], [], []
    for _, r in spans_df.iterrows():
        try:
            sf = int(r[spans_sf]); ef = int(r[spans_ef])
        except (TypeError, ValueError):
            matched_ids.append(None); matched_ious.append(np.nan); matched_dirs.append(None)
            continue
        best_iou, best_g = 0.0, None
        for g in gt_list:
            iou = _iou(sf, ef, g[gt_sf], g[gt_ef])
            if iou > best_iou:
                best_iou, best_g = iou, g
        if best_g is not None and best_iou >= min_iou:
            matched_ids.append(best_g[gt_id])
            matched_ious.append(best_iou)
            matched_dirs.append(best_g[gt_dir])
        else:
            matched_ids.append(None)
            matched_ious.append(best_iou)
            matched_dirs.append(None)

    out = spans_df.copy()
    out['matched_gt_id']  = matched_ids
    out['matched_gt_iou'] = matched_ious
    out['gt_direction']   = matched_dirs
    return out


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_direction_metrics(df: pd.DataFrame | str | Path,
                              actual_col: str = 'gt_direction',
                              pred_col:   str = 'predicted_direction') -> dict:
    """Accuracy + confusion matrix on a predictions DataFrame or CSV path."""
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df)
    scored = df.dropna(subset=[actual_col, pred_col])
    total = len(df)
    n = len(scored)
    if n == 0:
        return dict(total=total, scored=0, correct=0,
                    accuracy=None, confusion_matrix={})
    correct = int((scored[actual_col] == scored[pred_col]).sum())
    conf = pd.crosstab(scored[actual_col], scored[pred_col],
                       rownames=['actual'], colnames=['pred'], dropna=False)
    return dict(total=total,
                scored=n,
                correct=correct,
                accuracy=correct / n,
                confusion_matrix=conf.to_dict())
