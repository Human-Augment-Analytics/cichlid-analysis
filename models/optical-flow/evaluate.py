"""Event-level evaluation metrics (Step 3).

Two things live here:
  - mAP @ tIoU family (span-detection metric)
  - direction accuracy, both on predicted spans and on GT spans (the latter
    isolates the direction-prediction algorithm from boundary errors)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_iou(a_start: float, a_end: float,
                 b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


# -----------------------------------------------------------------------------
# Precision / recall / F1 at a single tIoU (thresholded)
# -----------------------------------------------------------------------------

def event_prf_at_tiou(preds: list[dict], gts: list[dict],
                      tiou_thr: float) -> dict:
    """Greedy matching, preds sorted by `score` descending.

    pred / gt dicts must have keys: start_frame, end_frame, score (preds only).
    """
    preds_sorted = sorted(preds, key=lambda p: -p.get('score', 0.0))
    matched = set()
    tp = fp = 0
    for p in preds_sorted:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched:
                continue
            v = temporal_iou(p['start_frame'], p['end_frame'],
                             g['start_frame'], g['end_frame'])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= tiou_thr and best_j >= 0:
            tp += 1
            matched.add(best_j)
        else:
            fp += 1
    fn = len(gts) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(tiou=tiou_thr, tp=tp, fp=fp, fn=fn,
                precision=prec, recall=rec, f1=f1)


# -----------------------------------------------------------------------------
# mAP @ tIoU (VOC-style AP with all-points interpolation)
# -----------------------------------------------------------------------------

def ap_at_tiou(preds: list[dict], gts: list[dict], tiou_thr: float) -> float:
    if not gts:
        return 0.0
    if not preds:
        return 0.0
    preds_sorted = sorted(preds, key=lambda p: -p.get('score', 0.0))
    matched = set()
    tps, fps = [], []
    for p in preds_sorted:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched:
                continue
            v = temporal_iou(p['start_frame'], p['end_frame'],
                             g['start_frame'], g['end_frame'])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= tiou_thr and best_j >= 0:
            tps.append(1); fps.append(0); matched.add(best_j)
        else:
            tps.append(0); fps.append(1)

    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    recall    = tp_cum / len(gts)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # VOC-style: monotone-decreasing right envelope, integrate against recall.
    rec = np.r_[0.0, recall, 1.0]
    prec = np.r_[1.0, precision, 0.0]
    for i in range(len(prec) - 1, 0, -1):
        prec[i - 1] = max(prec[i - 1], prec[i])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def map_at_tious(preds: list[dict], gts: list[dict],
                 tious: list[float] = (0.1, 0.3, 0.5, 0.7, 0.9)) -> pd.DataFrame:
    rows = [{'tIoU': t, 'AP': ap_at_tiou(preds, gts, t)} for t in tious]
    rows.append({'tIoU': 'mean', 'AP': float(np.mean([r['AP'] for r in rows]))})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Direction accuracy
# -----------------------------------------------------------------------------

def direction_accuracy_on_gt(df_flow_by_clip: dict,
                             gt_events: list[dict],
                             direction_fn,
                             activity_thr: float = 300) -> dict:
    """Evaluate direction prediction using **ground-truth** spans.

    gt_events: dicts with keys start_frame, end_frame, direction, clip (stem).
    direction_fn: callable(df_flow, start_frame, end_frame, activity_thr) -> str.

    Spans are expected in whichever frame coord system matches `df_flow`
    (e.g., clip-local). The caller ensures consistency.
    """
    rows = []
    for g in gt_events:
        df = df_flow_by_clip.get(g['clip'])
        if df is None:
            rows.append(dict(**g, predicted=None, correct=None))
            continue
        pred = direction_fn(df, g['start_frame'], g['end_frame'],
                            activity_thr=activity_thr)
        rows.append(dict(**g, predicted=pred,
                         correct=(pred == g['direction']) if pred else None))
    out = pd.DataFrame(rows)
    scored = out.dropna(subset=['correct'])
    acc = float(scored['correct'].mean()) if len(scored) else float('nan')
    return dict(per_event=out,
                accuracy=acc,
                correct=int(scored['correct'].sum()) if len(scored) else 0,
                scored=len(scored),
                total=len(out))


# -----------------------------------------------------------------------------
# Temporal-action-segmentation metrics: F1@k and segmental edit score
# -----------------------------------------------------------------------------

def _levenshtein(a: list, b: list) -> int:
    """Standard Levenshtein edit distance between two token sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    return dp[len(a)][len(b)]


def segmental_edit_score(preds: list[dict], gts: list[dict],
                         use_direction: bool = True) -> float:
    """Classical TAL segmental edit score (MS-TCN / ASRF convention).

    Collapses the event sequence to just its ordered labels (direction, if
    `use_direction`; otherwise a single 'E' token per event). Computes the
    Levenshtein distance between pred and GT label sequences, then normalises:
        score = 1 - lev / max(N_pred, N_gt)

    1 == perfect sequence match, 0 == completely wrong.
    """
    key = lambda e: e['start_frame']
    pred_labels = [p.get('direction', 'E') if use_direction else 'E'
                   for p in sorted(preds, key=key)]
    gt_labels   = [g.get('direction', 'E') if use_direction else 'E'
                   for g in sorted(gts, key=key)]
    if not pred_labels and not gt_labels:
        return 1.0
    if not pred_labels or not gt_labels:
        return 0.0
    lev = _levenshtein(pred_labels, gt_labels)
    return 1.0 - lev / max(len(pred_labels), len(gt_labels))


def f1_at_k(preds: list[dict], gts: list[dict], tiou_thr: float,
            require_direction_match: bool = False) -> dict:
    """F1 @ tIoU under greedy score-ranked matching.

    If `require_direction_match`, a match also requires predicted direction
    == GT direction (stricter; useful for the TAL-style 'correct detection
    of a labelled action' notion).
    """
    preds_sorted = sorted(preds, key=lambda p: -p.get('score', 0.0))
    matched = set()
    tp = fp = 0
    for p in preds_sorted:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched:
                continue
            if require_direction_match and p.get('direction') != g.get('direction'):
                continue
            v = temporal_iou(p['start_frame'], p['end_frame'],
                             g['start_frame'], g['end_frame'])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= tiou_thr and best_j >= 0:
            tp += 1
            matched.add(best_j)
        else:
            fp += 1
    fn = len(gts) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(tiou=tiou_thr, tp=tp, fp=fp, fn=fn,
                precision=prec, recall=rec, f1=f1)


def direction_accuracy_on_matched_preds(preds: list[dict], gts: list[dict],
                                        tiou_thr: float = 0.5) -> dict:
    """For each prediction that matches a GT at tIoU, check direction."""
    preds_sorted = sorted(preds, key=lambda p: -p.get('score', 0.0))
    matched = {}
    for p in preds_sorted:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched:
                continue
            v = temporal_iou(p['start_frame'], p['end_frame'],
                             g['start_frame'], g['end_frame'])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= tiou_thr and best_j >= 0:
            matched[best_j] = p
    if not matched:
        return dict(accuracy=float('nan'), correct=0, matched=0)
    correct = sum(1 for j, p in matched.items()
                  if p['direction'] == gts[j]['direction'])
    return dict(accuracy=correct / len(matched),
                correct=correct, matched=len(matched))
