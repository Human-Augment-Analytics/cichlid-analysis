"""Temporal action-segmentation metrics, in percent (MS-TCN conventions).

MoF includes background. Edit and segmental F1 exclude background by default.
Ignored labels and missing frames split evaluation into separate contiguous runs;
we never bridge those gaps. Edit is macro-averaged over runs; F1 pools TP/FP/FN.
"""
import numpy as np


def runs(labels, background=0):
    labels = np.asarray(labels)
    if not len(labels):
        return []
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    return [(int(labels[a]), int(a), int(b)) for a, b in zip(starts, ends)
            if background is None or labels[a] != background]


def edit_score(predicted, target, background=0):
    p = [r[0] for r in runs(predicted, background)]
    t = [r[0] for r in runs(target, background)]
    previous = list(range(len(t) + 1))
    for i, value in enumerate(p, 1):
        current = [i]
        for j, truth in enumerate(t, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (value != truth)))
        previous = current
    return 100.0 * (1 - previous[-1] / max(len(p), len(t))) if p or t else 100.0


def segment_counts(predicted, target, overlap):
    p, t = runs(predicted), runs(target)
    hits = set()
    tp = 0
    for label, start, end in p:
        ious = [max(0, min(end, b) - max(start, a)) / (max(end, b) - min(start, a))
                if label == other else 0 for other, a, b in t]
        if ious:
            best = int(np.argmax(ious))
            if ious[best] >= overlap and best not in hits:
                hits.add(best)
                tp += 1
    return tp, len(p) - tp, len(t) - tp


def temporal_metrics(predictions, threshold=0.5, ignore=-100):
    """Evaluate a table with video, sequence, FrameNum, label, circling_score."""
    edits, edits_bg = [], []
    totals = {k: np.zeros(3, dtype=np.int64) for k in (10, 25, 50)}
    correct = count = 0
    for group in predictions.sort('video', 'sequence', 'FrameNum').partition_by(['video', 'sequence']):
        frames = group['FrameNum'].to_numpy()
        truth = group['label'].to_numpy()
        pred = (group['circling_score'].to_numpy() >= threshold).astype(np.int64)
        valid = truth != ignore
        count += int(valid.sum())
        correct += int((pred[valid] == truth[valid]).sum())
        # Explicitly split on ignored frames as well as discontinuous frame IDs.
        ids = np.flatnonzero(valid)
        if not len(ids):
            continue
        cuts = np.flatnonzero((np.diff(ids) != 1) | (np.diff(frames[ids]) != 1)) + 1
        for part in np.split(ids, cuts):
            p, t = pred[part], truth[part]
            edits.append(edit_score(p, t))
            edits_bg.append(edit_score(p, t, background=None))
            for k in totals:
                totals[k] += segment_counts(p, t, k / 100)
    if not count:
        raise ValueError('No labeled frames for temporal evaluation')
    result = {'mof': 100 * correct / count, 'edit_score': float(np.mean(edits)),
              'edit_score_with_background': float(np.mean(edits_bg)), 'evaluation_runs': len(edits)}
    for k, (tp, fp, fn) in totals.items():
        result[f'segmental_f1_{k}'] = float(100 * 2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
        result[f'segmental_counts_{k}'] = {'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}
    return result
