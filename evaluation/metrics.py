"""Frame- and segment-level metrics for circling intervals.

Frame ranges are zero-based and inclusive (start, end) tuples. Direction and cycle
metrics take labelled (start, end, direction, cycles) tuples, with None for unknown.
"""

from math import isnan, nan, sqrt

# IoU thresholds, in percent, for segmental F1@tau.
TAUS = (10, 25, 50)
# Circling directions, and the IoU threshold (percent) that pairs predicted with
# ground-truth segments for direction and cycle metrics. It is the loosest one, so
# these metrics judge the labels of detected events rather than their boundaries.
DIRECTIONS = ('CW', 'CCW')
LABEL_TAU = 10


def merge(intervals):
    """Union of inclusive (start, end) intervals; adjacent ranges are joined."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(span) for span in merged]


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def segment_labels(spans, num_frames):
    """Ordered segment labels (1 circling, 0 background) for merged spans."""
    labels, cursor = [], 0
    for start, end in spans:
        if start > cursor:
            labels.append(0)
        labels.append(1)
        cursor = end + 1
    if cursor < num_frames:
        labels.append(0)
    return labels


def levenshtein(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def edit_score(gt_spans, pred_spans, num_frames):
    gt, pred = segment_labels(gt_spans, num_frames), segment_labels(pred_spans, num_frames)
    return 100 * (1 - levenshtein(gt, pred) / max(len(gt), len(pred), 1))


def iou(a, b):
    inter = overlap(a, b)
    return inter / (a[1] - a[0] + b[1] - b[0] + 2 - inter)


def segment_matches(gt_spans, pred_spans, tau):
    """Predicted segment index -> matched ground-truth segment index for F1@tau.

    Each predicted segment, in order, is compared with the ground-truth segment it
    overlaps most. It is a true positive if that IoU is at least tau / 100 and no
    earlier prediction already claimed the segment; otherwise a false positive.
    """
    matches = {}
    for index, pred in enumerate(pred_spans):
        if not gt_spans:
            break
        best = max(range(len(gt_spans)), key=lambda i: iou(gt_spans[i], pred))
        if 100 * iou(gt_spans[best], pred) >= tau and best not in matches.values():
            matches[index] = best
    return matches


def segment_true_positives(gt_spans, pred_spans, tau):
    """True positives for F1@tau; see `segment_matches`."""
    return len(segment_matches(gt_spans, pred_spans, tau))


def segment_f1(tp, pred_segments, gt_segments):
    """Harmonic mean of precision tp / pred and recall tp / gt; zero without true positives."""
    return 2 * tp / (pred_segments + gt_segments) if tp else 0.


def ratio(numerator, denominator):
    return numerator / denominator if denominator else float('nan')


def evaluate_video(gt_events, pred_events, num_frames):
    """Metrics for one video from inclusive (start, end) frame intervals."""
    gt_spans, pred_spans = merge(gt_events), merge(pred_events)
    gt_frames = sum(end - start + 1 for start, end in gt_spans)
    pred_frames = sum(end - start + 1 for start, end in pred_spans)
    tp = sum(overlap(g, p) for g in gt_spans for p in pred_spans)
    precision, recall = ratio(tp, pred_frames), ratio(tp, gt_frames)
    tps = {tau: segment_true_positives(gt_spans, pred_spans, tau) for tau in TAUS}
    return {
        'num_frames': num_frames,
        'mof': ratio(num_frames - gt_frames - pred_frames + 2 * tp, num_frames),
        'edit_score': edit_score(gt_spans, pred_spans, num_frames),
        'frame_tp': tp, 'frame_fp': pred_frames - tp, 'frame_fn': gt_frames - tp,
        'frame_precision': precision, 'frame_recall': recall,
        'frame_f1': ratio(2 * precision * recall, precision + recall) if tp else 0.,
        'frame_iou': ratio(tp, gt_frames + pred_frames - tp),
        'gt_segments': len(gt_spans), 'pred_segments': len(pred_spans),
        **{f'tp@{tau}': tps[tau] for tau in TAUS},
        **{f'f1@{tau}': segment_f1(tps[tau], len(pred_spans), len(gt_spans)) for tau in TAUS},
    }


def summarize(per_video):
    """Pool frame and segment counts across videos; edit score and frame IoU are means."""
    total = {key: sum(row[key] for row in per_video) for key in (
        'num_frames', 'frame_tp', 'frame_fp', 'frame_fn', 'gt_segments', 'pred_segments',
        *(f'tp@{tau}' for tau in TAUS),
    )}
    tp, fp, fn = total['frame_tp'], total['frame_fp'], total['frame_fn']
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {
        'num_frames': total['num_frames'],
        'mof': ratio(total['num_frames'] - fp - fn, total['num_frames']),
        'edit_score': sum(row['edit_score'] for row in per_video) / len(per_video),
        **{key: total[key] for key in ('frame_tp', 'frame_fp', 'frame_fn')},
        'frame_precision': precision, 'frame_recall': recall,
        'frame_f1': ratio(2 * precision * recall, precision + recall) if tp else 0.,
        'frame_iou': sum(row['frame_iou'] for row in per_video) / len(per_video),
        'gt_segments': total['gt_segments'], 'pred_segments': total['pred_segments'],
        **{f'tp@{tau}': total[f'tp@{tau}'] for tau in TAUS},
        **{f'f1@{tau}': segment_f1(total[f'tp@{tau}'], total['pred_segments'], total['gt_segments'])
           for tau in TAUS},
    }


def label_spans(spans, events):
    """Label (start, end) spans from overlapping (start, end, direction, cycles) events.

    A span's direction is the one covering most of its frames, and its cycles are the
    events' cycles prorated by the fraction of each event inside the span. Either is
    None when no overlapping event has it.
    """
    labelled = []
    for span in spans:
        direction_frames, cycles = {}, None
        for start, end, direction, count in events:
            shared = overlap(span, (start, end))
            if not shared:
                continue
            if direction is not None:
                direction_frames[direction] = direction_frames.get(direction, 0) + shared
            if count is not None:
                cycles = (cycles or 0) + count * shared / (end - start + 1)
        direction = max(direction_frames, key=direction_frames.get) if direction_frames else None
        labelled.append((*span, direction, cycles))
    return labelled


def f1_from_counts(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else nan


def mean(values):
    values = [value for value in values if not isnan(value)]
    return sum(values) / len(values) if values else nan


def direction_metrics(counts, correct, matched):
    """Per-direction precision/recall/F1, macro F1 and accuracy from pooled counts."""
    if counts is None:
        return {'direction_f1': nan, 'direction_accuracy': nan, **{
            f'{d.lower()}_{key}': nan for d in DIRECTIONS for key in ('precision', 'recall', 'f1')}}
    metrics = {}
    for d in DIRECTIONS:
        tp, fp, fn = counts[d]
        metrics |= {f'{d.lower()}_precision': ratio(tp, tp + fp), f'{d.lower()}_recall': ratio(tp, tp + fn),
                    f'{d.lower()}_f1': f1_from_counts(tp, fp, fn)}
    return {'direction_f1': mean([metrics[f'{d.lower()}_f1'] for d in DIRECTIONS]),
            'direction_accuracy': ratio(correct, matched), **metrics}


def evaluate_labels(gt, pred, predicts_direction=None, predicts_cycles=None, tau=LABEL_TAU):
    """Direction and cycle-count metrics for one video.

    `gt` and `pred` are non-overlapping (start, end, direction, cycles) segments, e.g.
    from `label_spans`. Predicted and ground-truth segments are paired as in F1@tau. For each direction d, a predicted d segment is a true
    positive if it is paired with a ground-truth d segment and a false positive
    otherwise (paired with a segment of unknown direction: ignored); ground-truth d
    segments without a true positive are false negatives. Direction accuracy and
    cycle errors use paired segments where both labels are known.

    `predicts_direction` / `predicts_cycles` default to whether any predicted event has
    that label; metrics for a label the model does not predict are NaN and its counts None.
    """
    if predicts_direction is None:
        predicts_direction = any(segment[2] is not None for segment in pred)
    if predicts_cycles is None:
        predicts_cycles = any(segment[3] is not None for segment in pred)
    matches = segment_matches([g[:2] for g in gt], [p[:2] for p in pred], tau)
    pairs = [(gt[j], pred[i]) for i, j in matches.items()]

    row = {'gt_cw_segments': sum(g[2] == 'CW' for g in gt),
           'gt_ccw_segments': sum(g[2] == 'CCW' for g in gt)}
    counts, correct, matched = None, None, None
    if predicts_direction:
        counts = {d: [0, 0, 0] for d in DIRECTIONS}
        for index, (_, _, direction, _) in enumerate(pred):
            if direction is None:
                continue
            if index not in matches:
                counts[direction][1] += 1
            elif (truth := gt[matches[index]][2]) is not None:
                counts[direction][0 if truth == direction else 1] += 1
        for d in DIRECTIONS:
            counts[d][2] = sum(g[2] == d for g in gt) - counts[d][0]
        known = [(g[2], p[2]) for g, p in pairs if g[2] is not None and p[2] is not None]
        correct, matched = sum(g == p for g, p in known), len(known)
    for d in DIRECTIONS:
        row |= {f'{d.lower()}_{key}': counts[d][k] if counts else None
                for k, key in enumerate(('tp', 'fp', 'fn'))}
    row |= {'direction_correct': correct, 'direction_matched': matched,
            **direction_metrics(counts, correct, matched)}

    errors = [p[3] - g[3] for g, p in pairs if g[3] is not None and p[3] is not None]
    row |= {
        'gt_cycles': sum(g[3] for g in gt if g[3] is not None),
        'pred_cycles': sum(p[3] for p in pred if p[3] is not None) if predicts_cycles else None,
        'cycle_pairs': len(errors) if predicts_cycles else None,
        **cycle_metrics(errors if predicts_cycles else None),
    }
    return row


def cycle_metrics(errors):
    """MAE, RMSE and mean signed error (predicted - ground truth) of cycle counts."""
    if not errors:
        return {'cycle_mae': nan, 'cycle_rmse': nan, 'cycle_bias': nan}
    return {'cycle_mae': sum(map(abs, errors)) / len(errors),
            'cycle_rmse': sqrt(sum(e * e for e in errors) / len(errors)),
            'cycle_bias': sum(errors) / len(errors)}


def summarize_labels(per_video):
    """Pool direction counts and cycle errors across videos (see `evaluate_labels`)."""
    def total(key):
        values = [row[key] for row in per_video]
        return None if not values or any(v is None for v in values) else sum(values)

    row = {key: total(key) for key in ('gt_cw_segments', 'gt_ccw_segments')}
    counts = {d: [total(f'{d.lower()}_{key}') for key in ('tp', 'fp', 'fn')] for d in DIRECTIONS}
    has_direction = all(v is not None for d in DIRECTIONS for v in counts[d])
    for d in DIRECTIONS:
        row |= {f'{d.lower()}_{key}': counts[d][k] for k, key in enumerate(('tp', 'fp', 'fn'))}
    correct, matched = total('direction_correct'), total('direction_matched')
    row |= {'direction_correct': correct, 'direction_matched': matched,
            **direction_metrics(counts if has_direction else None, correct, matched)}

    # Each video's errors weighted by its pair count give the pooled errors exactly.
    pairs = total('cycle_pairs')
    row |= {'gt_cycles': total('gt_cycles'), 'pred_cycles': total('pred_cycles'), 'cycle_pairs': pairs}
    if not pairs:
        return row | cycle_metrics(None)
    used = [r for r in per_video if r['cycle_pairs']]
    def weighted(key, power=1):
        return sum(r['cycle_pairs'] * r[key] ** power for r in used) / pairs
    return row | {'cycle_mae': weighted('cycle_mae'), 'cycle_rmse': sqrt(weighted('cycle_rmse', 2)),
                  'cycle_bias': weighted('cycle_bias')}
