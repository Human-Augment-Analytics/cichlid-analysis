"""Frame- and segment-level metrics for circling intervals.

Frame ranges are zero-based and inclusive (start, end) tuples. Direction and cycle
metrics take labelled (start, end, direction, cycles) tuples, with None for unknown.
"""

from math import isnan, nan, sqrt

# IoU thresholds, in percent, for segmental F1@tau.
TAUS = (10, 25, 50)
# FSH-TAL reports selected thresholds and averages AP over every 0.1 step.
MAP_TAUS = (10, 20, 30, 40, 50, 60, 70)
MAP_REPORT_TAUS = (10, 30, 50, 70)
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


def temporal_iou(a, b):
    """IoU of half-open continuous intervals, as used by temporal action localization."""
    inter = max(0., min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.


def average_precision(gt_spans, pred_scored, tau):
    """All-point interpolated AP with score-ordered, greedy one-to-one matching.

    Spans are half-open intervals in any consistent time unit. Predictions are
    ``(start, end, score)`` tuples and ``tau`` is a percentage.
    """
    if not gt_spans:
        return nan
    if not pred_scored:
        return 0.

    matched = set()
    true_positives, false_positives = [], []
    for start, end, _ in sorted(pred_scored, key=lambda row: row[2], reverse=True):
        available = [index for index in range(len(gt_spans)) if index not in matched]
        best = max(available, key=lambda i: temporal_iou(gt_spans[i], (start, end))) if available else None
        is_match = best is not None and 100 * temporal_iou(gt_spans[best], (start, end)) >= tau
        if is_match:
            matched.add(best)
        true_positives.append(int(is_match))
        false_positives.append(int(not is_match))

    tp_total = fp_total = 0
    recalls, precisions = [], []
    for tp, fp in zip(true_positives, false_positives):
        tp_total += tp
        fp_total += fp
        recalls.append(tp_total / len(gt_spans))
        precisions.append(tp_total / (tp_total + fp_total))

    # Precision envelope and all-point area under the precision-recall curve.
    envelope = precisions[:]
    for index in range(len(envelope) - 2, -1, -1):
        envelope[index] = max(envelope[index], envelope[index + 1])
    ap, previous_recall = 0., 0.
    for recall, precision in zip(recalls, envelope):
        if recall > previous_recall:
            ap += (recall - previous_recall) * precision
            previous_recall = recall
    return ap


def map_at_tiou(gt_spans, pred_scored, taus=MAP_TAUS):
    """Single-class AP at each tIoU and its mean across thresholds."""
    values = {tau: average_precision(gt_spans, pred_scored, tau) for tau in taus}
    valid = [value for value in values.values() if not isnan(value)]
    return {**{f'map@{tau}': value for tau, value in values.items()},
            'map_avg': sum(valid) / len(valid) if valid else nan}


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


def evaluate_attribute_quality(gt, pred, tau=50):
    """Attribute-only metrics on segments paired at a strict detection IoU.

    Unlike ``evaluate_labels``, this does not count missed or spurious detections
    as direction errors. It answers whether attributes are correct given a
    sufficiently accurate detected span.
    """
    matches = segment_matches([g[:2] for g in gt], [p[:2] for p in pred], tau)
    pairs = [(gt[j], pred[i]) for i, j in matches.items()]
    direction_eligible = [(g[2], p[2]) for g, p in pairs if g[2] is not None]
    directions = [(g[2], p[2]) for g, p in pairs if g[2] is not None and p[2] is not None]
    cycle_eligible = [(g[3], p[3]) for g, p in pairs if g[3] is not None]
    cycle_pairs = [(g, p) for g, p in cycle_eligible if p is not None]
    errors = [p - g for g, p in cycle_pairs]

    counts = {d: [0, 0, 0] for d in DIRECTIONS}
    for truth, predicted in directions:
        counts[predicted][0 if truth == predicted else 1] += 1
    for direction in DIRECTIONS:
        counts[direction][2] = sum(truth == direction for truth, _ in directions) - counts[direction][0]
    direction_summary = direction_metrics(
        counts, sum(g == p for g, p in directions), len(directions)) if directions else direction_metrics(None, 0, 0)

    row = {
        'attribute_pairs': len(pairs),
        'attribute_direction_eligible': len(direction_eligible),
        'attribute_direction_pairs': len(directions),
        'attribute_direction_correct': sum(g == p for g, p in directions),
        'attribute_direction_accuracy': ratio(sum(g == p for g, p in directions), len(directions)),
        'attribute_direction_coverage': ratio(len(directions), len(direction_eligible)),
        'attribute_direction_macro_f1': direction_summary['direction_f1'],
        'attribute_direction_balanced_accuracy': mean(
            [direction_summary[f'{d.lower()}_recall'] for d in DIRECTIONS]),
        'attribute_cycle_eligible': len(cycle_eligible),
        'attribute_cycle_pairs': len(errors),
        'attribute_cycle_coverage': ratio(len(errors), len(cycle_eligible)),
        **{f'attribute_{key}': value for key, value in cycle_metrics(errors).items()},
    }
    for direction in DIRECTIONS:
        row |= {f'attribute_{direction.lower()}_{key}': counts[direction][index]
                for index, key in enumerate(('tp', 'fp', 'fn'))}
    absolute_error = sum(map(abs, errors))
    row |= {
        'attribute_cycle_abs_error': absolute_error,
        'attribute_cycle_squared_error': sum(error * error for error in errors),
        'attribute_cycle_error': sum(errors),
        'attribute_cycle_gt_total': sum(g for g, _ in cycle_pairs),
        'attribute_cycle_wape': ratio(absolute_error, sum(g for g, _ in cycle_pairs)),
        'attribute_cycle_within_0p5': ratio(sum(abs(error) <= .5 for error in errors), len(errors)),
        'attribute_cycle_within_1': ratio(sum(abs(error) <= 1 for error in errors), len(errors)),
    }
    return row


def evaluate_oracle_labels(gt, pred):
    """Attribute quality when predictions use the exact ground-truth boundaries."""
    gt = sorted(gt, key=lambda row: row[:2])
    pred = sorted(pred, key=lambda row: row[:2])
    if [row[:2] for row in gt] != [row[:2] for row in pred]:
        raise ValueError('oracle label spans must exactly match ground-truth spans')
    return evaluate_attribute_quality(gt, pred, tau=100)


def cycle_metrics(errors):
    """MAE, RMSE and mean signed error (predicted - ground truth) of cycle counts."""
    if not errors:
        return {'cycle_mae': nan, 'cycle_rmse': nan, 'cycle_bias': nan}
    return {'cycle_mae': sum(map(abs, errors)) / len(errors),
            'cycle_rmse': sqrt(sum(e * e for e in errors) / len(errors)),
            'cycle_bias': sum(errors) / len(errors)}


def summarize_attribute_quality(per_video, prefix):
    """Pool prefixed outputs from ``evaluate_attribute_quality`` across videos."""
    def total(key):
        return sum(row[f'{prefix}{key}'] for row in per_video)

    direction_pairs = total('attribute_direction_pairs')
    cycle_pairs = total('attribute_cycle_pairs')
    row = {
        f'{prefix}attribute_pairs': total('attribute_pairs'),
        f'{prefix}attribute_direction_eligible': total('attribute_direction_eligible'),
        f'{prefix}attribute_direction_pairs': direction_pairs,
        f'{prefix}attribute_direction_correct': total('attribute_direction_correct'),
        f'{prefix}attribute_cycle_eligible': total('attribute_cycle_eligible'),
        f'{prefix}attribute_cycle_pairs': cycle_pairs,
    }
    counts = {}
    for direction in DIRECTIONS:
        counts[direction] = [total(f'attribute_{direction.lower()}_{key}')
                             for key in ('tp', 'fp', 'fn')]
        row |= {f'{prefix}attribute_{direction.lower()}_{key}': value
                for key, value in zip(('tp', 'fp', 'fn'), counts[direction])}
    row[f'{prefix}attribute_direction_accuracy'] = ratio(
        row[f'{prefix}attribute_direction_correct'], direction_pairs)
    row[f'{prefix}attribute_direction_coverage'] = ratio(
        direction_pairs, row[f'{prefix}attribute_direction_eligible'])
    direction_summary = direction_metrics(
        counts, row[f'{prefix}attribute_direction_correct'], direction_pairs)
    row[f'{prefix}attribute_direction_macro_f1'] = direction_summary['direction_f1']
    row[f'{prefix}attribute_direction_balanced_accuracy'] = mean(
        [direction_summary[f'{d.lower()}_recall'] for d in DIRECTIONS])
    row[f'{prefix}attribute_cycle_coverage'] = ratio(
        cycle_pairs, row[f'{prefix}attribute_cycle_eligible'])
    if not cycle_pairs:
        return row | {f'{prefix}attribute_{key}': nan for key in (
            'cycle_mae', 'cycle_rmse', 'cycle_bias', 'cycle_wape',
            'cycle_within_0p5', 'cycle_within_1')}

    absolute_error = total('attribute_cycle_abs_error')
    squared_error = total('attribute_cycle_squared_error')
    error = total('attribute_cycle_error')
    gt_total = total('attribute_cycle_gt_total')
    row |= {
        f'{prefix}attribute_cycle_abs_error': absolute_error,
        f'{prefix}attribute_cycle_squared_error': squared_error,
        f'{prefix}attribute_cycle_error': error,
        f'{prefix}attribute_cycle_gt_total': gt_total,
        f'{prefix}attribute_cycle_mae': absolute_error / cycle_pairs,
        f'{prefix}attribute_cycle_rmse': sqrt(squared_error / cycle_pairs),
        f'{prefix}attribute_cycle_bias': error / cycle_pairs,
        f'{prefix}attribute_cycle_wape': ratio(absolute_error, gt_total),
    }
    for key in ('cycle_within_0p5', 'cycle_within_1'):
        row[f'{prefix}attribute_{key}'] = sum(
            item[f'{prefix}attribute_cycle_pairs'] * item[f'{prefix}attribute_{key}']
            for item in per_video if item[f'{prefix}attribute_cycle_pairs']) / cycle_pairs
    return row


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
