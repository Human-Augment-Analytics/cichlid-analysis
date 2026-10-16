"""Frame- and segment-level metrics for circling intervals.

Frame ranges are zero-based and inclusive (start, end) tuples.
"""


# IoU thresholds, in percent, for segmental F1@tau.
TAUS = (10, 25, 50)


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


def segment_true_positives(gt_spans, pred_spans, tau):
    """True positives for F1@tau.

    Each predicted segment, in order, is compared with the ground-truth segment it
    overlaps most. It is a true positive if that IoU is at least tau / 100 and no
    earlier prediction already claimed the segment; otherwise a false positive.
    """
    claimed = set()
    for pred in pred_spans:
        if not gt_spans:
            break
        best = max(range(len(gt_spans)), key=lambda i: iou(gt_spans[i], pred))
        if 100 * iou(gt_spans[best], pred) >= tau and best not in claimed:
            claimed.add(best)
    return len(claimed)


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
        'gt_segments': len(gt_spans), 'pred_segments': len(pred_spans),
        **{f'tp@{tau}': tps[tau] for tau in TAUS},
        **{f'f1@{tau}': segment_f1(tps[tau], len(pred_spans), len(gt_spans)) for tau in TAUS},
    }


def summarize(per_video):
    """Pool frame and segment counts across videos; edit score is the mean."""
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
        'gt_segments': total['gt_segments'], 'pred_segments': total['pred_segments'],
        **{f'tp@{tau}': total[f'tp@{tau}'] for tau in TAUS},
        **{f'f1@{tau}': segment_f1(total[f'tp@{tau}'], total['pred_segments'], total['gt_segments'])
           for tau in TAUS},
    }
