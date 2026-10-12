"""Frame- and event-level metrics for circling intervals.

Frame ranges are zero-based and inclusive (start, end) tuples.
"""


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


def match_events(gt_events, pred_events, iou_threshold):
    """Greedy one-to-one matching by IoU; returns (gt_index, pred_index, iou)."""
    candidates = []
    for i, g in enumerate(gt_events):
        for j, p in enumerate(pred_events):
            if inter := overlap(g, p):
                iou = inter / (g[1] - g[0] + p[1] - p[0] + 2 - inter)
                if iou >= iou_threshold:
                    candidates.append((iou, i, j))
    used_gt, used_pred, matches = set(), set(), []
    for iou, i, j in sorted(candidates, reverse=True):
        if i not in used_gt and j not in used_pred:
            used_gt.add(i)
            used_pred.add(j)
            matches.append((i, j, iou))
    return matches


def ratio(numerator, denominator):
    return numerator / denominator if denominator else float('nan')


def evaluate_video(gt_events, pred_events, num_frames, iou_threshold=.5):
    """Metrics for one video from inclusive (start, end) frame intervals."""
    gt_spans, pred_spans = merge(gt_events), merge(pred_events)
    gt_frames = sum(end - start + 1 for start, end in gt_spans)
    pred_frames = sum(end - start + 1 for start, end in pred_spans)
    tp = sum(overlap(g, p) for g in gt_spans for p in pred_spans)
    precision, recall = ratio(tp, pred_frames), ratio(tp, gt_frames)
    matched = len(match_events(gt_events, pred_events, iou_threshold))
    return {
        'num_frames': num_frames,
        'mof': ratio(num_frames - gt_frames - pred_frames + 2 * tp, num_frames),
        'edit_score': edit_score(gt_spans, pred_spans, num_frames),
        'frame_tp': tp, 'frame_fp': pred_frames - tp, 'frame_fn': gt_frames - tp,
        'frame_precision': precision, 'frame_recall': recall,
        'frame_f1': ratio(2 * precision * recall, precision + recall) if tp else 0.,
        'gt_events': len(gt_events), 'pred_events': len(pred_events),
        'matched_events': matched,
        'missed_events': len(gt_events) - matched,
        'unmatched_predictions': len(pred_events) - matched,
    }


def summarize(per_video):
    """Pool frame and event counts across videos; edit score is the mean."""
    total = {key: sum(row[key] for row in per_video) for key in (
        'num_frames', 'frame_tp', 'frame_fp', 'frame_fn', 'gt_events', 'pred_events',
        'matched_events', 'missed_events', 'unmatched_predictions',
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
        **{key: total[key] for key in (
            'gt_events', 'pred_events', 'matched_events', 'missed_events', 'unmatched_predictions',
        )},
    }
