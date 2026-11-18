import numpy as np
import polars as pl

WINDOW = 1800
STEP = 900


def window_grid(frame_count, length=WINDOW, step=STEP):
    starts = np.arange(0, max(frame_count - length, 0) + 1, step)
    return starts, np.minimum(starts + length, frame_count)


def window_labels(spans, frame_count, length=WINDOW, step=STEP):
    positive = np.zeros(frame_count + 1, bool)
    for start, end in spans:
        positive[max(0, start):min(frame_count, end + 1)] = True
    covered = np.r_[0, np.cumsum(positive[:frame_count])]
    starts, ends = window_grid(frame_count, length, step)
    return (covered[ends] - covered[starts]) > 0


def scores(y, p):
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else float('nan')
    recall = tp / (tp + fn) if tp + fn else float('nan')
    specificity = tn / (tn + fp) if tn + fp else float('nan')
    return {'windows': len(y), 'circling_windows': int(y.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall,
            'f1': 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            'balanced_accuracy': (recall + specificity) / 2}


def video_spans(table, video):
    return table.filter(pl.col('video_id') == video).select('start', 'end').rows()


def window_metrics(gt, predicted, num_frames, length=WINDOW, step=STEP):
    rows, ys, ps = [], [], []
    for video in sorted(set(gt['video_id']) | set(num_frames)):
        count = num_frames[video]
        y = window_labels(video_spans(gt, video), count, length, step)
        p = window_labels(video_spans(predicted, video), count, length, step)
        ys.append(y)
        ps.append(p)
        rows.append({'video_id': video, **scores(y, p)})
    rows.append({'video_id': 'overall', **scores(np.concatenate(ys), np.concatenate(ps))})
    return pl.DataFrame(rows)
