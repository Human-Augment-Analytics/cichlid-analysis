# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.24.0",
#     "opencv-python-headless",
#     "polars",
# ]
# ///

"""Evaluate predicted circling intervals against ground-truth intervals.

Reports MoF, segmental edit score, frame precision/recall/F1 and the number of
ground-truth circling events matched by a prediction. Frame ranges are
zero-based and inclusive.

    uv run python evaluation/evaluate.py \\
        --ground-truth ground_truth_annotations/finalized_annotations/merged_spans.csv \\
        --predicted outputs/predicted_spans/*_spans_threshold_0.50.csv
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import argparse
    import json
    import re
    import sys
    from pathlib import Path

    import marimo as mo
    import polars as pl

    ROOT = Path(__file__).resolve().parent.parent
    INTERVAL_SCHEMA = {'video_id': pl.String, 'start': pl.Int64, 'end': pl.Int64}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Circling evaluation

    Compares predicted circling intervals with ground truth, per video and overall.

    - **MoF**: fraction of all frames (circling and background) labelled correctly.
    - **Edit score**: `100 × (1 − Levenshtein / max segments)` over the ordered
      circling/background segment sequence; overall is the mean across videos.
    - **Frame precision / recall / F1**: for the circling class.
    - **Matched events**: ground-truth events paired one-to-one with a prediction
      at IoU ≥ the threshold (greedy, highest IoU first).

    Pass arguments after `--` when editing, e.g.
    `marimo edit evaluation/evaluate.py -- --predicted path.csv`.

    ## Expected file schema
    Minimal headers expected: extra columbns will simply be ignored.
    Concatenate rows across all videos together.

    |Header name|Description|
    |-|-|
    |VideoID|`0028_vid` or `0031_vid` for now|
    |StartFrame| frame number |
    |EndFrame| frame number |
    """)
    return


@app.function
def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--ground-truth', nargs='+', type=Path,
        default=[ROOT / 'ground_truth_annotations/finalized_annotations/merged_spans.csv'],
        help='ground-truth interval files (default: ground_truth_annotations/finalized_annotations/merged_spans.csv)',
    )
    parser.add_argument(
        '--predicted', nargs='+', type=Path,
        default=sorted((ROOT / 'outputs/predicted_spans').glob('*_spans_threshold_0.50.csv')),
        help='predicted interval files (default: outputs/predicted_spans/*_spans_threshold_0.50.csv)',
    )
    parser.add_argument('--iou-threshold', type=float, default=.5, help='minimum IoU for an event match')
    parser.add_argument(
        '--num-frames', nargs='+', default=[], metavar='VIDEO=N',
        help='frame count per video; otherwise read from VIDEO-DIR/<video>.mp4, '
             'else the last annotated or predicted frame + 1',
    )
    parser.add_argument('--video-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, help='optional JSON path for the results')
    args = parser.parse_args(argv)
    if not args.ground_truth or not args.predicted:
        parser.error('ground-truth and predicted files are required')
    if not 0 < args.iou_threshold <= 1:
        parser.error('iou-threshold must be in (0, 1]')
    return args


@app.function
def read_intervals(path):
    """Read intervals as (video_id, start, end) from CSV or rule-based TSV.

    CSV files need a video ID column plus start/end frames ("Start Frame").
    Rows with Circling other than true or ShouldUse false are dropped when those
    columns exist. Headerless TSV rows are `<VVVV>-<n>  start  end`.
    """
    path = Path(path)
    if path.suffix == '.tsv':
        rows = pl.read_csv(path, separator='\t', has_header=False, new_columns=['id', 'start', 'end'])
        return rows.select(
            pl.col('id').str.slice(0, 4).add('_vid').alias('video_id'),
            pl.col('start').cast(pl.Int64), pl.col('end').cast(pl.Int64),
        )

    rows = pl.read_csv(path, infer_schema=False)
    columns = {re.sub(r'[^a-z]', '', column.lower()): column for column in rows.columns}
    if 'circling' in columns:
        rows = rows.filter(pl.col(columns['circling']).str.to_lowercase() == 'true')
    if 'shoulduse' in columns:
        rows = rows.filter(pl.col(columns['shoulduse']).fill_null('').str.to_lowercase() != 'false')
    if 'videoid' not in columns:
        raise ValueError(f'{path}: missing video ID column')
    if {'startframe', 'endframe'} <= columns.keys():
        start = rows[columns['startframe']].cast(pl.Int64)
        end = rows[columns['endframe']].cast(pl.Int64)
    else:
        raise ValueError(f'{path}: needs start/end frame')
    intervals = pl.DataFrame({'video_id': rows[columns['videoid']], 'start': start, 'end': end})
    invalid = intervals.filter((pl.col('start') < 0) | (pl.col('end') < pl.col('start')))
    if len(invalid):
        raise ValueError(f'{path}: invalid intervals\n{invalid}')
    return intervals


@app.function
def merge(intervals):
    """Union of inclusive (start, end) intervals; adjacent ranges are joined."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(span) for span in merged]


@app.function
def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


@app.function
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


@app.function
def levenshtein(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


@app.function
def edit_score(gt_spans, pred_spans, num_frames):
    gt, pred = segment_labels(gt_spans, num_frames), segment_labels(pred_spans, num_frames)
    return 100 * (1 - levenshtein(gt, pred) / max(len(gt), len(pred), 1))


@app.function
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


@app.function
def ratio(numerator, denominator):
    return numerator / denominator if denominator else float('nan')


@app.function
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


@app.function
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


@app.function
def video_frame_count(video_id, video_dir):
    path = Path(video_dir) / f'{video_id}.mp4'
    if not path.exists():
        return None
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return count or None


@app.cell
def _():
    args = parse_args(sys.argv[1:])
    return (args,)


@app.cell
def _(args):
    def load(paths):
        frames = [read_intervals(path) for path in paths]
        return pl.concat(frames) if frames else pl.DataFrame(schema=INTERVAL_SCHEMA)

    gt = load(args.ground_truth)
    predicted = load(args.predicted)
    return gt, predicted


@app.cell
def _(args, gt, predicted):
    explicit_frames = {}
    for _item in args.num_frames:
        _video, _, _count = _item.rpartition('=')
        explicit_frames[_video] = int(_count)

    warnings = []
    _extra = set(predicted['video_id']) - set(gt['video_id'])
    if _extra:
        warnings.append(f'Ignoring predictions for videos without ground truth: {sorted(_extra)}')

    rows = []
    for _video in sorted(set(gt['video_id'])):
        _gt = list(gt.filter(pl.col('video_id') == _video).select('start', 'end').iter_rows())
        _pred = list(predicted.filter(pl.col('video_id') == _video).select('start', 'end').iter_rows())
        _last = max(end for _, end in _gt + _pred) + 1
        _frames = explicit_frames.get(_video) or video_frame_count(_video, args.video_dir)
        if _frames is None:
            _frames = _last
            warnings.append(f'{_video}: frame count unknown, using last interval frame + 1 ({_last:,}) for MoF')
        elif _frames < _last:
            raise ValueError(f'{_video}: {_frames} frames but intervals reach frame {_last - 1}')
        rows.append({'video_id': _video, **evaluate_video(_gt, _pred, _frames, args.iou_threshold)})
    rows.append({'video_id': 'overall', **summarize(rows)})
    results = pl.DataFrame(rows)
    return results, warnings


@app.cell
def _(args, results, warnings):
    _overall = results.row(-1, named=True)
    mo.vstack([
        *(mo.callout(mo.md(_w), kind='warn') for _w in warnings),
        mo.hstack([
            mo.stat(f"{_overall['mof']:.1%}", label='MoF'),
            mo.stat(f"{_overall['edit_score']:.1f}", label='Edit score'),
            mo.stat(f"{_overall['frame_precision']:.1%}", label='Frame precision'),
            mo.stat(f"{_overall['frame_recall']:.1%}", label='Frame recall'),
            mo.stat(f"{_overall['frame_f1']:.1%}", label='Frame F1'),
            mo.stat(
                f"{_overall['matched_events']} / {_overall['gt_events']}",
                label=f'Events matched (IoU ≥ {args.iou_threshold:g})',
            ),
        ], widths='equal'),
        mo.ui.table(results, selection=None),
    ])
    return


@app.cell
def _(args, results, warnings):
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'ground_truth': [str(p) for p in args.ground_truth],
            'predicted': [str(p) for p in args.predicted],
            'iou_threshold': args.iou_threshold,
            'results': results.to_dicts(),
        }, indent=2))

    if not mo.running_in_notebook():
        for _w in warnings:
            print(f'warning: {_w}', file=sys.stderr)
        _metrics = (
            'num_frames', 'mof', 'edit_score', 'frame_precision', 'frame_recall', 'frame_f1',
            'gt_events', 'pred_events', 'matched_events',
        )
        _table = pl.DataFrame({'metric': _metrics, **{
            _row['video_id']: [
                f'{_row[_m]:,}' if isinstance(_row[_m], int) else f'{_row[_m]:.4f}' for _m in _metrics
            ]
            for _row in results.iter_rows(named=True)
        }})
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, tbl_hide_dataframe_shape=True,
                       tbl_hide_column_data_types=True, tbl_cell_alignment='RIGHT'):
            print(_table)
    return


if __name__ == "__main__":
    app.run()
