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
        --predicted predicted_spans.csv
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

    try:
        from .metrics import evaluate_video, summarize
    except ImportError:
        from metrics import evaluate_video, summarize

    ROOT = Path(__file__).resolve().parent.parent
    PREDICTIONS_DIR = ROOT / 'outputs/predicted_spans'


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

    Pick the prediction file from the dropdown, which lists `outputs/predicted_spans`
    plus any `--predicted` file. Pass arguments after `--` when editing, e.g.
    `marimo edit evaluation/evaluate.py -- --predicted path.csv`.

    ## Expected file schema
    Minimal headers expected: extra columns will simply be ignored.
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
        '--ground-truth', type=Path,
        default=ROOT / 'ground_truth_annotations/finalized_annotations/merged_spans.csv',
        help='ground-truth interval file, with rows for all videos '
             '(default: ground_truth_annotations/finalized_annotations/merged_spans.csv)',
    )
    parser.add_argument(
        '--predicted', type=Path, required=not mo.running_in_notebook(),
        help='predicted interval file, with rows for all videos (optional in the notebook)',
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
def display_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


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
    _candidates = sorted(p for p in PREDICTIONS_DIR.glob('*') if p.suffix in ('.csv', '.tsv'))
    if args.predicted:
        _candidates.append(args.predicted)
    predicted_file = mo.ui.dropdown(
        options={display_path(_p): _p for _p in _candidates},
        value=display_path(args.predicted) if args.predicted else None,
        label='Predictions', searchable=True,
    )
    predicted_file
    return (predicted_file,)


@app.cell
def _(args):
    gt = read_intervals(args.ground_truth)
    return (gt,)


@app.cell
def _(predicted_file):
    mo.stop(predicted_file.value is None, mo.callout('Select a prediction file.', kind='info'))
    predicted = read_intervals(predicted_file.value)
    return (predicted,)


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
        mo.md(f'Ground truth: `{display_path(args.ground_truth)}`'),
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
def _(args, predicted_file, results, warnings):
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'ground_truth': str(args.ground_truth),
            'predicted': str(predicted_file.value),
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
