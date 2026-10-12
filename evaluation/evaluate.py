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
zero-based and inclusive. Run interactively:

    uv run marimo edit evaluation/evaluate.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import re
    from pathlib import Path

    import marimo as mo
    import polars as pl

    try:
        from .metrics import evaluate_video, summarize
    except ImportError:
        from metrics import evaluate_video, summarize

    ROOT = Path(__file__).resolve().parent.parent
    PREDICTIONS_DIR = ROOT / 'outputs/predicted_spans'
    GROUND_TRUTH = ROOT / 'ground_truth_annotations/finalized_annotations/merged_spans.csv'
    VIDEO_DIR = ROOT / 'data'


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

    Pick one or more prediction files from `outputs/predicted_spans`; their rows
    are concatenated. Frame counts are read from `data/<video>.mp4`, else the last
    annotated or predicted frame + 1.

    ## Expected file schema
    Minimal headers expected: extra columns will simply be ignored.
    Concatenate rows across all videos together.

    |Header name|Description|
    |-|-|
    |VideoID|`0028_vid` or `0031_vid` for now|
    |StartFrame| frame number |
    |EndFrame| frame number |

    You may generate individual intervals for each video, but evaluation has to be done across the all videos.
    We will merge the individual prediction results to perform evaluation.
    """)
    return


@app.function(hide_code=True)
def read_intervals(path):
    """Read intervals as (video_id, start, end) from CSV or rule-based TSV.

    CSV files need a video ID column plus start/end frames ("Start Frame").
    Rows with Circling other than true or ShouldUse false are dropped when those
    columns exist. Headerless TSV rows are `<VVVV>-<n>  start  end`.
    """
    path = Path(path)
    rows = pl.read_csv(path, infer_schema=False)
    columns = {re.sub(r'[^a-z]', '', column.lower()): column for column in rows.columns}
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


@app.function(hide_code=True)
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


@app.function(hide_code=True)
def display_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


@app.cell(hide_code=True)
def _():
    _candidates = sorted(p for p in PREDICTIONS_DIR.glob('*') if p.suffix in ('.csv', '.tsv'))
    predicted_files = mo.ui.multiselect(
        options={display_path(_p): _p for _p in _candidates},
        label='Predictions',
    )
    iou_threshold = mo.ui.number(start=.05, stop=1, step=.05, value=.5, label='IoU threshold')
    mo.vstack([predicted_files, iou_threshold], justify='start')
    return iou_threshold, predicted_files


@app.cell
def _():
    gt = read_intervals(GROUND_TRUTH)
    return (gt,)


@app.cell
def _(predicted_files):
    mo.stop(not predicted_files.value, mo.callout('Select one or more prediction files.', kind='info'))
    predicted = pl.concat([read_intervals(_p) for _p in predicted_files.value])
    return (predicted,)


@app.cell
def _(gt, iou_threshold, predicted):
    warnings = []
    _extra = set(predicted['video_id']) - set(gt['video_id'])
    if _extra:
        warnings.append(f'Ignoring predictions for videos without ground truth: {sorted(_extra)}')

    rows = []
    for _video in sorted(set(gt['video_id'])):
        _gt = list(gt.filter(pl.col('video_id') == _video).select('start', 'end').iter_rows())
        _pred = list(predicted.filter(pl.col('video_id') == _video).select('start', 'end').iter_rows())
        _last = max(end for _, end in _gt + _pred) + 1
        _frames = video_frame_count(_video, VIDEO_DIR)
        if _frames is None:
            _frames = _last
            warnings.append(f'{_video}: frame count unknown, using last interval frame + 1 ({_last:,}) for MoF')
        elif _frames < _last:
            raise ValueError(f'{_video}: {_frames} frames but intervals reach frame {_last - 1}')
        rows.append({'video_id': _video, **evaluate_video(_gt, _pred, _frames, iou_threshold.value)})
    rows.append({'video_id': 'overall', **summarize(rows)})
    results = pl.DataFrame(rows)
    return results, warnings


@app.cell
def _(results):
    results
    return


@app.cell
def _(iou_threshold, predicted_files, results, warnings):
    _overall = results.row(-1, named=True)
    mo.vstack([
        mo.md(f'Ground truth: `{display_path(GROUND_TRUTH)}`'),
        mo.md('Predictions: ' + ', '.join(f'`{display_path(_p)}`' for _p in predicted_files.value)),
        *(mo.callout(mo.md(_w), kind='warn') for _w in warnings),
        mo.hstack([
            mo.stat(f"{_overall['mof']:.1%}", label='MoF'),
            mo.stat(f"{_overall['edit_score']:.1f}", label='Edit score'),
            mo.stat(f"{_overall['frame_precision']:.1%}", label='Frame precision'),
            mo.stat(f"{_overall['frame_recall']:.1%}", label='Frame recall'),
            mo.stat(f"{_overall['frame_f1']:.1%}", label='Frame F1'),
            mo.stat(
                f"{_overall['matched_events']} / {_overall['gt_events']}",
                label=f'Events matched (IoU ≥ {iou_threshold.value:g})',
            ),
        ], widths='equal'),
        mo.ui.table(results, selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()
