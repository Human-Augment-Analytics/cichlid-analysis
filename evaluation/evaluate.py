# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.24.0",
#     "opencv-python-headless",
#     "polars",
# ]
# ///

"""Evaluate predicted circling intervals against ground-truth intervals.

Reports MoF, segmental edit score, segmental F1@{10, 25, 50}, frame
precision/recall/F1/IoU, circling-direction precision/recall/F1 and cycle-count
errors. Frame ranges are zero-based and inclusive. Run interactively:

    uv run marimo edit evaluation/evaluate.py
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import re
    from pathlib import Path

    import marimo as mo
    import polars as pl

    try:
        from .metrics import (LABEL_TAU, TAUS, evaluate_labels, evaluate_video, label_spans, merge,
                              summarize, summarize_labels)
    except ImportError:
        from metrics import (LABEL_TAU, TAUS, evaluate_labels, evaluate_video, label_spans, merge,
                             summarize, summarize_labels)

    ROOT = Path(__file__).resolve().parent.parent
    PREDICTIONS_DIR = ROOT / 'outputs/predictions'
    GROUND_TRUTH = ROOT / 'ground_truth_annotations/finalized_annotations/merged_spans.csv'
    # Annotated events with rotation direction and number of circles, used to label
    # the merged ground-truth spans.
    GROUND_TRUTH_LABELS = sorted(
        (ROOT / 'ground_truth_annotations/finalized_annotations').glob('circling_annotations_vid*.csv'))
    DIRECTION_NAMES = {'cw': 'CW', 'clockwise': 'CW',
                       'ccw': 'CCW', 'counterclockwise': 'CCW', 'anticlockwise': 'CCW'}
    CYCLE_COLUMNS = ('cycles', 'numcycles', 'circles', 'turns', 'noofturnscircles')
    VIDEO_DIR = ROOT / 'data'
    # So MoF does not depend on the source videos, which are normally streamed.
    VIDEO_FRAMES = {'0028_vid': 1_080_175, '0031_vid': 1_080_398}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Circling evaluation

    Compares predicted circling intervals with ground truth, per video and overall.

    - **MoF**: fraction of all frames (circling and background) labelled correctly.
    - **Edit score**: `100 × (1 − Levenshtein / max segments)` over the ordered
      circling/background segment sequence; overall is the mean across videos.
    - **IoU** (intersection over union) of a predicted segment P and a ground-truth
      segment G, both inclusive frame ranges: `|P ∩ G| / |P ∪ G|`, where
      `|P ∪ G| = |P| + |G| − |P ∩ G|`. 1 is an exact match, 0 means no overlap.
    - **F1@τ** for τ ∈ {10, 25, 50}: each predicted segment, in order, is compared
      with the ground-truth segment it has the highest IoU with. It is a true
      positive (TP) if that IoU ≥ τ/100 and no earlier prediction claimed that
      segment; otherwise a false positive. Unclaimed ground-truth segments are false
      negatives. Then precision = TP / predicted segments, recall = TP /
      ground-truth segments and `F1@τ = 2 · precision · recall / (precision + recall)`.
      Counts are pooled across videos for the overall figure.
    - **Frame precision / recall / F1**: for the circling class.
    - **Frame IoU**: `|A ∩ B| / |A ∪ B|`, where A is all frames predicted as circling
      in a video and B is all frames annotated as circling. Overall is the mean across
      videos.
    - **Direction precision / recall / F1** (CW and CCW), when the model predicts a
      direction: segments are paired as in F1@10. A predicted CW segment is a TP if
      it is paired with a CW ground-truth segment, else a FP; CW ground-truth
      segments without a TP are FNs (likewise CCW). Direction F1 is the mean of the
      CW and CCW F1; direction accuracy is the fraction of paired segments whose
      direction is correct, so it ignores missed and spurious events.
    - **Cycle MAE / RMSE / bias**, when the model predicts cycle counts: error
      (predicted − annotated circles) over segments paired as in F1@10. Bias > 0 means
      over-counting. Total annotated and predicted cycles are also reported.

    Ground-truth direction and cycles come from the annotated events in
    `circling_annotations_vid*.csv`: each merged span takes the direction covering
    most of its frames, and the overlapping events' circles prorated by overlap. Spans
    without an overlapping annotated event are left out of these metrics.

    Pick a model from `outputs/predictions/`; the CSV files in its folder are
    concatenated. Frame counts are read from `data/<video>.mp4`, else the last
    annotated or predicted frame + 1.

    ## Expected file layout and schema
    Each model writes one file per video to `outputs/predictions/<model_name>/<video_id>.csv`,
    e.g. `outputs/predictions/rule_based/0031_vid.csv`. Minimal headers expected:
    extra columns will simply be ignored.

    |Header name|Description|
    |-|-|
    |VideoID|`0028_vid` or `0031_vid` for now|
    |StartFrame| frame number |
    |EndFrame| frame number |
    |Direction| optional: `CW` or `CCW` |
    |Cycles| optional: number of circles, may be fractional |

    Evaluation is done across all videos, so write a file for every video, even if
    it only has the header row.
    """)
    return


@app.function(hide_code=True)
def read_intervals(path, labels=False):
    """Read intervals as (video_id, start, end) from a CSV file.

    The file needs video ID and start/end frame columns; header case, spaces and
    punctuation are ignored ("Start Frame" == "StartFrame"). Every non-blank row is used.
    With `labels`, also returns `direction` (CW/CCW) and `cycles` columns, null when
    the file does not have them: a column containing "direction", and one of
    `CYCLE_COLUMNS` ("Cycles", "No. of turns/circles", ...).
    """
    path = Path(path)
    rows = pl.read_csv(path, infer_schema=False)
    rows = rows.filter(~pl.all_horizontal(pl.all().is_null()))
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
    if labels:
        intervals = intervals.with_columns(read_labels(path, rows, columns))
    return intervals


@app.function(hide_code=True)
def read_labels(path, rows, columns):
    """Direction and cycles columns of `rows`, keyed by normalized header in `columns`."""
    direction_column = next((columns[key] for key in columns if 'direction' in key), None)
    cycle_column = next((columns[key] for key in CYCLE_COLUMNS if key in columns), None)
    direction = pl.Series('direction', [None] * len(rows), pl.String)
    if direction_column:
        raw = rows[direction_column].str.strip_chars()
        names = raw.str.to_lowercase().str.replace_all(r'[^a-z]', '')
        unknown = raw.filter(names.is_not_null() & (names != '') & ~names.is_in(list(DIRECTION_NAMES)))
        if len(unknown):
            raise ValueError(f'{path}: unknown directions {sorted(set(unknown))}')
        direction = names.replace_strict(DIRECTION_NAMES, default=None, return_dtype=pl.String)
    cycles = pl.Series('cycles', [None] * len(rows), pl.Float64)
    if cycle_column:
        cycles = rows[cycle_column].str.strip_chars().replace('', None).cast(pl.Float64)
        if (cycles < 0).any():
            raise ValueError(f'{path}: negative cycle counts')
    return direction.alias('direction'), cycles.alias('cycles')


@app.function(hide_code=True)
def read_predictions(model_dir, labels=False):
    """Concatenate the per-video CSV files in a model's prediction folder."""
    paths = sorted(Path(model_dir).glob('*.csv'))
    if not paths:
        raise FileNotFoundError(f'{model_dir}: no prediction CSV files')
    return pl.concat([read_intervals(path, labels) for path in paths])


@app.function(hide_code=True)
def read_ground_truth_labels(paths=GROUND_TRUTH_LABELS):
    """Annotated events with direction and cycles, used to label ground-truth spans."""
    return pl.concat([read_intervals(path, labels=True) for path in paths])


@app.function(hide_code=True)
def prediction_models():
    """Model name -> prediction folder, for every folder in outputs/predictions with CSV files."""
    return {path.name: path for path in sorted(PREDICTIONS_DIR.glob('*/')) if any(path.glob('*.csv'))}


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
    model = mo.ui.dropdown(options=prediction_models(), label='Model')
    model
    return (model,)


@app.cell
def _():
    gt = read_intervals(GROUND_TRUTH)
    gt_labels = read_ground_truth_labels()
    return gt, gt_labels


@app.cell
def _(model):
    mo.stop(model.value is None, mo.callout('Select a model from outputs/predictions.', kind='info'))
    predicted = read_predictions(model.value, labels=True)
    return (predicted,)


@app.function(hide_code=True)
def evaluate(gt, predicted, num_frames=None, gt_labels=None):
    """Per-video and overall metrics plus warnings.

    `num_frames` maps video ID to frame count; otherwise it comes from `VIDEO_FRAMES`,
    then `data/<video>.mp4`, else the last annotated or predicted frame + 1.
    With `gt_labels` (annotated events with `direction` and `cycles`, as from
    `read_ground_truth_labels`), direction and cycle metrics are added for the
    labels `predicted` has non-null values for.
    """
    num_frames = num_frames or {}
    warnings = []
    if gt_labels is not None:
        predicts_direction = 'direction' in predicted.columns and predicted['direction'].is_not_null().any()
        predicts_cycles = 'cycles' in predicted.columns and predicted['cycles'].is_not_null().any()
        unlabelled = 0
    extra = set(predicted['video_id']) - set(gt['video_id'])
    if extra:
        warnings.append(f'Ignoring predictions for videos without ground truth: {sorted(extra)}')

    rows = []
    for video in sorted(set(gt['video_id'])):
        gt_spans = list(gt.filter(pl.col('video_id') == video).select('start', 'end').iter_rows())
        pred_spans = list(predicted.filter(pl.col('video_id') == video).select('start', 'end').iter_rows())
        last = max(end for _, end in gt_spans + pred_spans) + 1
        frames = num_frames.get(video) or VIDEO_FRAMES.get(video) or video_frame_count(video, VIDEO_DIR)
        if frames is None:
            frames = last
            warnings.append(f'{video}: frame count unknown, using last interval frame + 1 ({last:,}) for MoF')
        elif frames < last:
            raise ValueError(f'{video}: {frames} frames but intervals reach frame {last - 1}')
        row = {'video_id': video, **evaluate_video(gt_spans, pred_spans, frames)}
        if gt_labels is not None:
            # Segments as in evaluate_video, labelled from the events overlapping them.
            labelled_gt = label_spans(merge(gt_spans), label_rows(gt_labels, video))
            labelled_pred = label_spans(merge(pred_spans), label_rows(predicted, video))
            row |= evaluate_labels(labelled_gt, labelled_pred, predicts_direction, predicts_cycles)
            unlabelled += sum(segment[2] is None for segment in labelled_gt)
        rows.append(row)
    overall = summarize(rows)
    if gt_labels is not None:
        overall |= summarize_labels(rows)
        if unlabelled:
            warnings.append(f'{unlabelled} ground-truth spans have no annotated direction and are '
                            'left out of direction and cycle metrics')
    rows.append({'video_id': 'overall', **overall})
    return pl.DataFrame(rows, infer_schema_length=None), warnings


@app.function(hide_code=True)
def label_rows(intervals, video):
    """(start, end, direction, cycles) tuples for one video."""
    columns = [c for c in ('start', 'end', 'direction', 'cycles') if c in intervals.columns]
    rows = intervals.filter(pl.col('video_id') == video).select(columns).iter_rows()
    return [row + (None,) * (4 - len(row)) for row in rows]


@app.function(hide_code=True)
def label_stats(overall):
    """Direction and cycle figures, or none when they were not evaluated."""
    if 'direction_f1' not in overall:
        return []
    def percent(value):
        return '—' if value is None or value != value else f'{value:.1%}'
    def count(value):
        return '—' if value is None or value != value else f'{value:.2f}'
    return [
        mo.stat(percent(overall['direction_f1']), label='Direction F1',
                caption=f"CW {percent(overall['cw_f1'])} · CCW {percent(overall['ccw_f1'])}"),
        mo.stat(percent(overall['direction_accuracy']), label='Direction accuracy',
                caption=f"of {overall['direction_matched'] or 0} segments paired at IoU ≥ {LABEL_TAU / 100}"),
        mo.stat(count(overall['cycle_mae']), label='Cycle MAE',
                caption=f"bias {count(overall['cycle_bias'])} · RMSE {count(overall['cycle_rmse'])}"),
    ]


@app.function(hide_code=True)
def summary_view(results, warnings):
    """Overall metric figures, warnings and the per-video table."""
    overall = results.row(-1, named=True)
    return mo.vstack([
        *(mo.callout(mo.md(w), kind='warn') for w in warnings),
        mo.hstack([
            mo.stat(f"{overall['mof']:.1%}", label='MoF'),
            mo.stat(f"{overall['edit_score']:.1f}", label='Edit score'),
            mo.stat(f"{overall['frame_f1']:.1%}", label='Frame F1',
                    caption=f"P {overall['frame_precision']:.1%} · R {overall['frame_recall']:.1%}"),
            mo.stat(f"{overall['frame_iou']:.1%}", label='Frame IoU'),
            *(mo.stat(f"{100 * overall[f'f1@{tau}']:.1f}", label=f'F1@{tau}',
                      caption=f"{overall[f'tp@{tau}']} / {overall['gt_segments']} segments matched")
              for tau in TAUS),
        ], widths='equal'),
        *([mo.hstack(label_stats(overall), widths='equal')] if label_stats(overall) else []),
        mo.ui.table(results, selection=None),
    ])


@app.cell
def _(gt, gt_labels, predicted):
    results, warnings = evaluate(gt, predicted, gt_labels=gt_labels)
    return results, warnings


@app.cell
def _(results):
    results
    return


@app.cell
def _(model, results, warnings):
    mo.vstack([
        mo.md(f'Ground truth: `{display_path(GROUND_TRUTH)}`'),
        mo.md(f'Predictions: `{display_path(model.value)}/*.csv`'),
        summary_view(results, warnings),
    ])
    return


if __name__ == "__main__":
    app.run()
