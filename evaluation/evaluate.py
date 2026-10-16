# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.24.0",
#     "opencv-python-headless",
#     "polars",
# ]
# ///

"""Evaluate predicted circling intervals against ground-truth intervals.

Reports MoF, segmental edit score, segmental F1@{10, 25, 50} and frame
precision/recall/F1. Frame ranges are zero-based and inclusive. Run interactively:

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
        from .metrics import TAUS, evaluate_video, summarize
    except ImportError:
        from metrics import TAUS, evaluate_video, summarize

    ROOT = Path(__file__).resolve().parent.parent
    PREDICTIONS_DIR = ROOT / 'outputs/predictions'
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

    Evaluation is done across all videos, so write a file for every video, even if
    it only has the header row.
    """)
    return


@app.function(hide_code=True)
def read_intervals(path):
    """Read intervals as (video_id, start, end) from a CSV file.

    The file needs video ID and start/end frame columns; header case, spaces and
    punctuation are ignored ("Start Frame" == "StartFrame"). Every row is used.
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
def read_predictions(model_dir):
    """Concatenate the per-video CSV files in a model's prediction folder."""
    paths = sorted(Path(model_dir).glob('*.csv'))
    if not paths:
        raise FileNotFoundError(f'{model_dir}: no prediction CSV files')
    return pl.concat([read_intervals(path) for path in paths])


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
    return (gt,)


@app.cell
def _(model):
    mo.stop(model.value is None, mo.callout('Select a model from outputs/predictions.', kind='info'))
    predicted = read_predictions(model.value)
    return (predicted,)


@app.function(hide_code=True)
def evaluate(gt, predicted, num_frames=None):
    """Per-video and overall metrics plus warnings.

    `num_frames` maps video ID to frame count; otherwise it is read from
    `data/<video>.mp4`, else the last annotated or predicted frame + 1.
    """
    num_frames = num_frames or {}
    warnings = []
    extra = set(predicted['video_id']) - set(gt['video_id'])
    if extra:
        warnings.append(f'Ignoring predictions for videos without ground truth: {sorted(extra)}')

    rows = []
    for video in sorted(set(gt['video_id'])):
        gt_spans = list(gt.filter(pl.col('video_id') == video).select('start', 'end').iter_rows())
        pred_spans = list(predicted.filter(pl.col('video_id') == video).select('start', 'end').iter_rows())
        last = max(end for _, end in gt_spans + pred_spans) + 1
        frames = num_frames.get(video) or video_frame_count(video, VIDEO_DIR)
        if frames is None:
            frames = last
            warnings.append(f'{video}: frame count unknown, using last interval frame + 1 ({last:,}) for MoF')
        elif frames < last:
            raise ValueError(f'{video}: {frames} frames but intervals reach frame {last - 1}')
        rows.append({'video_id': video, **evaluate_video(gt_spans, pred_spans, frames)})
    rows.append({'video_id': 'overall', **summarize(rows)})
    return pl.DataFrame(rows), warnings


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
            *(mo.stat(f"{100 * overall[f'f1@{tau}']:.1f}", label=f'F1@{tau}',
                      caption=f"{overall[f'tp@{tau}']} / {overall['gt_segments']} segments matched")
              for tau in TAUS),
        ], widths='equal'),
        mo.ui.table(results, selection=None),
    ])


@app.cell
def _(gt, predicted):
    results, warnings = evaluate(gt, predicted)
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
