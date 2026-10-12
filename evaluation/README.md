# evaluation

`evaluate.py` is a marimo notebook that compares predicted circling intervals with
ground truth, per video and overall. It also runs as a command-line tool. The
metric functions live in `metrics.py` and have no marimo or polars dependency.

## Usage

```sh
uv run python evaluation/evaluate.py \
    --ground-truth ground_truth_annotations/finalized_annotations/merged_spans.csv \
    --predicted predicted_spans.csv \
    [--iou-threshold 0.5] [--num-frames 0031_vid=1080398] [--output results.json]
```

To open it as a notebook, pass arguments after `--`:

```sh
uv run marimo edit evaluation/evaluate.py -- --predicted rule-based.tsv
```

`--ground-truth` and `--predicted` each take a single file with rows for every
video. `--ground-truth` defaults to the merged spans shown above. `--predicted`
is required on the command line; in the notebook it is optional and only sets
the initial choice in a dropdown listing `outputs/predicted_spans/*.csv|tsv`.
From the command
line it prints a table and, with `--output`, writes the results as JSON. In the
notebook it shows summary figures and a per-video table.

## Input formats

- CSV files with a video ID column plus `Start Frame`/`End Frame` columns
  (header case, spaces and punctuation are ignored). Rows whose `Circling` is
  not `true`, or whose `ShouldUse` is `false`, are dropped when those columns
  exist. The merged ground-truth spans and predicted spans load as they are.
- Headerless TSV files in the `rule-based.tsv` format (`<VVVV>-<n>`, start, end).

Frame ranges are zero-based and inclusive.

## Metrics

- **MoF**: fraction of all frames (circling and background) labelled correctly.
  It needs each video's total frame count, which comes from `--num-frames`, else
  `data/<video>.mp4`, else the last frame in any interval + 1 (with a warning).
- **Edit score**: `100 × (1 − Levenshtein / max segments)` over the ordered
  circling/background segment sequence, so it ignores exact timing. The overall
  figure is the average across videos.
- **Frame precision / recall / F1**: for the circling class, pooled across videos
  for the overall figure.
- **Matched events**: each ground-truth event can be paired with at most one
  prediction, and only if their IoU is at least `--iou-threshold` (default 0.5;
  greedy, highest IoU first). Missed events and unmatched predictions are also
  reported.

Predictions for videos without ground truth are ignored with a warning.

## Tests

```sh
uv run python -m unittest tests.test_evaluation
```
