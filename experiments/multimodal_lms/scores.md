# Pilot model scores

References are the single overlapping human-annotated event per clip. All five events are circling. No labels were supplied to the models.

Circling: 1 if detected, 0 if missed. Direction: 1 if it matches the reference event, 0 otherwise (including a missing direction). Count difference: absolute predicted clip count minus annotated event count, in circles; lower is better.

**All five clip boundaries differ from their reference event boundaries. Count differences and their means are provisional event-reference comparisons, not valid exact-clip count errors. No duration-based rescaling is applied.** Direction agreement also assumes the overlapping event represents the clip's circling direction.

## Per clip

Each model cell is **circling / direction / absolute count difference**.

| Clip | Reference direction, event circles | Gemma | Qwen | Bonsai |
| --- | --- | --- | --- | --- |
| `F1_613_266809-267100` | CW, 2 | 1 / 1 / 1.75 | 1 / 1 / 2.75 | 1 / 1 / 0.75 |
| `F1_613_275040-275190` | CCW, 1 | 1 / 0 / 0.75 | 1 / 0 / 0.5 | 1 / 0 / 0.25 |
| `MC920_334703-334830` | CCW, 2 | 0 / 0 / 2 | 1 / 1 / 0.5 | 1 / 0 / 0.75 |
| `MC920_437259-437437` | CCW, 2 | 1 / 1 / 1.75 | 1 / 1 / 0.5 | 1 / 0 / 0.75 |
| `MC920_713035-713330` | CCW, 3.5 | 1 / 1 / 3.25 | 1 / 0 / 2 | 1 / 0 / 2.25 |

## Aggregate

| Model | Positive-clip recall | Direction agreement | Provisional count MAE |
| --- | ---: | ---: | ---: |
| Gemma 4 E4B | 4/5 (80%) | 3/5 (60%) | 1.90 circles |
| Qwen3.5 4B | 5/5 (100%) | 3/5 (60%) | 1.25 circles |
| Bonsai 27B | 5/5 (100%) | 1/5 (20%) | 0.95 circles |

All clips are positive examples, so these data cannot estimate false-positive rate, specificity, or balanced accuracy. Five clips are too few to establish a model ranking. Gemma failed a reversed-frame control; Qwen's final numeric count contradicts its explanation; Bonsai repeated the same decision/direction/count for every clip. Scores use the saved structured numeric answers without correcting them.

For exact count evaluation, manually annotate these exact five clip intervals or rerun models on the exact reference-event intervals.
