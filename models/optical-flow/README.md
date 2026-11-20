# Optical-flow circling detector

A pose-free, classical-computer-vision detector for male-female cichlid
**circling** events in overhead tank videos. Takes raw mp4 pixels in,
produces per-event rotation direction (CW / CCW) and span boundaries out.
Built entirely on dense Farneback optical flow and simple geometric
aggregation of the resulting flow field.

Headline result: **100% rotation-direction accuracy over 658 scored events
across 8 different-tank 10 hr videos (658 / 658 correct).**


## 1. Theory in brief

### Optical flow

Optical flow estimates the apparent per-pixel motion field `(u, v)`
between two consecutive frames `I_t` and `I_{t+1}` under a brightness-
constancy assumption. Two canonical families:

- **Sparse, feature-based (Lucas & Kanade, 1981).** Tracks a small set of
  salient corner points. Good for textured rigid scenes; poor for small,
  smooth-skinned fish bodies against a textured sand floor.
- **Dense, global (Horn & Schunck, 1981).** Estimates flow at every pixel
  by minimising a smoothness-regularised energy. Produces a full motion
  field but is sensitive to brightness noise and typically slow.

We use **Farneback's dense polynomial-expansion method** (Farneback 2003)
as implemented in OpenCV's `cv2.calcOpticalFlowFarneback`:

> Farneback, G. (2003). *"Two-Frame Motion Estimation Based on Polynomial
> Expansion."* Scandinavian Conference on Image Analysis (SCIA), LNCS 2749,
> 363-370. <https://doi.org/10.1007/3-540-45103-X_50>

Farneback approximates each neighbourhood of the image as a quadratic
polynomial, then solves for the motion that relates two polynomials.
It is comparatively insensitive to small image noise, needs no feature
detector, and scales cheaply thanks to the pyramidal implementation.
For our cichlid footage (small, smooth, poorly-textured fish bodies on
a high-texture sand background) it is a much better fit than LK.

Modern deep-learning methods (**RAFT**, Teed & Deng, ECCV 2020,
<https://arxiv.org/abs/2003.12039>; **SEA-RAFT**, Wang et al., ECCV 2024)
substantially improve accuracy for small, textureless objects but require
a GPU and are ~10x slower per frame. These may be explored in future for num_turns estimation where magnitude fidelity matters more.

### From flow field to rotation direction

Given the `(u, v)` field over active pixels inside a central region of the
frame, we compute per-frame:

- `activity` = count of pixels where `|flow| > 0.5` px/frame,
- `rot_omega` = best-fit angular velocity of the active region about its
  flow-magnitude-weighted centroid, defined as
  `sum_i (r_xi * v_i - r_yi * u_i) / sum_i (r_xi^2 + r_yi^2)`,
  bounded (roughly ±0.05 rad/frame).

Over the span of a GT (or detected) event, the direction decision uses
the **sign-majority** of per-frame `rot_omega`: for frames above a modest
activity floor, count how many have `rot_omega > 0` vs `< 0`, and return
the dominant sign. Noise contributes symmetric ± votes that cancel;
coherent orbital motion produces a strong majority.

On an exact tie between CW and CCW vote counts, the decision falls back
to the sign of the signed **mean** of `rot_omega` across the same frames.
This gives a deterministic answer whenever any residual rotation signal
exists; `None` is returned only if the mean is also exactly zero (never
seen in practice).


## 2. What the detector predicts

| output | status |
|---|---|
| **Rotation direction** (CW / CCW) | ✅ solved — 100% on 658 scored events |
| **Span start / end frames** (temporal localisation) | ⚠️ working baseline, strict-IoU still weak |
| **Number of turns** | 🚧 TODO (see §7) |


## 3. Current performance

### 3.1 Rotation direction

Evaluated by running `direction_predict.predict_direction` on every GT
span across the full 8-video set (10 h each).

| metric | value |
|---|---:|
| videos | 8 |
| GT events (total) | 659 |
| events scored | 658 |
| events correct | **658** |
| **accuracy** | **1.0000** |
| **macro F1** | **1.0000** |
| CW: precision / recall / F1 | 1.0000 / 1.0000 / 1.0000 |
| CCW: precision / recall / F1 | 1.0000 / 1.0000 / 1.0000 |

Confusion matrix (rows = GT, cols = prediction):

```
      Pred CW   Pred CCW
GT CW     326          0
GT CCW      0        332
```

Per-video breakdown is in
`outputs/optical-flow/direction_only/all_videos_metrics.json`.

### 3.2 Span detection

Evaluated on hour 1 of `vid0007_day05` (34 GT events) using the rule-based
event proposer in `detect.py` and the event-level metrics in `evaluate.py`,
at the detector's native **frame_stride = 1** (full 30 fps). Processing
stride > 1 was tried later and noticeably degrades boundary localisation,
so the numbers below are what the pipeline actually delivers on well-tuned
settings:

| metric | value |
|---|---:|
| mAP @ tIoU ∈ {0.1, 0.3, 0.5, 0.7, 0.9} | **0.518** (mean) |
| AP @ tIoU = 0.1 / 0.3 / 0.5 / 0.7 / 0.9 | 0.887 / 0.792 / 0.597 / 0.238 / 0.073 |
| F1 @ tIoU = 0.1 | 0.607 |
| F1 @ tIoU = 0.5 | 0.429 |
| F1 @ tIoU = 0.9 | 0.107 |
| Segmental edit score (direction-aware) | 0.436 |
| Direction accuracy on matched preds @ tIoU=0.5 | **1.000** |

The sharp drop from `AP@0.3 → AP@0.7` is the expected failure mode for
pose-free flow: events are localised well enough to overlap, but
boundaries drift by ~1 s because the sign-consistency window has ~2 s
inertia. Span detection on more than one video would require a per-video
calibration of the `activity_area_range` filter; see §7.


## 4. Repo layout

```
models/optical-flow/
  README.md             # this file
  requirements.txt      # Python dependencies (pip install -r)
  flow.py               # Farneback + per-frame flow features
  detect.py             # event proposer + direction + num_turns
  evaluate.py           # mAP, F1@k, edit score, direction accuracy
  pipeline.py           # single-video span + direction pipeline
  run_videos.py         # batch over a folder of videos + matching annots
  direction_predict.py  # standalone direction predictor (span -> direction)
  direction_eval.ipynb  # notebook: direction evaluation across all videos
```

Expected runtime input tree:

```
ground_truth_annotations/
  data/                             # source mp4 files
  finalized_annotations/            # one circling_annotations_vid<id>[-day<d>].csv per video
outputs/optical-flow/
  direction_only/                   # consolidated direction outputs
  results/                          # per-video span-detection outputs (pipeline.py)
```


## 5. Quick start

### 5.1 Install

From the repo root:

```bash
pip install -r models/optical-flow/requirements.txt
```

System-level: **ffmpeg** must be on the `PATH` for `cv2.VideoCapture` to
decode mp4 files.

### 5.2 Direction evaluation across all videos (the recommended workflow)

Place one mp4 per video under `ground_truth_annotations/data/` and the
matching annotation CSV under `ground_truth_annotations/finalized_annotations/`
(naming convention: `circling_annotations_vid<4-digit-id>[-day<N>].csv`),
then open `direction_eval.ipynb` and Run All.

It:

1. Auto-discovers every `*.mp4` / annotation pair.
2. Dispatches videos to a `ProcessPoolExecutor` with
   `MAX_PARALLEL_VIDEOS = 2` workers (edit at the top of cell 1 to tune).
3. For each video, calls `direction_predict.score_video_gt_events` which
   applies `predict_direction` to every GT event via `df.apply`.
   No full-video Farneback pass -- only the frames inside each GT span
   are processed.
4. Writes:
   - `outputs/optical-flow/direction_only/all_videos_direction.csv`
     (one row per GT event with
      `video, event_id, start_frame, end_frame, start_time, end_time,
       gt_direction, pred_direction, correct`),
   - `outputs/optical-flow/direction_only/all_videos_metrics.json`
     (accuracy, confusion matrix, per-class precision / recall / F1,
      macro + micro F1, per-video breakdown),
   - `outputs/optical-flow/direction_only/confusion_matrix.png`.

### 5.3 Single-span direction API

For scripting / notebook usage without the batch notebook:

```python
from pathlib import Path
import pandas as pd
import direction_predict as dp

video = Path('ground_truth_annotations/data/0007_vid_day05.mp4')
gt = pd.read_csv('ground_truth_annotations/finalized_annotations/circling_annotations_vid0007-day5.csv')

# Direction for a single span:
pred = dp.predict_direction(video, start_frame=20142, end_frame=21044)
# -> 'CW' / 'CCW' / None

# Batch over a DataFrame of spans (GT or detected):
gt['predicted_direction'] = gt.apply(
    lambda r: dp.predict_direction(video,
                                    int(r['Start Frame']),
                                    int(r['End Frame'])),
    axis=1,
)
```

For scoring predicted-span CSVs (where there is no row-wise GT direction),
match each predicted span to the best-overlapping GT span first:

```python
preds = pd.read_csv('.../predictions.csv')
preds = dp.match_to_gt_spans(preds, gt, min_iou=0.1)   # adds gt_direction
preds['predicted_direction'] = preds.apply(
    lambda r: dp.predict_direction(video,
                                    int(r['start_frame']),
                                    int(r['end_frame'])),
    axis=1,
)
print(dp.compute_direction_metrics(preds,
                                   actual_col='gt_direction',
                                   pred_col='predicted_direction'))
```

### 5.4 Span detection on a single video

```bash
python models/optical-flow/pipeline.py \
    --video       ground_truth_annotations/data/0007_vid_day05.mp4 \
    --annotations ground_truth_annotations/finalized_annotations/circling_annotations_vid0007-day5.csv \
    --out         outputs/optical-flow/results \
    --mode        both
```

Writes `flow.parquet`, `predictions.csv`, `direction_on_gt.csv`,
`direction_on_predicted.csv`, `direction_conf_matrix_on_gt.csv`,
`metrics_summary.json` under
`outputs/optical-flow/results/<video_stem>/`.


## 6. Key tuning knobs

Most defaults in `pipeline.py` and `direction_predict.py` transfer across
overhead tank footage with fish of roughly similar on-screen size. Review
if that assumption changes:

| knob | default | rationale |
|---|---|---|
| `downsample` | 2.0 | fish body ~40 px at 2x on a 1296x972 frame |
| `crop_frac`  | 0.1 | 10% off each side excludes most tank-wall fixtures / reflections |
| `mag_thr` (per-pixel flow) | 0.5 px/frame | sand-texture noise floor |
| `activity_thr` (per-frame) | **50** | sign-majority voter floor; works on the quietest tank in the 8-video set |
| `sign_win_s` | 2.0 | rolling-mean window for span proposer |
| `sign_thr`   | 0.6 | sustained \|mean sign\| above this triggers an event candidate |
| `min_dur_s`  | 3.0 | matches the >= 3 s rule in the annotation guidelines |
| `merge_gap_s`| 2.0 | merge adjacent candidates <2 s apart, same direction |
| `min_score`  | 0.75 | kills ~45% of spurious FPs with zero TP loss on hour 1 of vid0007 |
| `activity_area_range` | (200, 500) | pair-circling range in active pixels; **video-specific, must recalibrate per video** |

For the direction task the first four knobs matter; the rest tune span
detection.


## 7. TODO

- **Number of turns** (`detect.num_turns_on_span`). The current estimator
  integrates `rot_omega` over the span and divides by `2π`. It
  consistently *under*-counts by 5-10x because Farneback at `downsample=2`
  underestimates tangential velocities on small, smooth fish bodies.
  Planned next steps:
  1. **Replace Farneback with RAFT / SEA-RAFT** on only the candidate
     spans (GPU inference on minutes of footage, not full videos).
  2. **Trajectory-based counting** via improved classical segmentation
     (watershed splitting of merged-pair blobs + temporal linking) and
     tracking the angular position of each fish around the pair
     centroid.

- **Span detection generalisation.** The `activity_area_range` filter is
  calibrated to one video and does not transfer. Planned: an auto-
  calibration pass that measures typical single-fish active-pixel count
  from a short quiet segment of each video and sets the range to roughly
  `[2x, 4x]` that value.


## 8. Dependencies

Listed in `requirements.txt` (OpenCV, NumPy, pandas, SciPy, matplotlib,
pyarrow, Jupyter). `ffmpeg` is required as a system binary; the rest is
plain pip-installable on Python 3.11.
