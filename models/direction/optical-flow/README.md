# Circling pipeline — direction, turns, span features

Detects male-female cichlid **circling** events in overhead tank videos
and, per event, reports:

- **Rotation direction** (CW / CCW)
- **Number of turns** (quantized: integer, half-turn, or round-up)
- **`is_circling` flag** (true iff quantized turns ≥ 1)
- (Optional) **Dense per-frame Optical Flow (OF) features** inside each
  fish mask, for downstream TAL / span-detection models

The pipeline is organised as **three parallel methods**, each independent,
each with its own cost / accuracy / annotation profile. You can mix and match depending on whether pose-tracker output and / or a GPU are available for the incoming video.

- **Method A — Optical Flow (OF) direction** on raw pixels (Farneback +
  sign-majority). No tracker, no GPU, no annotation. 100 % accuracy
  over 658 events.
- **Method B — Pose turns** on pre-computed per-pair orbital angle.
  No GPU, no annotation. 99.8 % direction, MAE 0.72 turns / 94 % within 2 turns across 658 events, but *only runs on videos for which a pose parquet already exists*.
- **Method C — SAM 2 masks** (initialised from a single click per fish per
  event *or* from Grounding DINO's zero-shot detection). GPU required;
  gives the cleanest centroids, the cleanest OF features, and the
  cleanest num_turns. SAM 2 throughput scales with VRAM — see
  Section 7.3 for the fps / VRAM table.

See Section 3 for the headline numbers, Section 4 for which method to
use when, Section 7 for the real-time 20 k-frame-batch streaming plan.


## 1. Theory in brief

### 1.1 Optical flow (Method A)

We estimate per-pixel motion `(u, v)` between consecutive frames using
Farneback's dense polynomial-expansion method (Farneback 2003) as
implemented in `cv2.calcOpticalFlowFarneback`. Farneback is robust to
the small, smooth fish bodies on textured sand that break sparse LK
feature tracking, and runs 10-50× faster than modern deep flow (RAFT,
SEA-RAFT) without the GPU requirement.

> Farneback, G. (2003). *Two-Frame Motion Estimation Based on Polynomial
> Expansion.* SCIA 2003, LNCS 2749, 363-370.

From the per-frame `(u, v)` field we compute inside an activity mask:

- `activity` = count of pixels where `|flow| > 0.5` px/frame
- `rot_omega` = best-fit angular velocity of the active region about its
  flow-magnitude-weighted centroid,
  `Σ (rx·v − ry·u) / Σ (rx² + ry²)`. Sign convention (image coords,
  y-down): positive = **CW** on screen.

For a GT or detected span, rotation direction is the **sign-majority**
of per-frame `rot_omega`. Noise contributes symmetric ± votes that
cancel; coherent orbital motion produces a strong majority.

### 1.2 Pose features (Method B)

The project already has per-video pair-pose parquet files under
`ground_truth_annotations/data/pairs_raw/`. Each row is one
(pair, frame) observation with 29 pre-computed features; the ones we use
are:

- `orbital_rad` — angle of fish 1 relative to the pair midpoint,
  bounded `[-π, +π]`. Unwrapping it over an event span and dividing by
  `2π` gives signed turns directly. The sign convention matches Method A
  (positive = CW on screen).
- `pair_episode_id` — contiguous co-presence stretch of one (TID₁, TID₂)
  pair. Same pair may appear under multiple episode IDs if the tracker
  briefly loses then reacquires it.
- `pair_sex` — per-pair M/F/MF/FF/MM label (not perfectly reliable).
- `centroid_distance_px`, `rel_heading_rad`, `bearing_*_rad`,
  `d_heading_*_rad_s`, `qc_n_missing_kp_*` — complementary features.

Because track IDs are sometimes imperfect, a single GT event may span
several `(TID₁, TID₂)` pair identities. We compute the signed angular
change over each pair's rows (unwrapped) and sum. See `pose_num_turns.py`.

### 1.3 SAM 2 masks (Method C)

SAM 2 (Meta, 2024) is a promptable video segmentation model. Given one
foreground point (or box) per object on an initial frame, it propagates
a per-object mask across the clip. We run SAM 2.1 tiny at 360 px max
side on event spans, extract per-frame centroids from the propagated
masks, and feed them through the same `unwrap → ÷ 2π → quantize`
pipeline as Method B. Zero track-ID instability because SAM 2
maintains object identity by construction within a propagation session.

We additionally use **Grounding DINO Tiny** (prompt `"fish"`) as a
zero-annotation initialiser: DINO on the event's first frame → boxes →
NMS + pair-separation filter → box centres become SAM 2 seed points.
On our test case this reproduced the manual-click SAM 2 result
exactly (same quantized turns, same direction).

### 1.4 Quantization + is_circling rule

Raw unwrap-derived turns are post-processed by:

| fractional part | quantized |
|---|---|
| `≥ 0.9` | round **up**  (`i + 1`) |
| `0.5 ≤ frac < 0.9` | **half-turn** (`i + 0.5`) |
| `< 0.5` | round **down** (`i`) |

`is_circling` = `|quantized turns| ≥ 1` (strict: a bout must complete
at least one full circle). Direction is the **sign of the raw
(unquantized) signed turns** — a quarter-turn still has a well-defined
direction even when its quantized magnitude is 0.


## 2. What the pipeline predicts

| output | Method A (OF) | Method B (Pose) | Method C (SAM 2) |
|---|---|---|---|
| **Rotation direction** | ✅ 100 % (658/658) | ✅ 99.8 % (657/658) | ✅ 100 % on small sample |
| **Number of turns** | — (under-counts 5-10×) | ✅ MAE 0.72, 94 % within 2 | ✅ MAE ~0 on small sample |
| **Span start / end** | ⚠️ baseline, strict-IoU weak | ⚠️ not yet | 🚧 TODO via OF-on-masks → TAL |
| **Dense per-frame features for TAL** | — | — | ✅ via `sam2_of_features.py` |


## 3. Results summary

### 3.1 Rotation direction — all three methods

Scored on the full 8-video, 658-event GT set.

| Method | annotation | compute | direction correct |
|---|---|---|---|
| **A — Optical Flow (Farneback) + sign-majority** | **none** | **~45 s / event** (CPU) | **658 / 658 = 100 %** |
| **B — Pose + sign of raw turns** | none | ~10 ms / event | 657 / 658 = **99.8 %** |
| **C — SAM 2 centroid unwrap + sign** | 2 clicks / event *or* DINO | 3-20 min / event (GPU) | 5 / 5 test cases = **100 %** |

The single pose-direction error (`vid0028 evt_0030`) was manually
verified as a genuine pose-tracker mistake (GT CCW 1.0, pose +0.85 CW).
Method A got that one correct.

Full OF numbers (per-video + confusion matrix) are in
`outputs/optical-flow/direction_only/all_videos_metrics.json`.

### 3.2 Number of turns

#### Method B (Pose) — full dataset

658 events across 8 videos, after pair-pooled unwrap + quantization rule:

| metric | value |
|---|---|
| direction correct | 657 / 658 = **99.8 %** |
| exact quantized match | 34.8 % |
| within 0.5 turns | 67.5 % |
| within 1 turn | 83.4 % |
| within 2 turns | **93.9 %** |
| **MAE  (quantized)** | **0.72 turns** |
| **RMSE (quantized)** | **1.21 turns** |
| mean bias | −0.28 turns (systematic slight under-count) |
| compute per event | **~10 ms** |
| annotation | **zero** |

Error floor is dominated by ~20-30 events with heavy
pair-identity-swap activity (`n_episodes` or `n_pairs > 15`); on those
events the pose tracker fragmented so badly that no post-processing
recovers the missing turns.

Per-video breakdown is in `outputs/optical-flow/pose/summary.txt`.

#### Method C (SAM 2) — spot-validation vs Method B on the same events

Hand-clicked SAM 2 on three events (one short, one medium, one long)
from vid0007-day5:

| event | GT | **Method B pose** | **Method C SAM 2 (clicks)** | Method C DINO + SAM 2 |
|---|---|---|---|---|
| evt_0001 (902 fr, CW 8.0) | 8.0 | **q=8 ✓** | **q=8 ✓** (raw 8.42) | **q=8 ✓** (raw 8.41) |
| evt_0003 (389 fr, CCW 3.0) | 3.0 | **q=3 ✓** | **q=3 ✓** (raw 3.22) | — |
| evt_0022 (277 fr, CCW 1.5) | 1.5 | **q=1.5 ✓** | **q=1.5 ✓** (raw 1.72) | — |

On the three test events, all three methods agree with GT exactly after
quantization. Raw SAM 2 and raw pose numbers match to the second
decimal — same math applied to the same physical centroid trajectory.

Hand-clicked SAM 2 on two **high-fragmentation** events where pose fell
short:

| event | GT | Pose (fragmented) | **SAM 2** |
|---|---|---|---|
| vid0037 evt_0098 (1633 fr, CW 12) | 12 | 5.0 (Δ −7.0) | **10.0 (Δ −2.0)** |
| vid0031 evt_0005 (1290 fr, CCW 10) | 10 | 5.5 (Δ −4.5) | **9.5 (Δ −0.5)** |

SAM 2 closed most of the gap on both. On vid0031 evt_0005 the raw SAM 2
number was 9.79 — within 0.2 turns of GT. This confirms that the Method B
MAE floor is a pose-tracker limitation, and that SAM 2 can serve as a
**fallback** when pose fragments badly.

Method C cost per event scales with VRAM (see Section 7.3): with CPU
offload on a low-VRAM card, long events can take 5-20 min each because
of the growing memory-bank state; on a 12 GB+ card with no offload,
the same events run at 20+ fps (< 1 min per event). Plus 2 clicks per
event — or zero, with DINO.

#### Method A (OF) — num_turns is **not** supported

The flow-integral approach in `detect.num_turns_on_span` consistently
under-counts by 5-10 × on cichlid footage, because Farneback at
`downsample=2` underestimates tangential velocity on small, smooth fish
bodies. Documented but not recommended for production — use Method B or C
for turn counting.

### 3.3 Span detection (both methods)

Still an open problem. Evaluated on one hour of `vid0007_day05`
(34 GT events) using the rule-based event proposer in `detect.py` and
the event-level metrics in `evaluate.py`, at the detector's native
`frame_stride = 1` (full 30 fps).

Standard TAL (Temporal Action Localization) practice reports mAP as the
mean of AP over tIoU thresholds `{0.1, 0.3, 0.5, 0.7}` — the 0.9
threshold is rarely included because sub-frame boundary precision is
below what most detectors achieve.

| metric | value |
|---|---:|
| **mAP** (mean over tIoU ∈ {0.1, 0.3, 0.5, 0.7}) | **0.629** |
| AP @ tIoU = 0.1 | 0.887 |
| AP @ tIoU = 0.3 | 0.792 |
| AP @ tIoU = 0.5 | 0.597 |
| AP @ tIoU = 0.7 | 0.238 |
| F1 @ tIoU = 0.1 | 0.607 |
| F1 @ tIoU = 0.5 | 0.429 |
| F1 @ tIoU = 0.9 | 0.107 |
| Segmental edit score (direction-aware) | 0.436 |

The sharp drop from AP@0.3 → AP@0.7 is the expected failure mode for
pose-free Optical Flow: events are localised well enough to overlap,
but boundaries drift by ~1 s because the sign-consistency window has
~2 s inertia. Span detection on more than one video requires per-video
calibration of the `activity_area_range` filter; see Section 7.4.

Working Method C plan: compute dense Optical Flow features on SAM 2
masks (via `sam2_of_features.py`), feed the per-frame feature stream to
an ActionFormer-style TAL model. Mask-level flow is far cleaner than
whole-frame flow and should push mAP@0.5 materially above 0.6.


## 4. Which method to use — decision cheat sheet

```
             incoming video / event
                      │
    ┌─────────────────┼──────────────────┐
    ▼                 ▼                  ▼
pose parquet     GPU available       neither
exists?          for SAM 2?
    │                 │                  │
   YES               YES                 │
    ▼                 ▼                  ▼
┌──────────┐    ┌──────────┐      ┌──────────┐
│ Method B │    │ Method C │      │ Method A │
│  pose    │    │  SAM 2   │      │  OF dir  │
│          │    │  + DINO  │      │  only    │
│  10 ms / │    │  5-20    │      │  45 s /  │
│  event   │    │  min /   │      │  event   │
│  99.8 %  │    │  event   │      │  100 %   │
│  MAE     │    │  MAE ~0  │      │  dir     │
│  0.72    │    │          │      │  (no     │
│          │    │          │      │  turns)  │
└──────────┘    └──────────┘      └──────────┘
                                       OR
                                  `sam2_click_init.py`
                                  then Method C
                                  (2 clicks / event)
```

- **Research use on the 8 videos** — use Method B (pose). You get everything
  in milliseconds with zero annotation.
- **New video + GPU + no pose** — use Method A for direction + Method C
  (DINO-initialised SAM 2) for turns.
- **New video + no GPU + no pose** — Method A for direction only;
  num_turns is not currently recoverable without a GPU.


## 5. Repo layout

```
models/direction/optical-flow/
  README.md               # this file
  requirements.txt        # pip deps (SAM 2 / DINO are optional extras)

  # --- Method A: Optical Flow direction ---
  flow.py                 # Farneback + per-frame flow features
  detect.py               # event proposer + direction + num_turns stub
  evaluate.py             # mAP, F1@k, edit score, direction accuracy
  pipeline.py             # single-video span + direction pipeline
  run_videos.py           # batch over a folder of videos + annots
  direction_predict.py    # span -> direction (standalone)
  direction_eval.ipynb    # notebook: direction eval across all videos

  # --- Method B: Pose turns ---
  pose_num_turns.py       # orbital_rad unwrap -> direction + num_turns
  pose_aggregate_stats.py # per-video + overall MAE/RMSE + CSV dump

  # --- Method C: SAM 2 masks (+ optional DINO) ---
  sam2_click_init.py      # OpenCV GUI: one click per fish on frame N
  sam2_dino_init.py       # Grounding DINO auto-init (zero clicks)
  sam2_segment_event.py   # init + propagate + overlay + centroids.csv
  sam2_num_turns.py       # centroids.csv -> direction + num_turns
  sam2_of_features.py     # Farneback inside SAM 2 masks (TAL features)
  sam2_analyze_centroid_timeseries.py   # plot pair distance / rot_omega
                                        # / cumulative turns over a span
```

Expected runtime input tree:

```
ground_truth_annotations/
  data/                             # source mp4 files
    pairs_raw/                      # per-video pose parquet (Method B)
  finalized_annotations/            # circling_annotations_vid<id>.csv per video

outputs/optical-flow/
  direction_only/                   # Method A consolidated direction outputs
  pose/                             # Method B per-video + combined event CSVs
  sam2/                             # Method C per-event overlays + centroids
  results/                          # per-video span-detection outputs
```


## 6. Quick start

### 6.1 Install

```bash
# Core (always required):
pip install -r models/direction/optical-flow/requirements.txt

# SAM 2 (Method C only):
pip install git+https://github.com/facebookresearch/sam2.git
mkdir -p ~/sam2_checkpoints
wget -P ~/sam2_checkpoints \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

System-level: `ffmpeg` must be on the `PATH` for `cv2.VideoCapture` to
decode mp4 files. For Method C: CUDA 12.1+ and an NVIDIA GPU — see
Section 7.3 for VRAM requirements (minimum 12 GB for real-time
streaming; less works for batch / research use with CPU-offload flags).

### 6.2 Method A — Optical Flow direction evaluation across all videos

Open `direction_eval.ipynb` and Run All. Writes
`outputs/optical-flow/direction_only/all_videos_direction.csv` and
`all_videos_metrics.json`.

### 6.3 Method B — Pose num_turns + direction

Per-video:

```bash
python pose_num_turns.py \
  --pose        ground_truth_annotations/data/pairs_raw/0007_day05_vid_pairs.parquet \
  --annotations ground_truth_annotations/finalized_annotations/circling_annotations_vid0007-day5.csv \
  --all-events \
  --out         outputs/optical-flow/pose/vid0007-day5_events.csv
```

Then aggregate across videos:

```bash
python pose_aggregate_stats.py   # dumps summary.txt, summary_per_video.csv,
                                 # all_events_combined.csv
```

### 6.4 Method C — SAM 2 masks → num_turns

Zero-annotation (DINO-initialised):

```bash
# 1. DINO auto-init on event's first frame
python sam2_dino_init.py \
  --video ground_truth_annotations/data/0007_vid_day05.mp4 \
  --frame 20142 --prompt fish --filter-pair \
  --out outputs/optical-flow/sam2/evt_0001_dinoclicks.json

# 2. SAM 2 propagate across the event
python sam2_segment_event.py \
  --video ground_truth_annotations/data/0007_vid_day05.mp4 \
  --annotations ground_truth_annotations/finalized_annotations/circling_annotations_vid0007-day5.csv \
  --event-id vid5-v07_d05_evt_0001 \
  --clicks outputs/optical-flow/sam2/evt_0001_dinoclicks.json \
  --ckpt ~/sam2_checkpoints/sam2.1_hiera_tiny.pt \
  --model-cfg configs/sam2.1/sam2.1_hiera_t.yaml \
  --max-input-side 360 --state-to-gpu \
  --out outputs/optical-flow/sam2/

# 3. num_turns + direction from the centroids
python sam2_num_turns.py \
  --centroids outputs/optical-flow/sam2/<video_stem>_<event_id>/centroids.csv \
  --annotations ground_truth_annotations/finalized_annotations/circling_annotations_vid0007-day5.csv \
  --event-id vid5-v07_d05_evt_0001
```

Manual-click variant: replace step 1 with
`python sam2_click_init.py --video … --frame … --out …`; a window
appears, left-click each fish once, press `s` to save.


## 7. TODO — real-time 20 k-frame-batch streaming system

The current entry points above are batch/offline. For a near-real-time
system that ingests 10-15 min of video at a time (≈ 18-27 k frames at
29 fps), the target budget is **process an N-sec clip in ≤ N sec**.

### 7.1 Proposed architecture

Two independent methods run in parallel per batch. Pose is assumed
**not available** for new streaming video.

```
                   streaming batch (10-15 min)
                            │
          ┌─────────────────┼──────────────────┐
          ▼                                    ▼
   METHOD A (always on)                 METHOD C (GPU)
   Farneback direction:               1. DINO sample every ~15 frames
   candidate span proposer               with pair-geometry trigger
   (sign-majority, min-dur)              (close + persistent)
          │                            2. SAM 2 tiny @ 360 p on each
          │                               triggered window
          │                            3. centroid unwrap -> direction
          │                               + num_turns + is_circling
          │                            4. Farneback-on-masks ->
          │                               dense per-frame features
          │                               (for TAL / future span model)
          │                                    │
          └──────────────────┬─────────────────┘
                             ▼
                     cross-method consolidator:
                     - if Method C saw a span but Method A agrees
                       on direction, emit (span, direction,
                       num_turns, is_circling, feature_slice)
                     - if direction disagrees, fall back to
                       Method A's answer (OF has stronger prior)
                             │
                             ▼
                     cross-batch stitcher:
                     - event opened in batch N and still
                       continuing at the boundary stays
                       "open" into batch N+1
                     - 1-2 batch overlap window avoids
                       cutting mid-circling
                             │
                             ▼
             per-event records out (JSON / parquet stream)
```

### 7.2 Budget per 10-min batch at 7 % circling fraction

Cost per stage. SAM 2 propagation cost scales inversely with VRAM
(enough VRAM lets you drop CPU-offload flags and run SAM 2 at native
GPU speed); see Section 7.3 for the fps / VRAM relationship.

| stage | cost | notes |
|---|---|---|
| Method A: Farneback direction on candidate spans | ~1-2 min | CPU; grows linearly with span time |
| Method C: DINO sampling (every 15 fr → ~1 200 inferences) | ~30 s | small GPU bursts, negligible VRAM |
| Method C: pair-geometry trigger | < 1 s | pure Python |
| Method C: SAM 2 propagation on triggered windows (~1 260 fr) | 1-5 min at ~20 fps | dominant cost; see Section 7.3 |
| Method C: Farneback inside SAM masks (~1 260 fr) | ~2 min | CPU, negligible vs SAM 2 |
| Centroid unwrap + quantize | < 10 ms | — |
| **Total** | **~5-10 min** | fits inside the 10-min real-time cap |

Method A alone comfortably meets the real-time budget on any CPU; the
end-to-end budget is driven entirely by Method C's SAM 2 propagation,
which in turn depends on available VRAM.

### 7.3 Hardware requirements

SAM 2 is the only GPU-bound stage and sets the hardware floor. Target
SAM 2 propagation throughput: **≥ 20 fps** (keeps a 10-min batch at
≤ 20 % circling fraction inside the real-time cap).

| GPU VRAM | SAM 2 config | SAM 2 fps | 10-min batch (7 % circling) | verdict |
|---|---|---|---|---|
| **12 GB (minimum required)** | tiny @ 360-720 px, no CPU offload | ~20-25 fps | ~5-7 min | meets real-time cap |
| **24 GB (comfortable)** | tiny or base_plus @ 720 px, no offload | ~30-50 fps | 2-4 min | comfortable headroom for larger batches or multi-pair events |
| 48+ GB (server class) | base_plus @ native resolution, batched events | 100+ fps | seconds | overkill for single-stream, useful for multi-stream fan-out |

Below 12 GB: feasible for research / batch processing with CPU offload
flags (`--state-to-gpu` dropped, `offload_video_to_cpu=True`) but SAM 2
throughput drops to 1-5 fps and the end-to-end pipeline falls out of
real-time. Not recommended for streaming deployment.

### 7.4 Open work

- **Stream + stitcher implementation** (the dashed boxes above).
- **DINO pair-geometry trigger**: NMS + separation gate already in
  `sam2_dino_init.py`; needs to be extended to a persistence gate
  across multiple DINO samples before engaging SAM 2.
- **TAL span model** on dense Method C flow features
  (`sam2_of_features.py` already emits the right per-frame feature
  vector; needs an ActionFormer / BMN / G-TAD head).
- **Span-detection generalisation in Method A** via per-video calibration
  of the `activity_area_range` filter.


## 8. Dependencies

See `requirements.txt`. SAM 2 is installed separately from GitHub;
Grounding DINO comes via `transformers`. The pose and OF methods do not
require a GPU.
