"""Controlled ActionFormer proximity-hard-negative experiment."""
import importlib
from pathlib import Path

import numpy as np
import polars as pl

from models.common.config import (
    FINALIZED_ANNOTATIONS, HARD_NEGATIVES,
    mstcn_workdir, read_annotations, video_frames,
)
from models.common.segments import (
    load_meta, pair_closeness, rows_to_frames,
)
from models.mstcn_dlc2action.dlc_yolo_converter import (
    dlc2action_bridge as br,
)

train_module = importlib.import_module("models.actionformer.train")

BASELINE = Path(
    "outputs/experiment_backups/baseline_actionformer/sampled_segments.parquet"
)
EXPERIMENT = "actionformer_pairmotion_proximity_exp"
TARGET_FRACTION = 0.5
SEED = 42

if not BASELINE.exists():
    raise SystemExit(f"Missing baseline: {BASELINE}")

if HARD_NEGATIVES != 0:
    raise SystemExit(
        "Keep CIRCLING_HARD_NEGATIVES=0; this script controls sampling."
    )

baseline = pl.read_parquet(BASELINE)
videos = sorted(baseline["video"].unique().to_list())

if videos != ["0028_vid", "0031_vid"]:
    raise SystemExit(f"Unexpected baseline videos: {videos}")

ground_truth = read_annotations(
    FINALIZED_ANNOTATIONS, videos=videos
)
counts = video_frames(videos)
workdir = mstcn_workdir()
meta = load_meta(workdir, videos)
close = pair_closeness(workdir, meta, videos)

rng = np.random.default_rng(SEED)
parts = []

for video in videos:
    original = baseline.filter(pl.col("video") == video)

    test = original.filter(pl.col("split") == "test")
    training = original.filter(pl.col("split") == "train")
    positive = training.filter(pl.col("circling"))
    negative = training.filter(~pl.col("circling"))

    test_intervals = [
        (int(s), int(e))
        for s, e in test.select(
            "start_row", "end_row"
        ).iter_rows()
    ]

    existing_intervals = {
        (int(s), int(e))
        for s, e in original.select(
            "start_row", "end_row"
        ).iter_rows()
    }

    def overlaps_test(s, e):
        return any(s < te and e > ts for ts, te in test_intervals)

    def overlaps_circling(s, e):
        frames = rows_to_frames(
            s, e, meta[video], counts[video]
        )
        if len(frames) == 0:
            return True
        first, last = int(frames[0]), int(frames[-1])
        return any(
            first <= gt_end and last >= gt_start
            for gt_start, gt_end in ground_truth[video]
        )

    spans = br.segment_spans(
        int(meta[video]["n_frames"]), 128, 0
    )

    candidates = []

    for s, e in spans:
        s, e = int(s), int(e)

        if (s, e) in existing_intervals:
            continue
        if overlaps_test(s, e):
            continue
        if overlaps_circling(s, e):
            continue
        if close[video][s:e].mean() < 0.5:
            continue

        candidates.append((s, e))

    target = round(TARGET_FRACTION * negative.height)
    count = min(target, len(candidates))

    chosen_ids = (
        rng.choice(len(candidates), size=count, replace=False)
        if count else []
    )
    chosen = [candidates[int(i)] for i in chosen_ids]

    # Replace an equal number of ordinary training negatives.
    easy = negative.filter(~pl.col("hard_negative"))
    if count > easy.height:
        raise SystemExit("Not enough ordinary negatives to replace")

    drop_ids = (
        set(int(i) for i in rng.choice(
            easy.height, size=count, replace=False
        ))
        if count else set()
    )

    drop_intervals = {
        (int(s), int(e))
        for i, (s, e) in enumerate(
            easy.select("start_row", "end_row").iter_rows()
        )
        if i in drop_ids
    }

    kept = original.filter(
        ~(
            (pl.col("split") == "train")
            & pl.struct(["start_row", "end_row"]).is_in(
                [
                    {"start_row": s, "end_row": e}
                    for s, e in drop_intervals
                ]
            )
        )
    ) if drop_intervals else original

    additions = pl.DataFrame(
        {
            "video": [video] * count,
            "start_row": [s for s, e in chosen],
            "end_row": [e for s, e in chosen],
            "positive_rows": [0] * count,
            "circling": [False] * count,
            "hard_negative": [True] * count,
            "split": ["train"] * count,
        },
        schema=baseline.schema,
    )

    parts.append(pl.concat([kept, additions]))

    print(
        f"{video}: {len(candidates)} verified available; "
        f"replaced {count}/{negative.height} training negatives",
        flush=True,
    )

experiment = pl.concat(parts)

# Verify that the baseline test set is unchanged.
def keys(df):
    return sorted(
        df.select(
            "video", "start_row", "end_row",
            "circling", "split"
        ).iter_rows()
    )

old_test = baseline.filter(pl.col("split") == "test")
new_test = experiment.filter(pl.col("split") == "test")

assert keys(old_test) == keys(new_test)
assert experiment.height == baseline.height

old_positives = baseline.filter(
    (pl.col("split") == "train") & pl.col("circling")
)
new_positives = experiment.filter(
    (pl.col("split") == "train") & pl.col("circling")
)
assert keys(old_positives) == keys(new_positives)

for video in videos:
    before = baseline.filter(
        (pl.col("video") == video)
        & (pl.col("split") == "train")
        & (~pl.col("circling"))
    ).height
    after = experiment.filter(
        (pl.col("video") == video)
        & (pl.col("split") == "train")
        & (~pl.col("circling"))
    ).height
    assert before == after

print("\nVerified: original test segments and positive "
      "training segments are unchanged.")
print(
    experiment.group_by(
        "video", "split", "circling", "hard_negative"
    ).len().sort(
        "video", "split", "circling", "hard_negative"
    )
)

# The existing trainer normally generates a new split.
# Replace that operation with our frozen experimental split.
def frozen_segments(workdir_arg, spans_arg, output_dir=None):
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        experiment.write_parquet(
            destination / "sampled_segments.parquet"
        )
    return experiment, meta

train_module.shared_segments = frozen_segments

# Change the model's output name without touching baseline files.
train_module.NAMES["pairmotion"] = EXPERIMENT

print("\nStarting experimental ActionFormer training...", flush=True)

train_module.train(
    features="pairmotion",
    epochs=60,
    batch_size=4,
    learning_rate=1e-4,
    validation_fraction=0.2,
    patience=6,
    tune_postprocess=False,
)
