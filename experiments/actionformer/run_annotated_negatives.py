"""Experiment C: ActionFormer with Aaron's reviewed behavioral negatives."""
from collections import Counter
from pathlib import Path

import pandas as pd
import polars as pl

import models.actionformer.train as trainer
from models.common.config import (
    FINALIZED_ANNOTATIONS, mstcn_workdir,
    read_annotations, video_frames,
)
from models.common.segments import load_meta, rows_to_frames
from models.mstcn_dlc2action.dlc_yolo_converter import (
    dlc2action_bridge as br,
)

SOURCE = Path("experiments/actionformer/data/reviewed_hard_negatives.xlsx")
BASELINE = Path(
    "outputs/experiment_backups/baseline_actionformer/sampled_segments.parquet"
)
NAME = "actionformer_pairmotion_annotated_negatives"
VIDEOS = ["0028_vid", "0031_vid"]

annotations = pd.read_excel(SOURCE, sheet_name="Hard negatives")
annotations = annotations[
    annotations["VideoID"].isin(VIDEOS)
    & annotations["Primary label/behavior"].isin(
        ["no-circling", "partial circling"]
    )
].copy()

baseline = pl.read_parquet(BASELINE)
workdir = mstcn_workdir()
meta = load_meta(workdir, VIDEOS)
counts = video_frames(VIDEOS)
truth = read_annotations(FINALIZED_ANNOTATIONS, videos=VIDEOS)

def overlaps(a, b, c, d):
    return a <= d and c <= b

def frame_bounds(video, start, end):
    frames = rows_to_frames(
        int(start), int(end), meta[video], counts[video]
    )
    if not len(frames):
        return None
    return int(frames[0]), int(frames[-1])

replacement_rows = []
drop_keys = set()
stats = Counter()

for video in VIDEOS:
    events = annotations[annotations["VideoID"] == video]
    existing = baseline.filter(pl.col("video") == video)

    test_bounds = [
        frame_bounds(video, s, e)
        for s, e in existing.filter(
            pl.col("split") == "test"
        ).select("start_row", "end_row").iter_rows()
    ]

    existing_keys = {
        (int(s), int(e))
        for s, e in existing.select(
            "start_row", "end_row"
        ).iter_rows()
    }

    eligible = []

    for start, end in br.segment_spans(
        int(meta[video]["n_frames"]), 128, 0
    ):
        start, end = int(start), int(end)
        if (start, end) in existing_keys:
            continue

        bounds = frame_bounds(video, start, end)
        if bounds is None:
            continue

        first, last = bounds

        if any(
            overlaps(first, last, a, b)
            for a, b in test_bounds if a is not None
        ):
            stats[(video, "test_excluded")] += 1
            continue

        if any(
            overlaps(first, last, int(a), int(b))
            for a, b in truth[video]
        ):
            stats[(video, "circling_excluded")] += 1
            continue

        best_overlap = 0
        best_label = None

        for _, event in events.iterrows():
            a = int(event["Start Frame"])
            b = int(event["End Frame"])
            intersection = max(
                0, min(last, b) - max(first, a) + 1
            )
            event_length = b - a + 1

            if (
                intersection >= min(16, event_length)
                and intersection / event_length >= 0.5
                and intersection > best_overlap
            ):
                best_overlap = intersection
                best_label = event["Primary label/behavior"]

        if best_label is None:
            continue

        eligible.append((start, end, best_label, best_overlap))

    negatives = existing.filter(
        (pl.col("split") == "train")
        & (~pl.col("circling"))
    )

    # Keep training size and video balance unchanged.
    quota = min(len(eligible), negatives.height)
    selected = sorted(
        eligible, key=lambda item: (-item[3], item[0])
    )[:quota]

    # Replace the first ordinary negatives deterministically.
    dropped = list(
        negatives.sort("start_row")
        .select("start_row", "end_row")
        .iter_rows()
    )[:quota]

    drop_keys.update(
        (video, int(s), int(e)) for s, e in dropped
    )

    for start, end, label, overlap in selected:
        replacement_rows.append({
            "video": video,
            "start_row": start,
            "end_row": end,
            "positive_rows": 0,
            "circling": False,
            "hard_negative": True,
            "split": "train",
        })
        stats[(video, label)] += 1

    print(
        f"{video}: {len(eligible)} eligible new segments, "
        f"{quota} replacements",
        flush=True,
    )

kept = baseline.filter(
    ~pl.struct(["video", "start_row", "end_row"]).is_in(
        [
            {"video": v, "start_row": s, "end_row": e}
            for v, s, e in sorted(drop_keys)
        ]
    )
)

if not replacement_rows:
    raise SystemExit("No eligible new hard-negative segments found.")

added = pl.DataFrame(
    replacement_rows,
    schema=baseline.schema,
)

segments = pl.concat([kept, added]).sort(
    ["video", "start_row", "end_row"]
)

assert segments.height == baseline.height
assert segments.filter(pl.col("split") == "test").sort(
    ["video", "start_row", "end_row"]
).equals(
    baseline.filter(pl.col("split") == "test").sort(
        ["video", "start_row", "end_row"]
    )
)
assert segments.filter(pl.col("circling")).height == (
    baseline.filter(pl.col("circling")).height
)
assert segments.select(
    ["video", "start_row", "end_row"]
).unique().height == segments.height

print("\nSelected annotations:", dict(stats), flush=True)
print(
    "\nFinal segment distribution:\n",
    segments.group_by(
        ["video", "split", "circling", "hard_negative"]
    ).len().sort(["video", "split", "circling"]),
    flush=True,
)

output = Path("outputs") / NAME
output.mkdir(parents=True, exist_ok=True)
segments.write_parquet(output / "sampled_segments.parquet")

# Reuse the original trainer without regenerating the shared split.
def experiment_segments(workdir, spans, output_dir=None):
    return segments, meta

trainer.shared_segments = experiment_segments
trainer.NAMES["pairmotion"] = NAME

print("\nStarting Aaron hard-negative experiment...", flush=True)

trainer.train(
    features="pairmotion",
    epochs=60,
    batch_size=4,
    learning_rate=1e-4,
    validation_fraction=0.2,
    patience=6,
    tune_postprocess=False,
)
