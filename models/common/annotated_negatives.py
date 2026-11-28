"""Read reviewer-selected negative intervals without changing positive ground truth."""

import csv
from pathlib import Path

import numpy as np

DEFAULT_ANNOTATED_NEGATIVES = (
    Path(__file__).resolve().parents[2] / "ground_truth_annotations"
    / "hard_negatives" / "hard_negatives_annotated.csv"
)
VIDEO_ALIASES = {"0002_vid": "vid7-v02_d00", "0013_vid": "vid3-v13_d13"}


def read_annotated_negatives(path, videos):
    spans = {video: [] for video in videos}
    if path is None:
        return spans
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not {"VideoID", "StartFrame", "EndFrame"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected VideoID, StartFrame and EndFrame")
        for row in reader:
            video = row["VideoID"].strip()
            if video not in spans:
                video = VIDEO_ALIASES.get(video, video)
            if video in spans:
                start, end = int(row["StartFrame"]), int(row["EndFrame"])
                if start < 0 or end < start:
                    raise ValueError(f"{path}: invalid interval {video} {start}-{end}")
                spans[video].append((start, end))
    return spans


def annotated_negative_mask(count, intervals, positives):
    """Take the union of inclusive negative intervals, preserving finalized positives."""
    mask = np.zeros(count, dtype=bool)
    for start, end in intervals:
        if end >= count:
            raise ValueError(f"Annotated negative interval {start}-{end} exceeds {count} frames")
        mask[start:end + 1] = True
    for start, end in positives:
        mask[start:end + 1] = False
    return mask
