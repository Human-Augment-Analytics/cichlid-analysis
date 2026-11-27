from pathlib import Path

import polars as pl

from models.common.config import (
    FINALIZED_ANNOTATIONS,
    read_annotations,
    video_frames,
)
from models.actionformer.tuning import select_oof_inference_settings

output_dir = Path("outputs/actionformer_pairmotion")
proposal_dir = output_dir / "oof_proposals"

videos = ["0028_vid", "0031_vid"]

oof = {}

for video in videos:
    df = pl.read_parquet(proposal_dir / f"{video}.parquet")

    oof[video] = [
        ((float(start), float(end)), float(confidence))
        for start, end, confidence in df.select(
            "start", "end", "confidence"
        ).iter_rows()
    ]

    print(f"{video}: loaded {len(oof[video]):,} proposals", flush=True)

ground_truth = read_annotations(FINALIZED_ANNOTATIONS, videos=videos)
frame_counts = video_frames(videos)

settings = select_oof_inference_settings(
    oof,
    ground_truth,
    frame_counts,
    output_dir,
    folds=[
        {
            "fold": 1,
            "fit_videos": ["0028_vid"],
            "validation_videos": ["0031_vid"],
        },
        {
            "fold": 2,
            "fit_videos": ["0031_vid"],
            "validation_videos": ["0028_vid"],
        },
    ],
)

print("Recovered OOF settings:", settings)