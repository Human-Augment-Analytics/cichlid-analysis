"""Score videos with the saved centered-window XGBoost model and export its voted spans.

`models.xgboost.train` scores only its training videos. This scores any video, e.g. the unseen
test video of a generalizability fold (CIRCLING_VIDEOS), with the same 0.5 threshold and
30-frame majority vote:

    uv run python -m models.xgboost.predict --videos 0028_vid
"""

import argparse
import os
from pathlib import Path

from xgboost import XGBClassifier

from models.rule_based.circling_cleaning import default_data_dir
from .features import ROOT, frame_counts, load_series, selected_videos
from .train import OUTPUT_DIR, PREDICTION_DIR, export_video, score_series


def predict(videos, data_dir, model_dir=OUTPUT_DIR, prediction_dir=PREDICTION_DIR):
    model = XGBClassifier()
    model.load_model(Path(model_dir) / "model.ubj")
    for video, count in frame_counts(videos, data_dir).items():
        print(f"Scoring all {count:,} frames: {video}", flush=True)
        _, spans = export_video(video, score_series(model, load_series(data_dir, video, count)),
                                model_dir, prediction_dir)
        print(f"{video}: {len(spans)} spans -> {Path(prediction_dir) / f'{video}.csv'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", default=selected_videos(),
                        help="videos to score (default: CIRCLING_VIDEOS, else 0028_vid and 0031_vid)")
    parser.add_argument("--data-dir", type=Path,
                        default=os.environ.get("CIRCLING_DATA_DIR") or default_data_dir(ROOT))
    parser.add_argument("--model-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=PREDICTION_DIR)
    args = parser.parse_args()
    predict(args.videos, args.data_dir, args.model_dir, args.prediction_dir)
