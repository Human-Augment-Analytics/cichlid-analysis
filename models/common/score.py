"""Score the selected videos (CIRCLING_VIDEOS) with an already trained per-frame model.

Training scores only its own videos; this scores videos the model never saw, e.g. the
test video of a generalizability fold. Scores go to outputs/<name>/scores, where
`models.common.predict` turns them into spans with the model's saved postprocessing.
"""
import argparse

import joblib

from models.common.config import OUTPUTS, model_name, mstcn_workdir, video_frames
from models.common.frame_model import score_videos
from models.common.segments import load_meta


def score(name, feature=None):
    name = model_name(name)
    saved = joblib.load(OUTPUTS / name / 'model.joblib')
    feature = feature or saved['feature_set']
    counts = video_frames()
    workdir = mstcn_workdir()
    meta = load_meta(workdir, sorted(counts))
    return score_videos(saved['model'], saved['features'], workdir, feature, meta, counts,
                        OUTPUTS / name / 'scores')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', help='model folder in outputs/, e.g. xgboost_turns')
    args = parser.parse_args()
    score(args.name)
