"""Reusable training/evaluation steps for the marimo notebook.

Headless equivalent: uv run python -m models.xgboost.train

The CLI defaults follow the shared CIRCLING_* variables, so a generalizability fold can train it
on its training videos (CIRCLING_VIDEOS) under its own output root (CIRCLING_OUTPUTS);
`models.xgboost.predict` then scores the fold's unseen test video.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from models.common.annotated_negatives import (DEFAULT_ANNOTATED_NEGATIVES, annotated_negative_mask,
                                              read_annotated_negatives)

from evaluation.metrics import evaluate_video, summarize
from models.rule_based.circling_cleaning import default_data_dir
from .features import (FEATURE_DESCRIPTION, FEATURES, GROUND_TRUTH, ROOT, VIDEO_FRAMES, WINDOW, frame_counts,
                       frame_labels, load_series, read_spans, sample_frames, selected_videos, window_features)
from .postprocess import VOTE_FRACTION, VOTE_WINDOW, majority_vote_fraction

SEED = 42
THRESHOLD = 0.5
OUTPUTS = Path(os.environ.get("CIRCLING_OUTPUTS") or ROOT / "outputs").resolve()
OUTPUT_DIR = OUTPUTS / "xgboost"
PREDICTION_DIR = OUTPUTS / "predictions/xgboost"


def prepare_dataset(data_dir, spans, video_frames=VIDEO_FRAMES,
                    annotated_negatives=DEFAULT_ANNOTATED_NEGATIVES):
    series, tables, matrices = {}, [], []
    negatives = read_annotated_negatives(annotated_negatives, video_frames)
    for video, count in video_frames.items():
        print(f"Loading cleaned signals: {video}", flush=True)
        if video not in spans:
            raise ValueError(f"No positive annotations for {video}")
        series[video] = load_series(data_dir, video, count)
        frames, labels = sample_frames(count, spans[video])
        extras = annotated_negative_mask(count, negatives[video], spans[video])
        extras[frames] = False
        added = np.flatnonzero(extras)
        sampled_frames = np.r_[frames, added]
        tables.append(pd.DataFrame({
            "VideoID": video, "FrameNum": sampled_frames,
            "circling": np.r_[labels, np.zeros(len(added), dtype=labels.dtype)],
            "added_annotated_negative": extras[sampled_frames],
        }))
        matrices.append(window_features(series[video], sampled_frames))
    return series, pd.concat(tables, ignore_index=True), np.concatenate(matrices)


def split_dataset(table):
    extra = (table["added_annotated_negative"].to_numpy()
             if "added_annotated_negative" in table else np.zeros(len(table), dtype=bool))
    original = np.flatnonzero(~extra)
    train_ids, test_ids = train_test_split(original, test_size=0.3, random_state=SEED,
                                          stratify=table.circling.to_numpy()[original])
    train_ids = np.r_[train_ids, np.flatnonzero(extra)]
    table = table.copy()
    table["split"] = "train"
    table.loc[test_ids, "split"] = "test"
    return table, train_ids, test_ids


def fit_model(x, y):
    if len(np.unique(y)) != 2:
        raise ValueError("Training requires both positive and negative frames")
    model = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                          objective="binary:logistic", eval_metric="logloss",
                          tree_method="hist", random_state=SEED, n_jobs=4)
    model.fit(x, y)
    return model


def frame_metrics(y, scores, threshold=THRESHOLD):
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {"frames": len(y), "positive_frames": int(np.sum(y)),
            "prevalence": float(np.mean(y)), "accuracy": float(accuracy_score(y, predicted)),
            "average_precision": float(average_precision_score(y, scores)) if np.any(y) else None,
            "roc_auc": float(roc_auc_score(y, scores)) if both else None,
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def score_series(model, series, batch_size=20_000):
    """Build windows in bounded batches while retaining context across batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    scores = np.empty(len(series), dtype=np.float32)
    for start in range(0, len(series), batch_size):
        end = min(len(series), start + batch_size)
        scores[start:end] = model.predict_proba(window_features(series, np.arange(start, end)))[:, 1]
    return scores


def positive_spans(scores, threshold=THRESHOLD):
    """Contiguous threshold-positive frames; no duration filtering or gap merging."""
    padded = np.r_[False, scores >= threshold, False].astype(np.int8)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def export_video(video, scores, output_dir=OUTPUT_DIR, prediction_dir=PREDICTION_DIR, labels=None):
    """Write per-frame scores and the majority-voted spans of one full video; return the votes and spans."""
    scores_dir, prediction_dir = Path(output_dir) / "scores", Path(prediction_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    votes = majority_vote_fraction(scores, threshold=THRESHOLD)
    columns = {"FrameNum": np.arange(len(scores))}
    if labels is not None:
        columns["circling"] = labels
    pd.DataFrame({**columns, "circling_score": scores, "predicted_circling": scores >= THRESHOLD,
                  "circling_vote_fraction": votes, "postprocessed_circling": votes >= VOTE_FRACTION}
                 ).to_parquet(scores_dir / f"{video}_scores.parquet", index=False)
    processed = positive_spans(votes, threshold=VOTE_FRACTION)
    pd.DataFrame([(video, start, end) for start, end in processed],
                 columns=["VideoID", "StartFrame", "EndFrame"]
                 ).to_csv(prediction_dir / f"{video}.csv", index=False)
    return votes, processed


def evaluate_full_videos(model, series, spans, output_dir=OUTPUT_DIR, prediction_dir=PREDICTION_DIR):
    frames, segments, processed_frames, processed_segments = {}, {}, {}, {}
    for video, signals in series.items():
        print(f"Scoring all {len(signals):,} frames: {video}", flush=True)
        scores = score_series(model, signals)
        labels = frame_labels(len(signals), spans[video])
        votes, processed = export_video(video, scores, output_dir, prediction_dir, labels)
        predicted = positive_spans(scores)
        frames[video] = frame_metrics(labels, scores)
        segments[video] = evaluate_video(spans[video], predicted, len(signals))
        processed_frames[video] = frame_metrics(labels, votes, threshold=VOTE_FRACTION)
        processed_segments[video] = evaluate_video(spans[video], processed, len(signals))
    return {"frame_metrics": frames, "segment_metrics": segments,
            "overall_segments": summarize(list(segments.values())),
            "postprocessed": {"frame_metrics": processed_frames,
                              "segment_metrics": processed_segments,
                              "overall_segments": summarize(list(processed_segments.values()))}}


def save_artifacts(model, table, x, test_ids, test_scores, holdout, full, spans,
                   output_dir=OUTPUT_DIR, data_dir=None, annotated_negatives=DEFAULT_ANNOTATED_NEGATIVES,
                   ground_truth=GROUND_TRUTH, save_features=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "model.ubj")
    if save_features:  # ~1.5 GB for six videos, so the folds keep only the rows below
        sampled = pd.concat([table, pd.DataFrame(x, columns=FEATURES)], axis=1)
        sampled.to_parquet(output_dir / "sampled_frames.parquet", index=False)
    test = table.iloc[test_ids].copy()
    test["circling_score"], test["predicted_circling"] = test_scores, test_scores >= THRESHOLD
    test.to_parquet(output_dir / "test_predictions.parquet", index=False)
    pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}
                 ).sort_values("importance", ascending=False).to_csv(output_dir / "feature_importance.csv", index=False)
    report = {
        "window_frames": WINDOW, "features": FEATURES, "seed": SEED, "threshold": THRESHOLD,
        "feature_source": FEATURE_DESCRIPTION,
        "ground_truth": str(Path(ground_truth).resolve().relative_to(ROOT)
                            if Path(ground_truth).resolve().is_relative_to(ROOT) else ground_truth),
        "spans": spans,
        "data_dir": str(Path(data_dir).resolve()) if data_dir is not None else None,
        "sampling": "All positive centers; ceil(inclusive segment length / 2) negative context frames on each side. Union overlapping buffers; clip to video; positive labels take precedence. One row per (video, center frame).",
        "split": "Stratified random 70/30 split of the original sampled frames, seed 42. "
                 "Extra annotated negatives are assigned to training.",
        "annotated_negatives": {
            "source": str(annotated_negatives) if annotated_negatives is not None else None,
            "extra_training_frames": table.groupby("VideoID")["added_annotated_negative"]
                .sum().astype(int).to_dict() if "added_annotated_negative" in table else {},
        },
        "postprocessing": {"window_frames": VOTE_WINDOW, "positive_fraction": VOTE_FRACTION,
                           "description": "Centered rolling vote over thresholded predictions, offsets -15..+14; ties positive. Video boundaries use only available frames. Apply after concatenating full-video scores; export processed spans.",
                           "processed_ranking_metrics": "AP and ROC AUC use the rolling positive-vote fraction, not model probabilities.",
                           "holdout": "Raw classifier predictions; sampled holdout rows are not contiguous."},
        "limitations": ["Adjacent windows overlap across train/test; holdout is not independent-event or new-video evaluation.",
                        "Full-video evaluation includes training frames and assumes all unannotated frames are negative; annotations may be incomplete.",
                        "Centered windows use future frames. Scores trained on near-balanced data are not calibrated at full-video prevalence."],
        "model_parameters": {key: "NaN" if isinstance(value, float) and np.isnan(value) else value
                             for key, value in model.get_params().items()},
        "sample_counts": table.groupby(["VideoID", "split", "circling"]).size().rename("frames").reset_index().to_dict("records"),
        "holdout": holdout, "full_video": full,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def train(data_dir, output_dir=OUTPUT_DIR, prediction_dir=PREDICTION_DIR,
          annotated_negatives=DEFAULT_ANNOTATED_NEGATIVES, videos=None, ground_truth=GROUND_TRUTH,
          full_video=True):
    """Fit on the sampled frames of `videos`. full_video=False skips scoring the training videos
    end to end (the generalizability folds score only their unseen test video)."""
    counts = frame_counts(videos or sorted(VIDEO_FRAMES), data_dir)
    spans = read_spans(ground_truth)
    series, table, x = prepare_dataset(data_dir, spans, counts, annotated_negatives=annotated_negatives)
    table, train_ids, test_ids = split_dataset(table)
    print(f"Training: {len(train_ids):,}; holdout: {len(test_ids):,}", flush=True)
    model = fit_model(x[train_ids], table.circling.to_numpy()[train_ids])
    test_scores = model.predict_proba(x[test_ids])[:, 1]
    holdout = frame_metrics(table.circling.to_numpy()[test_ids], test_scores)
    print(json.dumps(holdout, indent=2), flush=True)
    full = evaluate_full_videos(model, series, spans, output_dir, prediction_dir) if full_video else None
    return save_artifacts(model, table, x, test_ids, test_scores, holdout, full,
                          {video: spans[video] for video in counts}, output_dir, data_dir,
                          annotated_negatives, ground_truth, save_features=full_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=os.environ.get("CIRCLING_DATA_DIR") or default_data_dir(ROOT))
    parser.add_argument("--videos", nargs="+", default=selected_videos(),
                        help="training videos (default: CIRCLING_VIDEOS, else 0028_vid and 0031_vid)")
    parser.add_argument("--ground-truth", type=Path, default=os.environ.get("CIRCLING_GROUND_TRUTH") or GROUND_TRUTH,
                        help="annotation CSV or folder of circling_annotations_vid*.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=PREDICTION_DIR)
    parser.add_argument("--skip-full-video", action="store_true",
                        help="do not score the training videos end to end or save sampled_frames.parquet")
    negatives = parser.add_mutually_exclusive_group()
    negatives.add_argument("--annotated-hard-negatives", type=Path, default=DEFAULT_ANNOTATED_NEGATIVES)
    negatives.add_argument("--no-annotated-hard-negatives", action="store_const", const=None,
                           dest="annotated_hard_negatives")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.prediction_dir, args.annotated_hard_negatives, args.videos,
          args.ground_truth, not args.skip_full_video)
