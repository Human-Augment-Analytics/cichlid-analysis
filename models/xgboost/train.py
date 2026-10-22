"""Reusable training/evaluation steps for the marimo notebook.

Headless equivalent: uv run python -m models.xgboost.train
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from evaluation.metrics import evaluate_video, summarize
from models.rule_based.circling_cleaning import default_data_dir
from .features import (FEATURE_DESCRIPTION, FEATURES, GROUND_TRUTH, ROOT, VIDEO_FRAMES, WINDOW, frame_labels,
                       load_series, read_spans, sample_frames, window_features)
from .postprocess import VOTE_FRACTION, VOTE_WINDOW, majority_vote_fraction

SEED = 42
THRESHOLD = 0.5
OUTPUT_DIR = ROOT / "outputs/xgboost"
PREDICTION_DIR = ROOT / "outputs/predictions/xgboost"


def prepare_dataset(data_dir, spans, video_frames=VIDEO_FRAMES):
    series, tables, matrices = {}, [], []
    for video, count in video_frames.items():
        print(f"Loading cleaned signals: {video}", flush=True)
        if video not in spans:
            raise ValueError(f"No positive annotations for {video}")
        series[video] = load_series(data_dir, video, count)
        frames, labels = sample_frames(count, spans[video])
        tables.append(pd.DataFrame({"VideoID": video, "FrameNum": frames, "circling": labels}))
        matrices.append(window_features(series[video], frames))
    return series, pd.concat(tables, ignore_index=True), np.concatenate(matrices)


def split_dataset(table):
    train_ids, test_ids = train_test_split(np.arange(len(table)), test_size=0.3,
                                          random_state=SEED, stratify=table.circling)
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


def evaluate_full_videos(model, series, spans, output_dir=OUTPUT_DIR, prediction_dir=PREDICTION_DIR):
    output_dir, prediction_dir = Path(output_dir), Path(prediction_dir)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    frames, segments, processed_frames, processed_segments = {}, {}, {}, {}
    for video, signals in series.items():
        print(f"Scoring all {len(signals):,} frames: {video}", flush=True)
        scores = score_series(model, signals)
        labels = frame_labels(len(signals), spans[video])
        votes = majority_vote_fraction(scores, threshold=THRESHOLD)
        pd.DataFrame({"FrameNum": np.arange(len(scores)), "circling": labels,
                      "circling_score": scores, "predicted_circling": scores >= THRESHOLD,
                      "circling_vote_fraction": votes,
                      "postprocessed_circling": votes >= VOTE_FRACTION}
                     ).to_parquet(scores_dir / f"{video}_scores.parquet", index=False)
        predicted = positive_spans(scores)
        processed = positive_spans(votes, threshold=VOTE_FRACTION)
        pd.DataFrame([(video, start, end) for start, end in processed],
                     columns=["VideoID", "StartFrame", "EndFrame"]
                     ).to_csv(prediction_dir / f"{video}.csv", index=False)
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
                   output_dir=OUTPUT_DIR, data_dir=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "model.ubj")
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
        "ground_truth": [str(path.relative_to(ROOT)) for path in GROUND_TRUTH], "spans": spans,
        "data_dir": str(Path(data_dir).resolve()) if data_dir is not None else None,
        "sampling": "All positive centers; ceil(inclusive segment length / 2) negative context frames on each side. Union overlapping buffers; clip to video; positive labels take precedence. One row per (video, center frame).",
        "split": "Stratified random 70/30 frame split, seed 42; fitted on training rows only.",
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


def train(data_dir, output_dir=OUTPUT_DIR, prediction_dir=PREDICTION_DIR):
    spans = read_spans()
    series, table, x = prepare_dataset(data_dir, spans)
    table, train_ids, test_ids = split_dataset(table)
    print(f"Training: {len(train_ids):,}; holdout: {len(test_ids):,}", flush=True)
    model = fit_model(x[train_ids], table.circling.to_numpy()[train_ids])
    test_scores = model.predict_proba(x[test_ids])[:, 1]
    holdout = frame_metrics(table.circling.to_numpy()[test_ids], test_scores)
    print(json.dumps(holdout, indent=2), flush=True)
    full = evaluate_full_videos(model, series, spans, output_dir, prediction_dir)
    return save_artifacts(model, table, x, test_ids, test_scores, holdout, full, spans, output_dir, data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(ROOT))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=PREDICTION_DIR)
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.prediction_dir)
