"""Held-out-video ActionFormer study for comparison with team's LOVO evaluation.

Run from repo root:
    uv run python -m experiments.actionformer.run_actionformer_lovo

Fits only on 0028 and predicts the *complete* 0031 video, then reverses.
Repeats for model seeds 42, 123, 456. Preprocessing and postprocessing
are fitted/tuned using training-video segments only.

WARNING: Full-video decoding on CPU may be slow. Keep the laptop awake.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import random
from pathlib import Path

import numpy as np
import polars as pl
import torch

from evaluation.metrics import evaluate_video, summarize
from models.actionformer.model import build_model, suppress_cross_window_duplicates
from models.actionformer.postprocess import prediction_table
from models.actionformer.predict import decode_video
from models.actionformer.tuning import select_inference_settings
from models.common.config import FINALIZED_ANNOTATIONS, mstcn_workdir, read_annotations, video_frames
from models.common.segments import load_meta

trainer = importlib.import_module("models.actionformer.train")

BASELINE = Path("outputs/experiment_backups/baseline_actionformer/sampled_segments.parquet")
OUTPUT = Path("outputs/actionformer_lovo_controlled")
PREDICTION_ROOT = Path("outputs/predictions")
VIDEOS = ("0028_vid", "0031_vid")
SEEDS = (42, 123, 456)
MAX_EPOCHS = 60


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_done(seed, test_video, study_id):
    folder = OUTPUT / f"seed_{seed}" / f"test_{test_video}"
    marker = folder / "DONE.json"
    if not marker.exists():
        return None
    row = json.loads(marker.read_text())
    if row.get("study_id") != study_id:
        raise RuntimeError(f"Incompatible previous results in {folder}")
    if not (folder / "model.pt").exists() or not (folder / "full_video_predictions.csv").exists():
        return None
    return row


def update_results(study_id):
    rows = [record for seed in SEEDS for video in VIDEOS
            if (record := read_done(seed, video, study_id)) is not None]
    fields = ("seed", "train_video", "test_video", "epochs_trained", "best_epoch",
              "frame_precision", "frame_recall", "frame_f1", "f1@10", "f1@25", "f1@50",
              "edit_score", "pred_segments", "gt_segments", "inference_min_score",
              "inference_min_seconds", "inference_merge_seconds", "run_directory")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "per_fold_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    pooled = []
    for seed in SEEDS:
        folds = [r for r in rows if r["seed"] == seed]
        if len(folds) != len(VIDEOS):
            continue
        metrics = summarize([{key: row[key] for key in (
            "num_frames", "mof", "edit_score", "frame_tp", "frame_fp", "frame_fn",
            "frame_iou", "gt_segments", "pred_segments", "tp@10", "tp@25", "tp@50")}
            for row in folds])
        pooled.append({"seed": seed, **metrics})
    if pooled:
        with (OUTPUT / "pooled_lovo_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["seed", *list(pooled[0].keys())[1:]], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(pooled)
    lines = ["# ActionFormer leave-one-video-out results", "",
             "Both videos are evaluated in full; each fold trains on the *other* video.",
             "Postprocessing is selected only from training-video validation segments.",
             "These are separate from the 63-run sampled-segment study.", "",
             "| Seed | Frame F1 | F1@10 | F1@25 | F1@50 | Edit score |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in pooled:
        lines.append(f"| {row['seed']} | {row['frame_f1']:.4f} | "
                     f"{row['f1@10']:.4f} | {row['f1@25']:.4f} | "
                     f"{row['f1@50']:.4f} | {row['edit_score']:.2f} |")
    lines.append("")
    lines.append("Predictions saved as per-video interval CSVs with Confidence scores for later mAP evaluation.")
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="CPU (default on Mac) or CUDA")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Stop after N new test-video folds; later runs resume")
    args = parser.parse_args()
    if args.max_folds is not None and args.max_folds < 1:
        parser.error("--max-folds must be positive")
    if not BASELINE.exists():
        parser.error(f"Missing frozen sampled data: {BASELINE}")
    data = pl.read_parquet(BASELINE)
    if tuple(sorted(data["video"].unique().to_list())) != VIDEOS:
        parser.error("Expected the 0028/0031 baseline sample file")
    videos = list(VIDEOS)
    counts = video_frames(videos)
    spans = read_annotations(FINALIZED_ANNOTATIONS, videos=videos)
    workdir = mstcn_workdir()
    meta = load_meta(workdir, videos)
    src = BASELINE.read_bytes()
    # Include the code itself and annotation files in study identity to ensure safe resume.
    gt_path = Path(FINALIZED_ANNOTATIONS)
    annotations = sorted(gt_path.glob("circling_annotations_vid*.csv")) if gt_path.is_dir() else [gt_path]
    h = hashlib.sha256(src + Path(__file__).read_bytes())
    for p in annotations:
        h.update(p.name.encode() + p.read_bytes())
    study_id = h.hexdigest()[:20]
    if args.dry_run:
        print("LOVO plan: 6 full-video evaluations (3 seeds x 2 held-out videos)")
        for seed in SEEDS:
            for test in VIDEOS:
                print(f"seed={seed}: train={next(v for v in VIDEOS if v != test)}, test={test}")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("study_id") != study_id:
        parser.error(f"Output directory belongs to different study; rename {OUTPUT}")
    if not manifest.exists():
        manifest.write_text(json.dumps({"study_id": study_id, "seeds": SEEDS,
                                        "videos": VIDEOS, "epochs": MAX_EPOCHS,
                                        "training": "only the other video, sampled 128-row segments",
                                        "validation": "20% stratified on training video; fixed seed 42",
                                        "evaluation": "complete held-out video, native proposals",
                                        "limitations": "two videos; no external video diversity; native ActionFormer postprocessing differs from other detectors' policies"}, indent=2) + "\n")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    done = update_results(study_id)
    print(f"LOVO progress: {done}/{len(SEEDS) * len(VIDEOS)} completed", flush=True)
    newly_completed = 0
    for seed in SEEDS:
        for test_video in VIDEOS:
            if read_done(seed, test_video, study_id) is not None:
                print(f"SKIP seed={seed}, test={test_video}", flush=True)
                continue
            train_video = next(v for v in VIDEOS if v != test_video)
            fold_dir = OUTPUT / f"seed_{seed}" / f"test_{test_video}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            # Use all baseline-sampled examples in the training video;
            # previously designated sampled 'test' rows are allowed here because
            # the LOVO test video is completely disjoint.
            train_rows = data.filter(pl.col("video") == train_video).with_columns(
                pl.lit("train").alias("split"))
            train_counts = {train_video: counts[train_video]}
            source_meta = {train_video: meta[train_video]}
            base_samples, _, _, _ = trainer.prepare_samples(
                train_rows, source_meta, spans, train_counts, workdir, "pairmotion")
            fitting, validating = trainer.training_validation_split(base_samples, 0.2, seed=42)
            validation_ids = {s["sample_id"] for s in validating}
            train_rows = train_rows.with_columns(pl.Series("split", [
                "validation" if i in validation_ids else "train"
                for i in range(train_rows.height)]))
            del base_samples, fitting, validating
            samples, columns, mean, std = trainer.prepare_samples(
                train_rows, source_meta, spans, train_counts, workdir, "pairmotion")
            fitting = [s for s in samples if s["split"] == "train"]
            validating = [s for s in samples if s["split"] == "validation"]
            if len(columns) != 132:
                raise RuntimeError(f"Feature count changed: {len(columns)}")
            seed_all(seed)
            model, config = build_model(len(columns))
            print(f"START LOVO seed={seed}, train={train_video} ({len(fitting)} fit, "
                  f"{len(validating)} validation), full-video test={test_video}", flush=True)
            history = trainer.train_model(model, fitting, device, epochs=MAX_EPOCHS,
                                          batch_size=4, learning_rate=1e-4,
                                          validation_samples=validating, patience=6,
                                          seed=seed)
            pl.DataFrame(history).write_csv(fold_dir / "training_history.csv")
            best = next((x for x in reversed(history) if x.get("is_best")), None)
            # Only training-video validation intervals determine postprocessing.
            settings = select_inference_settings(model, validating, device, fold_dir)
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                        "feature_columns": columns, "feature_set": "pairmotion",
                        "normalization_mean": mean, "normalization_std": std,
                        "window": 256, "seed": seed,
                        "best_epoch": best["epoch"] if best else len(history),
                        "best_validation_loss": best["validation_loss"] if best else None,
                        "inference_settings": settings}, fold_dir / "model.pt")
            print(f"Decoding ALL frames for held-out video {test_video}; this is slower than segment scoring", flush=True)
            proposals = decode_video(model, test_video, counts[test_video], columns,
                                     mean, std, workdir, "pairmotion", meta, device)
            deduplicated = suppress_cross_window_duplicates(proposals)
            selected = [(span, confidence) for span, confidence in deduplicated
                        if confidence >= float(settings["min_score"])]
            table = prediction_table(test_video, selected, counts[test_video],
                                     min_seconds=float(settings["min_seconds"]),
                                     merge_seconds=float(settings["merge_seconds"]))
            table = table.with_columns(pl.col("Confidence").alias("Score"))
            table.write_csv(fold_dir / "full_video_predictions.csv")
            pred_dir = PREDICTION_ROOT / f"actionformer_lovo_seed_{seed}"
            pred_dir.mkdir(parents=True, exist_ok=True)
            # Keep one CSV per video, compatible with evaluation's reader.
            table.write_csv(pred_dir / f"{test_video}.csv")
            predicted_spans = [(int(s), int(e)) for s, e in
                               table.select("StartFrame", "EndFrame").iter_rows()]
            measured = evaluate_video(spans[test_video], predicted_spans, counts[test_video])
            record = {"study_id": study_id, "seed": seed, "train_video": train_video,
                      "test_video": test_video, "epochs_trained": len(history),
                      "best_epoch": best["epoch"] if best else len(history),
                      "inference_min_score": settings["min_score"],
                      "inference_min_seconds": settings["min_seconds"],
                      "inference_merge_seconds": settings["merge_seconds"],
                      "raw_proposals": len(proposals),
                      "run_directory": str(fold_dir), **measured}
            (fold_dir / "DONE.json").write_text(json.dumps(record, indent=2) + "\n")
            completed = update_results(study_id)
            print(f"DONE LOVO seed={seed} test={test_video} | F1@50={measured['f1@50']:.4f} "
                  f"frame F1={measured['frame_f1']:.4f} | {completed}/6 folds complete", flush=True)
            newly_completed += 1
            del model, samples, fitting, validating, proposals, deduplicated, selected
            gc.collect()
            if args.max_folds is not None and newly_completed >= args.max_folds:
                print("Stopped at --max-folds; rerun to resume", flush=True)
                return
    print(f"\nALL {len(SEEDS) * len(VIDEOS)} LOVO FOLDS COMPLETE | {OUTPUT / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
