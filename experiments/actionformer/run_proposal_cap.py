"""Inference-only ActionFormer full-video proposal-cap sweep.

Place this file at experiments/actionformer/run_proposal_cap.py, then run:

    uv run python -m experiments.actionformer.run_proposal_cap \
      --cv-dir outputs/actionformer_cv_window1024 \
      --output outputs/actionformer_cap_sweep_1024 \
      --caps 1000 2000 3000 --max-folds 1

Rerun without --max-folds to finish; decoded proposals and completed folds are cached.
This script NEVER trains or scores the final test video.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


METRICS = (
    "map@10", "map@30", "map@50", "map@70", "map_avg",
    "f1@10", "f1@25", "f1@50", "frame_precision", "frame_recall", "frame_f1",
    "gt_segments", "pred_segments", "tp@50", "native_proposals",
)
REFERENCE_METRICS = (
    "map@10", "map@30", "map@50", "map@70", "map_avg",
    "f1@50", "frame_precision", "frame_recall", "frame_f1",
    "gt_segments", "pred_segments", "tp@50", "native_proposals",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_normalized_annotations(path, videos):
    """Read the exact frozen labels used by the source CV, not new annotations."""
    result = {v: [] for v in videos}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"VideoID", "StartFrame", "EndFrame", "Behavior"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Missing columns in {path}: {sorted(required)}")
        for row in reader:
            video = row["VideoID"]
            if video in result and row["Behavior"].strip().lower() == "circling":
                result[video].append((int(row["StartFrame"]), int(row["EndFrame"])))
    if any(not result[v] for v in videos):
        raise ValueError("Source CV annotations have missing validation videos")
    return {v: sorted(result[v]) for v in videos}


def save_raw(path, proposals, pl):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pl.DataFrame({
        "start": [float(span[0]) for span, _ in proposals],
        "end": [float(span[1]) for span, _ in proposals],
        "confidence": [float(score) for _, score in proposals],
    }, schema={"start": pl.Float64, "end": pl.Float64, "confidence": pl.Float64})
    temp = path.with_suffix(".tmp.parquet")
    table.write_parquet(temp, compression="zstd")
    temp.replace(path)


def read_raw(path, pl):
    table = pl.read_parquet(path)
    if table.columns != ["start", "end", "confidence"]:
        raise ValueError(f"Unexpected proposal schema in {path}")
    return [((float(start), float(end)), float(score))
            for start, end, score in table.iter_rows()]


def write_results(output, rows, caps):
    output.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["cap"], r["fold"]))
    if rows:
        fields = sorted({key for row in rows for key in row})
        dest = output / "cap_results.csv"
        tmp = output / "cap_results.tmp.csv"
        with tmp.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(dest)
    summary = {}
    for cap in caps:
        selected = [row for row in rows if row["cap"] == cap]
        if not selected:
            continue
        summary[str(cap)] = {
            "folds": len(selected),
            "mean": {key: statistics.mean(float(row[key]) for row in selected)
                     for key in METRICS if key in selected[0]},
            "total_pred_segments": sum(int(row["pred_segments"]) for row in selected),
            "total_gt_segments": sum(int(row["gt_segments"]) for row in selected),
            "total_tp_at_50": sum(int(row["tp@50"]) for row in selected),
        }
    temp = output / "cap_summary.tmp.json"
    temp.write_text(json.dumps(summary, indent=2) + "\n")
    temp.replace(output / "cap_summary.json")
    print("\nProgress (mean per-video scores, not pooled):", flush=True)
    for cap in caps:
        if str(cap) not in summary:
            continue
        m = summary[str(cap)]["mean"]
        print(f"  cap={cap:5d}: {summary[str(cap)]['folds']}/7 folds | "
              f"mAP={m['map_avg']:.4f}, F1@50={m['f1@50']:.4f}, "
              f"frame precision={m['frame_precision']:.4f}, "
              f"predicted events={summary[str(cap)]['total_pred_segments']}", flush=True)


def check_reference(measured, reference, video):
    """Ensure inference-only cap=1000 matches the original finished CV fold."""
    bad = []
    for key in REFERENCE_METRICS:
        actual, expected = float(measured[key]), float(reference[key])
        if not math.isclose(actual, expected, rel_tol=1e-4, abs_tol=1e-4):
            bad.append(f"{key}: new={actual:.7g}, original={expected:.7g}")
    if bad:
        raise RuntimeError(
            f"{video}: original 1000-cap results did not reproduce. "
            "Do not interpret the sweep until this is resolved.\n" + "\n".join(bad)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path, default=Path("outputs/actionformer_cv_window1024"),
                        help="Completed video-disjoint CV directory containing seven fold checkpoints")
    parser.add_argument("--output", type=Path, default=Path("outputs/actionformer_cap_sweep_1024"))
    parser.add_argument("--caps", nargs="+", type=int, default=[1000, 2000, 3000],
                        help="Full-video proposal limits; must include the original cap of 1000")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Evaluate at most N new folds this invocation; resume by rerunning")
    parser.add_argument("--device", default=None, help="PyTorch device; defaults to CUDA if available, else CPU")
    args = parser.parse_args()
    caps = sorted(set(args.caps))
    if not caps or caps[0] < 1 or 1000 not in caps:
        parser.error("--caps must be positive and include the baseline cap of 1000")
    if args.max_folds is not None and args.max_folds < 1:
        parser.error("--max-folds must be positive")
    cv_dir, output = args.cv_dir.resolve(), args.output.resolve()
    if cv_dir == output or cv_dir in output.parents and output.name in {"folds", "raw_proposals"}:
        parser.error("Use a separate --output directory; do not modify the CV source")
    old_manifest_path = cv_dir / "manifest.json"
    gt_path = cv_dir / "normalized_annotations.csv"
    if not old_manifest_path.is_file() or not gt_path.is_file():
        parser.error("--cv-dir must contain manifest.json and normalized_annotations.csv")
    source = json.loads(old_manifest_path.read_text())
    window = int(source.get("window", 256))
    if window != 1024:
        parser.error(f"Expected a 1024-frame source CV, got window={window}")
    if int(source.get("max_proposals", 1000)) != 1000:
        parser.error("Source CV did not use the expected 1,000-proposal baseline")
    if sha256_file(gt_path) != source["GT_SHA256"]:
        parser.error("Frozen normalized annotations differ from the original CV manifest")
    videos = list(source["videos"])
    holdout = source["holdout"]
    dev = [v for v in videos if v != holdout]
    if len(dev) != 7:
        parser.error(f"Expected seven development videos; got {dev}")
    fold_specs = [(i, video, cv_dir / f"fold_{i:02d}_{video}")
                  for i, video in enumerate(dev, 1)]
    for _, video, fold_dir in fold_specs:
        if not (fold_dir / "DONE.json").is_file() or not (fold_dir / "model.pt").is_file():
            parser.error(f"Missing completed checkpoint or DONE.json for {video}: {fold_dir}")
    if not (cv_dir / "normalized_data").is_dir():
        parser.error("Missing source CV normalized_data directory")

    # Resume-safe. A different checkpoint, annotation set, or cap list gets a new output dir.
    experiment = {
        "purpose": "frozen-checkpoint full-video proposal-cap sweep",
        "source_cv_manifest_sha256": sha256_file(old_manifest_path),
        "annotation_sha256": sha256_file(gt_path),
        "source_cv": str(cv_dir), "window": window, "holdout": holdout,
        "caps": caps,
        "checkpoint_sha256": {video: sha256_file(fold_dir / "model.pt")
                              for _, video, fold_dir in fold_specs},
    }
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != experiment:
        parser.error("Existing cap experiment has different inputs; use a new --output folder")
    output.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        manifest.write_text(json.dumps(experiment, indent=2) + "\n")

    # Import model code only after recreating the source CV's data configuration.
    os.environ["CIRCLING_DATA_DIR"] = str(cv_dir / "normalized_data")
    os.environ["CIRCLING_GROUND_TRUTH"] = str(gt_path)
    os.environ["CIRCLING_VIDEOS"] = ",".join(videos)
    os.environ["CIRCLING_LABELS"] = "ground_truth"
    os.environ["CIRCLING_HARD_NEGATIVES"] = "0"
    import polars as pl
    import torch
    from evaluation.metrics import evaluate_video, map_at_tiou
    from experiments.actionformer.run_video_cv import metrics_for_video
    from models.actionformer.model import suppress_cross_window_duplicates
    from models.actionformer.postprocess import prediction_table
    from models.actionformer.predict import decode_video, load_checkpoint
    from models.common.config import mstcn_workdir, video_frames
    from models.common.segments import load_meta

    spans = load_normalized_annotations(gt_path, dev)
    counts = video_frames(videos)
    for video in dev:
        if not all(0 <= a <= b < counts[video] for a, b in spans[video]):
            parser.error(f"Out-of-bounds labels for {video}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    workdir = mstcn_workdir()
    meta = load_meta(workdir, videos)
    print(f"Using pretrained 1024-frame checkpoints on {device}; NO training and NO {holdout} evaluation.")
    print(f"Testing caps: {caps}. Raw predictions cached under {output / 'raw_proposals'}")

    all_rows = []
    completed_new = 0
    for fold, video, fold_dir in fold_specs:
        done_path = output / "fold_results" / f"fold_{fold:02d}_{video}.json"
        if done_path.is_file():
            rows = json.loads(done_path.read_text())
            if {int(row["cap"]) for row in rows} != set(caps):
                raise RuntimeError(f"Incomplete or incompatible fold result: {done_path}")
            all_rows.extend(rows)
            print(f"SKIP {video}: already completed", flush=True)
            continue
        previous = json.loads((fold_dir / "DONE.json").read_text())
        if previous["validation_video"] != video:
            raise RuntimeError(f"Incorrect validation video in {fold_dir / 'DONE.json'}")
        raw_path = output / "raw_proposals" / f"fold_{fold:02d}_{video}.parquet"
        if raw_path.is_file():
            raw = read_raw(raw_path, pl)
            print(f"{video}: loaded {len(raw):,} cached native proposals", flush=True)
        else:
            print(f"{video}: decoding full recording with frozen checkpoint", flush=True)
            model, checkpoint = load_checkpoint(fold_dir / "model.pt", device)
            if int(checkpoint.get("window", 0)) != window:
                raise RuntimeError(f"{video}: checkpoint window differs from source CV")
            raw = decode_video(
                model, video, counts[video], checkpoint["feature_columns"],
                checkpoint["normalization_mean"], checkpoint["normalization_std"],
                workdir, checkpoint["feature_set"], meta, device, window=window)
            save_raw(raw_path, raw, pl)
            del model, checkpoint
            print(f"{video}: saved {len(raw):,} native proposals", flush=True)

        # The vendored hard-NMS selection is confidence-ordered and prefix-stable:
        # one NMS with the largest cap is equivalent to separate calls at each cap.
        ranked = suppress_cross_window_duplicates(raw, max_segments=caps[-1])
        print(f"{video}: NMS kept {len(ranked):,} proposals at maximum cap", flush=True)
        settings = {
            "min_score": float(previous["post_min_score"]),
            "min_seconds": float(previous["post_min_seconds"]),
            "merge_seconds": float(previous["post_merge_seconds"]),
        }
        fold_rows = []
        for cap in caps:
            selected = ranked[:cap]
            tools = (lambda proposals: proposals, prediction_table, evaluate_video, map_at_tiou)
            table, measurements = metrics_for_video(
                video, selected, settings, spans, counts, tools)
            if cap == 1000:
                check_reference(measurements, previous, video)
            pred_dir = output / f"cap_{cap}" / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            table.write_csv(pred_dir / f"{video}.csv")
            row = {"fold": fold, "validation_video": video, "cap": cap,
                   "raw_proposals": len(raw), "post_min_score": settings["min_score"],
                   "post_min_seconds": settings["min_seconds"],
                   "post_merge_seconds": settings["merge_seconds"], **measurements}
            fold_rows.append(row)
            print(f"  cap={cap:5d}: mAP={row['map_avg']:.4f}, F1@50={row['f1@50']:.4f}, "
                  f"precision={row['frame_precision']:.4f}, events={row['pred_segments']}",
                  flush=True)
        done_path.parent.mkdir(parents=True, exist_ok=True)
        temp = done_path.with_suffix(".tmp.json")
        temp.write_text(json.dumps(fold_rows, indent=2) + "\n")
        temp.replace(done_path)
        all_rows.extend(fold_rows)
        write_results(output, all_rows, caps)
        completed_new += 1
        if args.max_folds is not None and completed_new >= args.max_folds:
            print("Stopped at --max-folds; rerun without it to finish.")
            return
    write_results(output, all_rows, caps)
    print(f"\nDONE: evaluated {len(dev)} folds at caps {caps}. "
          f"Results: {output / 'cap_results.csv'}", flush=True)
    print(f"Final holdout {holdout} was not scored.", flush=True)


if __name__ == "__main__":
    main()
