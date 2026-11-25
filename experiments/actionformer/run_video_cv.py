"""Video-disjoint ActionFormer CV, built on existing cichlid-analysis modules.

From repo root (on an ActionFormer branch):
  uv run python -m experiments.actionformer.run_video_cv --holdout 0037_vid --dry-run
  uv run python -m experiments.actionformer.run_video_cv --holdout 0037_vid --max-folds 1
  uv run python -m experiments.actionformer.run_video_cv --holdout 0037_vid
  uv run python -m experiments.actionformer.run_video_cv --holdout 0037_vid --final-test

Do not run --final-test until a common team holdout is agreed and the CV design
has been frozen. Model tuning never uses the final holdout labels.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from functools import partial

VIDEOS = ("0002_vid", "0007a_vid", "0007b_vid", "0013_vid",
          "0018_vid", "0028_vid", "0031_vid", "0037_vid")
SPECIAL = {"0007a_vid": 6, "0007b_vid": 5}
RECORDING_PROJECTS = {
    "0007a_vid": "MCYHRF1_535_t015_tr1",  # day 06
    "0007b_vid": "MCYHRF1_538_t021_tr1",  # day 05
}
MAP_TAUS = (10, 30, 50, 70)


def fail(message):
    raise SystemExit("VIDEO CV: " + str(message))


def read_gt_csvs(annotation_dir: Path) -> dict[str, list[tuple[int, int]]]:
    """Separate day-5/day-6 0007 recordings using CSV filename, NOT VideoID."""
    files = sorted(annotation_dir.glob("circling_annotations_vid*.csv"))
    if not files:
        fail(f"No finalized annotation CSVs in {annotation_dir}")
    result = {v: [] for v in VIDEOS}
    seen = {v: [] for v in VIDEOS}
    for path in files:
        filename = path.name.lower()
        if "0007" in filename and ("day5" in filename or "day05" in filename):
            expected = "0007b_vid"
        elif "0007" in filename and ("day6" in filename or "day06" in filename):
            expected = "0007a_vid"
        else:
            m = re.search(r"vid(\d{1,4})(?=\.csv$)", filename)
            expected = (m.group(1).zfill(4) + "_vid") if m else None
            if expected == "0007_vid":
                fail(f"Ambiguous 0007 annotations in {path.name}; filename must specify day5/day6")
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                continue
            def get(row, *names):
                keys = {re.sub(r"[^a-z0-9]", "", str(k).lower()): k for k in row}
                for name in names:
                    key = keys.get(re.sub(r"[^a-z0-9]", "", name.lower()))
                    if key is not None:
                        return (row[key] or "").strip()
                return ""
            for row in reader:
                video = expected or get(row, "VideoID")
                if video not in result:
                    continue
                behavior = get(row, "Primary label/behavior", "Behavior")
                if behavior and behavior.lower() != "circling":
                    continue
                if any(get(row, flag).lower() not in ("", "true", "1")
                       for flag in ("ShouldUse", "Circling")):
                    continue
                start = get(row, "StartFrame", "Start Frame")
                end = get(row, "EndFrame", "End Frame", "StopFrame")
                if not start or not end:
                    continue
                try:
                    a, b = int(start), int(end)
                except ValueError:
                    fail(f"Noninteger annotations in {path}: {start}, {end}")
                if a < 0 or b < a:
                    fail(f"Bad interval {a}-{b} in {path}")
                result[video].append((a, b))
                seen[video].append(path.name)
    missing = [v for v in VIDEOS if not result[v]]
    if missing:
        fail(f"Missing circling annotations for {missing}. Check the finalized CSV filename/VideoID mapping.")
    return {v: sorted(set(spans)) for v, spans in result.items()}


def write_gt_csv(spans, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["VideoID", "StartFrame", "EndFrame", "Behavior"])
        for video in VIDEOS:
            for a, b in spans[video]:
                w.writerow([video, a, b, "Circling"])


def input_paths(data_dir):
    return {v: (data_dir / f"{v}.parquet", data_dir / f"{v}_pairs.parquet") for v in VIDEOS}


def inspect_inputs(data_dir):
    files = input_paths(data_dir)
    missing = [str(p) for pair in files.values() for p in pair if not p.is_file()]
    if missing:
        fail("Missing files: " + ", ".join(missing))
    print("All eight matching pose/pair *filenames* are present.")
    return files


def rewrite_0007(src: Path, dst: Path, day_label: str, day_index: int):
    """Stream-rewrite labels, preserving original Parquet schema/metadata.

    Only affects copies in outputs/actionformer_cv_data/; never mutates inputs.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    signature = {"src": str(src.resolve()), "bytes": src.stat().st_size,
                 "mtime_ns": src.stat().st_mtime_ns, "label": day_label, "day_index": day_index}
    stamp = dst.with_suffix(dst.suffix + ".source.json")
    if dst.is_file() and stamp.is_file() and json.loads(stamp.read_text()) == signature:
        return
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    for key in ("day_label", "day_index", "project_id"):
        if key not in schema.names:
            fail(f"{src.name} missing {key}")
    tmp = dst.with_suffix(".part.parquet")
    tmp.unlink(missing_ok=True)
    try:
        with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
            for batch in pf.iter_batches(batch_size=100_000):
                tbl = pa.Table.from_batches([batch], schema=schema)
                for field_name, val in (("day_label", day_label), ("day_index", day_index)):
                    idx = schema.get_field_index(field_name)
                    field = schema.field(idx)
                    values = pa.array([val] * len(tbl), type=field.type)
                    tbl = tbl.set_column(idx, field, values)
                writer.write_table(tbl)
        tmp.replace(dst)
        stamp.write_text(json.dumps(signature, indent=2) + "\n")
    finally:
        tmp.unlink(missing_ok=True)


def prepare_data_dir(data_dir, derived_dir):
    """Symlink standard inputs, but normalize the two ambiguous 0007 labels."""
    files = inspect_inputs(data_dir)
    derived_dir.mkdir(parents=True, exist_ok=True)
    for v, pair in files.items():
        for src in pair:
            dst = derived_dir / src.name
            if v in SPECIAL:
                rewrite_0007(src, dst, v, SPECIAL[v])
            elif not dst.exists():
                dst.symlink_to(src.resolve())
            elif dst.resolve() != src.resolve():
                fail(f"Existing {dst} points to a different source; remove it to rebuild")
    # Validate project IDs and the per-file row count/label using Parquet metadata.
    import pyarrow.parquet as pq
    for v, (track, pairs) in input_paths(derived_dir).items():
        values = []
        for p in (track, pairs):
            pf = pq.ParquetFile(p)
            cols = ["project_id", "day_label", "FrameNum"]
            if not set(cols) <= set(pf.schema_arrow.names):
                fail(f"{p.name} missing project_id/day_label/FrameNum")
            batch = next(pf.iter_batches(batch_size=1, columns=cols), None)
            if batch is None:
                fail(f"Empty dataset {p}")
            row = batch.to_pydict()
            if row["day_label"][0] != v:
                fail(f"{p.name} day_label={row['day_label'][0]!r}, expected {v}")
            values.append((row["project_id"][0], pf.metadata.num_rows))
        if values[0][0] != values[1][0]:
            fail(f"{v}: pose/pair project IDs disagree: {values}")
        if v in RECORDING_PROJECTS and values[0][0] != RECORDING_PROJECTS[v]:
            fail(f"{v}: wrong project {values[0][0]}")
        print(f"{v}: project={values[0][0]}, tracks={values[0][1]:,}, pairs={values[1][1]:,}")


def seed_all(torch, numpy, seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics_for_video(video, raw, settings, spans, counts, tools):
    """AP from scored native proposals; F1 from train-tuned hard decisions."""
    import numpy as np
    import polars as pl

    suppress, prediction_table, evaluate_video, map_at_tiou = tools
    proposals = suppress(raw)
    n = counts[video]
    # mAP is confidence-ranked: do not apply a hard decision cutoff.
    scored = []
    for (start, end), score in proposals:
        a, b = max(0, int(np.floor(start))), min(n, int(np.ceil(end)))
        if b > a:
            scored.append((a, b, float(score)))
    gt_half_open = [(int(a), int(b) + 1) for a, b in spans[video]]
    ap = map_at_tiou(gt_half_open, scored, taus=MAP_TAUS)
    selected = [(span, conf) for span, conf in proposals if conf >= float(settings["min_score"])]
    table = prediction_table(video, selected, n,
                             min_seconds=float(settings["min_seconds"]),
                             merge_seconds=float(settings["merge_seconds"]))
    predicted = [(int(a), int(b)) for a, b in table.select("StartFrame", "EndFrame").iter_rows()]
    scores = evaluate_video(spans[video], predicted, n)
    return table, {**ap, **scores, "native_proposals": len(proposals),
                   "ranked_proposals": len(scored)}


def save_results(output, holdout, folds):
    if not folds:
        return
    out_rows = []
    for row in folds:
        out_rows.append({k: v for k, v in row.items() if k != "study_id"})
    fields = sorted({k for r in out_rows for k in r})
    with (output / "cv_results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    if len(folds) == 7:
        names = [*(f"map@{t}" for t in MAP_TAUS), "map_avg", "f1@50", "frame_f1"]
        summary = {k: {"mean": statistics.mean(float(r[k]) for r in folds),
                        "std": statistics.stdev(float(r[k]) for r in folds)} for k in names}
        (output / "cv_summary.json").write_text(json.dumps({
            "holdout": holdout, "folds": len(folds), "metrics": summary,
            "definition": "unweighted mean/std across seven separate held-out development videos",
        }, indent=2) + "\n")
        print("CV complete; mean four-threshold mAP:", round(summary["map_avg"]["mean"], 4))


def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--holdout", choices=VIDEOS, required=True,
                   help="Final untouched test recording, agreed with team")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Folder with 16 pose+pair Parquets")
    p.add_argument("--annotations", type=Path, default=Path("ground_truth_annotations/finalized_annotations"))
    p.add_argument("--output", type=Path, default=Path("outputs/actionformer_video_cv"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--window", type=int, choices=(256, 512, 1024), default=256,
                   help="Temporal window in original video frames (30 FPS)")
    p.add_argument("--max-proposals", type=int, default=1000,
                   help="Maximum proposals after full-video cross-window NMS")
    p.add_argument("--hard-negative-fraction", type=float, default=0.0,
                   help="Fraction of sampled negative training segments drawn from nearby-pair cases (0 to 1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dry-run", action="store_true", help="Check inputs, labels and print fold schedule")
    p.add_argument("--prepare-only", action="store_true", help="Create normalized inputs and feature caches only")
    p.add_argument("--max-folds", type=int, help="Run at most this many incomplete folds (resume later)")
    p.add_argument("--final-test", action="store_true", help="After CV, fit all 7 dev videos, score untouched holdout once")
    args = p.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.max_proposals < 1
            or not 0 <= args.hard_negative_fraction <= 1
            or (args.max_folds is not None and args.max_folds < 1)):
        p.error("epochs, batch-size, max-proposals and max-folds must be positive; hard-negative-fraction must be 0 to 1")
    data_dir, output = args.data_dir.resolve(), args.output.resolve()
    derived_dir = output / "normalized_data"
    if data_dir == derived_dir:
        p.error("--data-dir must be original input folder, not normalized_data")
    inspect_inputs(data_dir)
    spans = read_gt_csvs(args.annotations)
    dev = [v for v in VIDEOS if v != args.holdout]
    print(f"FINAL HOLDOUT (never used for development): {args.holdout}")
    for i, video in enumerate(dev, 1):
        print(f"Fold {i}: train={[v for v in dev if v != video]}, validate={video}")
    print("Annotations:", ", ".join(f"{v}:{len(spans[v])}" for v in VIDEOS))
    if args.dry_run:
        return

    output.mkdir(parents=True, exist_ok=True)
    gt_csv = output / "normalized_annotations.csv"
    write_gt_csv(spans, gt_csv)
    cfg = {"holdout": args.holdout, "videos": list(VIDEOS), "epochs": args.epochs,
           "batch_size": args.batch_size, "seed": args.seed, "feature": "pairmotion",
           "window": args.window, "max_proposals": args.max_proposals,
           "GT_SHA256": hashlib.sha256(gt_csv.read_bytes()).hexdigest()}
    if args.hard_negative_fraction > 0:
        cfg["hard_negative_fraction"] = args.hard_negative_fraction
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != cfg:
        fail(f"Config or ground truth changed; use a new --output folder, not {output}")
    if not manifest.exists():
        manifest.write_text(json.dumps(cfg, indent=2) + "\n")

    prepare_data_dir(data_dir, derived_dir)
    # These environment vars must precede importing models.common.config.
    os.environ["CIRCLING_DATA_DIR"] = str(derived_dir)
    os.environ["CIRCLING_GROUND_TRUTH"] = str(gt_csv)
    os.environ["CIRCLING_VIDEOS"] = ",".join(VIDEOS)
    os.environ["CIRCLING_LABELS"] = "ground_truth"
    os.environ["CIRCLING_HARD_NEGATIVES"] = str(args.hard_negative_fraction)

    import numpy as np
    import polars as pl
    import torch
    from evaluation.metrics import evaluate_video, map_at_tiou
    from models.actionformer.model import build_model, suppress_cross_window_duplicates
    from models.actionformer.postprocess import prediction_table
    from models.actionformer.predict import decode_video
    from models.actionformer.tuning import select_inference_settings
    from models.common.config import mstcn_workdir, video_frames
    from models.common.segments import STRIDE as DATA_STRIDE, load_meta, shared_segments
    from models.mstcn_dlc2action.train import prepare_data
    trainer = importlib.import_module("models.actionformer.train")

    tools = (partial(suppress_cross_window_duplicates, max_segments=args.max_proposals),
             prediction_table, evaluate_video, map_at_tiou)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    counts = video_frames(VIDEOS)
    for video in VIDEOS:
        if not all(0 <= a <= b < counts[video] for a, b in spans[video]):
            fail(f"Annotations outside frame bounds of {video}")
    workdir = mstcn_workdir()
    print(f"Preparing existing 132-feature pairmotion pipeline in {workdir}")
    prepare_data(workdir, derived_dir, "pairmotion", videos=VIDEOS)
    if args.prepare_only:
        print("Feature preparation complete; rerun without --prepare-only")
        return
    # Important: only *development* annotations enter the sampling/splitting process.
    dev_spans = {v: spans[v] for v in dev}
    if args.window % DATA_STRIDE:
        fail(f"--window must be a multiple of data stride {DATA_STRIDE}")
    segments, _ = shared_segments(
        workdir, dev_spans, output / "development_segments",
        len_segment=args.window // DATA_STRIDE)
    meta = load_meta(workdir, VIDEOS)
    results = []
    finished_new = 0
    for fold_num, val_video in enumerate(dev, 1):
        fold_dir = output / f"fold_{fold_num:02d}_{val_video}"
        done = fold_dir / "DONE.json"
        if done.is_file():
            row = json.loads(done.read_text())
            if row.get("study_id") != cfg["GT_SHA256"]:
                fail(f"Incompatible previous fold: {done}")
            results.append(row)
            print(f"SKIP {val_video}: completed fold")
            continue
        fold_dir.mkdir(parents=True, exist_ok=True)
        fit_videos = [v for v in dev if v != val_video]
        train_rows = segments.filter(pl.col("video").is_in(fit_videos)).with_columns(
            pl.lit("train").alias("split"))
        fit_meta = {v: meta[v] for v in fit_videos}
        fit_counts = {v: counts[v] for v in fit_videos}
        initial_samples, _, _, _ = trainer.prepare_samples(
            train_rows, fit_meta, dev_spans, fit_counts, workdir, "pairmotion")
        train_split, val_split = trainer.training_validation_split(initial_samples, 0.2, seed=args.seed)
        validation_ids = {s["sample_id"] for s in val_split}
        train_rows = train_rows.with_columns(pl.Series("split", [
            "validation" if i in validation_ids else "train" for i in range(train_rows.height)]))
        del initial_samples, train_split, val_split
        samples, columns, mean, std = trainer.prepare_samples(
            train_rows, fit_meta, dev_spans, fit_counts, workdir, "pairmotion")
        fitting = [s for s in samples if s["split"] == "train"]
        checking = [s for s in samples if s["split"] == "validation"]
        if len(columns) != 132:
            fail(f"Expected 132 features; found {len(columns)}. Check pairmotion configuration.")
        seed_all(torch, np, args.seed)
        model, model_config = build_model(len(columns), dataset={"max_seq_len": args.window})
        start_time = time.monotonic()
        print(f"FOLD {fold_num}/7: fit {fit_videos}; evaluate FULL {val_video} | "
              f"{len(fitting)} training segments, {len(checking)} inner-validation", flush=True)
        history = trainer.train_model(model, fitting, device, epochs=args.epochs,
                                      batch_size=args.batch_size, learning_rate=1e-4,
                                      validation_samples=checking, patience=6, seed=args.seed)
        pl.DataFrame(history).write_csv(fold_dir / "training_history.csv")
        settings = select_inference_settings(model, checking, device, fold_dir)
        best = next((r for r in reversed(history) if r.get("is_best")), None)
        torch.save({"model_state_dict": model.state_dict(), "config": model_config,
                    "feature_columns": columns, "feature_set": "pairmotion",
                    "normalization_mean": mean, "normalization_std": std,
                    "window": args.window, "seed": args.seed,
                    "best_epoch": best["epoch"] if best else len(history),
                    "inference_settings": settings}, fold_dir / "model.pt")
        proposals = decode_video(model, val_video, counts[val_video], columns, mean, std,
                                 workdir, "pairmotion", meta, device, window=args.window)
        table, measurements = metrics_for_video(val_video, proposals, settings, spans, counts, tools)
        table.write_csv(fold_dir / "predictions.csv")
        record = {"study_id": cfg["GT_SHA256"], "fold": fold_num,
                  "train_videos": ";".join(fit_videos), "validation_video": val_video,
                  "fit_segments": len(fitting), "best_epoch": best["epoch"] if best else len(history),
                  "train_and_infer_seconds": time.monotonic() - start_time,
                  **{f"post_{k}": v for k, v in settings.items()}, **measurements}
        done.write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
        save_results(output, args.holdout, results)
        print(f"DONE {val_video}: map_avg={record['map_avg']:.4f}, "
              f"F1@50={record['f1@50']:.4f} | {len(results)}/7", flush=True)
        del model, samples, fitting, checking, proposals
        gc.collect()
        finished_new += 1
        if args.max_folds is not None and finished_new >= args.max_folds:
            print("Stopped at --max-folds; rerun same command to resume.")
            return
    save_results(output, args.holdout, results)
    if not args.final_test:
        print("All CV folds complete. Holdout remains untouched; add --final-test to run final evaluation.")
        return
    if len(results) != 7:
        fail("Seven completed development folds required before --final-test")
    final_dir = output / f"final_test_{args.holdout}"
    if (final_dir / "DONE.json").exists():
        print("Final test already completed, refusing to rescore it.")
        return
    final_dir.mkdir(parents=True, exist_ok=True)
    # Mean best epoch is determined on development folds, before seeing test data.
    epochs_final = max(1, round(statistics.mean(int(r["best_epoch"]) for r in results)))
    settings = {key: float(statistics.median(float(r[f"post_{key}"]) for r in results))
                for key in ("min_score", "min_seconds", "merge_seconds")}
    # All dev segments, including those used for inner validation in CV, are training here.
    train_rows = segments.with_columns(pl.lit("train").alias("split"))
    samples, columns, mean, std = trainer.prepare_samples(
        train_rows, {v: meta[v] for v in dev}, dev_spans,
        {v: counts[v] for v in dev}, workdir, "pairmotion")
    seed_all(torch, np, args.seed)
    model, model_config = build_model(len(columns), dataset={"max_seq_len": args.window})
    began = time.monotonic()
    history = trainer.train_model(model, samples, device, epochs=epochs_final,
                                  batch_size=args.batch_size, learning_rate=1e-4,
                                  validation_samples=[], patience=0, seed=args.seed)
    torch.save({"model_state_dict": model.state_dict(), "config": model_config,
                "feature_columns": columns, "feature_set": "pairmotion",
                "normalization_mean": mean, "normalization_std": std,
                "window": args.window, "seed": args.seed, "best_epoch": epochs_final,
                "inference_settings": settings}, final_dir / "model.pt")
    proposals = decode_video(model, args.holdout, counts[args.holdout], columns, mean,
                             std, workdir, "pairmotion", meta, device, window=args.window)
    table, measurements = metrics_for_video(args.holdout, proposals, settings, spans, counts, tools)
    table.write_csv(final_dir / "predictions.csv")
    (final_dir / "DONE.json").write_text(json.dumps({
        "test_video": args.holdout, "trained_on": dev, "epochs": epochs_final,
        "settings_from": "medians of 7 development folds",
        "train_and_infer_seconds": time.monotonic() - began, **measurements,
    }, indent=2) + "\n")
    print(f"FINAL TEST: mAP={measurements['map_avg']:.4f}, "
          f"F1@50={measurements['f1@50']:.4f}")


if __name__ == "__main__":
    run()
