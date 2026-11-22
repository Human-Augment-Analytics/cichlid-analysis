"""Controlled, resumable ActionFormer feature x negative-sampling study.

Run from the cichlid-analysis repository root:
    uv run python -m experiments.actionformer.run_large_actionformer_study --dry-run
    uv run python -m experiments.actionformer.run_large_actionformer_study --tier full

Full plan: 6 feature sets x 3 negative sets x 3 seeds (54 models), plus
3 additive-negative controls x 3 seeds (9 models) = 63 trained models.
Core plan: 3 feature sets x 3 negative sets x 3 seeds = 27 models.

This study uses a frozen *segment-level*, same-video test set; it is NOT LOVO.
It deliberately uses only fit-segment data for normalization. The validation
set is fixed before hard negatives are selected, and the test set is never used
to pick thresholds or models. Earlier single-run results used a slightly
different preprocessing protocol and should not be treated as directly paired.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import random
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support)

from feature_engineering.pair_motion.feature_groups import FEATURE_GROUPS
from models.actionformer.model import build_model
from models.common.config import (FINALIZED_ANNOTATIONS, HARD_NEGATIVES,
                                  mstcn_workdir, read_annotations, video_frames)
from models.common.pair_motion import PREFIX, ROLE_NAMES
from models.common.report import write_report
from models.common.segments import load_meta, pair_closeness
from models.mstcn_dlc2action.dlc_yolo_converter import dlc2action_bridge as br

trainer = importlib.import_module("models.actionformer.train")

BASELINE = Path("outputs/experiment_backups/baseline_actionformer/sampled_segments.parquet")
REVIEWED = Path("experiments/actionformer/data/reviewed_hard_negatives.xlsx")
OUT = Path("outputs/actionformer_large_controlled_study")
SEEDS = (42, 123, 456)
SPLIT_SEED = 42
CANDIDATE_SEED = 2026
EPOCHS = 60
BATCH_SIZE = 4
LR = 1e-4
PATIENCE = 6
VIDEO_IDS = ("0028_vid", "0031_vid")

FULL_FEATURES = ("full", *["no_" + name for name in FEATURE_GROUPS])
CORE_FEATURES = ("full", "no_angular_movement", "no_temporal_consistency")
REPLACE_STRATEGIES = ("original", "proximity_replace", "reviewed_replace")
ADD_STRATEGIES = ("random_add", "proximity_add", "reviewed_add")

RUN_FIELDS = ("seed", "negatives", "features", "n_features", "fit_segments",
              "validation_segments", "test_segments", "epochs_trained", "best_epoch",
              "best_validation_loss", "average_precision", "roc_auc",
              "val_threshold", "validation_f1", "test_precision_at_val_threshold",
              "test_recall_at_val_threshold", "test_f1_at_val_threshold",
              "test_tp", "test_fp", "test_fn", "test_tn", "run_directory")
SUMMARY_FIELDS = ("negatives", "features", "n_seeds", "ap_mean", "ap_std",
                  "roc_auc_mean", "roc_auc_std", "val_threshold_f1_mean",
                  "val_threshold_f1_std", "mean_delta_ap_vs_original_full",
                  "mean_delta_ap_vs_same_negative_full")
PAIR_FIELDS = ("seed", "negatives", "features", "ap_difference_vs_original_full",
               "roc_auc_difference_vs_original_full", "ap_difference_vs_same_negative_full")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def source_signature() -> dict:
    ground_truth = Path(FINALIZED_ANNOTATIONS)
    sources = sorted(ground_truth.glob("circling_annotations_vid*.csv")) if ground_truth.is_dir() else [ground_truth]
    return {
        "baseline_sha256": sha256_file(BASELINE),
        "annotations_sha256": {path.name: sha256_file(path) for path in sources},
        "reviewed_sha256": sha256_file(REVIEWED) if REVIEWED.exists() else None,
        "script_sha256": sha256_file(Path(__file__)),
    }


def key(row) -> tuple[str, int, int]:
    return (str(row["video"]), int(row["start_row"]), int(row["end_row"]))


def bounds(video: str, start: int, end: int, meta: dict, counts: dict):
    m = meta[video]
    first = int(start) * int(m.get("stride", 1)) + int(m.get("frame_min", 0))
    stop = min(int(end) * int(m.get("stride", 1)) + int(m.get("frame_min", 0)), int(counts[video]))
    return (first, stop - 1) if first < stop else None


def intersects(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def fixed_roles(baseline: pl.DataFrame, meta: dict, spans: dict, counts: dict, workdir: Path):
    # Recreate exactly the original training-validation split with the repo's
    # stratification function and its segment event targets, once, before changes.
    base_samples, _, _, _ = trainer.prepare_samples(
        baseline, meta, spans, counts, workdir, "pairmotion")
    base_train = [s for s in base_samples if s["split"] == "train"]
    fitting, validating = trainer.training_validation_split(base_train, 0.2, seed=SPLIT_SEED)
    fit_ids = {int(s["sample_id"]) for s in fitting}
    val_ids = {int(s["sample_id"]) for s in validating}
    rows = list(baseline.iter_rows(named=True))
    fit_keys, val_keys, test_keys = set(), set(), set()
    for i, row in enumerate(rows):
        k = key(row)
        if row["split"] == "test":
            test_keys.add(k)
        elif i in fit_ids:
            fit_keys.add(k)
        elif i in val_ids:
            val_keys.add(k)
        else:
            raise AssertionError(f"No role for baseline row {i}: {k}")
    assert len(fit_keys) == len(fitting) and len(val_keys) == len(validating)
    assert not (fit_keys & val_keys or fit_keys & test_keys or val_keys & test_keys)
    print(f"Frozen roles: {len(fit_keys)} fit, {len(val_keys)} validation, "
          f"{len(test_keys)} test segments", flush=True)
    del base_samples, base_train, fitting, validating
    gc.collect()
    return fit_keys, val_keys, test_keys


def checked_baseline_rows(baseline: pl.DataFrame, val_keys: set):
    rows = []
    for r in baseline.iter_rows(named=True):
        row = dict(r)
        if key(row) in val_keys:
            row["split"] = "validation"
        rows.append(row)
    return rows


def build_candidate_banks(baseline, meta, spans, counts, val_keys, test_keys, workdir):
    original = {key(row) for row in baseline.iter_rows(named=True)}
    protected = {v: [] for v in VIDEO_IDS}
    for r in baseline.iter_rows(named=True):
        if key(r) in val_keys or key(r) in test_keys:
            region = bounds(r["video"], r["start_row"], r["end_row"], meta, counts)
            if region:
                protected[r["video"]].append(region)
    close = pair_closeness(workdir, meta, VIDEO_IDS)
    bank = {v: [] for v in VIDEO_IDS}
    for video in VIDEO_IDS:
        for s, e in br.segment_spans(int(meta[video]["n_frames"]), 128, 0):
            s, e = int(s), int(e)
            if (video, s, e) in original:
                continue
            region = bounds(video, s, e, meta, counts)
            if region is None:
                continue
            if any(intersects(region, x) for x in protected[video]):
                continue
            if any(intersects(region, x) for x in spans[video]):
                continue
            closeness = float(np.mean(close[video][s:e])) if e > s else 0.0
            bank[video].append({"video": video, "start_row": s, "end_row": e,
                                "first": region[0], "last": region[1],
                                "proximity": closeness >= 0.5,
                                "close_fraction": closeness,
                                "review_label": None, "review_overlap": 0})
        print(f"{video}: {len(bank[video])} available negative candidates, "
              f"{sum(c['proximity'] for c in bank[video])} proximity", flush=True)
    return bank


def attach_reviewed_labels(bank):
    annotations = pd.read_excel(REVIEWED, sheet_name="Hard negatives")
    required = {"VideoID", "Start Frame", "End Frame", "Primary label/behavior"}
    if not required <= set(annotations.columns):
        raise ValueError(f"Reviewed spreadsheet missing columns: {sorted(required - set(annotations.columns))}")
    by_video = {v: [] for v in VIDEO_IDS}
    for _, row in annotations.iterrows():
        video = str(row["VideoID"]).strip()
        label = str(row["Primary label/behavior"]).strip().lower()
        if video not in by_video or label not in ("no-circling", "partial circling"):
            continue
        try:
            start, end = int(row["Start Frame"]), int(row["End Frame"])
        except (ValueError, TypeError):
            continue
        if end >= start:
            by_video[video].append((start, end, label))
    for video in VIDEO_IDS:
        for candidate in bank[video]:
            best_overlap, best_label = 0, None
            a, b = candidate["first"], candidate["last"]
            for start, end, label in by_video[video]:
                overlap = max(0, min(b, end) - max(a, start) + 1)
                if overlap >= min(16, end - start + 1) and overlap / (end - start + 1) >= 0.5 and overlap > best_overlap:
                    best_overlap, best_label = overlap, label
            candidate["review_label"] = best_label
            candidate["review_overlap"] = best_overlap
        labels = Counter(c["review_label"] for c in bank[video] if c["review_label"])
        print(f"{video}: reviewed candidate matches {dict(labels)}", flush=True)


def new_row(candidate, schema):
    out = {column: None for column in schema}
    out.update({"video": candidate["video"], "start_row": candidate["start_row"],
                "end_row": candidate["end_row"], "positive_rows": 0,
                "circling": False, "hard_negative": True, "split": "train"})
    return out


def sample_candidates(candidates, number, rng):
    number = min(number, len(candidates))
    if not number:
        return []
    ids = sorted(int(i) for i in rng.choice(len(candidates), size=number, replace=False))
    return [candidates[i] for i in ids]


def build_strategies(baseline, fit_keys, val_keys, test_keys, bank, use_reviewed):
    originals = checked_baseline_rows(baseline, val_keys)
    fitting_negatives = {v: sorted(
        [r for r in originals if r["video"] == v and r["split"] == "train"
         and not r["circling"] and not r["hard_negative"]], key=key)
        for v in VIDEO_IDS}
    selected = {"proximity": [], "reviewed": [], "random": []}
    select_info = {}
    for video in VIDEO_IDS:
        neg = fitting_negatives[video]
        proximity_pool = [x for x in bank[video] if x["proximity"]]
        prox_quota = min(round(0.5 * len(neg)), len(proximity_pool))
        proximity = sample_candidates(proximity_pool, prox_quota,
                                      np.random.default_rng(CANDIDATE_SEED + VIDEO_IDS.index(video)))
        reviewed_pool = sorted([x for x in bank[video] if x["review_label"]],
                               key=lambda x: (-x["review_overlap"], x["start_row"]))
        reviewed = reviewed_pool[:min(len(neg), len(reviewed_pool))] if use_reviewed else []
        # Same number of additional ordinary (non-proximity) negatives as
        # proximity_add, per video; tests data volume vs hard-negative selection.
        easy_pool = [x for x in bank[video] if not x["proximity"] and x not in reviewed]
        random_extra = sample_candidates(easy_pool, len(proximity),
                                         np.random.default_rng(CANDIDATE_SEED + 100 + VIDEO_IDS.index(video)))
        if len(random_extra) != len(proximity):
            raise RuntimeError(f"Not enough ordinary negatives to match proximity_add in {video}")
        for name, values in (("proximity", proximity), ("reviewed", reviewed), ("random", random_extra)):
            selected[name].extend(values)
        select_info[video] = {
            "fit_negative_count": len(neg), "proximity_candidates": len(proximity_pool),
            "proximity_selected": len(proximity), "reviewed_candidates": len(reviewed_pool),
            "reviewed_selected": len(reviewed), "reviewed_partial_selected": sum(
                c["review_label"] == "partial circling" for c in reviewed),
            "random_added_control": len(random_extra),
        }
        print(f"{video}: fit negatives={len(neg)}, proximity={len(proximity)}, "
              f"reviewed={len(reviewed)}, random add control={len(random_extra)}", flush=True)

    def replacing(kind):
        drop = set()
        for v in VIDEO_IDS:
            chosen = [x for x in selected[kind] if x["video"] == v]
            negatives = fitting_negatives[v]
            if kind == "reviewed":
                dropped = negatives[:len(chosen)]  # reproduce deterministic reviewed-negative replacement
            else:
                dropped = sample_candidates(negatives, len(chosen),
                                             np.random.default_rng(CANDIDATE_SEED + 300 + VIDEO_IDS.index(v)))
            drop.update(map(key, dropped))
        return [r.copy() for r in originals if key(r) not in drop] + [
            new_row(c, baseline.schema) for c in selected[kind]]

    definitions = {
        "original": [r.copy() for r in originals],
        "proximity_replace": replacing("proximity"),
        "proximity_add": [r.copy() for r in originals] + [new_row(c, baseline.schema) for c in selected["proximity"]],
        "random_add": [r.copy() for r in originals] + [new_row(c, baseline.schema) for c in selected["random"]],
    }
    if use_reviewed:
        definitions["reviewed_replace"] = replacing("reviewed")
        definitions["reviewed_add"] = [r.copy() for r in originals] + [new_row(c, baseline.schema) for c in selected["reviewed"]]

    test_baseline = {key(r): (bool(r["circling"]), int(r["positive_rows"]))
                     for r in originals if r["split"] == "test"}
    val_baseline = {key(r): (bool(r["circling"]), int(r["positive_rows"]))
                    for r in originals if r["split"] == "validation"}
    positives_baseline = {key(r) for r in originals if r["split"] == "train" and r["circling"]}
    tables = {}
    for name, rows in definitions.items():
        if len(rows) != len({key(r) for r in rows}):
            raise AssertionError(f"{name}: duplicate segment keys")
        test = {key(r): (bool(r["circling"]), int(r["positive_rows"]))
                for r in rows if r["split"] == "test"}
        val = {key(r): (bool(r["circling"]), int(r["positive_rows"]))
               for r in rows if r["split"] == "validation"}
        pos = {key(r) for r in rows if r["split"] == "train" and r["circling"]}
        assert test == test_baseline, f"{name}: test changed"
        assert val == val_baseline, f"{name}: validation changed"
        assert pos == positives_baseline, f"{name}: fit positives changed"
        assert {key(r) for r in rows if r["split"] == "train"}.isdisjoint(val_keys | test_keys)
        table = pl.DataFrame(rows, schema=baseline.schema)
        tables[name] = table
        print(f"{name}: {table.filter(pl.col('split') == 'train').height} fitting, "
              f"{table.filter(pl.col('split') == 'validation').height} validation, "
              f"{table.filter(pl.col('split') == 'test').height} test", flush=True)
    return tables, select_info


def feature_indices(columns, feature_name):
    if feature_name == "full":
        return list(range(len(columns)))
    group = feature_name.removeprefix("no_")
    if group not in FEATURE_GROUPS or feature_name == group:
        raise ValueError(f"Invalid feature variant: {feature_name}")
    remove = {PREFIX + ROLE_NAMES.get(x, x) for x in FEATURE_GROUPS[group]}
    missing = remove - set(columns)
    if missing:
        raise ValueError(f"Cannot ablate {group}; missing columns: {sorted(missing)}")
    keep = [i for i, c in enumerate(columns) if c not in remove]
    if not keep or len(keep) == len(columns):
        raise ValueError(f"Invalid ablation {feature_name}")
    return keep


def choose_validation_threshold(table):
    y = table["circling"].to_numpy().astype(int)
    scores = table["circling_score"].to_numpy().astype(float)
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1 = np.divide(2 * precision[:-1] * recall[:-1],
                   precision[:-1] + recall[:-1],
                   out=np.zeros_like(precision[:-1]),
                   where=(precision[:-1] + recall[:-1]) > 0)
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])


def threshold_test_metrics(table, threshold):
    y = table["circling"].to_numpy().astype(int)
    scores = table["circling_score"].to_numpy().astype(float)
    pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"test_precision_at_val_threshold": float(p),
            "test_recall_at_val_threshold": float(r),
            "test_f1_at_val_threshold": float(f1),
            "test_tn": int(tn), "test_fp": int(fp),
            "test_fn": int(fn), "test_tp": int(tp)}


def csv_write(path, records, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def completed_record(out_dir, study_id, negatives, features, seed, require_checkpoint):
    p = out_dir / negatives / features / f"seed_{seed}"
    marker = p / "DONE.json"
    if not marker.is_file() or not (p / "metrics.json").is_file():
        return None
    row = json.loads(marker.read_text())
    if row.get("study_id") != study_id:
        raise RuntimeError(f"Stale completed run at {p}; study inputs or script changed. "
                           "Use a different output directory to avoid mixing studies.")
    if require_checkpoint and not (p / "model.pt").is_file():
        return None
    return row


def summarize(out_dir, plan, study_id, require_checkpoint):
    rows = []
    for negatives, features, seed in plan:
        record = completed_record(out_dir, study_id, negatives, features, seed,
                                  require_checkpoint)
        if record is not None:
            rows.append(record)
    csv_write(out_dir / "per_run_results.csv", rows, RUN_FIELDS)
    by_key = {(r["negatives"], r["features"], r["seed"]): r for r in rows}
    combos = list(dict.fromkeys((n, f) for n, f, _ in plan))
    summary = []
    for negatives, features in combos:
        subset = [r for r in rows if r["negatives"] == negatives and r["features"] == features]
        if not subset:
            continue
        def avg(field):
            return statistics.mean(float(x[field]) for x in subset)
        def sd(field):
            return statistics.stdev(float(x[field]) for x in subset) if len(subset) > 1 else ""
        delta_baseline = []
        delta_within = []
        for row in subset:
            ref = by_key.get(("original", "full", row["seed"]))
            internal = by_key.get((negatives, "full", row["seed"]))
            if ref:
                delta_baseline.append(float(row["average_precision"]) - float(ref["average_precision"]))
            if internal:
                delta_within.append(float(row["average_precision"]) - float(internal["average_precision"]))
        summary.append({"negatives": negatives, "features": features, "n_seeds": len(subset),
                        "ap_mean": avg("average_precision"), "ap_std": sd("average_precision"),
                        "roc_auc_mean": avg("roc_auc"), "roc_auc_std": sd("roc_auc"),
                        "val_threshold_f1_mean": avg("test_f1_at_val_threshold"),
                        "val_threshold_f1_std": sd("test_f1_at_val_threshold"),
                        "mean_delta_ap_vs_original_full": statistics.mean(delta_baseline) if delta_baseline else "",
                        "mean_delta_ap_vs_same_negative_full": statistics.mean(delta_within) if delta_within else ""})
    csv_write(out_dir / "summary.csv", summary, SUMMARY_FIELDS)
    paired = []
    for row in rows:
        full = by_key.get(("original", "full", row["seed"]))
        internal = by_key.get((row["negatives"], "full", row["seed"]))
        paired.append({"seed": row["seed"], "negatives": row["negatives"],
                       "features": row["features"],
                       "ap_difference_vs_original_full": float(row["average_precision"]) - float(full["average_precision"]) if full else "",
                       "roc_auc_difference_vs_original_full": float(row["roc_auc"]) - float(full["roc_auc"]) if full else "",
                       "ap_difference_vs_same_negative_full": float(row["average_precision"]) - float(internal["average_precision"]) if internal else ""})
    csv_write(out_dir / "paired_differences.csv", paired, PAIR_FIELDS)
    complete = [r for r in summary if r["n_seeds"] == len(SEEDS)]
    complete.sort(key=lambda r: r["ap_mean"], reverse=True)
    lines = ["# Controlled ActionFormer study — sampled-segment evaluation", "",
             f"**Completed models:** {len(rows)} / {len(plan)}", "",
             "**Important:** Same-video segment-level holdout, NOT leave-one-video-out.",
             "Frame scores and validation-selected frame thresholds are not comparable",
             "to the team's full-video LOVO F1@50 or event mAP.", "",
             "Normalization fitted on *fitting* segments only. Validation and test never",
             "participate in hard-negative selection, model fitting, or normalization.", "",
             "| Negatives | Feature set | Seeds | AP mean ± SD | ROC-AUC mean | Val-threshold test F1 |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in complete:
        lines.append(f"| {r['negatives']} | {r['features']} | {r['n_seeds']} | "
                     f"{r['ap_mean']:.4f} ± {r['ap_std']:.4f} | "
                     f"{r['roc_auc_mean']:.4f} | {r['val_threshold_f1_mean']:.4f} |")
    lines += ["", "Incomplete configurations and individual seeds: `summary.csv` and",
              "`per_run_results.csv`. Paired differences: `paired_differences.csv`.", "",
              "Hard negatives were selected outside all original baseline samples;"]
    lines += ["only the fitting partition changes. `reviewed_*` combines 'no-circling'",
              "and 'partial circling' examples, the latter being potentially ambiguous.", ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return len(rows)


def run_one(args, out_dir, study_id, negatives, features, seed, samples,
            columns, mean, std, spans, device):
    run_dir = out_dir / negatives / features / f"seed_{seed}"
    if completed_record(out_dir, study_id, negatives, features, seed,
                        require_checkpoint=not args.no_checkpoints):
        print(f"SKIP completed: {negatives}/{features}/seed_{seed}", flush=True)
        return False
    indices = feature_indices(columns, features)
    chosen = [columns[i] for i in indices]
    # Normalization was already fitted to fitting samples, once per strategy;
    # subsetting standardized features is algebraically identical to normalizing
    # only the selected columns with the same fitting examples.
    subset = samples if features == "full" else [
        {**s, "feats": s["feats"][indices, :].contiguous()} for s in samples]
    fitting = [s for s in subset if s["split"] == "train"]
    validation = [s for s in subset if s["split"] == "validation"]
    testing = [s for s in subset if s["split"] == "test"]
    if not fitting or not validation or not testing:
        raise ValueError(f"Empty fitting/validation/test partition in {negatives}")
    set_seed(seed)  # CRITICAL: before build_model, not only inside train_model.
    model, config = build_model(len(chosen))
    run_dir.mkdir(parents=True, exist_ok=True)
    run_spec = {"study_id": study_id, "seed": seed, "negatives": negatives,
                "features": features, "feature_columns": chosen,
                "excluded_columns": sorted(set(columns) - set(chosen)),
                "fitting_segments": len(fitting), "validation_segments": len(validation),
                "test_segments": len(testing), "epochs_max": EPOCHS,
                "batch_size": BATCH_SIZE, "learning_rate": LR,
                "patience": PATIENCE, "training_seed": seed,
                "fixed_validation_split_seed": SPLIT_SEED,
                "normalization": "fitting segments only",
                "full_video_postprocess_tuning": False,
                "test_unit": "sampled held-out frames (same-video)",
                "notes": "Only validated thresholds selected on validation frames; "
                         "not an external-video experiment."}
    (run_dir / "experiment.json").write_text(json.dumps(run_spec, indent=2) + "\n")
    print(f"\nSTART {negatives}/{features}/seed_{seed} "
          f"({len(chosen)} features, {len(fitting)} fit, "
          f"{len(validation)} validation, {len(testing)} test, {device})", flush=True)
    history = trainer.train_model(model, fitting, device, epochs=EPOCHS,
                                  batch_size=BATCH_SIZE, learning_rate=LR,
                                  validation_samples=validation,
                                  patience=PATIENCE, seed=seed)
    pl.DataFrame(history).write_csv(run_dir / "training_history.csv")
    best = next((row for row in reversed(history) if row.get("is_best")), None)
    if not args.no_checkpoints:
        torch.save({"model_state_dict": model.state_dict(), "config": config,
                    "feature_columns": chosen, "feature_set": "pairmotion",
                    "normalization_mean": mean[indices],
                    "normalization_std": std[indices],
                    "window": 256, "seed": seed,
                    "best_epoch": best["epoch"] if best else len(history),
                    "best_validation_loss": best["validation_loss"] if best else None,
                    "inference_settings": None}, run_dir / "model.pt")
        np.savez(run_dir / "normalization.npz", columns=np.asarray(chosen),
                 mean=mean[indices], std=std[indices])
    val_as_test = [{**s, "split": "test"} for s in validation]
    val_table = trainer.heldout_test(model, val_as_test, spans, device)
    chosen_threshold, validation_f1 = choose_validation_threshold(val_table)
    test = trainer.heldout_test(model, testing, spans, device)
    tuned_metrics = threshold_test_metrics(test, chosen_threshold)
    test.with_columns(pl.Series("predicted_at_validation_threshold",
                                test["circling_score"].to_numpy() >= chosen_threshold)).write_parquet(
                                    run_dir / "heldout_frames.parquet")
    report = {"model": "ActionFormer", "features": f"pairmotion_{features}",
              "negatives": negatives, "seed": seed,
              "feature_columns": chosen, "excluded_columns": sorted(set(columns) - set(chosen)),
              "baseline_split": str(BASELINE),
              "evaluation": "same-video heldout segments; frame scores",
              "limitation": "Test segments share source videos with fitting segments. Not LOVO.",
              "score_warning": "Original threshold-0.5 F1 may be zero; a separate "
                               "threshold was tuned on validation frames only."}
    result = write_report(run_dir, f"ActionFormer {negatives} / {features} / seed {seed}",
                          "ActionFormer", test, report,
                          notes=["Frozen test and validation samples shared by all configurations.",
                                 "Feature normalization uses fitting frames only.",
                                 "Frame-F1 threshold chosen without test labels; no full-video tuning."])
    metrics_path = run_dir / "metrics.json"
    metrics_data = json.loads(metrics_path.read_text())
    metrics_data["validation_threshold_selection"] = {
        "threshold": chosen_threshold, "validation_frame_f1": validation_f1,
        "validation_frame_count": val_table.height}
    metrics_data["test_at_validation_threshold"] = tuned_metrics
    metrics_path.write_text(json.dumps(metrics_data, indent=2) + "\n")
    record = {"seed": seed, "negatives": negatives, "features": features,
              "n_features": len(chosen), "fit_segments": len(fitting),
              "validation_segments": len(validation), "test_segments": len(testing),
              "epochs_trained": len(history),
              "best_epoch": best["epoch"] if best else len(history),
              "best_validation_loss": best["validation_loss"] if best else None,
              "average_precision": float(result["average_precision"]),
              "roc_auc": float(result["roc_auc"]),
              "val_threshold": chosen_threshold, "validation_f1": validation_f1,
              **tuned_metrics, "run_directory": str(run_dir), "study_id": study_id}
    # DONE.json is written LAST: a crash cannot make an incomplete run look finished.
    (run_dir / "DONE.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"DONE {negatives}/{features}/seed_{seed}: "
          f"AP={record['average_precision']:.4f} "
          f"AUC={record['roc_auc']:.4f} "
          f"F1@validation_threshold={record['test_f1_at_val_threshold']:.4f}", flush=True)
    del model, subset, fitting, validation, testing, val_table, test
    gc.collect()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("full", "core"), default="full",
                        help="full: 63 planned models; core: 27 planned models")
    parser.add_argument("--dry-run", action="store_true", help="Show plan and selected sample counts, no models")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="Stop after this many newly completed models; restart later to resume")
    parser.add_argument("--skip-reviewed", action="store_true",
                        help="Omit human-reviewed strategies if the local spreadsheet is unavailable")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="Save metrics and predictions, but no model weights")
    parser.add_argument("--device", default=None, help="cpu or cuda; default CUDA if available else CPU")
    args = parser.parse_args()
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be at least 1")
    if HARD_NEGATIVES != 0:
        parser.error("Set CIRCLING_HARD_NEGATIVES=0; this study controls negative sampling")
    if not BASELINE.is_file():
        parser.error(f"Missing frozen baseline split: {BASELINE}")
    if not REVIEWED.is_file() and not args.skip_reviewed:
        parser.error(f"Missing reviewed negatives spreadsheet: {REVIEWED}. "
                     "Restore the original local xlsx, or pass --skip-reviewed.")
    use_reviewed = not args.skip_reviewed
    feature_plan = FULL_FEATURES if args.tier == "full" else CORE_FEATURES
    negative_plan = [x for x in REPLACE_STRATEGIES if use_reviewed or not x.startswith("reviewed")]
    extra_plan = [x for x in ADD_STRATEGIES if use_reviewed or not x.startswith("reviewed")] if args.tier == "full" else []
    plan = [(n, f, seed) for n in negative_plan for f in feature_plan for seed in SEEDS]
    plan += [(n, "full", seed) for n in extra_plan for seed in SEEDS]
    print(f"Study: {len(plan)} runs, tier={args.tier}, {len(SEEDS)} controlled seeds", flush=True)
    print(f"Output directory: {OUT}", flush=True)

    baseline = pl.read_parquet(BASELINE)
    videos = tuple(sorted(baseline["video"].unique().to_list()))
    if videos != VIDEO_IDS:
        raise SystemExit(f"Unexpected baseline videos {videos}; expected {VIDEO_IDS}")
    workdir = mstcn_workdir()
    meta = load_meta(workdir, VIDEO_IDS)
    counts = video_frames(VIDEO_IDS)
    spans = read_annotations(FINALIZED_ANNOTATIONS, videos=VIDEO_IDS)
    fit_keys, val_keys, test_keys = fixed_roles(baseline, meta, spans, counts, workdir)
    bank = build_candidate_banks(baseline, meta, spans, counts, val_keys, test_keys, workdir)
    if use_reviewed:
        attach_reviewed_labels(bank)
    strategies, selection = build_strategies(baseline, fit_keys, val_keys, test_keys,
                                             bank, use_reviewed)
    if args.dry_run:
        print("\nDRY RUN: no models trained. Strategy sample counts above.", flush=True)
        for negatives, features, seed in plan:
            print(f"PLAN {negatives}/{features}/seed_{seed}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    signature = source_signature()
    settings = {"protocol": "large_study_v1_fit_only_normalization",
                "tier": args.tier, "seeds": SEEDS, "split_seed": SPLIT_SEED,
                "candidate_seed": CANDIDATE_SEED, "epochs_max": EPOCHS,
                "batch_size": BATCH_SIZE, "learning_rate": LR, "patience": PATIENCE,
                "feature_groups": dict(FEATURE_GROUPS),
                "source_signature": signature, "use_reviewed": use_reviewed,
                "selection_counts": selection,
                "important_limitations": [
                    "The heldout evaluation is same-video sampled-segment, not LOVO.",
                    "Labels treat all frames outside annotated events as background.",
                    "Reviewed partial-circling intervals may have ambiguous labels.",
                    "Frame average precision is measured on a balanced sampled test set.",
                    "Normalization uses fitting frames only, unlike early single-run scripts.",
                    "Full-video postprocessing / event F1 and mAP are not evaluated.",
                ]}
    study_id = hashlib.sha256(json.dumps(settings, sort_keys=True, default=list).encode()).hexdigest()[:20]
    existing_manifest = OUT / "manifest.json"
    if existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text())
        if previous.get("study_id") != study_id:
            parser.error("Existing output directory belongs to a different study protocol/data. "
                         f"Rename {OUT} before running the modified study.")
    else:
        settings["study_id"] = study_id
        try:
            settings["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            settings["git_commit"] = None
        existing_manifest.write_text(json.dumps(settings, indent=2, default=list) + "\n")
    for name, table in strategies.items():
        table.write_parquet(OUT / f"segments_{name}.parquet")
    done = summarize(OUT, plan, study_id, require_checkpoint=not args.no_checkpoints)
    print(f"Existing completed runs: {done}/{len(plan)}", flush=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    new_runs = 0
    for negatives in dict.fromkeys(n for n, _, _ in plan):
        work = [(n, f, seed) for n, f, seed in plan if n == negatives and
                completed_record(OUT, study_id, n, f, seed,
                                 require_checkpoint=not args.no_checkpoints) is None]
        if not work:
            continue
        samples, columns, mean, std = trainer.prepare_samples(
            strategies[negatives], meta, spans, counts, workdir, "pairmotion")
        if len(columns) != 132:
            raise RuntimeError(f"Expected 132 base features, got {len(columns)}")
        for _, features, seed in work:
            trained = run_one(args, OUT, study_id, negatives, features, seed,
                              samples, columns, mean, std, spans, device)
            if trained:
                new_runs += 1
                completed_count = summarize(OUT, plan, study_id,
                                            require_checkpoint=not args.no_checkpoints)
                print(f"PROGRESS: {completed_count}/{len(plan)} complete", flush=True)
                if args.max_runs is not None and new_runs >= args.max_runs:
                    print(f"Reached --max-runs={args.max_runs}; safe to resume with same command", flush=True)
                    return
        del samples, columns, mean, std
        gc.collect()
    completed_count = summarize(OUT, plan, study_id,
                                require_checkpoint=not args.no_checkpoints)
    print(f"\nALL {completed_count} RUNS COMPLETE | summary: {OUT / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
