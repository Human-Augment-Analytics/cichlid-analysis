"""Leave-one-video-out (LOVO) experiments for shared features.

One fold per video: hold it out and fit on the remaining videos (with the current
two videos, fit 0031 -> test 0028 and fit 0028 -> test 0031). Everything that is
fitted lives inside a fold and sees only the training videos: imputer, scaler,
model weights, and the threshold / minimum-length postprocessing (chosen from
blocked out-of-fold scores inside each training video). Only the combined
held-out predictions of all folds are scored.

    uv run python -m experiments.shared_features.lovo --variants all
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering.pair_motion.feature_groups import ALL_MODEL_FEATURES, FEATURE_GROUPS

from .common import (
    EXPERIMENT_DIR, MODEL_FRAMES, PREDICTION_ROOT, SEED, VIDEO_FRAMES, VIDEOS,
    evaluate_video, read_spans, scores_to_spans, summarize,
)
from .baselines import BASELINE_FRAMES, WIN_COLUMNS
from .dataset import AGGREGATE_COLUMNS

LOVO_DIR = EXPERIMENT_DIR / "lovo"
INNER_FOLDS = 4
INNER_BUFFER_FRAMES = 300  # unused frames between inner train and inner validation blocks
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
MIN_LENGTHS = (0, 90, 150, 300)
RULE_GRID = {"w4s_close_frac": (0.5, 0.6, 0.7, 0.8, 0.9), "w4s_orbit_turns_abs": (0.25, 0.5, 0.75, 1.0, 1.5)}


@dataclass(frozen=True)
class Variant:
    name: str
    model: str  # "rule", "logreg", "hgb" (sklearn boosting) or "xgb"
    features: tuple[str, ...]
    description: str = ""


def _group_features(*groups: str) -> tuple[str, ...]:
    return tuple(column for group in groups for column in FEATURE_GROUPS[group])


def lr_columns() -> list[str]:
    """Names of the existing logistic-regression features (read from the baseline table)."""
    if not BASELINE_FRAMES.exists():
        return []
    return [c for c in pd.read_parquet(BASELINE_FRAMES).columns if c.startswith("lr_")]


def build_variants() -> dict[str, Variant]:
    groups = list(FEATURE_GROUPS)
    everything = tuple(ALL_MODEL_FEATURES)
    variants = {"rule_shared": Variant("rule_shared", "rule", tuple(RULE_GRID), "4 s close fraction + 4 s orbit turns")}
    for model in ("logreg", "hgb", "xgb"):
        variants[f"{model}_all"] = Variant(f"{model}_all", model, everything, "closest pair, all groups")
        variants[f"{model}_all_agg"] = Variant(
            f"{model}_all_agg", model, everything + tuple(AGGREGATE_COLUMNS), "closest pair + all-pair aggregates"
        )
        variants[f"{model}_v1_features"] = Variant(
            f"{model}_v1_features", model,
            _group_features("distance_proximity", "orientation_facing", "speed_closing"),
            "first-half features only (no angular movement, no windows)",
        )
        for group in groups:
            others = tuple(g for g in groups if g != group)
            variants[f"{model}_no_{group}"] = Variant(f"{model}_no_{group}", model, _group_features(*others), f"all but {group}")
        lr_cols, win_cols = tuple(lr_columns()), tuple(WIN_COLUMNS)
        variants[f"{model}_their_lr102"] = Variant(f"{model}_their_lr102", model, lr_cols, "existing logistic-regression features")
        variants[f"{model}_their_win155"] = Variant(f"{model}_their_win155", model, win_cols, "existing XGBoost window signals (stride 4)")
        variants[f"{model}_their_both"] = Variant(f"{model}_their_both", model, lr_cols + win_cols, "both existing feature sets")
        variants[f"{model}_mine_plus_their"] = Variant(
            f"{model}_mine_plus_their", model, everything + tuple(AGGREGATE_COLUMNS) + lr_cols + win_cols, "shared + existing features")
        if model == "logreg":
            for group in groups:
                variants[f"{model}_only_{group}"] = Variant(f"{model}_only_{group}", model, _group_features(group), f"only {group}")
    return variants


def make_model(kind: str, quick: bool = False):
    if kind == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=400),
        )
    if kind == "hgb":
        # scikit-learn's gradient boosting: handles NaN natively and needs no OpenMP install.
        return HistGradientBoostingClassifier(
            max_iter=40 if quick else 150, learning_rate=0.1, max_depth=4, min_samples_leaf=40, random_state=SEED,
        )
    if kind == "xgb":
        from xgboost import XGBClassifier  # needs the OpenMP runtime (brew install libomp)

        return XGBClassifier(
            n_estimators=80 if quick else 300, max_depth=4, learning_rate=0.08, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=8,
        )
    raise ValueError(kind)


def full_length_scores(frames: np.ndarray, scores: np.ndarray, count: int) -> np.ndarray:
    out = np.zeros(count, dtype=np.float32)
    out[frames] = scores
    return out


def evaluate_spans(video: str, spans, predicted):
    return evaluate_video(spans[video], predicted, VIDEO_FRAMES[video])


def objective(metrics: dict) -> tuple[float, float]:
    return (metrics["f1@50"], metrics["frame_f1"])


def folds(videos=VIDEOS):
    """Each video is held out once and the remaining videos are the training set."""
    for test_video in videos:
        yield [video for video in videos if video != test_video], test_video


def full_scores_by_video(table: pd.DataFrame, scores: np.ndarray) -> dict[str, np.ndarray]:
    videos, frames = table["VideoID"].to_numpy(), table["FrameNum"].to_numpy()
    return {
        video: full_length_scores(frames[videos == video], scores[videos == video], VIDEO_FRAMES[video])
        for video in pd.unique(videos)
    }


def pooled_objective(full_by_video: dict[str, np.ndarray], spans, threshold: float, min_length: int):
    """Objective pooled over the training videos, as `summarize` pools the held-out ones."""
    per_video = [
        evaluate_spans(video, spans, scores_to_spans(full, threshold, min_length))
        for video, full in full_by_video.items()
    ]
    return objective(summarize(per_video))


def choose_postprocess(table: pd.DataFrame, scores: np.ndarray, spans):
    """Pick (threshold, min_length) on the training videos only."""
    full = full_scores_by_video(table, scores)
    best, best_key = (0.5, 0), (-1.0, -1.0)
    for threshold, min_length in itertools.product(THRESHOLDS, MIN_LENGTHS):
        key = pooled_objective(full, spans, float(threshold), min_length)
        if key > best_key:
            best, best_key = (float(threshold), int(min_length)), key
    return best, best_key


def inner_blocks(frame_numbers: np.ndarray, folds: int = INNER_FOLDS):
    edges = np.quantile(frame_numbers, np.linspace(0, 1, folds + 1))
    for index in range(folds):
        lo, hi = edges[index], edges[index + 1] + (1 if index == folds - 1 else 0)
        valid = (frame_numbers >= lo) & (frame_numbers < hi)
        train = (frame_numbers < lo - INNER_BUFFER_FRAMES) | (frame_numbers >= hi + INNER_BUFFER_FRAMES)
        yield train, valid


def inner_folds(table: pd.DataFrame, folds: int = INNER_FOLDS):
    """Blocked inner folds, cut separately inside each training video."""
    videos, frames = table["VideoID"].to_numpy(), table["FrameNum"].to_numpy()
    blocks = {video: list(inner_blocks(frames[videos == video], folds)) for video in pd.unique(videos)}
    for index in range(folds):
        train, valid = np.zeros(len(table), dtype=bool), np.zeros(len(table), dtype=bool)
        for video, video_blocks in blocks.items():
            mask = videos == video
            train[mask], valid[mask] = video_blocks[index]
        yield train, valid


def rule_scores(table: pd.DataFrame, close_min: float, turns_min: float) -> np.ndarray:
    return ((table["w4s_close_frac"] >= close_min) & (table["w4s_orbit_turns_abs"] >= turns_min)).to_numpy(dtype=np.float32)


def fit_rule(train: pd.DataFrame, spans):
    best, best_key = None, (-1.0, -1.0)
    for close_min, turns_min in itertools.product(*RULE_GRID.values()):
        full = full_scores_by_video(train, rule_scores(train, close_min, turns_min))
        for min_length in MIN_LENGTHS:
            key = pooled_objective(full, spans, 0.5, min_length)
            if key > best_key:
                best, best_key = (close_min, turns_min, min_length), key
    return {"close_min": best[0], "turns_min": best[1], "min_length": best[2], "train_objective": best_key}


def run_fold(variant: Variant, frames: pd.DataFrame, train_videos: list[str], test_video: str, spans, quick: bool):
    train = frames[frames["VideoID"].isin(train_videos)].reset_index(drop=True)
    test = frames[frames["VideoID"] == test_video].reset_index(drop=True)
    info: dict = {"train_videos": train_videos, "test_video": test_video, "features": list(variant.features)}
    if variant.model == "rule":
        params = fit_rule(train, spans)
        info.update(params)
        test_scores = rule_scores(test, params["close_min"], params["turns_min"])
        threshold, min_length = 0.5, params["min_length"]
        return test, test_scores, threshold, min_length, info, None

    columns = list(variant.features)
    x_train, y_train = train[columns].to_numpy(dtype=np.float32), train["label"].to_numpy()
    oof = np.full(len(train), np.nan, dtype=np.float32)
    for inner_train, inner_valid in inner_folds(train):
        inner = make_model(variant.model, quick)
        inner.fit(x_train[inner_train], y_train[inner_train])
        oof[inner_valid] = inner.predict_proba(x_train[inner_valid])[:, 1]
    known = np.isfinite(oof)
    (threshold, min_length), key = choose_postprocess(train.loc[known], oof[known], spans)
    info.update({"threshold": threshold, "min_length": min_length, "inner_objective": key})

    model = make_model(variant.model, quick)
    model.fit(x_train, y_train)
    x_test = test[columns].to_numpy(dtype=np.float32)
    test_scores = model.predict_proba(x_test)[:, 1].astype(np.float32)
    return test, test_scores, threshold, min_length, info, model


def feature_importance(model, variant: Variant, test: pd.DataFrame, repeats: int = 2):
    """Held-out permutation importance: drop in average precision when a column is shuffled."""
    rng = np.random.default_rng(SEED)
    sample = test.sample(min(len(test), 60_000), random_state=SEED)
    columns = list(variant.features)
    x = sample[columns].to_numpy(dtype=np.float32)
    y = sample["label"].to_numpy()
    if y.sum() == 0:
        return pd.DataFrame(columns=["feature", "importance"])
    base = average_precision_score(y, model.predict_proba(x)[:, 1])
    rows = []
    for index, column in enumerate(columns):
        drops = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[:, index] = rng.permutation(shuffled[:, index])
            drops.append(base - average_precision_score(y, model.predict_proba(shuffled)[:, 1]))
        rows.append({"feature": column, "importance": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def run_variant(variant: Variant, frames: pd.DataFrame, spans, quick: bool, importance: bool, out_dir: Path):
    per_video, fold_info, imports, coefs = [], [], [], []
    prediction_dir = PREDICTION_ROOT / f"lovo_{variant.name}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for train_videos, test_video in folds():
        test, scores, threshold, min_length, info, model = run_fold(variant, frames, train_videos, test_video, spans, quick)
        full = full_length_scores(test["FrameNum"].to_numpy(), scores, VIDEO_FRAMES[test_video])
        predicted = scores_to_spans(full, threshold, min_length)
        pd.DataFrame([(test_video, s, e) for s, e in predicted], columns=["VideoID", "StartFrame", "EndFrame"]).to_csv(
            prediction_dir / f"{test_video}.csv", index=False
        )
        metrics = evaluate_spans(test_video, spans, predicted)
        metrics["video"] = test_video
        labels = test["label"].to_numpy()
        metrics["frame_ap_rows_with_pair"] = float(average_precision_score(labels, scores)) if labels.any() else float("nan")
        per_video.append(metrics)
        fold_info.append(info)
        if model is not None:
            if importance:
                table = feature_importance(model, variant, test)
                table["fold_test_video"] = test_video
                imports.append(table)
            if variant.model == "logreg":
                weights = model[-1].coef_[0]
                coefs.append(pd.DataFrame({"feature": list(variant.features), "coefficient": weights, "fold_test_video": test_video}))
    overall = summarize(per_video)
    (out_dir / "folds").mkdir(parents=True, exist_ok=True)
    (out_dir / "folds" / f"{variant.name}.json").write_text(json.dumps(fold_info, indent=2, default=float) + "\n")
    if imports:
        pd.concat(imports).to_csv(out_dir / "folds" / f"{variant.name}_permutation_importance.csv", index=False)
    if coefs:
        pd.concat(coefs).to_csv(out_dir / "folds" / f"{variant.name}_coefficients.csv", index=False)
    rows = [{"variant": variant.name, "video": m["video"], **{k: v for k, v in m.items() if k != "video"}} for m in per_video]
    rows.append({"variant": variant.name, "video": "overall", **overall})
    return rows


RESULT_COLUMNS = ["variant", "video", "f1@10", "f1@25", "f1@50", "edit_score", "frame_precision", "frame_recall",
                  "frame_f1", "gt_segments", "pred_segments", "mof"]


def write_summary(results: pd.DataFrame, variants: dict[str, Variant], out_dir: Path) -> None:
    overall = results[results["video"] == "overall"].set_index("variant")
    lines = [
        "# Leave-one-video-out results (held-out predictions only)",
        "",
        f"Folds: each of {', '.join(VIDEOS)} is held out once. Overall pools the held-out videos.",
        "Annotations: `merged_spans.csv` (139 events). Segment F1 is IoU based (see `evaluation/`).",
        "",
        "| Variant | F1@10 | F1@25 | F1@50 | Edit | Frame P | Frame R | Frame F1 | Pred segs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in overall.sort_values("f1@50", ascending=False).iterrows():
        lines.append(
            f"| `{name}` | {row['f1@10']:.3f} | {row['f1@25']:.3f} | {row['f1@50']:.3f} | {row['edit_score']:.1f} | "
            f"{row['frame_precision']:.3f} | {row['frame_recall']:.3f} | {row['frame_f1']:.3f} | {int(row['pred_segments'])} |"
        )
    lines += ["", "Per-video numbers are in `results.csv`.", ""]
    (out_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=MODEL_FRAMES)
    parser.add_argument("--variants", nargs="+", default=["all"], help="variant names, or 'all'")
    parser.add_argument("--quick", action="store_true", help="fewer trees, for smoke tests")
    parser.add_argument("--no-importance", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=LOVO_DIR)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    variants = build_variants()
    if args.list:
        print("\n".join(f"{v.name:42s} {v.model:7s} {len(v.features):3d} features  {v.description}" for v in variants.values()))
        return
    names = list(variants) if args.variants == ["all"] else args.variants
    unknown = [n for n in names if n not in variants]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")
    frames = pd.read_parquet(args.frames)
    if BASELINE_FRAMES.exists():
        frames = frames.merge(pd.read_parquet(BASELINE_FRAMES), on=["VideoID", "FrameNum"], how="left", validate="1:1")
    spans = read_spans()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.csv"
    existing = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    for name in names:
        print(f"== {name}", flush=True)
        rows = run_variant(
            variants[name], frames, spans, args.quick,
            importance=(name.endswith("_all") or name.endswith("_all_agg")) and not args.no_importance,
            out_dir=args.out_dir,
        )
        new = pd.DataFrame(rows)
        existing = pd.concat([existing[existing.get("variant") != name] if len(existing) else existing, new], ignore_index=True)
        existing.to_csv(results_path, index=False)
        row = new[new["video"] == "overall"].iloc[0]
        print(f"   F1@50 {row['f1@50']:.3f}  edit {row['edit_score']:.1f}  frameF1 {row['frame_f1']:.3f}  segs {int(row['pred_segments'])}", flush=True)
    write_summary(existing, variants, args.out_dir)


if __name__ == "__main__":
    main()
