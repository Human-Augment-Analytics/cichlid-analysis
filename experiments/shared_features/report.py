"""Circling-versus-background feature report (exploration only).

Per feature and per video: missingness, class means, standardized effect size
(Cohen's d) and single-feature ROC AUC. Features that rank the same way in both
videos are the candidates worth testing; held-out results decide what is kept.

    uv run python -m experiments.shared_features.report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from feature_engineering.pair_motion.feature_groups import ALL_MODEL_FEATURES, FEATURE_GROUPS

from .common import EXPERIMENT_DIR, MODEL_FRAMES, SEED
from .dataset import AGGREGATE_COLUMNS

REPORT_DIR = EXPERIMENT_DIR / "feature_report"
GROUP_OF = {column: group for group, columns in FEATURE_GROUPS.items() for column in columns}


def cohens_d(positive: np.ndarray, negative: np.ndarray) -> float:
    if len(positive) < 2 or len(negative) < 2:
        return float("nan")
    pooled = np.sqrt((positive.var(ddof=1) + negative.var(ddof=1)) / 2.0)
    return float((positive.mean() - negative.mean()) / pooled) if pooled > 0 else float("nan")


def feature_table(frames: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for video, part in frames.groupby("VideoID"):
        label = part["label"].to_numpy() == 1
        for feature in features:
            values = part[feature].to_numpy(dtype=float)
            ok = np.isfinite(values)
            pos, neg = values[ok & label], values[ok & ~label]
            auc = float("nan")
            if len(pos) and len(neg):
                auc = float(roc_auc_score(label[ok], values[ok]))
            rows.append(
                {
                    "feature": feature,
                    "group": GROUP_OF.get(feature, "all_pair_aggregate"),
                    "video": video,
                    "missing_pos": float(1 - ok[label].mean()) if label.any() else float("nan"),
                    "missing_neg": float(1 - ok[~label].mean()),
                    "mean_pos": float(pos.mean()) if len(pos) else float("nan"),
                    "mean_neg": float(neg.mean()) if len(neg) else float("nan"),
                    "cohens_d": cohens_d(pos, neg),
                    "auc": auc,
                }
            )
    return pd.DataFrame(rows)


def summarize(per_video: pd.DataFrame) -> pd.DataFrame:
    wide = per_video.pivot(index="feature", columns="video", values=["auc", "cohens_d", "missing_pos"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    out = wide.reset_index()
    out["group"] = out["feature"].map(lambda f: GROUP_OF.get(f, "all_pair_aggregate"))
    auc_cols = [c for c in out if c.startswith("auc_")]
    # Signed strength: distance from chance, positive when both videos agree in direction.
    centered = out[auc_cols] - 0.5
    out["same_direction"] = (np.sign(centered).nunique(axis=1) == 1)
    out["min_abs_auc_gap"] = centered.abs().min(axis=1)
    return out.sort_values("min_abs_auc_gap", ascending=False)


def redundancy(frames: pd.DataFrame, features: list[str], threshold: float = 0.95) -> pd.DataFrame:
    sample = frames[features].sample(min(len(frames), 100_000), random_state=SEED)
    corr = sample.corr().abs()
    pairs = []
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            value = corr.loc[a, b]
            if value >= threshold:
                pairs.append({"feature_a": a, "feature_b": b, "abs_corr": float(value)})
    return pd.DataFrame(pairs, columns=["feature_a", "feature_b", "abs_corr"]).sort_values("abs_corr", ascending=False)


def write_report(frames: pd.DataFrame, out_dir: Path = REPORT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    features = ALL_MODEL_FEATURES + AGGREGATE_COLUMNS
    per_video = feature_table(frames, features)
    summary = summarize(per_video)
    redundant = redundancy(frames, features)
    per_video.to_csv(out_dir / "feature_stats_per_video.csv", index=False)
    summary.to_csv(out_dir / "feature_summary.csv", index=False)
    redundant.to_csv(out_dir / "redundant_pairs.csv", index=False)

    prevalence = frames.groupby("VideoID")["label"].agg(["size", "sum"])
    lines = [
        "# Circling vs background: shared feature report",
        "",
        "Exploration only. Rows are video frames with a valid pair. AUC is the chance that a",
        "circling frame has a higher feature value than a background frame (0.5 = no signal).",
        "Do not pick features with the held-out video; use this to choose candidates.",
        "",
        "| Video | Frames with a valid pair | Circling frames |",
        "| --- | ---: | ---: |",
        *[f"| {v} | {int(r['size']):,} | {int(r['sum']):,} |" for v, r in prevalence.iterrows()],
        "",
        "## Strongest features (weakest-video distance from AUC 0.5, same direction in both)",
        "",
        "| Feature | Group | AUC 0028 | AUC 0031 | d 0028 | d 0031 | Missing (circling) 0028/0031 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    top = summary[summary["same_direction"]].head(25)
    for _, r in top.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['group']} | {r['auc_0028_vid']:.3f} | {r['auc_0031_vid']:.3f} | "
            f"{r['cohens_d_0028_vid']:.2f} | {r['cohens_d_0031_vid']:.2f} | "
            f"{r['missing_pos_0028_vid']:.1%} / {r['missing_pos_0031_vid']:.1%} |"
        )
    flipped = summary[~summary["same_direction"]]
    lines += ["", f"## Features whose direction differs between videos: {len(flipped)}", ""]
    lines += [f"- `{f}`" for f in flipped["feature"].head(20)]
    lines += ["", f"## Near-duplicate features (|r| >= 0.95): {len(redundant)} pairs", ""]
    lines += [f"- `{r.feature_a}` ~ `{r.feature_b}` ({r.abs_corr:.3f})" for r in redundant.head(25).itertuples()]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=MODEL_FRAMES)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    path = write_report(pd.read_parquet(args.frames), args.out_dir)
    print(path.read_text())


if __name__ == "__main__":
    main()
