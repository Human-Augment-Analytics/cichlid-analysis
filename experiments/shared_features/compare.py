"""Side-by-side comparison of shipped detectors and held-out (LOVO) variants.

Shipped detectors (`outputs/predictions/{rule_based,logistic_regression,xgboost}`) were
fitted or tuned on the annotated videos, so their scores are in-sample references, not
held-out evidence (the rule-based authors report 0028 as their held-out video).

    uv run python -m experiments.shared_features.compare
"""

from __future__ import annotations

import pandas as pd

from .common import EXPERIMENT_DIR, PREDICTION_ROOT, VIDEO_FRAMES, VIDEOS, evaluate_video, read_spans, summarize
from .lovo import LOVO_DIR

SHIPPED = ["rule_based", "logistic_regression", "xgboost"]
COLUMNS = ["f1@10", "f1@25", "f1@50", "edit_score", "frame_precision", "frame_recall", "frame_f1", "pred_segments"]


def shipped_scores(name: str, spans) -> list[dict]:
    rows = []
    for video in VIDEOS:
        path = PREDICTION_ROOT / name / f"{video}.csv"
        table = pd.read_csv(path)
        predicted = list(table[["StartFrame", "EndFrame"]].itertuples(index=False, name=None))
        rows.append({"video": video, **evaluate_video(spans[video], predicted, VIDEO_FRAMES[video])})
    rows.append({"video": "overall", **summarize([{k: v for k, v in r.items() if k != "video"} for r in rows])})
    return rows


def fmt(row) -> str:
    return " | ".join(f"{row[c]:.3f}" if c != "pred_segments" else f"{int(row[c])}" for c in COLUMNS)


def main() -> None:
    spans = read_spans()
    lovo = pd.read_csv(LOVO_DIR / "results.csv")
    header = "| Detector | " + " | ".join(["F1@10", "F1@25", "F1@50", "Edit", "Frame P", "Frame R", "Frame F1", "Segs"]) + " |"
    rule = "| --- |" + " ---: |" * len(COLUMNS)
    lines = ["# Shipped detectors vs shared features", "", "## Shipped detectors (in-sample reference, not held out)", "", header, rule]
    for name in SHIPPED:
        rows = shipped_scores(name, spans)
        for row in rows:
            lines.append(f"| `{name}` ({row['video']}) | {fmt(row)} |")
    lines += ["", "## Held-out leave-one-video-out (pooled over both held-out videos)", "", header, rule]
    overall = lovo[lovo["video"] == "overall"].sort_values("f1@50", ascending=False)
    for _, row in overall.iterrows():
        lines.append(f"| `{row['variant']}` | {fmt(row)} |")
    lines += ["", "## Per held-out video, best variants", "", header, rule]
    best = overall.head(8)["variant"].tolist()
    for name in best:
        for video in VIDEOS:
            row = lovo[(lovo["variant"] == name) & (lovo["video"] == video)].iloc[0]
            lines.append(f"| `{name}` (test {video}) | {fmt(row)} |")
    path = EXPERIMENT_DIR / "comparison.md"
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text())


if __name__ == "__main__":
    main()
