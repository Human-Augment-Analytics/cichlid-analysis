"""Single entry point for reproducible ActionFormer circling experiments.

Examples (run from the repository root):

    uv run python -m experiments.actionformer.run_study window --window 512 --max-folds 1
    uv run python -m experiments.actionformer.run_study window --window 1024
    uv run python -m experiments.actionformer.run_study proposal-caps --caps 1000 2000 3000 --max-folds 1
    uv run python -m experiments.actionformer.run_study summary

This entry point never requests --final-test; 0037_vid is excluded from development.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


CV_PATHS = {
    256: Path("outputs/actionformer_video_cv"),
    512: Path("outputs/actionformer_cv_window512"),
    1024: Path("outputs/actionformer_cv_window1024"),
}


def _run_module(name: str, arguments: list[str]) -> None:
    cmd = [sys.executable, "-m", name, *arguments]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _window(args: argparse.Namespace) -> None:
    out = args.output
    if out is None:
        # Original baseline was produced before the new manifest fields were added.
        # Do not try to resume/overwrite it using the new experiment runner.
        out = (Path("outputs/actionformer_cv_window256_reproduction")
               if args.window == 256 else CV_PATHS[args.window])
    batch = args.batch_size if args.batch_size is not None else (2 if args.window == 1024 else 4)
    flags = ["--holdout", "0037_vid", "--window", str(args.window),
             "--batch-size", str(batch), "--output", str(out)]
    if args.epochs is not None:
        flags.extend(["--epochs", str(args.epochs)])
    if args.max_folds is not None:
        flags.extend(["--max-folds", str(args.max_folds)])
    if args.device is not None:
        flags.extend(["--device", args.device])
    if args.dry_run:
        flags.append("--dry-run")
    print("Development-only seven-fold video CV; final holdout 0037_vid is not evaluated.", flush=True)
    _run_module("experiments.actionformer.run_video_cv", flags)


def _proposal_caps(args: argparse.Namespace) -> None:
    flags = ["--cv-dir", str(args.cv_dir), "--output", str(args.output),
             "--caps", *map(str, args.caps)]
    if args.max_folds is not None:
        flags.extend(["--max-folds", str(args.max_folds)])
    if args.device is not None:
        flags.extend(["--device", args.device])
    _run_module("experiments.actionformer.run_proposal_cap", flags)


def _postprocess(args: argparse.Namespace) -> None:
    flags = ["--cv-dir", str(args.cv_dir), "--cap-dir", str(args.cap_dir),
             "--output", str(args.output),
             "--scores", *map(str, args.scores),
             "--durations", *map(str, args.durations),
             "--merges", *map(str, args.merges)]
    if args.max_folds is not None:
        flags.extend(["--max-folds", str(args.max_folds)])
    _run_module("experiments.actionformer.run_postprocess_sweep", flags)


def _hard_negatives(args: argparse.Namespace) -> None:
    """Train with the same 1024-frame setup, changing only negative sampling."""
    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit('--fraction must be > 0 and <= 1; 0.0 is the existing baseline')
    percent = round(args.fraction * 100)
    if abs(percent / 100 - args.fraction) > 1e-9:
        raise SystemExit('Use a percentage in whole-number increments (e.g., 0.25 or 0.50)')
    out = args.output or Path(f"outputs/actionformer_cv_window1024_hn{percent}")
    flags = ["--holdout", "0037_vid", "--window", "1024",
             "--batch-size", str(args.batch_size),
             "--hard-negative-fraction", str(args.fraction),
             "--output", str(out)]
    if args.epochs is not None:
        flags.extend(["--epochs", str(args.epochs)])
    if args.max_folds is not None:
        flags.extend(["--max-folds", str(args.max_folds)])
    if args.device:
        flags.extend(["--device", args.device])
    if args.dry_run:
        flags.append("--dry-run")
    print("Hard-negative CV, frozen protocol: window=1024, batch=2 by default; NO final test.", flush=True)
    _run_module("experiments.actionformer.run_video_cv", flags)


def _mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)



def _review(args: argparse.Namespace) -> None:
    flags = ["--cv-dir", str(args.cv_dir), "--post-dir", str(args.post_dir),
             "--output", str(args.output), "--videos", *args.videos,
             "--cases-per-type", str(args.cases_per_type),
             "--context-seconds", str(args.context_seconds)]
    _run_module("experiments.actionformer.review_predictions", flags)


def _summary(args: argparse.Namespace) -> None:
    lines = [
        "# ActionFormer circling behavior experiments",
        "",
        "Seven-fold video-disjoint cross-validation. Development videos: 0002, 0007a,",
        "0007b, 0013, 0018, 0028, 0031. Final holdout `0037_vid` is excluded.",
        "",
        "## Temporal window comparison",
        "",
        "| Window (frames) | Mean mAP (tIoU 10/30/50/70) | Mean event F1@50 | Mean frame precision | Folds |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for window, path in CV_PATHS.items():
        filename = path / "cv_results.csv"
        if not filename.is_file():
            lines.append(f"| {window} | — | — | — | not available |")
            continue
        with filename.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            lines.append(f"| {window} | — | — | — | 0 |")
            continue
        lines.append(
            f"| {window} | {_mean(rows, 'map_avg'):.4f} | {_mean(rows, 'f1@50'):.4f} | "
            f"{_mean(rows, 'frame_precision'):.4f} | {len(rows)}/7 |"
        )
    lines.extend([
        "",
        "Window experiments were not identical except for window duration: sampled",
        "training segments, selected postprocessing and the 1024-frame batch size (2",
        "rather than 4) also differed. Interpret comparisons accordingly.",
        "",
        "## Proposal-cap comparison (frozen 1024-frame checkpoints)",
        "",
    ])
    cap_summary = args.cap_dir / "cap_summary.json"
    if cap_summary.exists():
        summaries = json.loads(cap_summary.read_text())
        lines.extend([
            "| Cap | Mean mAP | Mean event F1@50 | Mean frame precision | Predicted events | Folds |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for cap in sorted(summaries, key=int):
            row = summaries[cap]
            m = row["mean"]
            lines.append(
                f"| {cap} | {m['map_avg']:.4f} | {m['f1@50']:.4f} | "
                f"{m['frame_precision']:.4f} | {row['total_pred_segments']} | "
                f"{row['folds']}/7 |"
            )
    else:
        lines.append("Not run yet.")
    pp = Path("outputs/actionformer_postprocess_1024/summary.md")
    if pp.exists():
        report = pp.read_text().strip().splitlines()
        if report and report[0].startswith("# "):
            report = report[1:]
        lines.extend(["", "## Postprocessing experiment", "", *report, ""])
    lines.extend([
        "",
        "## Hard-negative training comparison (1024 frames)",
        "",
        "| Hard-negative fraction | Mean mAP | Mean event F1@50 | Mean frame precision | Folds |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for fraction, path in (
        (0.0, CV_PATHS[1024]),
        (0.25, Path("outputs/actionformer_cv_window1024_hn25")),
        (0.50, Path("outputs/actionformer_cv_window1024_hn50")),
    ):
        file = path / "cv_results.csv"
        if not file.is_file():
            lines.append(f"| {fraction:.2f} | — | — | — | unavailable |")
            continue
        with file.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            lines.append(f"| {fraction:.2f} | — | — | — | 0/7 |")
        else:
            lines.append(
                f"| {fraction:.2f} | {_mean(rows, 'map_avg'):.4f} | "
                f"{_mean(rows, 'f1@50'):.4f} | {_mean(rows, 'frame_precision'):.4f} | "
                f"{len(rows)}/7 |"
            )
    lines.extend([
        "",
        "Hard-negative rows use the standard per-fold postprocessing selection;",
        "they should be compared with the original 1024-frame CV row, rather than",
        "the optimistically selected postprocessing grid winner.",
    ])
    lines.extend([
        "",
        "Proposal-cap results reuse the same trained checkpoints and fold-specific",
        "postprocessing settings. The 1,000-cap row must reproduce original CV",
        "results before interpreting other caps.",
        "",
        "## Reproduce experiments",
        "",
        "```bash",
        "uv run python -m experiments.actionformer.run_study window --window 512",
        "uv run python -m experiments.actionformer.run_study window --window 1024",
        "uv run python -m experiments.actionformer.run_study proposal-caps --caps 1000 2000 3000",
        "uv run python -m experiments.actionformer.run_study postprocess",
        "uv run python -m experiments.actionformer.run_study hard-negatives --fraction 0.25",
        "uv run python -m experiments.actionformer.run_study summary",
        "```",
        "",
        "The original 256-frame baseline is retained under",
        "`outputs/actionformer_video_cv`. The unified runner uses a different output",
        "folder when reproducing the old 256-frame experiment to avoid modifying it.",
        "",
    ])
    body = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body)
    print(body)
    print("\nSummary written to", args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="experiment", required=True)

    windows = sub.add_parser("window", help="Run/resume video-disjoint window-size CV")
    windows.add_argument("--window", type=int, choices=(256, 512, 1024), required=True)
    windows.add_argument("--output", type=Path, default=None)
    windows.add_argument("--batch-size", type=int, default=None)
    windows.add_argument("--epochs", type=int, default=None)
    windows.add_argument("--max-folds", type=int, default=None)
    windows.add_argument("--device", default=None)
    windows.add_argument("--dry-run", action="store_true")
    windows.set_defaults(action=_window)

    caps = sub.add_parser("proposal-caps", help="No-retraining proposal-cap study")
    caps.add_argument("--cv-dir", type=Path, default=CV_PATHS[1024])
    caps.add_argument("--output", type=Path, default=Path("outputs/actionformer_cap_sweep_1024"))
    caps.add_argument("--caps", type=int, nargs="+", default=[1000, 2000, 3000])
    caps.add_argument("--max-folds", type=int, default=None)
    caps.add_argument("--device", default=None)
    caps.set_defaults(action=_proposal_caps)

    pp = sub.add_parser("postprocess", help="No-retraining confidence/duration/merge sweep")
    pp.add_argument("--cv-dir", type=Path, default=CV_PATHS[1024])
    pp.add_argument("--cap-dir", type=Path, default=Path("outputs/actionformer_cap_sweep_1024"))
    pp.add_argument("--output", type=Path, default=Path("outputs/actionformer_postprocess_1024"))
    pp.add_argument("--scores", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.35, 0.5])
    pp.add_argument("--durations", type=float, nargs="+", default=[0, 2, 4])
    pp.add_argument("--merges", type=float, nargs="+", default=[0, 1, 2, 4])
    pp.add_argument("--max-folds", type=int, default=None)
    pp.set_defaults(action=_postprocess)

    hn = sub.add_parser("hard-negatives", help="Train 1024-frame ActionFormer with harder negative examples")
    hn.add_argument("--fraction", type=float, required=True,
                    help="Fraction of negative segments sampled as hard negatives (e.g., 0.25)")
    hn.add_argument("--output", type=Path, default=None)
    hn.add_argument("--batch-size", type=int, default=2)
    hn.add_argument("--epochs", type=int, default=None)
    hn.add_argument("--max-folds", type=int, default=None)
    hn.add_argument("--device", default=None)
    hn.add_argument("--dry-run", action="store_true")
    hn.set_defaults(action=_hard_negatives)

    review = sub.add_parser("review", help="Create HTML timelines of predicted and annotated events (no training)")
    review.add_argument("--cv-dir", type=Path, default=CV_PATHS[1024])
    review.add_argument("--post-dir", type=Path, default=Path("outputs/actionformer_postprocess_1024"))
    review.add_argument("--output", type=Path, default=Path("outputs/actionformer_visual_review_1024"))
    review.add_argument("--videos", nargs="+", default=["0002_vid", "0013_vid", "0028_vid"])
    review.add_argument("--cases-per-type", type=int, default=3)
    review.add_argument("--context-seconds", type=float, default=30)
    review.set_defaults(action=_review)

    summary = sub.add_parser("summary", help="Generate a README-ready Markdown results table")
    summary.add_argument("--cap-dir", type=Path, default=Path("outputs/actionformer_cap_sweep_1024"))
    summary.add_argument("--output", type=Path, default=Path("outputs/actionformer_study_summary.md"))
    summary.set_defaults(action=_summary)

    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
