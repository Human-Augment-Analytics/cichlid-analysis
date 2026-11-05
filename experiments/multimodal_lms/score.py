"""Score pilot judgments against overlapping annotated events, with bounds caveats."""
import csv
import html
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    with (HERE / "results.csv").open() as file:
        predictions = list(csv.DictReader(file))
    scored = []
    for prediction in predictions:
        start, end = int(prediction["start_frame"]), int(prediction["end_frame"])
        source = ROOT / f"ground_truth_annotations/finalized_annotations/circling_annotations_vid{prediction['video_id'][:4]}.csv"
        with source.open(newline="") as file:
            matches = [row for row in csv.DictReader(file)
                       if row["Start Frame"] and row["End Frame"]
                       and max(start, int(row["Start Frame"])) <= min(end, int(row["End Frame"]))]
        if len(matches) != 1:
            raise ValueError(f"Expected one reference event for {prediction['clip']}, got {len(matches)}")
        reference = matches[0]
        ref_start, ref_end = int(reference["Start Frame"]), int(reference["End Frame"])
        true_direction = {"CW": "clockwise", "CCW": "anticlockwise"}[reference["Rotation direction"].strip().upper()]
        if reference["Behavior"].strip().lower() != "circling":
            raise ValueError("This pilot scorer expects circling reference events")
        true_count = float(reference["No. of turns/circles"])
        count = float(prediction["circle_count"]) if prediction["circle_count"] else None
        scored.append({
            "clip": prediction["clip"], "model": prediction["model"],
            "reference_event": reference["EventID"],
            "clip_start_frame": start, "clip_end_frame": end,
            "reference_start_frame": ref_start, "reference_end_frame": ref_end,
            "exact_bounds": start == ref_start and end == ref_end,
            "reference_circling": True, "predicted_circling": prediction["circling"],
            "circling_correct": int(prediction["circling"].lower() == "true"),
            "reference_direction": true_direction, "predicted_direction": prediction["direction"],
            "direction_match": int(prediction["direction"] == true_direction),
            "reference_event_circles": true_count, "predicted_clip_circles": count,
            "count_signed_difference_provisional": None if count is None else count - true_count,
            "count_absolute_difference_provisional": None if count is None else abs(count - true_count),
        })
    models = list(dict.fromkeys(row["model"] for row in scored))
    clips = list(dict.fromkeys(row["clip"] for row in scored))
    assert len(scored) == 15 and len(clips) == 5
    with (HERE / "scores.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    summaries = []
    for model in models:
        rows = [row for row in scored if row["model"] == model]
        summaries.append({"model": model, "clips": len(rows),
                          "circling_detected": sum(row["circling_correct"] for row in rows),
                          "positive_clip_recall": mean(row["circling_correct"] for row in rows),
                          "direction_matches": sum(row["direction_match"] for row in rows),
                          "direction_agreement": mean(row["direction_match"] for row in rows),
                          "count_mae_provisional": mean(row["count_absolute_difference_provisional"] for row in rows
                                                        if row["count_absolute_difference_provisional"] is not None)})
    with (HERE / "score_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    lines = ["# Pilot model scores", "",
             "References are the single overlapping human-annotated event per clip. All five events are circling. No labels were supplied to the models.", "",
             "Circling: 1 if detected, 0 if missed. Direction: 1 if it matches the reference event, 0 otherwise (including a missing direction). Count difference: absolute predicted clip count minus annotated event count, in circles; lower is better.", "",
             "**All five clip boundaries differ from their reference event boundaries. Count differences and their means are provisional event-reference comparisons, not valid exact-clip count errors. No duration-based rescaling is applied.** Direction agreement also assumes the overlapping event represents the clip's circling direction.", "",
             "## Per clip", "", "Each model cell is **circling / direction / absolute count difference**.", "",
             "| Clip | Reference direction, event circles | Gemma | Qwen | Bonsai |",
             "| --- | --- | --- | --- | --- |"]
    for clip in clips:
        rows = [row for row in scored if row["clip"] == clip]
        reference = rows[0]
        direction = "CW" if reference["reference_direction"] == "clockwise" else "CCW"
        cells = [f"{row['circling_correct']} / {row['direction_match']} / {row['count_absolute_difference_provisional']:g}" for row in rows]
        lines.append(f"| `{Path(clip).stem}` | {direction}, {reference['reference_event_circles']:g} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Aggregate", "", "| Model | Positive-clip recall | Direction agreement | Provisional count MAE |", "| --- | ---: | ---: | ---: |"])
    for row in summaries:
        lines.append(f"| {row['model']} | {row['circling_detected']}/5 ({row['positive_clip_recall']:.0%}) | {row['direction_matches']}/5 ({row['direction_agreement']:.0%}) | {row['count_mae_provisional']:.2f} circles |")
    lines.extend(["", "All clips are positive examples, so these data cannot estimate false-positive rate, specificity, or balanced accuracy. Five clips are too few to establish a model ranking. Gemma failed a reversed-frame control; Qwen's final numeric count contradicts its explanation; Bonsai repeated the same decision/direction/count for every clip. Scores use the saved structured numeric answers without correcting them.", "",
                  "For exact count evaluation, manually annotate these exact five clip intervals or rerun models on the exact reference-event intervals."])
    (HERE / "scores.md").write_text("\n".join(lines) + "\n")
    chart = ['<!doctype html><html><head><meta charset="utf-8"><style>body{font-family:var(--font-sans,system-ui);color:var(--foreground,#222);font-size:13px}.group{margin:18px 0}.row{display:flex;align-items:center;gap:10px;margin:7px 0}.model{width:90px;flex-shrink:0}.track{height:16px;flex:1;background:var(--secondary,#ddd)}.bar{height:16px}.value{width:60px;font-variant-numeric:tabular-nums}.note{font-size:12px;color:var(--muted-foreground,#666);line-height:1.5}</style></head><body>']
    for key, label in [("positive_clip_recall", "Circling detected · five positive clips"), ("direction_agreement", "Direction agreement · overlapping event reference")]:
        chart.append(f'<div class="group"><p>{label}</p>')
        for index, row in enumerate(summaries, 1):
            value = row[key]
            chart.append(f'<div class="row"><span class="model">{html.escape(row["model"].split()[0])}</span><div class="track"><div class="bar" style="width:{value*100:g}%;background:var(--chart-{index},#888)"></div></div><span class="value">{value:.0%}</span></div>')
        chart.append('</div>')
    chart.append('<p class="note">These are five positive clips, not a general accuracy benchmark. Direction uses overlapping event labels. Exact clip-count scores require matching annotation boundaries; provisional count differences are listed in scores.md.</p></body></html>')
    (HERE / "scores.html").write_text("".join(chart))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
