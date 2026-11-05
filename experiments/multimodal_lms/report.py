"""Summarize the three-model, five-clip pilot, checking identical image inputs."""
import csv
import html
import json
from pathlib import Path
import re

from .run import ROOT, parse_answer


def main():
    base = ROOT / "outputs/experiments/multimodal_lms"
    with (base / "qwen/results.csv").open() as file:
        selected = [{"clip": row["clip"], "duration_seconds": float(row["duration_seconds"])}
                    for row in csv.DictReader(file)]
    if len(selected) != 5:
        raise ValueError("The pilot requires five completed Qwen results")
    models = [("gemma", "Gemma 4 E4B"), ("qwen", "Qwen3.5 4B"), ("bonsai", "Bonsai 27B")]
    control_path = base / "gemma/reversed_control.json"
    reverse_failed = False
    if control_path.exists():
        forward = json.loads((base / "gemma/F1_613_266809-267100.json").read_text())["answer"]
        reverse = parse_answer(json.loads(control_path.read_text())["raw_response"])
        reverse_failed = forward == reverse and forward["circling"] is True and forward["direction"] in {"clockwise", "anticlockwise"}
    rows = []
    references = []
    lines = ["# Five-clip local multimodal pilot", "", "Model judgments, not verified labels or accuracy measurements.", "",
             "Gemma 4 E4B (4-bit), Qwen3.5 4B (4-bit), Bonsai 27B (1-bit).", "",
             "| Clip | Duration | Gemma | Qwen | Bonsai |", "| --- | ---: | --- | --- | --- |"]
    for clip in selected:
        name = Path(clip["clip"]).stem
        video = "0031" if name.startswith("F1_613") else "0028"
        start, end = map(int, re.search(r"_(\d+)-(\d+)$", name).groups())
        cells, hashes = [], set()
        for folder, label in models:
            record = json.loads((base / folder / (name + ".json")).read_text())
            answer = record["answer"]
            hashes.add(record["input_sha256"])
            value = "uncertain" if answer["circling"] is None else "yes" if answer["circling"] else "no"
            count = "?" if answer["circle_count"] is None else str(answer["circle_count"])
            cells.append(f"{value}; {answer['direction']}; {count} circles ({answer['confidence']})")
            flags = "Failed time-reversal direction control" if folder == "gemma" and reverse_failed else ""
            if folder == "qwen" and answer["circle_count"] == 1.5 and "ninth revolution" in answer.get("evidence", "").lower():
                flags = "Numeric count 1.5 conflicts with nine revolutions stated in evidence"
            rows.append({"clip": clip["clip"], "video_id": video + "_vid", "start_frame": start, "end_frame": end,
                         "model": label, "model_identifier": record["model"],
                         "duration_seconds": round(record["duration_seconds"], 3),
                         **answer, "sample_count": record["sample_count"],
                         "effective_sample_fps": round(record["effective_sample_fps"], 4),
                         "elapsed_seconds": record["elapsed_seconds"],
                         "input_sha256": record["input_sha256"], "review_flags": flags})
        if len(hashes) != 1:
            raise ValueError(f"Models did not receive identical images: {name}")
        lines.append(f"| `{name}` | {clip['duration_seconds']:.2f} s | " + " | ".join(cells) + " |")
        annotations = ROOT / f"ground_truth_annotations/finalized_annotations/circling_annotations_vid{video}.csv"
        with annotations.open(newline="") as file:
            for annotation in csv.DictReader(file):
                if not annotation["Start Frame"] or not annotation["End Frame"]:
                    continue
                a, b = int(annotation["Start Frame"]), int(annotation["End Frame"])
                if max(start, a) <= min(end, b):
                    references.append(f"| `{name}` | `{annotation['EventID']}` | {a}–{b} | {annotation['Behavior']} | {annotation['Rotation direction']} | {annotation['No. of turns/circles']} |")
    quality_notes = []
    if reverse_failed:
        quality_notes.append("Gemma's pipeline failed a time-reversal control on the first clip: it returned the same clockwise direction and identical explanation after the sampled frame order was reversed. Its direction/count outputs are unvalidated.")
    if any("Numeric count" in row["review_flags"] for row in rows):
        quality_notes.append("Qwen's last clip has internally inconsistent counting: its numeric circle_count is 1.5, while its evidence describes nine complete revolutions. The report preserves the numeric answer and the raw explanation; it does not reconcile them into a new label.")
    for _, model in models:
        model_rows = [row for row in rows if row["model"] == model]
        if len({(row["circling"], row["direction"], row["circle_count"]) for row in model_rows}) == 1:
            quality_notes.append(f"{model} returned the same circling, direction, and count judgment for all five clips. This does not establish that those judgments are correct.")
            for row in model_rows:
                row["review_flags"] = "; ".join(filter(None, [row["review_flags"], "Identical circling/direction/count across all five clips"]))
    image_control = base / "bonsai/image_order_control.json"
    if image_control.exists() and json.loads(image_control.read_text()).get("passed"):
        quality_notes.append("Bonsai passed a two-image color-order control in both orders. This verifies basic image transport/order, not motion understanding.")
    lines.extend(["", "All three models received identical timestamped JPEG frames for each clip (verified by SHA-256).",
                  "Sampling: approximately 4 fps, first and last frame included, 640 px maximum frame width; four timestamped frames per 2x2 contact sheet.",
                  "Direction is screen-space orbital motion; counts include partial revolutions rounded to quarter-turns.",
                  "Unknown circling and circle counts remain null; a null direction is normalized to uncertain.", "",
                  *[text for note in quality_notes for text in (note, "")],
                  "Counts and directions are model estimates. Confidence is self-reported and uncalibrated. No accuracy ranking can be inferred from these five comparisons.", "",
                  "## Existing annotation references", "",
                  "All five clips overlap manually annotated circling events. These labels were read separately for reporting and were never supplied to a model. Event boundaries differ from the clip boundaries above: event counts are contextual references, not exact clip-count targets. CW means clockwise; CCW means anticlockwise.", "",
                  "| Clip | Annotated event | Event frames | Behavior | Direction | Event circles |",
                  "| --- | --- | --- | --- | --- | ---: |", *references, "",
                  "Sources: `ground_truth_annotations/finalized_annotations/circling_annotations_vid0028.csv` and `circling_annotations_vid0031.csv`.", "",
                  "## Model evidence", ""])
    for row in rows:
        lines.extend([f"### {Path(row['clip']).stem} — {row['model']}", "",
                      f"Pair: {row.get('pair_description', '')}", "", row.get("evidence", ""), ""])
    dest = Path(__file__).parent
    (dest / "results.md").write_text("\n".join(lines) + "\n")
    with (dest / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    maximum = max(1, max(row["circle_count"] or 0 for row in rows))
    groups = []
    for clip in selected:
        group = [row for row in rows if row["clip"] == clip["clip"]]
        label = f"{group[0]['video_id']} · {group[0]['start_frame']}–{group[0]['end_frame']}"
        bars = []
        for index, row in enumerate(group, 1):
            value = row["circle_count"]
            direction = {"clockwise": "CW", "anticlockwise": "CCW", "mixed": "mixed", "none": "no circling", "uncertain": "uncertain"}[row["direction"]]
            text = "unknown" if value is None else f"{value:g} · {direction}"
            bars.append(f'<div class="row"><span class="model">{html.escape(row["model"].split()[0])}</span><div class="track"><div class="bar" style="width:{100 * (value or 0) / maximum:.2f}%;background:var(--chart-{index},#888)"></div></div><span class="value">{text}</span></div>')
        groups.append(f'<div class="group"><div class="label">{label}</div>{"".join(bars)}</div>')
    page = '''<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:var(--font-sans,system-ui);color:var(--foreground,#222);font-size:13px}
.group{margin:18px 0}.label{font-weight:600;margin-bottom:8px}.row{display:flex;align-items:center;gap:8px;margin:5px 0}
.model{width:55px;flex-shrink:0}.track{flex:1;height:14px;background:var(--secondary,#ddd)}.bar{height:14px}
.value{width:116px;flex-shrink:0;font-variant-numeric:tabular-nums}.note{color:var(--muted-foreground,#666);font-size:12px;line-height:1.5}
</style></head><body><p>Estimated circles per clip · CW = clockwise, CCW = anticlockwise</p>''' + "".join(groups) + '<p class="note">Local model outputs, not verified labels. Identical timestamped frames were supplied to all three models.<br>' + '<br>'.join(html.escape(note) for note in quality_notes) + '<br>Counts include partial revolutions. Human reference events have different boundaries; see results.md.</p></body></html>'
    (dest / "results.html").write_text(page)
    print(f"Wrote {len(rows)} judgments; identical inputs verified for {len(selected)} clips.")


if __name__ == "__main__":
    main()
