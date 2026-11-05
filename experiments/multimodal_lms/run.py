"""Send chronological, timestamped clip frames to a local LM Studio vision model."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROMPT = """Analyze the chronological video frames below as ONE continuous fish clip.
Decide whether a pair of fish is mutually circling: sustained orbital motion around
one another, not a single fish turning, parallel swimming, a crossing, or simple chasing.
Direction is the rotation of the vector between the SAME two fish, as seen on the
screen: clockwise or anticlockwise. Do not infer biological sex. Ignore reflections.
Count revolutions of this relative-position vector, not the sum of both fish's paths.
One circle is 360 degrees. Estimate total revolutions including partial turns to the
nearest quarter-turn. If direction reverses, use mixed. If no circling, direction is
none and circle_count is 0. If visibility, temporal sampling, or identities prevent
a determination, use null for circling/count and uncertain for direction as needed.
Do not invent unseen turns. Base the answer only on visible motion.
Return ONLY a JSON object with these keys:
circling (boolean or null), direction (clockwise, anticlockwise, mixed, none, uncertain),
circle_count (nonnegative number or null), confidence (low, medium, high),
pair_description (short visual description identifying the pair),
evidence (short explanation of visible trajectory and uncertainty).
The timestamps are seconds from the beginning of the clip. No labels are supplied.
"""


def request_json(url, payload=None, timeout=600):
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"},
                                 data=None if payload is None else json.dumps(payload).encode())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()[:3000]}") from error


def sample_frames(path, sample_fps, max_frames, width):
    cap = cv2.VideoCapture(str(path))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0 or count < 1:
            raise ValueError(f"Invalid video: {path}")
        wanted = min(count, max_frames, max(2, math.ceil((count - 1) / fps * sample_fps) + 1))
        indices = sorted({round(i * (count - 1) / (wanted - 1)) for i in range(wanted)}) if wanted > 1 else [0]
        inputs = []
        for frame in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {frame} in {path}")
            h, w = image.shape[:2]
            if w > width:
                image = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError("JPEG encoding failed")
            inputs.extend([{"type": "text", "content": f"t={frame / fps:.3f} s"},
                           {"type": "image", "data_url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode()}])
        return inputs, {"source_fps": fps, "source_frames": count, "duration_seconds": count / fps,
                        "sample_frame_indices": indices, "sample_count": len(indices),
                        "effective_sample_fps": (len(indices) - 1) * fps / max(1, count - 1)}
    finally:
        cap.release()


def parse_answer(response):
    message = "\n".join(item["content"] for item in response.get("output", []) if item["type"] == "message")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", message, re.DOTALL)
    text = fenced.group(1) if fenced else message.strip()
    answer = json.loads(text)
    if answer.get("direction") is None and "direction" in answer:
        answer["direction"] = "uncertain"
    if type(answer.get("circling")) not in (bool, type(None)):
        raise ValueError("Invalid circling value")
    if answer.get("direction") not in {"clockwise", "anticlockwise", "mixed", "none", "uncertain"}:
        raise ValueError("Invalid direction")
    count = answer.get("circle_count")
    if count is not None and (type(count) not in (int, float) or not math.isfinite(count) or count < 0):
        raise ValueError("Invalid circle count")
    if answer.get("confidence") not in {"low", "medium", "high"}:
        raise ValueError("Invalid confidence")
    if answer["circling"] is False and (count != 0 or answer["direction"] != "none"):
        raise ValueError("Inconsistent non-circling answer")
    return answer


def contact_sheets(inputs):
    """Group four consecutive frames in row-major order to bound encoder memory."""
    result = [{"type": "text", "content":
        "Each following image is a 2x2 contact sheet of consecutive video frames. "
        "Read top-left, top-right, bottom-left, bottom-right, then the next sheet. "
        "Each tile shows its time in seconds. Black unused tiles are not frames. "
        "These are the SAME fish over time, not four different pairs."}]
    for offset in range(0, len(inputs), 8):
        tiles = []
        for index in range(offset, min(offset + 8, len(inputs)), 2):
            image = cv2.imdecode(np.frombuffer(base64.b64decode(inputs[index + 1]["data_url"].split(",", 1)[1]), dtype=np.uint8), cv2.IMREAD_COLOR)
            tile = cv2.copyMakeBorder(image, 28, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(tile, inputs[index]["content"], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        while len(tiles) < 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("Contact sheet JPEG encoding failed")
        result.append({"type": "image", "data_url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-e4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--clips-dir", type=Path, default=ROOT / "data/clips")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/multimodal_lms")
    parser.add_argument("--sample-fps", type=float, default=4)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--context-length", type=int, default=0, help="0 keeps the loaded model's context")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--clips", nargs="+", help="Explicit clip filenames within --clips-dir")
    parser.add_argument("--contact-sheets", action="store_true", help="Pack four timestamped frames per image")
    parser.add_argument("--omit-reasoning-parameter", action="store_true", help="For a model whose prepared chat template already disables thinking")
    args = parser.parse_args()
    if args.sample_fps <= 0 or args.max_frames < 2 or args.width < 32:
        parser.error("Invalid sampling settings")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = request_json(args.base_url + "/api/v1/models")
    (args.output_dir / "available_models.json").write_text(json.dumps(models, indent=2))
    clips = ([args.clips_dir / name for name in args.clips] if args.clips else sorted(args.clips_dir.glob("*.mp4")))
    if args.limit is not None:
        clips = clips[:args.limit]
    if not clips:
        parser.error("No MP4 clips found")
    config = {"model": args.model, "sample_fps": args.sample_fps, "max_frames": args.max_frames,
              "width": args.width, "context_length": args.context_length, "prompt": PROMPT,
              "contact_sheets": args.contact_sheets,
              "omit_reasoning_parameter": args.omit_reasoning_parameter}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    results = []
    for number, clip in enumerate(clips, 1):
        output = args.output_dir / (clip.stem + ".json")
        record = json.loads(output.read_text()) if output.exists() else None
        signature = {"size": clip.stat().st_size, "mtime_ns": clip.stat().st_mtime_ns}
        if (record and record.get("config_hash") == config_hash
                and record.get("source_signature") == signature and "raw_response" in record):
            try:
                record["answer"] = parse_answer(record["raw_response"])
                record.pop("error", None)
                output.write_text(json.dumps(record, indent=2) + "\n")
            except (ValueError, KeyError, TypeError):
                pass
        if not record or record.get("config_hash") != config_hash or record.get("source_signature") != signature or "answer" not in record:
            print(f"[{number}/{len(clips)}] {clip.name}: sampling", flush=True)
            inputs, metadata = sample_frames(clip, args.sample_fps, args.max_frames, args.width)
            if args.contact_sheets:
                inputs = contact_sheets(inputs)
            metadata["input_image_count"] = sum(item["type"] == "image" for item in inputs)
            record = {"clip": str(clip.relative_to(ROOT) if clip.is_relative_to(ROOT) else clip),
                      "model": args.model, "config_hash": config_hash, "source_signature": signature, **metadata}
            record["input_sha256"] = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
            start = time.monotonic()
            try:
                response = request_json(args.base_url + "/api/v1/chat", {
                    "model": args.model, "input": [{"type": "text", "content": PROMPT}, *inputs],
                    **({} if args.omit_reasoning_parameter else {"reasoning": "off"}),
                    "store": False, "temperature": 0,
                    "max_output_tokens": 600,
                    **({"context_length": args.context_length} if args.context_length else {})})
                record["raw_response"] = response
                record["answer"] = parse_answer(response)
            except Exception as error:
                record["error"] = str(error)
            record["elapsed_seconds"] = round(time.monotonic() - start, 2)
            output.write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
        print(f"[{number}/{len(clips)}] {clip.name}: {record.get('answer', record.get('error'))}", flush=True)
        fields = ["clip", "model", "duration_seconds", "sample_count", "effective_sample_fps", "circling", "direction", "circle_count", "confidence", "pair_description", "evidence", "error"]
        with (args.output_dir / "results.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in results:
                writer.writerow({key: {**row, **row.get("answer", {})}.get(key) for key in fields})
        if "error" in record:
            raise RuntimeError(record["error"])


if __name__ == "__main__":
    main()
