"""Grounding-DINO-based auto-init for SAM 2.

Replaces manual clicking: given a video + frame index, runs Grounding DINO
with the text prompt "fish" to detect bounding boxes, then writes a
clicks.json whose "points" are box CENTERS, one per detected fish.

The downstream sam2_segment_event.py already supports N init points (one
object per point).  So this is a drop-in replacement for manually-clicked
init -- the only difference is the "frame" and "points" come from DINO.

Usage
-----
    python sam2_dino_init.py \
        --video ../../../ground_truth_annotations/data/0007_vid_day05.mp4 \
        --frame 20142 \
        --out ../../../../outputs/optical-flow/sam2/vid5-v07_d05_evt_0001_dinoclicks.json

Flags of interest:
    --prompt "fish"                     text prompt for DINO
    --box-threshold 0.30                DINO box confidence threshold
    --text-threshold 0.25               DINO text alignment threshold
    --max-boxes 5                       cap on #boxes returned
    --filter-pair                       keep only the TOP TWO boxes by score
                                        (for circling-event init we want 2 fish)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


DEFAULT_MODEL = 'IDEA-Research/grounding-dino-tiny'


def _load_frame(video: Path, frame_idx: int) -> tuple:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'could not read frame {frame_idx} from {video}')
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_bgr, Image.fromarray(rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',          type=Path, required=True)
    ap.add_argument('--frame',          type=int,  required=True)
    ap.add_argument('--out',            type=Path, required=True)
    ap.add_argument('--model',          type=str,  default=DEFAULT_MODEL)
    ap.add_argument('--prompt',         type=str,  default='fish')
    ap.add_argument('--box-threshold',  type=float, default=0.30)
    ap.add_argument('--text-threshold', type=float, default=0.25)
    ap.add_argument('--max-boxes',      type=int,  default=5)
    ap.add_argument('--filter-pair',    action='store_true',
                    help='Keep only the top-TWO boxes by score (for pair init).')
    ap.add_argument('--nms-iou',        type=float, default=0.3,
                    help='Boxes with IoU above this against a higher-scored box '
                         'are suppressed (classic detector NMS).  0 = disable.')
    ap.add_argument('--min-pair-sep-px', type=float, default=80.0,
                    help='With --filter-pair, require top-2 boxes to have '
                         'centroid distance >= this many px.  Otherwise write '
                         'is_valid_pair=False and SAM 2 should not engage.')
    ap.add_argument('--device',         type=str,  default='cuda')
    ap.add_argument('--vis',            type=Path, default=None,
                    help='If given, save an annotated PNG with the boxes '
                         'drawn (for manual sanity check).')
    args = ap.parse_args()

    frame_bgr, img = _load_frame(args.video, args.frame)
    H, W = frame_bgr.shape[:2]
    print(f'[dino] video={args.video.name} frame={args.frame} size={W}x{H}')

    print(f'[dino] loading model {args.model} on {args.device}')
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(args.device)
    model.eval()

    # Prompt must end with a "." for the grounded-detection pipeline.
    prompt = args.prompt.strip()
    if not prompt.endswith('.'):
        prompt = prompt + '.'

    with torch.inference_mode():
        inputs = processor(images=img, text=prompt, return_tensors='pt').to(args.device)
        outputs = model(**inputs)
        # post_process_grounded_object_detection expects a list of target sizes.
        target_sizes = torch.tensor([[H, W]], device=args.device)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs['input_ids'],
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=target_sizes,
        )
    r = results[0]
    scores = r['scores'].tolist()
    boxes  = r['boxes'].tolist()             # [x0, y0, x1, y1] in original pixels
    labels = r.get('text_labels') or r.get('labels')
    if hasattr(labels, 'tolist'):
        labels = labels.tolist()
    labels = labels or ['?'] * len(scores)
    print(f'[dino] raw detections: {len(scores)}')
    for s, b, l in zip(scores, boxes, labels):
        print(f'  score={s:.3f}  box={[round(x,1) for x in b]}  label={l}')

    # Sort by score desc.
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    scores = [scores[i] for i in order]
    boxes  = [boxes[i]  for i in order]
    labels = [labels[i] for i in order]

    # Non-max suppression on boxes (standard detector NMS).
    def _iou(a, b):
        xa0, ya0, xa1, ya1 = a
        xb0, yb0, xb1, yb1 = b
        ix0 = max(xa0, xb0); iy0 = max(ya0, yb0)
        ix1 = min(xa1, xb1); iy1 = min(ya1, yb1)
        iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
        inter = iw * ih
        aarea = max(0.0, xa1 - xa0) * max(0.0, ya1 - ya0)
        barea = max(0.0, xb1 - xb0) * max(0.0, yb1 - yb0)
        return inter / (aarea + barea - inter) if (aarea + barea - inter) > 0 else 0.0

    if args.nms_iou > 0 and len(boxes) > 1:
        keep = []
        for i, b in enumerate(boxes):
            if any(_iou(b, boxes[j]) > args.nms_iou for j in keep):
                continue
            keep.append(i)
        scores = [scores[i] for i in keep]
        boxes  = [boxes[i]  for i in keep]
        labels = [labels[i] for i in keep]
        print(f'[nms]  kept {len(keep)} / after IoU<={args.nms_iou} suppression')

    is_valid_pair = None
    if args.filter_pair:
        scores = scores[:2]; boxes = boxes[:2]; labels = labels[:2]
        if len(boxes) >= 2:
            c1 = ((boxes[0][0] + boxes[0][2]) / 2, (boxes[0][1] + boxes[0][3]) / 2)
            c2 = ((boxes[1][0] + boxes[1][2]) / 2, (boxes[1][1] + boxes[1][3]) / 2)
            sep = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
            is_valid_pair = sep >= args.min_pair_sep_px
            print(f'[pair] separation={sep:.1f} px  '
                  f'threshold={args.min_pair_sep_px:.1f}  '
                  f'valid_pair={is_valid_pair}')
        else:
            is_valid_pair = False
            print(f'[pair] only {len(boxes)} boxes after NMS -> invalid pair')
    else:
        scores = scores[:args.max_boxes]
        boxes  = boxes[:args.max_boxes]
        labels = labels[:args.max_boxes]

    # Convert each box -> center point.
    points = [[int(round((b[0] + b[2]) / 2)),
               int(round((b[1] + b[3]) / 2))] for b in boxes]
    print(f'[auto-init] kept {len(points)} boxes; points (box centers):')
    for i, (p, s) in enumerate(zip(points, scores)):
        print(f'  #{i + 1}  point={p}  score={s:.3f}')

    payload = dict(
        video=str(args.video),
        frame=int(args.frame),
        image_size=[W, H],
        points=points,
        labels=[1] * len(points),
        is_valid_pair=is_valid_pair,
        _source=f'auto-init from Grounding DINO ({args.model}) prompt={args.prompt!r}',
        _scores=scores,
        _boxes=boxes,
        _text_labels=labels,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f'wrote {args.out}')

    if args.vis is not None:
        vis = frame_bgr.copy()
        for i, (b, s) in enumerate(zip(boxes, scores)):
            x0, y0, x1, y1 = map(int, b)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(vis, f'{i + 1} {s:.2f}', (x0 + 4, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cx, cy = points[i]
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
        cv2.imwrite(str(args.vis), vis)
        print(f'wrote {args.vis}')


if __name__ == '__main__':
    main()
