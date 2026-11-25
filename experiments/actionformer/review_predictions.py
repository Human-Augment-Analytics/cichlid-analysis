"""Generate HTML and CSV cases for human review of 1024-frame OOF circling events.

Uses only cached NMS proposals from the seven development CV folds. Does not train,
run inference, or examine the final holdout. See README included in the update ZIP.

From the repository root:
    uv run python -m experiments.actionformer.review_predictions
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter
from pathlib import Path

FPS = 30.0
POST = {'min_score': 0.2, 'min_seconds': 4.0, 'merge_seconds': 4.0}


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ts(frame):
    secs = int(round(frame / FPS))
    return f'{secs // 3600:02d}:{(secs // 60) % 60:02d}:{secs % 60:02d}'


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = a[1] - a[0] + b[1] - b[0] + 2 - inter
    return inter / union if union else 0.0


def best_overlap(interval, intervals):
    if not intervals:
        return 0.0
    return max(iou(interval, other) for other in intervals)


def rects(intervals, left, right, y, color, opacity=1.0):
    result = []
    den = max(1, right - left)
    for start, end in intervals:
        if end < left or start > right:
            continue
        x = 20 + 920 * max(0, start - left) / den
        x2 = 20 + 920 * min(den, end + 1 - left) / den
        result.append(f'<rect x="{x:.2f}" y="{y}" width="{max(1,x2-x):.2f}" height="17" fill="{color}" opacity="{opacity}" rx="2"/>')
    return ''.join(result)


def timeline(gt, predictions, left, right):
    left, right = max(0, left), max(left + 1, right)
    lines = ''.join(f'<line x1="{20+920*j/4:.1f}" y1="9" x2="{20+920*j/4:.1f}" y2="87" stroke="#dbe2ea"/>' for j in range(5))
    axis = ''.join(f'<text x="{20+920*j/4:.1f}" y="103" text-anchor="middle" font-size="11" fill="#64748b">{html.escape(ts(left + (right-left)*j/4))}</text>' for j in range(5))
    return (f'<svg viewBox="0 0 960 110" role="img" aria-label="Ground truth and predicted events from {ts(left)} to {ts(right)}">'
            f'{lines}<text x="20" y="22" fill="#334155" font-size="13">Ground truth</text>'
            f'<text x="20" y="62" fill="#334155" font-size="13">Prediction</text>'
            f'<rect x="20" y="28" width="920" height="17" fill="#eff6f7" rx="2"/>'
            f'<rect x="20" y="68" width="920" height="17" fill="#fff2e8" rx="2"/>'
            f'{rects(gt,left,right,28,"#0f766e")}{rects(predictions,left,right,68,"#ea580c")}{axis}</svg>')


def case_item(kind, idx, span, overlap, video, gt, pred, count, context_frames):
    a, b = span
    left = max(0, a - context_frames)
    right = min(count - 1, b + context_frames)
    return {'video': video, 'kind': kind, 'case': idx, 'start_frame': a, 'end_frame': b,
            'start_time': ts(a), 'end_time': ts(b), 'best_iou': round(overlap, 4),
            'view_start': left, 'view_end': right,
            'svg': timeline(gt, pred, left, right)}


def inspect_video(video, gt, pred, count, context_frames, cases_per_type):
    from evaluation.metrics import merge, segment_matches
    truth = merge(gt)
    predictions = merge(pred)
    matches = segment_matches(truth, predictions, 50)
    matched_gt = set(matches.values())
    fp = [(predictions[j], best_overlap(predictions[j], truth))
          for j in range(len(predictions)) if j not in matches]
    fn = [(truth[j], best_overlap(truth[j], predictions))
          for j in range(len(truth)) if j not in matched_gt]
    tp = [(predictions[j], iou(predictions[j], truth[gj])) for j, gj in matches.items()]
    # Feature informative near misses and isolated false positives alongside best matches.
    fp.sort(key=lambda item: (-item[1], item[0][0]))
    fn.sort(key=lambda item: (-item[1], item[0][0]))
    tp.sort(key=lambda item: (item[1], item[0][0]))  # boundary-challenging TPs
    picks = {'false positive / partial overlap': fp[:cases_per_type],
             'missed annotated event': fn[:cases_per_type],
             'matched event / boundary check': tp[:cases_per_type]}
    cases = []
    for kind, group in picks.items():
        for i, (span, overlap) in enumerate(group, 1):
            cases.append(case_item(kind, i, span, overlap, video, truth, predictions,
                                   count, context_frames))
    return cases, {'video': video, 'gt_events': len(truth), 'pred_events': len(predictions),
                   'matched_events_50': len(matches), 'false_positives_50': len(predictions)-len(matches),
                   'missed_gt_events_50': len(truth)-len(matches),
                   'event_f1_50': 2*len(matches)/(len(truth)+len(predictions)) if truth or predictions else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cv-dir', type=Path, default=Path('outputs/actionformer_cv_window1024'))
    parser.add_argument('--post-dir', type=Path, default=Path('outputs/actionformer_postprocess_1024'))
    parser.add_argument('--output', type=Path, default=Path('outputs/actionformer_visual_review_1024'))
    parser.add_argument('--videos', nargs='*', default=['0002_vid', '0013_vid', '0028_vid'],
                        help='Subset of the seven development videos')
    parser.add_argument('--cases-per-type', type=int, default=3)
    parser.add_argument('--context-seconds', type=float, default=30.0)
    args = parser.parse_args()
    if args.cases_per_type < 1 or not math.isfinite(args.context_seconds) or args.context_seconds < 0:
        parser.error('Invalid case count or context duration')
    cv, post, out = map(Path.resolve, (args.cv_dir, args.post_dir, args.output))
    if any(out == p or out in p.parents or p in out.parents for p in (cv, post)):
        parser.error('Output directory must be separate from source folders')
    cv_manifest = cv / 'manifest.json'
    post_manifest = post / 'manifest.json'
    gt_path = cv / 'normalized_annotations.csv'
    for source in (cv_manifest, post_manifest, gt_path):
        if not source.is_file():
            parser.error(f'Missing source file: {source}')
    cfg = json.loads(cv_manifest.read_text())
    pp_cfg = json.loads(post_manifest.read_text())
    holdout = cfg['holdout']
    allowed = [v for v in cfg['videos'] if v != holdout]
    if int(cfg.get('window', -1)) != 1024 or len(allowed) != 7:
        parser.error('Require the original 1024-frame seven-development-video CV')
    if sha256_file(gt_path) != cfg['GT_SHA256']:
        parser.error('Source annotation SHA256 mismatch; not using changed labels')
    if pp_cfg.get('source_cv_manifest_sha256') != sha256_file(cv_manifest):
        parser.error('Postprocess results do not correspond to the original CV run')
    if set(args.videos) - set(allowed) or not args.videos:
        parser.error(f'Only development recordings allowed: {allowed}')
    if holdout in args.videos:
        parser.error('Final holdout is excluded from visual development review')
    # Use the same optimized fixed settings as the reported study, not per-video tuning.
    from experiments.actionformer.run_proposal_cap import read_raw, load_normalized_annotations
    from evaluation.metrics import evaluate_video
    from models.actionformer.postprocess import prediction_table
    import polars as pl
    truth = load_normalized_annotations(gt_path, allowed)
    with (cv / 'cv_results.csv').open(newline='') as fh:
        cvrows = {r['validation_video']: r for r in csv.DictReader(fh)}
    cases, metrics = [], []
    for video in args.videos:
        fold = allowed.index(video) + 1
        file = post / 'nms_proposals' / f'fold_{fold:02d}_{video}.parquet'
        if not file.exists():
            parser.error(f'Missing cached 1000-cap NMS proposals: {file}. Run postprocess study first.')
        proposals = read_raw(file, pl)
        filtered = [item for item in proposals if item[1] >= POST['min_score']]
        frame_count = int(cvrows[video]['num_frames'])
        intervals = prediction_table(video, filtered, frame_count,
                                     min_seconds=POST['min_seconds'], merge_seconds=POST['merge_seconds'])
        pred = [(int(a), int(b)) for a,b in intervals.select('StartFrame','EndFrame').rows()]
        video_cases, row = inspect_video(video, truth[video], pred, frame_count,
                                        round(args.context_seconds * FPS), args.cases_per_type)
        reference = evaluate_video(truth[video], pred, frame_count)
        if abs(reference['f1@50'] - row['event_f1_50']) > 1e-10:
            raise RuntimeError(f'{video}: event scoring does not match the official evaluation')
        cases.extend(video_cases)
        metrics.append(row)
        print(f"{video}: {row['gt_events']} GT, {row['pred_events']} predictions, "
              f"{row['matched_events_50']} matches; {len(video_cases)} review examples", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'cases.csv').open('w', newline='') as fh:
        fields = ['video', 'kind', 'case', 'start_frame', 'end_frame', 'start_time',
                  'end_time', 'best_iou', 'view_start', 'view_end']
        writer = csv.DictWriter(fh, fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({k: case[k] for k in fields})
    with (out / 'per_video.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    sections = []
    for row in metrics:
        v = row['video']
        blocks = [f'<section><h2>{html.escape(v)}</h2><p>{row["gt_events"]} annotated events · '
                  f'{row["pred_events"]} predictions · {row["matched_events_50"]} matched at IoU ≥ 0.5'
                  f' · F1@50 = {row["event_f1_50"]:.3f}</p>']
        for case in (c for c in cases if c['video'] == v):
            blocks.append(f'<article><h3>{html.escape(case["kind"])} #{case["case"]}</h3>'
                          f'<p>Event {case["start_time"]}–{case["end_time"]} '
                          f'(frames {case["start_frame"]}–{case["end_frame"]}); '
                          f'best overlap IoU={case["best_iou"]:.2f}. '
                          f'Inspect the underlying recording around this time.</p>'
                          f'{case["svg"]}</article>')
        blocks.append('</section>')
        sections.append('\n'.join(blocks))
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8"/>'
           '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
           '<title>ActionFormer circling review</title><style>'
           'body{font-family:system-ui,-apple-system,sans-serif;max-width:1060px;margin:35px auto;padding:0 22px;color:#172033;background:#f8fafc}'
           'section{margin:28px 0;padding:16px;background:white;border:1px solid #ddd;border-radius:13px}'
           'article{padding:15px;margin:16px 0;border:1px solid #e5e7eb;border-radius:9px}'
           'h1,h2{margin-bottom:7px}h3{font-size:17px;margin:0}p{color:#475569;line-height:1.45}svg{width:100%;height:auto}'
           '.tag{background:#e2e8f0;padding:5px 9px;border-radius:5px;display:inline-block}'
           '</style></head><body><h1>ActionFormer · visual review</h1>'
           '<p>1024-frame original model · score ≥ 0.20 · minimum proposal duration 4 s · merge gap 4 s. '
           'Seven-video development CV only; no final holdout. '
           'Green = annotated circling; orange = predicted circling.</p>'
           '<p><span class="tag">Review checklist</span> Do the predictions overlap actual circling in the recording? '
           'Are event boundaries too wide or too short? Are distinct bouts incorrectly merged? '
           'Some “false positives” may be boundary mismatches rather than wholly unrelated behaviors.</p>'
           + '\n'.join(sections) + '</body></html>')
    (out / 'review.html').write_text(doc, encoding='utf-8')
    (out / 'settings.json').write_text(json.dumps({'source_cv':str(cv),'source_postprocess':str(post),
        'videos':args.videos,'holdout_excluded':holdout,**POST},indent=2)+'\n')
    print(f'Wrote {out / "review.html"}, {out / "cases.csv"}, and {out / "per_video.csv"}')


if __name__ == '__main__':
    main()
