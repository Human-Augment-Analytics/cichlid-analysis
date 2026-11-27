"""Batch runner: direction-detection pipeline over multiple videos.

For each video .mp4 under --videos-dir, finds the matching annotation CSV
under --annots-dir, then runs `pipeline.py` on the full video.

Filename matching rule (customise `_find_annotation_file` if different):
    video stem e.g. "0007_vid_day05"  -> annotation name contains "vid0007-day5"

Videos are processed sequentially by default. Use --parallel K to run K
videos concurrently (each on its own Python process).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent


def _find_annotation_file(video: Path, annots_dir: Path) -> Path | None:
    """Heuristic: pull the leading numeric id from the stem, optionally a day."""
    stem = video.stem
    m_num = re.match(r'(\d+)', stem)
    if not m_num:
        return None
    vid_num = int(m_num.group(1))
    m_day = re.search(r'day(\d+)', stem, re.IGNORECASE)
    day = int(m_day.group(1)) if m_day else None

    candidates = []
    if day is not None:
        candidates += [
            annots_dir / f'circling_annotations_vid{vid_num:04d}-day{day}.csv',
            annots_dir / f'circling_annotations_vid{vid_num:04d}-day{day:02d}.csv',
        ]
    candidates += [
        annots_dir / f'circling_annotations_vid{vid_num:04d}.csv',
        annots_dir / f'circling_annotations_vid{vid_num:03d}.csv',
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: fuzzy match on vid number appearing in the stem
    for p in annots_dir.glob('*.csv'):
        if re.search(rf'\b0*{vid_num}\b', p.stem) or f'vid{vid_num:04d}' in p.stem:
            if day is None or f'day{day}' in p.stem.lower():
                return p
    return None


def _run_one(video: Path, annots: Path, out: Path, extra: list[str]) -> int:
    cmd = [
        sys.executable, str(HERE / 'pipeline.py'),
        '--video', str(video),
        '--annotations', str(annots),
        '--out', str(out),
        *extra,
    ]
    print('>>', ' '.join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos-dir', type=Path, required=True)
    ap.add_argument('--annots-dir', type=Path, required=True)
    ap.add_argument('--out',        type=Path, required=True)
    ap.add_argument('--parallel',   type=int,  default=1,
                    help='Run up to N videos concurrently.')
    ap.add_argument('--mode', choices=('gt_span', 'predicted_span', 'both'),
                    default='both')
    ap.add_argument('--skip-flow', action='store_true')
    # Any trailing args are forwarded to pipeline.py
    args, extra = ap.parse_known_args()

    extra = extra + ['--mode', args.mode]
    if args.skip_flow:
        extra.append('--skip-flow')

    videos = sorted(args.videos_dir.glob('*.mp4'))
    print(f'found {len(videos)} videos under {args.videos_dir}')

    jobs = []
    for v in videos:
        ann = _find_annotation_file(v, args.annots_dir)
        if ann is None:
            print(f'[skip] no annotation found for {v.name}')
            continue
        jobs.append((v, ann))
    print(f'running {len(jobs)} jobs (parallel={args.parallel})')

    if args.parallel <= 1:
        for v, ann in jobs:
            _run_one(v, ann, args.out, extra)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = [ex.submit(_run_one, v, ann, args.out, extra) for v, ann in jobs]
            for f in as_completed(futs):
                _ = f.result()


if __name__ == '__main__':
    main()
