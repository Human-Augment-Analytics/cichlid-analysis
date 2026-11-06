import hashlib
import json
import os
import re
from pathlib import Path

import polars as pl

from models.logistic_regression.train import CIRCLING_SPANS_BY_VIDEO, ROOT, VIDEO_FRAMES as KNOWN_VIDEO_FRAMES

GROUND_TRUTH = os.environ.get('CIRCLING_GROUND_TRUTH') or str(
    ROOT / 'ground_truth_annotations' / 'finalized_annotations' / 'merged_spans.csv')
LABELS = (os.environ.get('CIRCLING_LABELS') or 'ground_truth').strip().lower()
if LABELS not in ('ground_truth', 'baseline'):
    raise SystemExit(f'CIRCLING_LABELS must be ground_truth or baseline, not {LABELS!r}')
BASELINE = LABELS == 'baseline'
BASELINE_SUFFIX = '_15spans'
DATA_DIR = os.environ.get('CIRCLING_DATA_DIR') or str(
    next((d for d in (ROOT / 'data', ROOT.parent / 'data') if d.is_dir()), ROOT / 'data'))
VIDEOS = sorted({v.strip() for v in os.environ.get('CIRCLING_VIDEOS', '').split(',') if v.strip()}
                or KNOWN_VIDEO_FRAMES)
if BASELINE and VIDEOS != sorted(KNOWN_VIDEO_FRAMES):
    raise SystemExit(f'CIRCLING_LABELS=baseline uses the 15 spans of {sorted(KNOWN_VIDEO_FRAMES)} only; '
                     f'CIRCLING_VIDEOS is {VIDEOS}')

OUTPUTS = ROOT / 'outputs'
FRAME_COUNTS = OUTPUTS / 'video_frames.json'
MSTCN_ROOT = OUTPUTS / 'mstcn_dlc2action'
DINOV2_DIR = MSTCN_ROOT / 'emb_fish_full'
VIDEOMAE_DIR = MSTCN_ROOT / 'emb_videomae_full'
BASELINE_SPANS = OUTPUTS / 'baseline_spans.csv'

def write_baseline_spans(path=BASELINE_SPANS):
    rows = [{'VideoID': v, 'StartFrame': s, 'EndFrame': e}
            for v, spans in CIRCLING_SPANS_BY_VIDEO.items() for s, e in spans]
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(path)
    return str(path)

FINALIZED_ANNOTATIONS = write_baseline_spans() if BASELINE else GROUND_TRUTH

def model_name(name):
    return f'{name}{BASELINE_SUFFIX}' if BASELINE else name

def label_description(annotations):
    if BASELINE:
        return ('the 15 spans in CIRCLING_SPANS_BY_VIDEO (models/logistic_regression/train.py), '
                'the logistic regression\'s training labels')
    return Path(annotations).name

def video_set_name(videos=None):
    videos = sorted(videos or VIDEOS)
    if videos == sorted(KNOWN_VIDEO_FRAMES):
        return ''
    digest = hashlib.sha1(','.join(videos).encode()).hexdigest()[:8]
    return f'{len(videos)}videos_{digest}'

def mstcn_workdir(videos=None):
    name = video_set_name(videos)
    return MSTCN_ROOT / 'sets' / name if name else MSTCN_ROOT

def count_frames(video, data_dir=None):
    data_dir = Path(data_dir or DATA_DIR)
    counts = []
    movie = data_dir / f'{video}.mp4'
    if movie.exists():
        import cv2
        capture = cv2.VideoCapture(str(movie))
        counts.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.release()
    poses = data_dir / f'{video}.parquet'
    if poses.exists():
        counts.append(int(pl.scan_parquet(poses).select(pl.col('FrameNum').max()).collect().item()) + 1)
    if not counts or max(counts) <= 0:
        raise SystemExit(f'{video}: cannot count frames, need {movie} or {poses}')
    return max(counts)

def video_frames(videos=None, data_dir=None):
    videos = sorted(videos or VIDEOS)
    cache = json.loads(FRAME_COUNTS.read_text()) if FRAME_COUNTS.exists() else {}
    counts, changed = {}, False
    for video in videos:
        if video in KNOWN_VIDEO_FRAMES:
            counts[video] = KNOWN_VIDEO_FRAMES[video]
        elif video in cache:
            counts[video] = int(cache[video])
        else:
            counts[video] = cache[video] = count_frames(video, data_dir)
            changed = True
    if changed:
        FRAME_COUNTS.parent.mkdir(parents=True, exist_ok=True)
        FRAME_COUNTS.write_text(json.dumps(cache, indent=2, sort_keys=True) + '\n')
    return counts

def read_annotations(path=None, videos=None):
    path = FINALIZED_ANNOTATIONS if path is None else path
    if not str(path).strip():
        raise SystemExit('Set FINALIZED_ANNOTATIONS in models/common/config.py or CIRCLING_GROUND_TRUTH')
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f'FINALIZED_ANNOTATIONS: {path} not found')
    rows = pl.read_csv(path, infer_schema=False)
    columns = {re.sub(r'[^a-z]', '', c.lower()): c for c in rows.columns}
    if not {'videoid', 'startframe', 'endframe'} <= columns.keys():
        raise SystemExit(f'{path}: needs VideoID, StartFrame and EndFrame columns')
    frames = video_frames(videos)
    spans = {video: [] for video in frames}
    ignored = set()
    for video, start, end in rows.select(columns['videoid'], columns['startframe'],
                                         columns['endframe']).iter_rows():
        if video not in spans:
            ignored.add(video)
            continue
        start, end = int(start), int(end)
        if start < 0 or end < start or end >= frames[video]:
            raise SystemExit(f'{path}: invalid interval {video} {start}-{end}')
        spans[video].append((start, end))
    if ignored:
        print(f'Ignoring {sorted(ignored)}: not in the selected videos {sorted(frames)}', flush=True)
    empty = [video for video, intervals in spans.items() if not intervals]
    if len(empty) == len(spans):
        raise SystemExit(f'{path}: no circling intervals for {sorted(spans)}')
    if empty:
        print(f'No circling intervals for {empty}: their segments are all negative', flush=True)
    return {video: sorted(intervals) for video, intervals in spans.items()}

if __name__ == '__main__':
    print(f'Training labels: {LABELS} ({FINALIZED_ANNOTATIONS})')
    print(f'Evaluation ground truth: {GROUND_TRUTH}')
    print(f'Data: {DATA_DIR}')
    print(f'MS-TCN workdir: {mstcn_workdir()}')
    for video, count in video_frames().items():
        print(f'{video}: {count:,} frames')
