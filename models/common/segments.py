import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from models.common.config import video_frames
from models.logistic_regression.train import THRESHOLD, frame_labels
from models.mstcn_dlc2action.dlc_yolo_converter import dlc2action_bridge as br

LEN_SEGMENT = 128
OVERLAP = 0
SEED = 42
STRIDE = 2
SPLIT = 'repo_split'

def sample_segments(positives, n_rows, len_segment=LEN_SEGMENT, seed=SEED):
    rng = np.random.default_rng(seed)
    tables = []
    for video in sorted(n_rows):
        n = n_rows[video]
        positive = np.zeros(n, dtype=bool)
        for start, end in positives.get(video, []):
            positive[start:end] = True
        spans = np.array(br.segment_spans(n, len_segment, OVERLAP), dtype=np.int64)
        counts = np.array([positive[s:e].sum() for s, e in spans], dtype=np.int64)
        with_circling = np.flatnonzero(counts > 0)
        without = np.flatnonzero(counts == 0)
        if not len(with_circling):
            continue
        if len(without) < len(with_circling):
            raise ValueError(f'{video}: not enough segments without circling to sample')
        picked = np.sort(np.concatenate([
            with_circling, rng.choice(without, size=len(with_circling), replace=False)]))
        tables.append(pl.DataFrame({
            'video': [video] * len(picked), 'start_row': spans[picked, 0], 'end_row': spans[picked, 1],
            'positive_rows': counts[picked], 'circling': counts[picked] > 0}))
    if not tables:
        raise SystemExit('no segment with circling in any selected video')
    return pl.concat(tables)

def split_segments(segments, seed=SEED):
    ids = np.arange(segments.height)
    _, test_ids = train_test_split(ids, test_size=0.2, random_state=seed,
                                   stratify=segments['circling'].to_numpy())
    split = np.full(segments.height, 'train', dtype='<U5')
    split[test_ids] = 'test'
    return segments.with_columns(pl.Series('split', split))

def write_split(segments, path, n_rows):
    ids = {k: [br.segment_id(v, s, e) for v, s, e in segments.filter(pl.col('split') == k)
               .select('video', 'start_row', 'end_row').iter_rows()] for k in ('train', 'test')}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for header, key in (('Training videos:', 'train'), ('Validation videos:', 'test'),
                            ('Test videos:', 'test')):
            f.write(header + '\n')
            f.writelines(sid + '\n' for sid in ids[key])
    Path(str(path) + '.json').write_text(json.dumps({
        'scheme': 'repo', 'len_segment': LEN_SEGMENT, 'overlap': OVERLAP, 'seed': SEED,
        'val_is_test': True, 'n_frames': n_rows,
        'sampling': 'every segment with circling plus an equal number without, per video',
        'split': 'stratified random 80/20 train/test, random_state=42'}, indent=2), encoding='utf-8')

def rows_to_frames(start_row, end_row, meta, frame_count):
    stride, first = int(meta.get('stride', 1)), int(meta.get('frame_min', 0))
    return np.arange(start_row * stride + first, min(end_row * stride + first, frame_count))

def frame_scores(probs, meta, frame_count):
    stride, first = int(meta.get('stride', 1)), int(meta.get('frame_min', 0))
    frames = np.arange(frame_count, dtype=np.int64)
    rows = (frames - first) // stride
    valid = (frames >= first) & (rows < len(probs))
    scores = np.zeros(frame_count)
    scores[valid] = probs[rows[valid]]
    return pl.DataFrame({'FrameNum': frames, 'circling_score': scores})

def load_meta(workdir, videos):
    data = Path(workdir) / 'data'
    meta = br.load_meta(data) if (data / br.META_FILENAME).exists() else {}
    missing = sorted(set(videos) - set(meta))
    if missing:
        raise SystemExit(f'{missing} not converted in {data}: run '
                         '`python -m models.mstcn_dlc2action.train --prepare-only` first')
    return meta

def shared_segments(workdir, spans, output_dir=None):
    workdir = Path(workdir)
    data = workdir / 'data'
    videos = sorted(spans)
    meta = load_meta(workdir, videos)
    spans_csv = workdir / 'training_spans.csv'
    pl.DataFrame([{'VideoID': v, 'StartFrame': s, 'EndFrame': e}
                  for v, intervals in spans.items() for s, e in intervals],
                 schema={'VideoID': pl.String, 'StartFrame': pl.Int64, 'EndFrame': pl.Int64}
                 ).write_csv(spans_csv)
    br.ground_truth_to_annotations(spans_csv, data, meta)
    (data / '_annotation_source.json').write_text(json.dumps({
        'ground_truth': str(spans_csv.resolve()), 'label': 'circling', 'fps': None, 'merged': False,
    }, indent=2), encoding='utf-8')
    n_rows = {v: int(meta[v]['n_frames']) for v in videos}
    positives = br.read_positive_intervals(data, videos)
    segments = split_segments(sample_segments(positives, n_rows))
    write_split(segments, workdir / 'splits' / f'{SPLIT}.txt', n_rows)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        segments.write_parquet(Path(output_dir) / 'sampled_segments.parquet')
    counts = segments.group_by('split', 'circling').len().sort('split', 'circling')
    print(f'Sampled segments:\n{counts}', flush=True)
    return segments, {v: meta[v] for v in videos}

def segment_frames(segments, meta, split=None):
    frames = video_frames(sorted(meta))
    if split is not None:
        segments = segments.filter(pl.col('split') == split)
    tables = []
    for index, (video, start, end, part) in enumerate(
            segments.select('video', 'start_row', 'end_row', 'split').iter_rows()):
        rows = rows_to_frames(start, end, meta[video], frames[video])
        tables.append(pl.DataFrame({'video': [video] * len(rows), 'FrameNum': rows,
                                    'segment': np.full(len(rows), index), 'split': [part] * len(rows)}))
    return pl.concat(tables)

def test_frames(segments, meta, scores, spans):
    test = segment_frames(segments, meta, 'test')
    tables = []
    for (video,), rows in test.group_by('video', maintain_order=True):
        frames = rows['FrameNum'].to_numpy()
        score = scores[video]['circling_score'].to_numpy()[frames]
        tables.append(pl.DataFrame({
            'video': [video] * len(frames), 'FrameNum': frames,
            'circling': frame_labels(frames, spans[video]),
            'circling_score': score, 'predicted_circling': score >= THRESHOLD}))
    return pl.concat(tables)
