import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from models.common.config import (DATA_DIR, FINALIZED_ANNOTATIONS, OUTPUTS, model_name, mstcn_workdir,
                                  read_annotations, video_frames)
from models.common.report import protocol, segment_counts, write_report
from models.common.segments import segment_frames, shared_segments
from models.logistic_regression.features import FEATURE_SOURCE, WINDOW, frame_features
from models.logistic_regression.train import frame_labels

FEATURE_DIR = OUTPUTS / 'frame_features'
CHUNK = 60_000
SCORE_CHUNK = 200_000
PAIR_CLIP = 'ind0+ind1'
DESCRIPTIONS = {
    'pair': 'the logistic regression\'s 50 pose + 52 pair frame features',
    'pairemb': 'the 102 pose + pair frame features plus 32 DINOv2 embedding PCs and emb_present',
    'pairvmae': 'the 102 pose + pair frame features plus 32 VideoMAE embedding PCs and emb_present',
}

def feature_table(video, frame_count, data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    inputs = [data_dir / f'{video}.parquet', data_dir / f'{video}_pairs.parquet']
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise SystemExit(f'missing input file(s): {missing} (set CIRCLING_DATA_DIR)')
    signature = {'frames': frame_count, 'feature_source': FEATURE_SOURCE,
                 'inputs': {p.name: p.stat().st_size for p in inputs}}
    path = FEATURE_DIR / f'{video}.parquet'
    info = path.with_suffix('.json')
    if path.exists() and info.exists() and json.loads(info.read_text()) == signature:
        return path
    print(f'Building frame features: {video}', flush=True)
    chunks = []
    for start in range(0, frame_count, CHUNK):
        end = min(frame_count - 1, start + CHUNK - 1)
        poses = (pl.scan_parquet(inputs[0])
                 .filter(pl.col('FrameNum').is_between(max(0, start - WINDOW), end)).collect())
        pairs = (pl.scan_parquet(inputs[1])
                 .filter(pl.col('FrameNum').is_between(start, end)).collect())
        chunks.append(frame_features(poses, pairs, start, end))
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(chunks).write_parquet(path)
    info.write_text(json.dumps(signature, indent=2) + '\n')
    return path

def embedding_table(workdir, video, feature, meta, frame_count):
    data = Path(workdir) / 'data'
    path = data / f'{video}_{feature}.pickle'
    if not path.exists():
        raise SystemExit(f'{path} not found: run `python -m models.mstcn_dlc2action.train '
                         f'--prepare-only --features {feature}` first')
    with open(path, 'rb') as f:
        values = np.asarray(pickle.load(f)[PAIR_CLIP], np.float32)
    n_pair = len((meta.get('pair_stats') or {}).get('pair_features') or [])
    info = data / f'_{feature}_info.json'
    names = json.loads(info.read_text())['feature_names'] if info.exists() else None
    names = (names or [f'col{i}' for i in range(values.shape[1])])[n_pair:]
    values = values[:, n_pair:]
    stride, first = int(meta.get('stride', 1)), int(meta.get('frame_min', 0))
    frames = np.arange(frame_count, dtype=np.int64)
    rows = (frames - first) // stride
    valid = (frames >= first) & (rows < len(values))
    out = np.full((frame_count, values.shape[1]), np.nan, np.float32)
    out[valid] = values[rows[valid]]
    return pl.DataFrame({'FrameNum': frames, **{f'{feature}_{n}': out[:, i] for i, n in enumerate(names)}})

def video_features(workdir, video, feature, meta, frame_count, frames=None):
    table = pl.scan_parquet(feature_table(video, frame_count))
    if frames is not None:
        table = table.filter(pl.col('FrameNum').is_in(frames))
    table = table.collect()
    if feature != 'pair':
        table = table.join(embedding_table(workdir, video, feature, meta, frame_count),
                           on='FrameNum', how='left')
    return table

def score_videos(model, columns, workdir, feature, meta, frames, scores_dir):
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for video, count in frames.items():
        print(f'Scoring {video}', flush=True)
        table = pl.scan_parquet(feature_table(video, count))
        embedding = (embedding_table(workdir, video, feature, meta[video], count)
                     if feature != 'pair' else None)
        parts = []
        for start in range(0, count, SCORE_CHUNK):
            chunk = table.slice(start, SCORE_CHUNK).collect()
            if embedding is not None:
                chunk = chunk.join(embedding.slice(start, SCORE_CHUNK), on='FrameNum', how='left')
            score = model.predict_proba(chunk.select(columns).to_numpy())[:, 1]
            parts.append(pl.DataFrame({'FrameNum': chunk['FrameNum'], 'circling_score': score.astype(np.float64)}))
        scores = pl.concat(parts)
        if scores.height != count or not np.isfinite(scores['circling_score'].to_numpy()).all():
            raise ValueError(f'{video}: expected {count} finite scores, got {scores.height}')
        scores.write_parquet(scores_dir / f'{video}_scores.parquet')
        (scores_dir / f'{video}_scores.json').write_text(json.dumps(
            {'frames': count, 'feature_source': FEATURE_SOURCE, 'features': feature}, indent=2) + '\n')
        out[video] = scores
    return out

def train_frame_model(name, feature, fit, title, model_label, details, annotations=FINALIZED_ANNOTATIONS):
    name = model_name(name)
    output_dir = OUTPUTS / name
    workdir = mstcn_workdir()
    spans = read_annotations(annotations)
    segments, meta = shared_segments(workdir, spans, output_dir)
    counts = video_frames(sorted(meta))
    rows = segment_frames(segments, meta)
    tables = []
    for (video,), selected in rows.group_by('video', maintain_order=True):
        features = video_features(workdir, video, feature, meta[video], counts[video],
                                  selected['FrameNum'].to_numpy())
        table = selected.join(features, on='FrameNum', how='left', validate='1:1')
        tables.append(table.with_columns(pl.Series(
            'circling', frame_labels(table['FrameNum'].to_numpy(), spans[video]))))
    table = pl.concat(tables)
    columns = [c for c in table.columns if c not in ('video', 'FrameNum', 'segment', 'split', 'circling')]
    train, test = table.filter(pl.col('split') == 'train'), table.filter(pl.col('split') == 'test')
    print(f'Fitting {train.height:,} train frames ({len(columns)} features); '
          f'evaluating {test.height:,} test frames', flush=True)
    model, fitted = fit(train.select(columns).to_numpy(), train['circling'].to_numpy(),
                        train['segment'].to_numpy())
    joblib.dump({'model': model, 'features': columns, 'feature_set': feature, 'seed': 42,
                 'feature_source': FEATURE_SOURCE}, output_dir / 'model.joblib')
    score = model.predict_proba(test.select(columns).to_numpy())[:, 1]
    test = test.select('video', 'FrameNum', 'circling').with_columns(pl.Series('circling_score', score))
    report = protocol(annotations, model_label, features=feature, feature_columns=columns,
                      segments=segment_counts(segments), train_frames=train.height,
                      train_positive_frames=int(train['circling'].sum()), **fitted)
    notes = [report['label_assumption'], report['sampling'], report['split'],
             f'Per-frame model on {DESCRIPTIONS[feature]}: every frame of the train segments for fitting, '
             f'every frame of the test segments for the metrics. {details}']
    result = write_report(output_dir, title, model_label, test, report, notes)
    score_videos(model, columns, workdir, feature, meta, counts, output_dir / 'scores')
    return result

