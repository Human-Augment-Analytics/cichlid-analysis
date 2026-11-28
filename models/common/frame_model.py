import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from models.common.annotated_negatives import annotated_negative_mask, read_annotated_negatives
from models.common.config import (DATA_DIR, FINALIZED_ANNOTATIONS, OUTPUTS, model_name, mstcn_workdir,
                                  read_annotations, video_frames)
from models.common.report import protocol, segment_counts, write_report
from models.common.segments import segment_frames, shared_segments
from models.common.turns import turn_table
from models.logistic_regression.features import (FEATURE_SOURCE, PAIR_ANGLES, PAIR_DISTANCES, PAIR_RATES,
                                                   WINDOW, frame_features)
from models.logistic_regression.train import frame_labels

FEATURE_DIR = OUTPUTS / 'frame_features'
CHUNK = 60_000
SCORE_CHUNK = 200_000
PAIR_CLIP = 'ind0+ind1'
DESCRIPTIONS = {
    'pair': 'the logistic regression\'s 50 pose + 52 pair frame features',
    'pairemb': 'the 102 pose + pair frame features plus 32 DINOv2 embedding PCs and emb_present',
    'pairvmae': 'the 102 pose + pair frame features plus 32 VideoMAE embedding PCs and emb_present',
    'pairturns': 'the 102 pose + pair frame features plus the rule-based detector\'s inputs (closest cleaned pair '
                 'distance and orbit rate, turns and close fraction over centred 2/4/8 s windows, start/keep flags)',
    'pairturnsvmae': 'the 102 pose + pair frame features plus the rule-based detector\'s inputs and 32 VideoMAE '
                     'embedding PCs',
    'pairmotion': 'the 50 causal pose features plus body-length-normalized male/partner distances, orientation, '
                  'speed, cleaned angular movement and gap-aware centered 0.5/1/2/4/8/16 s window features, '
                  'including sustained close/orbital-motion fractions',
    'pairmotionemb': 'pairmotion plus 32 DINOv2 embedding PCs and emb_present',
    'pairmotionvmae': 'pairmotion plus 32 VideoMAE embedding PCs and emb_present',
}
COMPONENTS = {'pair': (), 'pairemb': ('pairemb',), 'pairvmae': ('pairvmae',), 'pairturns': ('turns',),
              'pairturnsvmae': ('turns', 'pairvmae'), 'pairmotion': ('pairmotion',),
              'pairmotionemb': ('pairmotion', 'pairemb'), 'pairmotionvmae': ('pairmotion', 'pairvmae')}
BASE_PAIR_COLUMNS = ['pair_count'] + [
    f'{signal}_{stat}'
    for signal in [*PAIR_DISTANCES, *PAIR_RATES,
                   *[f'{angle}_{part}' for angle in PAIR_ANGLES for part in ('sin', 'cos')]]
    for stat in ('mean', 'min', 'max')
]


def base_feature_columns(table, feature):
    """Pair-motion variants replace arbitrary-pair aggregates, retaining pose."""
    return table.drop(BASE_PAIR_COLUMNS) if feature.startswith('pairmotion') else table


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


def extra_tables(workdir, video, feature, meta, frame_count):
    from models.common.pair_motion import pair_motion_table

    return [turn_table(video, frame_count) if part == 'turns'
            else pair_motion_table(workdir, video, meta, frame_count) if part == 'pairmotion'
            else embedding_table(workdir, video, part, meta, frame_count) for part in COMPONENTS[feature]]


def video_features(workdir, video, feature, meta, frame_count, frames=None):
    table = pl.scan_parquet(feature_table(video, frame_count))
    if frames is not None:
        table = table.filter(pl.col('FrameNum').is_in(frames))
    table = base_feature_columns(table.collect(), feature)
    for extra in extra_tables(workdir, video, feature, meta, frame_count):
        table = table.join(extra, on='FrameNum', how='left')
    return table


def score_videos(model, columns, workdir, feature, meta, frames, scores_dir=None):
    scores_dir = Path(scores_dir) if scores_dir is not None else None
    if scores_dir is not None:
        scores_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for video, count in frames.items():
        print(f'Scoring {video}', flush=True)
        table = pl.scan_parquet(feature_table(video, count))
        extras = extra_tables(workdir, video, feature, meta[video], count)
        parts = []
        for start in range(0, count, SCORE_CHUNK):
            chunk = base_feature_columns(table.slice(start, SCORE_CHUNK).collect(), feature)
            for extra in extras:
                chunk = chunk.join(extra.slice(start, SCORE_CHUNK), on='FrameNum', how='left')
            score = model.predict_proba(chunk.select(columns).to_numpy())[:, 1]
            parts.append(pl.DataFrame({'FrameNum': chunk['FrameNum'], 'circling_score': score.astype(np.float64)}))
        scores = pl.concat(parts)
        if scores.height != count or not np.isfinite(scores['circling_score'].to_numpy()).all():
            raise ValueError(f'{video}: expected {count} finite scores, got {scores.height}')
        if scores_dir is not None:
            scores.write_parquet(scores_dir / f'{video}_scores.parquet')
            (scores_dir / f'{video}_scores.json').write_text(json.dumps(
                {'frames': count, 'feature_source': FEATURE_SOURCE, 'features': feature}, indent=2) + '\n')
        out[video] = scores
    return out


def tune_frame_postprocess(train, fit, columns, workdir, feature, meta, counts, spans, output_dir):
    """Fit video-disjoint folds and tune span rules on full held-out videos."""
    from models.common.postprocess_tuning import select_postprocess

    videos = sorted(counts)
    if len(videos) < 2:
        raise ValueError('video-level OOF postprocessing needs at least two videos')
    shuffled = np.asarray(videos)
    np.random.default_rng(42).shuffle(shuffled)
    fold_count = min(3, len(videos))
    folds = [sorted(shuffled[index::fold_count].tolist()) for index in range(fold_count)]
    oof = {}
    for index, validation_videos in enumerate(folds, 1):
        fold_train = train.filter(~pl.col('video').is_in(validation_videos))
        y = fold_train['circling'].to_numpy()
        if len(np.unique(y)) < 2:
            raise ValueError(f'OOF fold {index} has only one training class')
        print(f'OOF postprocessing fold {index}/{fold_count}: validation videos {validation_videos}', flush=True)
        model, _ = fit(fold_train.select(columns).to_numpy(), y, fold_train['segment'].to_numpy())
        scores = score_videos(model, columns, workdir, feature, meta,
                              {video: counts[video] for video in validation_videos})
        oof.update(scores)
    score_dir = Path(output_dir) / 'oof_scores'
    score_dir.mkdir(parents=True, exist_ok=True)
    for video, scores in oof.items():
        scores.write_parquet(score_dir / f'{video}_scores.parquet')
    return select_postprocess(oof, spans, counts, output_dir, folds=folds)


# The widest centred feature window is 8 s (pairturns), so a frame's features reach 4 s to each side.
TEST_CONTEXT_FRAMES = 4 * 30


def near_frames(count, frames, radius=TEST_CONTEXT_FRAMES):
    """Mask of frames within `radius` frames (inclusive) of any of `frames`."""
    edges = np.zeros(count + 1, dtype=np.int64)
    frames = np.asarray(frames, dtype=np.int64)
    np.add.at(edges, np.clip(frames - radius, 0, count), 1)
    np.add.at(edges, np.clip(frames + radius + 1, 0, count), -1)
    return np.cumsum(edges[:-1]) > 0


def append_annotated_training_rows(rows, counts, positives, negatives, radius=TEST_CONTEXT_FRAMES):
    """Add unsampled negatives after the repository split, grouping each annotated run.

    Frames within `radius` of a test frame are left out, so no added training frame shares
    feature-window context with a held-out test frame.
    """
    added = {}
    extra = []
    next_group = int(rows['segment'].max()) + 1
    for video, count in counts.items():
        mask = annotated_negative_mask(count, negatives[video], positives[video])
        starts = mask & ~np.r_[False, mask[:-1]]
        groups = np.cumsum(starts) - 1 + next_group
        next_group += int(starts.sum())
        video_rows = rows.filter(pl.col('video') == video)
        mask[video_rows['FrameNum'].to_numpy()] = False
        mask &= ~near_frames(count, video_rows.filter(pl.col('split') == 'test')['FrameNum'].to_numpy(), radius)
        frames = np.flatnonzero(mask)
        added[video] = len(frames)
        if len(frames):
            extra.append(pl.DataFrame({'video': [video] * len(frames), 'FrameNum': frames,
                                       'segment': groups[frames], 'split': ['train'] * len(frames)},
                                      schema=rows.schema))
    return pl.concat([rows, *extra]), added


def train_frame_model(name, feature, fit, title, model_label, details, annotations=FINALIZED_ANNOTATIONS,
                      tune_postprocess=True, annotated_negatives=None):
    name = model_name(name)
    output_dir = OUTPUTS / name
    workdir = mstcn_workdir()
    spans = read_annotations(annotations)
    segments, meta = shared_segments(workdir, spans, output_dir)
    counts = video_frames(sorted(meta))
    rows = segment_frames(segments, meta)
    if annotated_negatives is not None:
        negatives = read_annotated_negatives(annotated_negatives, counts)
        rows, added = append_annotated_training_rows(rows, counts, spans, negatives)
        print(f'Additional annotated negative training frames: {added}', flush=True)
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
    if annotated_negatives is not None:
        report['annotated_negatives'] = {'source': str(annotated_negatives), 'extra_training_frames': added}
        report['sampling'] += (' Add unsampled annotated negative frames to training after the split, except '
                               f'frames within {TEST_CONTEXT_FRAMES} frames of a test frame.')
    notes = [report['label_assumption'], report['sampling'], report['split'],
             f'Per-frame model on {DESCRIPTIONS[feature]}: every frame of the train segments for fitting, '
             f'every frame of the test segments for the metrics. {details}']
    result = write_report(output_dir, title, model_label, test, report, notes)
    score_videos(model, columns, workdir, feature, meta, counts, output_dir / 'scores')
    if tune_postprocess:
        try:
            settings = tune_frame_postprocess(train, fit, columns, workdir, feature, meta,
                                              counts, spans, output_dir)
            report_path = output_dir / 'metrics.json'
            saved = json.loads(report_path.read_text())
            saved['postprocess_selection'] = {'status': 'selected', 'settings': settings}
            report_path.write_text(json.dumps(saved, indent=2) + '\n')
        except ValueError as error:
            print(f'Postprocess tuning unavailable: {error}; using defaults', flush=True)
            (output_dir / 'postprocess.json').write_text(json.dumps({
                'status': 'unavailable', 'reason': str(error), 'settings': {},
            }, indent=2) + '\n')
    else:
        (output_dir / 'postprocess.json').write_text(json.dumps({
            'status': 'disabled', 'settings': {},
        }, indent=2) + '\n')
    return result

