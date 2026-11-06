import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from models.common.config import (DATA_DIR, FINALIZED_ANNOTATIONS, LABELS, OUTPUTS, label_description, model_name,
                                  read_annotations, video_frames)
from models.common.report import write_report

NAME = model_name('sartab')
SAMPLE_EVERY = 6
WINDOW = 1800
STEP = 900
ROI = 400
BIN = 20
OUTPUT_DIR = OUTPUTS / NAME

def densest_square(x, y, size=ROI, bin_px=BIN):
    width, height = int(np.nanmax(x)) + size, int(np.nanmax(y)) + size
    grid, _, _ = np.histogram2d(x, y, bins=(width // bin_px + 1, height // bin_px + 1),
                                range=((0, (width // bin_px + 1) * bin_px), (0, (height // bin_px + 1) * bin_px)))
    k = size // bin_px
    total = np.pad(grid, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = total[k:, k:] - total[:-k, k:] - total[k:, :-k] + total[:-k, :-k]
    i, j = np.unravel_index(np.argmax(sums), sums.shape)
    return float(i * bin_px), float(j * bin_px), float(i * bin_px + size), float(j * bin_px + size)

def fish_counts(video, frame_count, data_dir=DATA_DIR):
    path = Path(data_dir) / f'{video}.parquet'
    if not path.exists():
        raise SystemExit(f'{path} not found (set CIRCLING_DATA_DIR)')
    columns = pl.scan_parquet(path).collect_schema().names()
    fish = (pl.scan_parquet(path)
            .select('FrameNum', 'X_center', 'Y_center', *(['Sex'] if 'Sex' in columns else []))
            .filter(pl.col('X_center').is_finite() & pl.col('Y_center').is_finite())
            .collect())
    male = fish.filter(pl.col('Sex').cast(pl.String).str.to_lowercase().is_in(['male', 'm'])) \
        if 'Sex' in columns else fish
    if male.is_empty():
        male = fish
    x0, y0, x1, y1 = densest_square(male['X_center'].to_numpy(), male['Y_center'].to_numpy())
    inside = (fish.filter((pl.col('FrameNum') % SAMPLE_EVERY == 0)
                  & pl.col('X_center').is_between(x0, x1, closed='left')
                  & pl.col('Y_center').is_between(y0, y1, closed='left'))
              .group_by('FrameNum').len())
    counts = np.zeros((frame_count - 1) // SAMPLE_EVERY + 1, np.int32)
    index = inside['FrameNum'].to_numpy() // SAMPLE_EVERY
    keep = index < len(counts)
    counts[index[keep]] = inside['len'].to_numpy()[keep]
    return counts, {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}

def window_starts(frame_count):
    return np.arange(0, max(frame_count - WINDOW, 0) + 1, STEP)

def window_table(video, frame_count, counts, spans):
    starts = window_starts(frame_count)
    ends = np.minimum(starts + WINDOW, frame_count)
    two = np.r_[0, np.cumsum(counts == 2)]
    first, last = starts // SAMPLE_EVERY, -(-ends // SAMPLE_EVERY)
    positive = np.zeros(frame_count, bool)
    for start, end in spans:
        positive[start:end + 1] = True
    covered = np.r_[0, np.cumsum(positive)]
    return pl.DataFrame({
        'video': [video] * len(starts), 'start': starts, 'end': ends - 1,
        'two_fish_fraction': (two[last] - two[first]) / np.maximum(last - first, 1),
        'circling': ((covered[ends] - covered[starts]) > 0).astype(np.int8)})

def frame_scores(window_scores, frame_count):
    frames = np.arange(frame_count, dtype=np.int64)
    k = np.clip(np.rint((frames - WINDOW / 2) / STEP).astype(np.int64), 0, len(window_scores) - 1)
    return pl.DataFrame({'FrameNum': frames, 'circling_score': np.asarray(window_scores, np.float64)[k]})

def train(annotations=FINALIZED_ANNOTATIONS, output_dir=OUTPUT_DIR, data_dir=DATA_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spans = read_annotations(annotations)
    frames = video_frames(sorted(spans))
    tables, rois = [], {}
    for video, count in frames.items():
        print(f'Counting fish in the ROI: {video}', flush=True)
        counts, rois[video] = fish_counts(video, count, data_dir)
        tables.append(window_table(video, count, counts, spans[video]))
    windows = pl.concat(tables)
    ids = np.arange(windows.height)
    _, test_ids = train_test_split(ids, test_size=0.2, random_state=42, stratify=windows['circling'].to_numpy())
    split = np.full(windows.height, 'train', dtype='<U5')
    split[test_ids] = 'test'
    windows = windows.with_columns(pl.Series('split', split))
    train_rows, test_rows = windows.filter(pl.col('split') == 'train'), windows.filter(pl.col('split') == 'test')
    model = LogisticRegressionCV()
    model.fit(train_rows.select('two_fish_fraction').to_numpy(), train_rows['circling'].to_numpy())
    windows = windows.with_columns(pl.Series('circling_score', model.predict_proba(
        windows.select('two_fish_fraction').to_numpy())[:, 1]))
    windows.write_parquet(output_dir / 'windows.parquet')
    joblib.dump({'model': model, 'features': ['two_fish_fraction'], 'roi': rois}, output_dir / 'model.joblib')
    (output_dir / 'roi.json').write_text(json.dumps(rois, indent=2) + '\n')

    test = windows.filter(pl.col('split') == 'test').select('video', 'start', 'end', 'circling', 'circling_score')
    report = {
        'label_assumption': f'A window is circling when it overlaps an interval in {label_description(annotations)}.',
        'labels': LABELS,
        'annotations': str(Path(annotations).resolve()),
        'method': 'Replication of SARTAB (Lancaster et al. 2024, Front. Behav. Neurosci. 18:1509369): fish '
                  f'counted in a {ROI} px ROI every {SAMPLE_EVERY}th frame; feature = fraction of sampled frames '
                  f'with exactly two fish in {WINDOW // 30} s windows every {STEP // 30} s; default '
                  'LogisticRegressionCV.',
        'deviations': 'The ROI is the densest 400 px square of the male\'s positions instead of a detector; '
                      'there is no detection-confidence column; labels are windows overlapping annotated circling.',
        'split': 'Stratified random 80/20 train/test split of the windows, random_state=42.',
        'limitation': 'Overlapping windows can occur in both splits; this is not an independent-event or '
                      'new-video evaluation.',
        'score_warning': 'Every frame gets the score of the window centred nearest to it, so spans have '
                         '30 s resolution.',
        'model': 'LogisticRegressionCV (scikit-learn defaults)', 'C': float(model.C_[0]),
        'videos': sorted(frames), 'roi': rois,
        'windows': {s: {'circling': int(g['circling'].sum()), 'other': int((g['circling'] == 0).sum())}
                    for s, g in (('train', train_rows), ('test', test_rows))},
    }
    notes = [report['method'], report['deviations'], report['label_assumption'], report['split']]
    result = write_report(output_dir, 'Circling SARTAB replication', 'SARTAB', test, report, notes, unit='windows')

    scores_dir = output_dir / 'scores'
    scores_dir.mkdir(parents=True, exist_ok=True)
    for video, count in frames.items():
        rows = windows.filter(pl.col('video') == video).sort('start')
        frame_scores(rows['circling_score'].to_numpy(), count).write_parquet(scores_dir / f'{video}_scores.parquet')
        (scores_dir / f'{video}_scores.json').write_text(json.dumps(
            {'frames': count, 'windows': rows.height, 'window': WINDOW, 'step': STEP}, indent=2) + '\n')
    return result

if __name__ == '__main__':
    argparse.ArgumentParser(description='SARTAB replication (two fish in the ROI over 60 s windows)').parse_args()
    train()
