import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.common.config import (DATA_DIR, FINALIZED_ANNOTATIONS, LABELS, OUTPUTS, label_description, model_name,
                                  mstcn_workdir, read_annotations, video_frames)
from models.common.report import write_report

NAME = model_name('sartab')
SAMPLE_EVERY = 6
WINDOW = 1800
STEP = 900
ROI = 400
BIN = 20
OUTPUT_DIR = OUTPUTS / NAME
PM_COLUMNS = [
    'pm_pair_present',
    'pm_centroid_distance_bl',
    'pm_speed_mean_bl_s',
    'pm_closing_speed_bl_s',
    'pm_abs_d_orbital_rad_s',
    'pm_w4s_close_frac',
    'pm_w4s_mutual_facing_frac',
    'pm_w4s_coverage',
    'pm_w4s_orbit_turns_net',
    'pm_w4s_orbit_turns_abs',
    'pm_w4s_corotation_mean',
]
PM_WINDOW_COLUMNS = [f'{column}_60s_mean' for column in PM_COLUMNS]
FEATURES = ['two_fish_fraction', *PM_WINDOW_COLUMNS]


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


def pair_motion_window_features(motion, starts, ends):
    """Aggregate cached engineered frame features over exact 60 s windows."""
    result = {}
    for name in PM_COLUMNS:
        values = motion[name].to_numpy().astype(np.float64)
        valid = np.isfinite(values)
        total = np.r_[0.0, np.cumsum(np.where(valid, values, 0.0))]
        count = np.r_[0, np.cumsum(valid, dtype=np.int64)]
        valid_count = count[ends] - count[starts]
        sums = total[ends] - total[starts]
        result[f'{name}_60s_mean'] = np.where(valid_count, sums / np.maximum(valid_count, 1), np.nan)
    return result


def window_table(video, frame_count, counts, spans, motion):
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
        # Fit/evaluate on nonoverlapping windows; overlapping windows are still
        # used for dense inference and are never split across train and test.
        'fit_window': starts % WINDOW == 0,
        'two_fish_fraction': (two[last] - two[first]) / np.maximum(last - first, 1),
        **pair_motion_window_features(motion, starts, ends),
        'circling': ((covered[ends] - covered[starts]) > 0).astype(np.int8)})


def frame_scores(window_scores, starts, ends, frame_count):
    """Linearly interpolate between window centers instead of 30 s steps."""
    frames = np.arange(frame_count, dtype=np.int64)
    centers = (starts + ends - 1) / 2.0
    scores = np.interp(frames, centers, np.asarray(window_scores, dtype=np.float64))
    return pl.DataFrame({'FrameNum': frames, 'circling_score': scores})


def train(annotations=FINALIZED_ANNOTATIONS, output_dir=OUTPUT_DIR, data_dir=DATA_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spans = read_annotations(annotations)
    frames = video_frames(sorted(spans))
    workdir = mstcn_workdir()
    from models.intervals.mstcn_dlc2action.train import prepare_data
    from models.common.pair_motion import pair_motion_table
    from models.common.segments import load_meta

    prepare_data(workdir, data_dir, 'pairmotion', sorted(spans))
    meta = load_meta(workdir, sorted(spans))
    tables, rois = [], {}
    for video, count in frames.items():
        print(f'Counting fish in the ROI and aggregating pair motion: {video}', flush=True)
        counts, rois[video] = fish_counts(video, count, data_dir)
        motion = pair_motion_table(workdir, video, meta[video], count, data_dir)
        tables.append(window_table(video, count, counts, spans[video], motion))
    windows = pl.concat(tables)
    fit_windows = windows.filter(pl.col('fit_window'))
    ids = np.arange(fit_windows.height)
    _, test_ids = train_test_split(ids, test_size=0.2, random_state=42,
                                   stratify=fit_windows['circling'].to_numpy())
    split = np.full(fit_windows.height, 'train', dtype='<U5')
    split[test_ids] = 'test'
    fit_windows = fit_windows.with_columns(pl.Series('split', split))
    train_rows, test_rows = fit_windows.filter(pl.col('split') == 'train'), fit_windows.filter(pl.col('split') == 'test')
    model = make_pipeline(
        SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegressionCV(Cs=np.logspace(-3, 2, 11), scoring='average_precision', class_weight='balanced',
                             max_iter=2000, solver='lbfgs', n_jobs=-1),
    )
    model.fit(train_rows.select(FEATURES).to_numpy(), train_rows['circling'].to_numpy())
    windows = windows.with_columns(pl.Series('circling_score', model.predict_proba(
        windows.select(FEATURES).to_numpy())[:, 1]))
    windows.write_parquet(output_dir / 'windows.parquet')
    joblib.dump({'model': model, 'features': FEATURES, 'feature_set': 'pairmotion', 'roi': rois},
                output_dir / 'model.joblib')
    (output_dir / 'roi.json').write_text(json.dumps(rois, indent=2) + '\n')

    test = test_rows.select('video', 'start', 'end', 'circling').with_columns(
        pl.Series('circling_score', model.predict_proba(test_rows.select(FEATURES).to_numpy())[:, 1]))
    search = model[-1]
    report = {
        'label_assumption': f'A window is circling when it overlaps an interval in {label_description(annotations)}.',
        'labels': LABELS,
        'annotations': str(Path(annotations).resolve()),
        'method': 'SARTAB-inspired fish occupancy plus shared pair-motion features: fish counted in a '
                  f'{ROI} px ROI every {SAMPLE_EVERY}th frame; {WINDOW // 30} s windows every {STEP // 30} s; '
                  '60 s means of normalized pair motion and 4 s temporal-consistency features; balanced '
                  'LogisticRegressionCV with median imputation and standard scaling.',
        'deviations': 'The ROI is the densest 400 px square of the male positions; pair-motion aggregates are '
                      'added to the original two-fish fraction. This is an augmented SARTAB variant, not a '
                      'strict replication.',
        'split': 'Stratified random 80/20 train/test split of NONOVERLAPPING 60 s windows, random_state=42. '
                 'Overlapping inference windows are excluded from fitting and internal metrics.',
        'limitation': 'Neighboring nonoverlapping windows and events can occur in both splits; this is not a '
                      'new-video evaluation. Whole-video repository metrics remain the primary comparison.',
        'score_warning': 'Frame scores are linearly interpolated between 60 s window centers spaced 30 s apart.',
        'model': 'LogisticRegressionCV (balanced, average-precision CV)', 'C': float(search.C_[0]),
        'features': FEATURES, 'videos': sorted(frames), 'roi': rois,
        'windows': {s: {'circling': int(g['circling'].sum()), 'other': int((g['circling'] == 0).sum())}
                    for s, g in (('train', train_rows), ('test', test_rows))},
    }
    notes = [report['method'], report['deviations'], report['label_assumption'], report['split']]
    result = write_report(output_dir, 'Circling SARTAB + pair-motion windows', 'SARTAB + pair motion',
                          test, report, notes, unit='windows')

    scores_dir = output_dir / 'scores'
    scores_dir.mkdir(parents=True, exist_ok=True)
    for video, count in frames.items():
        rows = windows.filter(pl.col('video') == video).sort('start')
        frame_scores(rows['circling_score'].to_numpy(), rows['start'].to_numpy(), rows['end'].to_numpy(),
                     count).write_parquet(scores_dir / f'{video}_scores.parquet')
        (scores_dir / f'{video}_scores.json').write_text(json.dumps(
            {'frames': count, 'windows': rows.height, 'window': WINDOW, 'step': STEP,
             'interpolation': 'linear between window centers', 'feature_set': 'pairmotion'}, indent=2) + '\n')
    return result


if __name__ == '__main__':
    argparse.ArgumentParser(description='SARTAB + engineered pair-motion features over 60 s windows').parse_args()
    train()
