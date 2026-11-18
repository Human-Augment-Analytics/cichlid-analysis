import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from models.common.config import DATA_DIR, OUTPUTS
from models.rule_based import circling_cleaning as cc

FPS = 30
CLOSE_PX = 160
MAX_RATE = 15
WINDOWS = (2, 4, 8)
START_TURNS, KEEP_TURNS, CLOSE_FRACTION = 0.75, 0.4, 0.6
VERSION = 1
FEATURE_DIR = OUTPUTS / 'frame_features'
PAIR_COLUMNS = ['FrameNum', 'TrackID_1', 'TrackID_2', 'centroid_distance_px', 'd_orbital_rad_s']


def flag_fish(fish, code):
    if code in cc.FLOOR:
        return cc.flag_fish(fish, code), True
    fish = fish.copy()
    fish['reflection'] = False
    offset = np.hypot(fish.X_center - fish.midline_centroid_x, fish.Y_center - fish.midline_centroid_y)
    fish['keypoint_jump'] = offset > fish.body_length_px.median()
    return fish, False


def closest_pair(video, frame_count, data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    fish = pd.read_parquet(data_dir / f'{video}.parquet', columns=cc.FISH_COLS)
    fish, floor_known = flag_fish(fish, video.split('_')[0])
    pairs = cc.flag_pairs(pd.read_parquet(data_dir / f'{video}_pairs.parquet', columns=PAIR_COLUMNS), fish)
    pairs = (pairs[pairs.keep].sort_values(['FrameNum', 'centroid_distance_px'])
             .drop_duplicates('FrameNum').set_index('FrameNum').reindex(pd.RangeIndex(0, frame_count)))
    return pairs, floor_known


def turn_features(video, frame_count, data_dir=DATA_DIR):
    pairs, floor_known = closest_pair(video, frame_count, data_dir)
    distance = pairs.centroid_distance_px.to_numpy(float)
    rate = pairs.d_orbital_rad_s.where(pairs.d_orbital_rad_s.abs() <= MAX_RATE).fillna(0).to_numpy(float)
    close = np.nan_to_num(distance, nan=1e9) < CLOSE_PX
    columns = {'FrameNum': np.arange(frame_count, dtype=np.int64), 'rule_closest_dist_px': distance,
               'rule_orbit_rate': rate, 'rule_close': close.astype(float)}
    for seconds in WINDOWS:
        window = int(seconds * FPS)
        turns = (pd.Series(rate * close / FPS).rolling(window, center=True, min_periods=window // 2).sum().abs()
                 .to_numpy() / (2 * np.pi))
        fraction = (pd.Series(close.astype(float)).rolling(window, center=True, min_periods=window // 2).mean()
                    .to_numpy())
        columns[f'rule_turns_{seconds}s'] = turns
        columns[f'rule_close_frac_{seconds}s'] = fraction
    mostly_close = np.nan_to_num(columns['rule_close_frac_4s']) >= CLOSE_FRACTION
    turns = np.nan_to_num(columns['rule_turns_4s'])
    columns['rule_start'] = ((turns >= START_TURNS) & mostly_close).astype(float)
    columns['rule_keep'] = ((turns >= KEEP_TURNS) & mostly_close).astype(float)
    columns['rule_floor_known'] = np.full(frame_count, float(floor_known))
    return pl.DataFrame(columns).with_columns(pl.exclude('FrameNum').cast(pl.Float32))


def turn_table(video, frame_count, data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    inputs = [data_dir / f'{video}.parquet', data_dir / f'{video}_pairs.parquet']
    signature = {'frames': frame_count, 'version': VERSION, 'close_px': CLOSE_PX, 'windows': list(WINDOWS),
                 'inputs': {p.name: p.stat().st_size for p in inputs}}
    path = FEATURE_DIR / f'{video}_turns.parquet'
    info = path.with_suffix('.json')
    if path.exists() and info.exists() and json.loads(info.read_text()) == signature:
        return pl.read_parquet(path)
    print(f'Building rule-based turn features: {video}', flush=True)
    table = turn_features(video, frame_count, data_dir)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(path)
    info.write_text(json.dumps(signature, indent=2) + '\n')
    return table
