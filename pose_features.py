"""Causal per-frame pose features used alongside pair-Parquet measurements."""

import numpy as np
import polars as pl
from pose_change import KEYPOINTS, TRACK_KEYS, keypoint_displacements

WINDOW = 30
MIDLINE = ('Nose', 'Head', 'Spine1', 'Spine2', 'Spine3', 'Spine4', 'Peduncle', 'TailTip')


def pose_frame_features(poses, start, end):
    if start < 0 or end < start:
        raise ValueError('Invalid frame range')
    if poses.select('project_id', 'day_label').unique().height > 1:
        raise ValueError('Provide poses for exactly one video')
    coords = [f'{k}_{axis}' for k in KEYPOINTS for axis in ('x', 'y')]
    poses = poses.select(*TRACK_KEYS, 'FrameNum', *coords).with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c).cast(pl.Float64)).otherwise(None).alias(c)
        for c in coords
    ]).sort([*TRACK_KEYS, 'FrameNum'])
    lengths = [((pl.col(f'{a}_x') - pl.col(f'{b}_x')).pow(2)
                + (pl.col(f'{a}_y') - pl.col(f'{b}_y')).pow(2)).sqrt()
               for a, b in zip(MIDLINE[:-1], MIDLINE[1:])]
    length = sum(lengths)
    chord = ((pl.col('Nose_x') - pl.col('TailTip_x')).pow(2)
             + (pl.col('Nose_y') - pl.col('TailTip_y')).pow(2)).sqrt()
    poses = poses.with_columns(length.alias('_length'),
        pl.arctan2(pl.col('Nose_y') - pl.col('Spine1_y'),
                   pl.col('Nose_x') - pl.col('Spine1_x')).alias('_heading'))
    delta = pl.col('_heading').diff().over(TRACK_KEYS)
    adjacent = pl.col('FrameNum').diff().over(TRACK_KEYS) == 1
    poses = poses.with_columns(
        pl.when(pl.col('_length') > 1).then(1 - chord / pl.col('_length')).alias('bend'),
        pl.when(adjacent).then(pl.arctan2(delta.sin(), delta.cos()).abs()).alias('turn'))
    movement = keypoint_displacements(poses)
    rows = movement.join(poses.select(*TRACK_KEYS, 'FrameNum', '_length', 'bend', 'turn'),
                         on=[*TRACK_KEYS, 'FrameNum'], how='left')
    rows = rows.with_columns([
        pl.when(pl.col('_length') > 1)
        .then((pl.col(k) / pl.col('_length')).log1p()).otherwise(None).alias(k)
        for k in KEYPOINTS
    ])
    signals = [*KEYPOINTS, 'bend', 'turn']
    aggregate = rows.group_by('FrameNum').agg(
        pl.len().cast(pl.Float64).alias('track_count'),
        *[expr for k in signals for expr in
          (pl.col(k).mean().alias(f'{k}_mean'), pl.col(k).max().alias(f'{k}_max'))])
    first = max(0, start - WINDOW + 1)
    frames = pl.DataFrame({'FrameNum': np.arange(first, end + 1, dtype=np.int64)})
    frames = frames.join(aggregate, on='FrameNum', how='left').sort('FrameNum')
    frames = frames.with_columns(pl.col('track_count').fill_null(0))
    base = [c for c in frames.columns if c != 'FrameNum']
    frames = frames.with_columns([
        pl.col(c).rolling_mean(WINDOW, min_samples=1).alias(f'{c}_trailing30') for c in base
    ])
    return frames.filter(pl.col('FrameNum') >= start).with_columns(pl.exclude('FrameNum').cast(pl.Float32))
