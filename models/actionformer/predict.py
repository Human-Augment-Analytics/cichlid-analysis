"""Overlapping-window ActionFormer inference with native interval confidence."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from models.actionformer.model import WINDOW, infer_windows, suppress_cross_window_duplicates
from models.actionformer.postprocess import prediction_table
from models.actionformer.train import NAMES, normalize
from models.actionformer.tuning import DECODE_MIN_SCORE, load_inference_settings, proposal_summary
from models.common.config import OUTPUTS, model_name, mstcn_workdir, video_frames
from models.common.frame_model import (COMPONENTS, base_feature_columns, embedding_table, feature_table)
from models.common.pair_motion import pair_motion_cache_file
from models.common.segments import load_meta
from models.common.turns import turn_table

STRIDE = WINDOW // 2
CHUNK_WINDOWS = 32


def window_starts(frame_count, window=WINDOW, stride=STRIDE):
    if frame_count <= window:
        return [0]
    starts = list(range(0, frame_count - window + 1, stride))
    if starts[-1] != frame_count - window:
        starts.append(frame_count - window)
    return starts


def feature_chunk(workdir, video, feature, meta, frame_count, first, last):
    """Read the half-open frame range [first, last) from cached features."""
    frames = np.arange(first, last, dtype=np.int64)
    in_chunk = (pl.col('FrameNum') >= first) & (pl.col('FrameNum') < last)
    video_meta = meta.get(video, meta)
    table = pl.scan_parquet(feature_table(video, frame_count)).filter(in_chunk).collect()
    table = base_feature_columns(table, feature)
    for part in COMPONENTS[feature]:
        if part == 'pairmotion':
            path = pair_motion_cache_file(workdir, video, video_meta, frame_count)
            extra = pl.scan_parquet(path).filter(in_chunk).collect()
        elif part == 'turns':
            extra = turn_table(video, frame_count).filter(in_chunk)
        else:
            extra = embedding_table(workdir, video, part, video_meta, frame_count).filter(in_chunk)
        table = table.join(extra, on='FrameNum', how='left')
    if table.height != len(frames) or table['FrameNum'].to_numpy().tolist() != frames.tolist():
        raise ValueError(f'{video}: incomplete feature rows for frames {first}:{last}')
    return table


def decode_video(model, video, frame_count, columns, mean, std, workdir, features, meta, device):
    """Decode raw proposals over a complete video using fold-specific normalization."""
    starts = window_starts(frame_count)
    decoded = []
    for chunk_start in range(0, len(starts), CHUNK_WINDOWS):
        chunk = starts[chunk_start:chunk_start + CHUNK_WINDOWS]
        first, last = chunk[0], min(frame_count, chunk[-1] + WINDOW)
        table = feature_chunk(workdir, video, features, meta, frame_count, first, last)
        positions = {int(frame): i for i, frame in enumerate(table['FrameNum'].to_numpy())}
        array = table.select(columns).to_numpy().astype(np.float32)
        windows = []
        for start in chunk:
            end = min(frame_count, start + WINDOW)
            values = normalize(array[[positions[frame] for frame in range(start, end)]], mean, std)
            windows.append((start, values))
        decoded.extend(infer_windows(model, windows, device, min_score=DECODE_MIN_SCORE))
    return decoded


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    from models.actionformer.vendor.libs.modeling import make_meta_arch
    model = make_meta_arch('LocPointTransformer', **checkpoint['config']['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    # Older checkpoints hid low-confidence proposals inside the official decoder.
    # Preserve raw proposals here; selection is applied after cross-window NMS.
    model.test_pre_nms_thresh = min(float(model.test_pre_nms_thresh), DECODE_MIN_SCORE)
    model.test_min_score = 0.0
    model.test_duration_thresh = 0.0
    model.to(device).eval()
    return model, checkpoint


def predict(features='pairmotion', min_score=None, min_seconds=None, merge_seconds=None,
            output_dir=None, device=None):
    name = model_name(NAMES[features])
    model_dir = OUTPUTS / name
    settings = load_inference_settings(model_dir, min_score, min_seconds, merge_seconds)
    min_score, min_seconds, merge_seconds = (settings['min_score'], settings['min_seconds'],
                                             settings['merge_seconds'])
    checkpoint_path = model_dir / 'model.pt'
    if not checkpoint_path.exists():
        raise SystemExit(f'{checkpoint_path} not found: run `python -m models.actionformer.train '
                         f'--features {features}` first')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    columns = checkpoint['feature_columns']
    mean, std = checkpoint['normalization_mean'], checkpoint['normalization_std']
    frames = video_frames()
    workdir = mstcn_workdir()
    meta = load_meta(workdir, sorted(frames))
    output_dir = Path(output_dir or OUTPUTS / 'predictions' / name)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposal_dir = model_dir / 'proposals'
    proposal_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob('*.csv'):
        if stale.stem not in frames:
            stale.unlink()
    paths = []
    diagnostics = {}
    for video, count in frames.items():
        decoded = decode_video(model, video, count, columns, mean, std, workdir, features, meta, device)
        pl.DataFrame({
            'start': [float(item[0][0]) for item in decoded],
            'end': [float(item[0][1]) for item in decoded],
            'confidence': [item[1] for item in decoded],
        }, schema={'start': pl.Float64, 'end': pl.Float64, 'confidence': pl.Float64}).write_parquet(
            proposal_dir / f'{video}.parquet')
        deduplicated = suppress_cross_window_duplicates(decoded)
        proposals = [item for item in deduplicated if item[1] >= min_score]
        predictions = prediction_table(video, proposals, count, min_seconds=min_seconds,
                                       merge_seconds=merge_seconds)
        path = output_dir / f'{video}.csv'
        predictions.write_csv(path)
        paths.append(path)
        diagnostics[video] = proposal_summary(decoded, deduplicated, predictions)
        print(f'{video}: {len(decoded)} raw, {len(deduplicated)} post-NMS, '
              f'{predictions.height} retained intervals -> {path}', flush=True)
        if not decoded:
            print(f'WARNING: {video}: ActionFormer decoded no proposals above {DECODE_MIN_SCORE}; '
                  'the checkpoint may have collapsed to background.', flush=True)
        elif predictions.is_empty():
            print(f'WARNING: {video}: proposals exist but min_score={min_score} and '
                  f'min_seconds={min_seconds} removed every interval. See {proposal_dir / f"{video}.parquet"}',
                  flush=True)
    (output_dir / 'diagnostics.json').write_text(json.dumps({
        'settings': settings, 'decode_min_score': DECODE_MIN_SCORE, 'videos': diagnostics,
    }, indent=2) + '\n')
    (output_dir / 'inference.json').write_text(json.dumps({
        'model': name, 'features': features, 'window_frames': WINDOW, 'window_stride': STRIDE,
        'min_score': min_score, 'min_seconds': min_seconds, 'merge_seconds': merge_seconds,
        'decode_min_score': DECODE_MIN_SCORE, 'diagnostics': diagnostics,
        'nms': 'Gaussian Soft-NMS within windows; class-agnostic hard NMS across windows (IoU 0.3)',
        'intervals': 'native ActionFormer segments with Confidence; inclusive StartFrame/EndFrame',
    }, indent=2) + '\n')
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ActionFormer native temporal-interval prediction')
    parser.add_argument('--features', choices=sorted(NAMES), default='pairmotion')
    parser.add_argument('--min-score', type=float, default=None,
                        help='Minimum confidence (default: validation-selected, otherwise 0.001)')
    parser.add_argument('--min-seconds', type=float, default=None,
                        help='Minimum interval duration (default: validation-selected, otherwise 0)')
    parser.add_argument('--merge-seconds', type=float, default=None,
                        help='Gap to merge (default: validation-selected, otherwise 0)')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    predict(args.features, args.min_score, args.min_seconds, args.merge_seconds, args.output_dir, args.device)
