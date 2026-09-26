"""Find sustained positive predictions and render review clips."""
import json
import math
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import polars as pl

from circling_timeline import score_window
from circling_clips import export_circling_clips


def score_video(model_path, video_path, frame_count, output_dir):
    """Score in bounded chunks; cache by model and input file fingerprints."""
    model_path, video_path, output_dir = Path(model_path), Path(video_path), Path(output_dir)
    poses = video_path.with_suffix('.parquet')
    pairs = video_path.with_name(f'{video_path.stem}_pairs.parquet')
    signature = {'frames': frame_count, 'files': [
        [str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns]
        for p in (model_path, poses, pairs, Path(__file__), Path(__file__).with_name('circling_model.py'),
                  Path(__file__).with_name('pose_features.py'), Path(__file__).with_name('circling_timeline.py'))]}
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / f'{video_path.stem}_scores.parquet'
    meta = cache.with_suffix('.json')
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == signature:
        return pl.read_parquet(cache)
    chunks = []
    for start in range(0, frame_count, 60000):
        chunks.append(score_window(model_path, poses, start, min(frame_count - 1, start + 59999)))
    result = pl.concat(chunks)
    result.write_parquet(cache)
    meta.write_text(json.dumps(signature))
    return result


def find_positive_spans(predictions, fps=30., threshold=.5, fraction=.9, seconds=5.):
    """Greedily merge qualifying windows, retaining >=fraction over each union.

    Start from the shortest window strictly longer than `seconds`. Merge
    overlapping or touching windows only when the combined interval qualifies.
    Skip overlapping windows that would invalidate the previous interval.
    Results are disjoint and every returned interval satisfies both conditions.
    """
    if not np.isfinite(fps) or fps <= 0 or seconds < 0 or not 0 < fraction <= 1 or not 0 <= threshold <= 1:
        raise ValueError('Invalid span detection settings')
    frames = predictions['FrameNum'].to_numpy()
    scores = predictions['circling_score'].to_numpy()
    if not np.isfinite(scores).all() or (len(frames) > 1 and not np.all(np.diff(frames) == 1)):
        raise ValueError('Need finite scores for consecutive frames')
    length = math.floor(seconds * fps) + 1
    # Source FPS is approximately 29.999994; use the nominal 30 FPS passed
    # by the notebook so 300 frames (10 seconds) do not pass the strict limit.
    prefix = np.r_[0, np.cumsum(scores >= threshold)]
    candidates = np.flatnonzero(prefix[length:] - prefix[:-length] >= math.ceil(fraction * length))
    intervals = []
    for start in candidates:
        end = int(start + length - 1)
        start = int(start)
        if intervals and start <= intervals[-1][1] + 1:
            previous_start, previous_end = intervals[-1]
            count = int(prefix[end + 1] - prefix[previous_start])
            if count >= math.ceil(fraction * (end - previous_start + 1)):
                intervals[-1] = (previous_start, end)
            elif start > previous_end:
                intervals.append((start, end))
        else:
            intervals.append((start, end))
    rows = []
    for index, (start, end) in enumerate(intervals, 1):
        n = end - start + 1
        count = int(prefix[end + 1] - prefix[start])
        rows.append(dict(span=index, start_frame=int(frames[start]), end_frame=int(frames[end]),
                         duration_seconds=n / fps, positive_frames=count, frame_count=n,
                         positive_fraction=count / n))
    return pl.DataFrame(rows, schema={'span': pl.Int64, 'start_frame': pl.Int64, 'end_frame': pl.Int64,
        'duration_seconds': pl.Float64, 'positive_frames': pl.Int64, 'frame_count': pl.Int64,
        'positive_fraction': pl.Float64})


def annotate_spans(spans, video_path, fps=30.):
    """Add review metadata; TrackIDs are observed, not assigned participants."""
    video_path = Path(video_path)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid frame rate')

    def timestamp(frame):
        milliseconds = round(frame * 1000 / fps)
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, remainder = divmod(remainder, 1_000)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{remainder:03d}'

    metadata = []
    poses = None
    if not spans.is_empty():
        poses = (pl.scan_parquet(video_path.with_suffix('.parquet'))
                 .select('FrameNum', 'TrackID')
                 .filter(pl.col('FrameNum').is_between(
                     spans['start_frame'].min(), spans['end_frame'].max()))
                 .collect())
    for row in spans.iter_rows(named=True):
        start, end = row['start_frame'], row['end_frame']
        track_ids = (poses.filter(pl.col('FrameNum').is_between(start, end))
                     ['TrackID'].drop_nulls().unique().sort().to_list())
        metadata.append({
            'Video ID': video_path.stem,
            'Event ID': f'{video_path.stem}-{start:07d}-{end:07d}',
            'Start Frame': start,
            'End Frame': end,
            'Start Time': timestamp(start),
            'End Time': timestamp(end + 1),
            'Behavior': 'Predicted circling',
            'Fish IDs': ', '.join(map(str, track_ids)) if track_ids else 'Unknown',
            'Confidence / Notes': (
                f"{row['positive_fraction']:.1%} positive frames; "
                'fish involvement unverified'),
        })
    schema = {
        'Video ID': pl.String, 'Event ID': pl.String,
        'Start Frame': pl.Int64, 'End Frame': pl.Int64,
        'Start Time': pl.String, 'End Time': pl.String,
        'Behavior': pl.String, 'Fish IDs': pl.String,
        'Confidence / Notes': pl.String,
    }
    return pl.concat([pl.DataFrame(metadata, schema=schema),
                      spans.drop('start_frame', 'end_frame')], how='horizontal_extend')


def load_sustained_spans(predictions, video_path, csv_path, fps=30., threshold=.5, seconds=7.5):
    """Read a saved review table, or compute and save it on a cache miss."""
    csv_path = Path(csv_path)
    parquet_path = csv_path.with_suffix('.parquet')
    if csv_path.exists():
        spans = pl.read_csv(csv_path, schema_overrides={
            'Video ID': pl.String, 'Event ID': pl.String,
            'Start Frame': pl.Int64, 'End Frame': pl.Int64,
            'Start Time': pl.String, 'End Time': pl.String,
            'Behavior': pl.String, 'Fish IDs': pl.String,
            'Confidence / Notes': pl.String, 'span': pl.Int64,
            'duration_seconds': pl.Float64, 'positive_frames': pl.Int64,
            'frame_count': pl.Int64, 'positive_fraction': pl.Float64,
        })
        if not parquet_path.exists() or parquet_path.stat().st_mtime_ns < csv_path.stat().st_mtime_ns:
            spans.write_parquet(parquet_path)
        return spans

    spans = annotate_spans(
        find_positive_spans(predictions, fps=fps, threshold=threshold, seconds=seconds),
        video_path, fps=fps,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    spans.write_csv(csv_path)
    spans.write_parquet(parquet_path)
    return spans


def render_span(video_path, start, end, output_dir):
    """Cache a full-resolution pose-overlay clip plus a small notebook preview."""
    video_path, output_dir = Path(video_path), Path(output_dir)
    folder = output_dir / f'{video_path.stem}_{start}-{end}'
    clip = folder / f'{video_path.stem}_{start}-{end}_overlay.mp4'
    source_time = max(video_path.stat().st_mtime_ns, video_path.with_suffix('.parquet').stat().st_mtime_ns)
    if not clip.exists() or clip.stat().st_mtime_ns < source_time:
        export_circling_clips(video_path.parent, folder, {video_path.name: [(start, end)]})
    preview = folder / 'preview.mp4'
    if not preview.exists() or preview.stat().st_mtime_ns < clip.stat().st_mtime_ns:
        # Bound the encoded preview to about 2 MB even for long spans.
        bitrate = max(1000, min(250000, int(12_000_000 / ((end - start + 1) / 30))))
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
            '-i', str(clip), '-vf', 'scale=480:-2', '-an', '-c:v', 'libx264',
            '-preset', 'fast', '-b:v', str(bitrate), '-maxrate', str(bitrate),
            '-bufsize', str(2 * bitrate), '-movflags', '+faststart', str(preview)], check=True)
    return clip, preview
