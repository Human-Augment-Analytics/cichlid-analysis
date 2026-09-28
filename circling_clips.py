"""Export every annotated circling interval with points-only pose overlays.

Run: python circling_clips.py
Frame ranges are zero-based and inclusive, matching the pose table.
"""

import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import polars as pl

from pose_overlay import overlay_frame, overlay_track_legend

CIRCLING_SPANS_BY_VIDEO = {
    "0028_vid.mp4": [
        (334703, 334830), (334985, 335118), (335409, 335502),
        (336153, 336285), (436636, 436895), (437259, 437437),
        (713035, 713330),
    ],
    "0031_vid.mp4": [
        (265510, 266227), (266809, 267100), (267242, 267837),
        (269089, 271817), (272241, 273018), (273550, 274787),
        (275040, 275190), (276439, 276530),
    ],
}


def export_circling_clips(data_dir, output_dir, spans_by_video=CIRCLING_SPANS_BY_VIDEO):
    """Write H.264 clips and a manifest, failing on missing source frames."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for video_name, spans in spans_by_video.items():
        video_path = data_dir / video_name
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open {video_path}")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0:
                raise ValueError(f"Invalid frame rate for {video_path}")
            for start, end in spans:
                if not 0 <= start <= end < total:
                    raise ValueError(f"Invalid interval {start}-{end} for {video_path}")
                poses = (
                    pl.scan_parquet(video_path.with_suffix('.parquet'))
                    .filter(pl.col('FrameNum').is_between(start, end))
                    .collect()
                )
                if poses.select(pl.struct('project_id', 'day_label').n_unique()).item() > 1:
                    raise ValueError(f"Pose table contains multiple videos: {video_path}")
                groups = poses.partition_by('FrameNum', as_dict=True)
                empty = poses.head(0)
                track_ids = sorted(poses['TrackID'].drop_nulls().unique().to_list())
                name = f"{video_path.stem}_{start}-{end}_overlay.mp4"
                destination = output_dir / name
                partial = destination.with_suffix('.part.mp4')
                writer = imageio_ffmpeg.write_frames(
                    str(partial), (width, height), fps=fps,
                    pix_fmt_in='bgr24', pix_fmt_out='yuv420p',
                    codec='libx264', macro_block_size=1, ffmpeg_log_level='error',
                    output_params=['-crf', '20', '-preset', 'fast', '-movflags', '+faststart'],
                )
                missing = 0
                print(f"Exporting {name} ({end - start + 1} frames)", flush=True)
                try:
                    writer.send(None)
                    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
                    for frame_number in range(start, end + 1):
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError(f"Could not decode {video_name} frame {frame_number}")
                        frame_poses = groups.get((frame_number,), empty)
                        missing += frame_poses.is_empty()
                        annotated = overlay_frame(frame, frame_poses, labels=False)
                        writer.send(overlay_track_legend(annotated, track_ids))
                    writer.close()
                    check = cv2.VideoCapture(str(partial))
                    try:
                        if (not check.isOpened()
                                or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != end - start + 1):
                            raise RuntimeError(f"Encoded clip has incorrect frame count: {partial}")
                    finally:
                        check.release()
                    partial.replace(destination)
                finally:
                    writer.close()
                    partial.unlink(missing_ok=True)
                manifest.append({
                    'file': name, 'source': video_name,
                    'start_frame': start, 'end_frame': end,
                    'frame_count': end - start + 1, 'fps': fps,
                    'duration_seconds': (end - start + 1) / fps,
                    'frames_without_poses': missing, 'track_ids': track_ids,
                })
        finally:
            capture.release()
    (output_dir / 'clips.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    clips = export_circling_clips(root / 'data', root / 'outputs' / 'circling_clips')
    print(f"Exported {len(clips)} clips.")
