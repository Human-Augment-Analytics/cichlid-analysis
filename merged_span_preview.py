"""Render buffered merged-span previews with pose points and TrackIDs."""

from pathlib import Path

import cv2
import imageio_ffmpeg
import polars as pl

from pose_overlay import overlay_frame, overlay_track_legend


def buffered_bounds(start, end, buffer_frames, frame_count):
    """Return an inclusive span extended within the source video's bounds."""
    if not 0 <= start <= end < frame_count or buffer_frames < 0:
        raise ValueError('Invalid span or buffer for video')
    return max(0, start - buffer_frames), min(frame_count - 1, end + buffer_frames)


def render_merged_preview(video_path, start, end, buffer_frames, output_dir, max_width=640):
    """Cache a small video with pose points and a per-frame TrackID legend."""
    video_path, output_dir = Path(video_path), Path(output_dir)
    pose_path = video_path.with_suffix('.parquet')
    if not pose_path.exists():
        raise FileNotFoundError(f'Pose data not found: {pose_path}')

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f'Could not open {video_path}')
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0 or max_width < 2:
            raise ValueError(f'Invalid video metadata: {video_path}')
        preview_start, preview_end = buffered_bounds(start, end, buffer_frames, frame_count)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Keep the notebook payload small; older 960 px previews can exceed
        # marimo's output limit when VS Code serializes the video.
        preview = output_dir / (
            f'{video_path.stem}_{start}-{end}_buffer_{buffer_frames}'
            f'_w{max_width}_crf30_overlay.mp4'
        )
        sources = (video_path, pose_path, Path(__file__), Path(__file__).with_name('pose_overlay.py'))
        if preview.exists() and preview.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in sources):
            return preview, preview_start, preview_end

        poses = (pl.scan_parquet(pose_path)
                 .filter(pl.col('FrameNum').is_between(preview_start, preview_end))
                 .collect())
        if (not poses.is_empty()
                and poses.select(pl.struct('project_id', 'day_label').n_unique()).item() > 1):
            raise ValueError(f'Pose table contains multiple videos: {pose_path}')
        groups = poses.partition_by('FrameNum', as_dict=True)
        empty = poses.head(0)

        scale = min(1., max_width / width)
        output_width = max(2, round(width * scale / 2) * 2)
        output_height = max(2, round(height * scale / 2) * 2)
        partial = preview.with_suffix('.part.mp4')
        writer = imageio_ffmpeg.write_frames(
            str(partial), (output_width, output_height), fps=fps,
            pix_fmt_in='bgr24', pix_fmt_out='yuv420p', codec='libx264',
            macro_block_size=1, ffmpeg_log_level='error',
            output_params=['-crf', '30', '-preset', 'veryfast', '-movflags', '+faststart'],
        )
        try:
            writer.send(None)
            capture.set(cv2.CAP_PROP_POS_FRAMES, preview_start)
            for frame_number in range(preview_start, preview_end + 1):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f'Could not decode frame {frame_number} from {video_path}')
                frame_poses = groups.get((frame_number,), empty)
                annotated = overlay_frame(frame, frame_poses, labels=False)
                annotated = cv2.resize(annotated, (output_width, output_height),
                                       interpolation=cv2.INTER_AREA)
                track_ids = frame_poses['TrackID'].drop_nulls().unique().sort().to_list()
                annotated = overlay_track_legend(annotated, track_ids)
                label = f'Frame {frame_number:,} | {"span" if start <= frame_number <= end else "buffer"}'
                cv2.rectangle(annotated, (0, 0), (360, 36), (24, 24, 24), -1)
                cv2.putText(annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 1, cv2.LINE_AA)
                writer.send(annotated)
            writer.close()
            check = cv2.VideoCapture(str(partial))
            try:
                if (not check.isOpened()
                        or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != preview_end - preview_start + 1):
                    raise RuntimeError(f'Encoded preview has incorrect frame count: {partial}')
            finally:
                check.release()
            partial.replace(preview)
        finally:
            writer.close()
            partial.unlink(missing_ok=True)
        return preview, preview_start, preview_end
    finally:
        capture.release()
