import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import imageio_ffmpeg
import numpy as np

import video_source
from video_source import VIDEO_URLS, VideoReader, resolve_video, video_frame_count


REAL_CAPTURE = cv2.VideoCapture


class FlakyCapture:
    """Local capture whose first opened instance fails after three reads."""
    opened = 0

    def __init__(self, source):
        self.capture = REAL_CAPTURE(source)
        FlakyCapture.opened += 1
        self.reads_left = 3 if FlakyCapture.opened == 1 else None

    def read(self):
        if self.reads_left == 0:
            return False, None
        if self.reads_left is not None:
            self.reads_left -= 1
        return self.capture.read()

    def __getattr__(self, name):
        return getattr(self.capture, name)


def write_video(path, frames=10):
    writer = imageio_ffmpeg.write_frames(
        str(path), (64, 48), fps=30, pix_fmt_in='bgr24', codec='libx264',
        macro_block_size=1, ffmpeg_log_level='error',
        output_params=['-crf', '0'],
    )
    writer.send(None)
    for value in range(frames):
        writer.send(np.full((48, 64, 3), value * 20, dtype=np.uint8))
    writer.close()


class VideoSourceTest(unittest.TestCase):
    def test_prefers_local_then_url(self):
        with TemporaryDirectory() as directory:
            local = Path(directory) / '0028_vid.mp4'
            self.assertEqual(resolve_video(local), VIDEO_URLS['0028_vid.mp4'])
            local.touch()
            self.assertEqual(resolve_video(local), str(local))
            with self.assertRaises(FileNotFoundError):
                resolve_video(Path(directory) / 'unknown.mp4')

    def test_known_frame_count_does_not_open_video(self):
        with patch.object(video_source, 'VideoReader') as reader:
            self.assertEqual(video_frame_count('data/0028_vid.mp4'), 1_080_175)
        reader.assert_not_called()

    def test_local_read_failure_is_not_retried(self):
        with TemporaryDirectory() as directory:
            video = Path(directory) / 'sample.mp4'
            write_video(video, frames=3)
            with VideoReader(video) as reader, self.assertRaises(RuntimeError):
                list(reader.frames(0, 5))

    def test_failed_stream_resumes_at_same_frame(self):
        with TemporaryDirectory() as directory:
            video = Path(directory) / 'sample.mp4'
            write_video(video)
            with VideoReader(video) as reader:
                expected = list(reader.frames(2, 8))
            FlakyCapture.opened = 0
            with (patch.object(video_source.cv2, 'VideoCapture', FlakyCapture),
                  patch.object(video_source, 'sleep')):
                with VideoReader(video) as reader:
                    reader.retries = 1
                    actual = list(reader.frames(2, 8))
            self.assertEqual(FlakyCapture.opened, 2)
            self.assertEqual([n for n, _ in actual], list(range(2, 9)))
            for (_, want), (_, got) in zip(expected, actual):
                self.assertTrue(np.array_equal(want, got))

    def test_exhausted_retries_raise(self):
        with TemporaryDirectory() as directory:
            video = Path(directory) / 'sample.mp4'
            write_video(video, frames=3)
            with patch.object(video_source, 'sleep') as sleep, VideoReader(video) as reader:
                reader.retries = 2
                with self.assertRaises(RuntimeError):
                    list(reader.frames(0, 5))
            self.assertEqual(sleep.call_count, 2)


if __name__ == '__main__':
    unittest.main()
