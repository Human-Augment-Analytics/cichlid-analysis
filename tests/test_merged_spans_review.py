import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import imageio_ffmpeg
import numpy as np
import polars as pl

from merged_spans_review import buffered_bounds, load_review_spans, render_merged_preview


class MergedSpanPreviewTest(unittest.TestCase):
    def test_auto_annotations_supply_optional_review_fields(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / '0018_vid.csv'
            path.write_text('VideoID,EventID,StartFrame,EndFrame,Behavior,FishIDs,Confidence/Notes\n'
                            '0018_vid,event_1,10,69,circling,1|2,0.5\n')
            spans = load_review_spans(path)
            self.assertEqual(spans['Source'].to_list(), ['auto_annotations'])
            self.assertEqual(spans['Notes'].to_list(), [''])
            self.assertEqual(spans['FishIDs'].to_list(), ['1|2'])
            self.assertEqual(spans['StartFrame'].dtype, pl.Int64)
            self.assertEqual(spans['EndFrame'].to_list(), [69])

    def test_existing_review_fields_are_preserved(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'reviewed.csv'
            path.write_text('VideoID,EventID,StartFrame,EndFrame,Source,Circling,ShouldUse,Notes\n'
                            '0018_vid,event_1,10,69,finalized,true,false,check bounds\n')
            row = load_review_spans(path).row(0, named=True)
            self.assertEqual(row['Source'], 'finalized')
            self.assertEqual(row['Circling'], 'true')
            self.assertEqual(row['ShouldUse'], 'false')
            self.assertEqual(row['Notes'], 'check bounds')

    def test_buffer_clamps_to_video(self):
        self.assertEqual(buffered_bounds(1, 3, 5, 10), (0, 8))
        self.assertEqual(buffered_bounds(7, 9, 5, 10), (2, 9))
        with self.assertRaises(ValueError):
            buffered_bounds(3, 4, -1, 10)

    def test_preview_has_buffered_frames_and_pose_overlay(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'sample.mp4'
            writer = imageio_ffmpeg.write_frames(
                str(video), (640, 480), fps=30, pix_fmt_in='bgr24',
                codec='libx264', macro_block_size=1, ffmpeg_log_level='error',
            )
            writer.send(None)
            for _ in range(10):
                writer.send(np.full((480, 640, 3), 20, dtype=np.uint8))
            writer.close()

            pose = {'project_id': 'p', 'day_label': 'd', 'FrameNum': 0,
                    'TrackID': 7, 'Nose_x': 100., 'Nose_y': 120.}
            for keypoint in ('LeftEye', 'RightEye', 'Head', 'Spine1', 'Spine2',
                             'Spine3', 'Spine4', 'Peduncle', 'TailTip'):
                pose[f'{keypoint}_x'] = None
                pose[f'{keypoint}_y'] = None
            pl.DataFrame([pose]).write_parquet(video.with_suffix('.parquet'))

            preview, start, end = render_merged_preview(
                video, 1, 3, 5, root / 'previews')
            self.assertEqual((start, end), (0, 8))
            capture = cv2.VideoCapture(str(preview))
            try:
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 9)
                ok, first = capture.read()
                self.assertTrue(ok)
                self.assertGreater(first[120, 100].max(), 100)
                self.assertGreater(first[10:90, 450:635].max(), 150)
            finally:
                capture.release()

            timestamp = preview.stat().st_mtime_ns
            self.assertEqual(render_merged_preview(video, 1, 3, 5, root / 'previews')[0],
                             preview)
            self.assertEqual(preview.stat().st_mtime_ns, timestamp)
