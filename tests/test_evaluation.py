import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.evaluate import edit_score, evaluate_video, match_events, read_intervals, summarize


class EvaluationTest(unittest.TestCase):
    def test_perfect_prediction(self):
        result = evaluate_video([(10, 19), (40, 49)], [(10, 19), (40, 49)], 100)
        self.assertEqual(result['mof'], 1)
        self.assertEqual(result['edit_score'], 100)
        self.assertEqual(result['frame_f1'], 1)
        self.assertEqual(result['matched_events'], 2)

    def test_partial_prediction(self):
        # GT 20 frames, prediction 10 frames of which 5 overlap.
        result = evaluate_video([(0, 9), (20, 29)], [(5, 14)], 40)
        self.assertEqual((result['frame_tp'], result['frame_fp'], result['frame_fn']), (5, 5, 15))
        self.assertAlmostEqual(result['mof'], 20 / 40)
        self.assertAlmostEqual(result['frame_precision'], .5)
        self.assertAlmostEqual(result['frame_recall'], .25)
        self.assertAlmostEqual(result['frame_f1'], 1 / 3)
        self.assertEqual(result['matched_events'], 0)  # IoU 5/15 < 0.5

    def test_empty_prediction(self):
        result = evaluate_video([(0, 9)], [], 20)
        self.assertTrue(math.isnan(result['frame_precision']))
        self.assertEqual(result['frame_f1'], 0)
        self.assertEqual(result['missed_events'], 1)

    def test_edit_score_counts_segments(self):
        # GT: bg c bg c bg; prediction: bg c bg -> two deletions over five segments.
        self.assertAlmostEqual(edit_score([(10, 19), (40, 49)], [(10, 49)], 100), 60)

    def test_matching_is_one_to_one(self):
        matches = match_events([(0, 9)], [(0, 9), (0, 8)], .5)
        self.assertEqual([(i, j) for i, j, _ in matches], [(0, 0)])

    def test_summary_pools_frames(self):
        rows = [evaluate_video([(0, 9)], [(0, 9)], 10), evaluate_video([(0, 9)], [], 30)]
        overall = summarize(rows)
        self.assertAlmostEqual(overall['mof'], 30 / 40)
        self.assertAlmostEqual(overall['frame_recall'], .5)
        self.assertEqual(overall['edit_score'], (100 + 50) / 2)

    def test_reads_time_and_frame_csvs(self):
        with TemporaryDirectory() as directory:
            times = Path(directory, 'gt.csv')
            times.write_text('VideoID,StartTime,EndTime\n0028_vid,00:00:01,00:00:02.5\n')
            frames = Path(directory, 'pred.csv')
            frames.write_text('Video ID,Start Frame,End Frame\n0028_vid,3,7\n')
            self.assertEqual(read_intervals(times).rows(), [('0028_vid', 30, 75)])
            self.assertEqual(read_intervals(frames).rows(), [('0028_vid', 3, 7)])


if __name__ == '__main__':
    unittest.main()
