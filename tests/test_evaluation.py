import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.evaluate import read_intervals, read_predictions
from evaluation.metrics import edit_score, evaluate_video, segment_true_positives, summarize


class EvaluationTest(unittest.TestCase):
    def test_perfect_prediction(self):
        result = evaluate_video([(10, 19), (40, 49)], [(10, 19), (40, 49)], 100)
        self.assertEqual(result['mof'], 1)
        self.assertEqual(result['edit_score'], 100)
        self.assertEqual(result['frame_f1'], 1)
        self.assertEqual([result[f'f1@{tau}'] for tau in (10, 25, 50)], [1, 1, 1])

    def test_partial_prediction(self):
        # GT 20 frames, prediction 10 frames of which 5 overlap.
        result = evaluate_video([(0, 9), (20, 29)], [(5, 14)], 40)
        self.assertEqual((result['frame_tp'], result['frame_fp'], result['frame_fn']), (5, 5, 15))
        self.assertAlmostEqual(result['mof'], 20 / 40)
        self.assertAlmostEqual(result['frame_precision'], .5)
        self.assertAlmostEqual(result['frame_recall'], .25)
        self.assertAlmostEqual(result['frame_f1'], 1 / 3)
        # IoU 5/15: a true positive at tau 10 and 25, not 50. One of one predicted
        # and two ground-truth segments gives F1 = 2 * 1 / (1 + 2).
        self.assertEqual([result[f'tp@{tau}'] for tau in (10, 25, 50)], [1, 1, 0])
        self.assertAlmostEqual(result['f1@25'], 2 / 3)
        self.assertEqual(result['f1@50'], 0)

    def test_empty_prediction(self):
        result = evaluate_video([(0, 9)], [], 20)
        self.assertTrue(math.isnan(result['frame_precision']))
        self.assertEqual(result['frame_f1'], 0)
        self.assertEqual(result['f1@10'], 0)

    def test_edit_score_counts_segments(self):
        # GT: bg c bg c bg; prediction: bg c bg -> two deletions over five segments.
        self.assertAlmostEqual(edit_score([(10, 19), (40, 49)], [(10, 49)], 100), 60)

    def test_one_true_positive_per_ground_truth_segment(self):
        # Both predictions overlap the same segment; the second is a false positive.
        self.assertEqual(segment_true_positives([(0, 9)], [(0, 4), (5, 9)], 10), 1)

    def test_iou_threshold_is_inclusive(self):
        # IoU 5/10 is exactly 0.5.
        self.assertEqual(segment_true_positives([(0, 9)], [(0, 4)], 50), 1)
        self.assertEqual(segment_true_positives([(0, 9)], [(0, 3)], 50), 0)

    def test_summary_pools_frames(self):
        rows = [evaluate_video([(0, 9)], [(0, 9)], 10), evaluate_video([(0, 9)], [], 30)]
        overall = summarize(rows)
        self.assertAlmostEqual(overall['mof'], 30 / 40)
        self.assertAlmostEqual(overall['frame_recall'], .5)
        self.assertEqual(overall['edit_score'], (100 + 50) / 2)
        # Pooled: one true positive, one predicted and two ground-truth segments.
        self.assertAlmostEqual(overall['f1@50'], 2 / 3)

    def test_reads_frame_csvs(self):
        with TemporaryDirectory() as directory:
            frames = Path(directory, 'pred.csv')
            frames.write_text('Video ID,Start Frame,End Frame\n0028_vid,3,7\n')
            times = Path(directory, 'gt.csv')
            times.write_text('VideoID,StartTime,EndTime\n0028_vid,00:00:01,00:00:02.5\n')
            self.assertEqual(read_intervals(frames).rows(), [('0028_vid', 3, 7)])
            with self.assertRaises(ValueError):
                read_intervals(times)

    def test_reads_model_folder(self):
        with TemporaryDirectory() as directory:
            Path(directory, '0028_vid.csv').write_text('VideoID,StartFrame,EndFrame,extra\n0028_vid,3,7,x\n')
            Path(directory, '0031_vid.csv').write_text('VideoID,StartFrame,EndFrame,extra\n')
            self.assertEqual(read_predictions(directory).rows(), [('0028_vid', 3, 7)])
        with TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            read_predictions(directory)


if __name__ == '__main__':
    unittest.main()
