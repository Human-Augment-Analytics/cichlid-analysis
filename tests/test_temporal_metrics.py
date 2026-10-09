import unittest
import polars as pl
from temporal_metrics import edit_score, segment_counts, temporal_metrics


class TemporalMetricsTest(unittest.TestCase):
    def test_perfect_and_fragmented(self):
        truth = [0, 1, 1, 1, 1, 1, 0]
        self.assertEqual(segment_counts(truth, truth, .5), (1, 0, 0))
        self.assertEqual(edit_score(truth, truth), 100)
        self.assertEqual(segment_counts([0, 1, 1, 0, 1, 1, 0], truth, .25), (1, 1, 0))
        self.assertEqual(edit_score([0, 1, 1, 0, 1, 1, 0], truth), 50)

    def test_overlap_threshold_and_empty(self):
        self.assertEqual(segment_counts([1, 0, 0, 0], [1, 1, 1, 1], .25), (1, 0, 0))
        self.assertEqual(segment_counts([1, 0, 0, 0], [1, 1, 1, 1], .5), (0, 1, 1))
        self.assertEqual(segment_counts([1, 1], [0, 0], .1), (0, 1, 0))
        self.assertEqual(edit_score([0, 0], [0, 0]), 100)

    def test_ignored_and_missing_frames_split_runs(self):
        table = pl.DataFrame({'video': ['v'] * 5, 'sequence': ['s'] * 5,
                              'FrameNum': [0, 1, 2, 4, 5], 'label': [1, -100, 1, 1, 0],
                              'circling_score': [1., 0., 1., 1., 0.]})
        result = temporal_metrics(table)
        self.assertEqual(result['mof'], 100)
        self.assertEqual(result['evaluation_runs'], 3)
        self.assertEqual(result['segmental_counts_50'], {'tp': 3, 'fp': 0, 'fn': 0})


if __name__ == '__main__':
    unittest.main()
