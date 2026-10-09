"""Checks for annotation alignment, masked supervision and segment boundaries."""
import unittest

import numpy as np
import polars as pl

from dlc2action_segmentation import context_intervals, make_windows, predicted_segments, span_labels


class SegmentationTests(unittest.TestCase):
    def test_inclusive_labels_and_ambiguous_overlap(self):
        rows = [{'Start Frame': 2, 'End Frame': 5, 'Verified': True},
                {'Start Frame': 5, 'End Frame': 6, 'Verified': False},
                {'Start Frame': 8, 'End Frame': 8, 'Verified': None}]
        np.testing.assert_array_equal(span_labels(np.arange(10), rows),
                                      [0, 0, 1, 1, 1, -100, -100, 0, -100, 0])

    def test_context_union_and_clipping(self):
        rows = [{'Start Frame': a, 'End Frame': b, 'Merge ID': str(a)}
                for a, b in [(2, 5), (8, 10), (20, 24)]]
        self.assertEqual(context_intervals(rows, 2, 25), [(0, 12), (18, 25)])
        with self.assertRaises(ValueError):
            context_intervals(rows, 2, 23)

    def test_window_padding_never_joins_sequences(self):
        sequences = [dict(x=np.full((n, 2), i + 1), y=np.ones(n, dtype=np.int64))
                     for i, n in enumerate([5, 2])]
        x, y, locations = make_windows(sequences, 4)
        self.assertEqual(tuple(x.shape), (3, 2, 4))
        np.testing.assert_array_equal(y.numpy(), [[1, 1, 1, 1], [1, -100, -100, -100], [1, 1, -100, -100]])
        self.assertEqual([item[2] for item in locations], [4, 1, 2])
        self.assertEqual(x[2, 0, 0].item(), 2)

    def test_predicted_runs_respect_gaps(self):
        predictions = pl.DataFrame({'video': ['v'] * 7, 'sequence': ['s'] * 7,
                                    'FrameNum': [10, 11, 12, 15, 16, 17, 18],
                                    'circling_score': [0.1, 0.8, 0.9, 0.7, 0.8, 0.1, 0.9]})
        self.assertEqual(predicted_segments(predictions).select('start_frame', 'end_frame').rows(),
                         [(11, 12), (15, 16), (18, 18)])
        self.assertEqual(predicted_segments(predictions, 1.0).height, 0)


if __name__ == '__main__':
    unittest.main()
