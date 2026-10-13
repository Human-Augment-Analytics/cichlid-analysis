"""Sampling, window alignment and inference regression tests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from models.xgboost.features import (FEATURES, closest_pair_series, read_spans,
                                     sample_frames, window_features)
from models.xgboost.postprocess import majority_vote_fraction
from models.xgboost.train import (evaluate_full_videos, fit_model, positive_spans,
                                 score_series, split_dataset)


class SamplingTests(unittest.TestCase):
    def test_half_length_buffers_overlap_and_positive_precedence(self):
        frames, labels = sample_frames(30, [(5, 8), (10, 13)])
        np.testing.assert_array_equal(frames, np.arange(3, 16))
        np.testing.assert_array_equal(frames[labels == 1], [5, 6, 7, 8, 10, 11, 12, 13])
        self.assertEqual(len(frames), len(set(frames)))

    def test_odd_lengths_boundaries_and_duplicate_spans(self):
        frames, labels = sample_frames(10, [(0, 2), (0, 2), (8, 9)])
        np.testing.assert_array_equal(frames, [0, 1, 2, 3, 4, 7, 8, 9])
        self.assertEqual(int(labels.sum()), 5)
        with self.assertRaises(ValueError):
            sample_frames(10, [(8, 10)])

    def test_read_only_usable_positive_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spans.csv"
            pd.DataFrame({"VideoID": ["a"] * 4, "StartFrame": [3, 4, 20, 30],
                          "EndFrame": [5, 8, 21, 31], "Circling": [True, True, False, True],
                          "ShouldUse": [True, True, True, False]}).to_csv(path, index=False)
            self.assertEqual(read_spans(path), {"a": [(3, 8)]})

    def test_split_seventy_thirty_is_repeatable_and_disjoint(self):
        table = pd.DataFrame({"circling": [0, 1] * 50})
        result, train, test = split_dataset(table)
        self.assertEqual((len(train), len(test)), (70, 30))
        self.assertFalse(set(train) & set(test))
        self.assertEqual(result.loc[test, "circling"].sum(), 15)
        np.testing.assert_array_equal(test, split_dataset(table)[2])


class WindowTests(unittest.TestCase):
    def test_cleaned_closest_pair_and_invalid_rate_match_baseline(self):
        pairs = pd.DataFrame({"FrameNum": [0, 0, 0, 2, 3],
                              "centroid_distance_px": [10, 80, 100, 90, 95],
                              "d_orbital_rad_s": [1, 2, 3, 16, np.nan],
                              "keep": [False, True, True, True, True]})
        series = closest_pair_series(pairs, 5)
        np.testing.assert_allclose(series[:, 0], [80, np.nan, 90, 95, np.nan], equal_nan=True)
        np.testing.assert_array_equal(series[:, 1], [2, 0, 0, 0, 0])

    def test_center_alignment_feature_order_and_boundary_padding(self):
        series = np.column_stack((np.arange(130), 1000 + np.arange(130)))
        x = window_features(series, [0, 60, 129])
        self.assertEqual(x.shape, (3, 242))
        self.assertEqual(len(FEATURES), x.shape[1])
        np.testing.assert_array_equal(x[1, :121], np.arange(121))
        np.testing.assert_array_equal(x[1, 121:], 1000 + np.arange(121))
        self.assertTrue(np.isnan(x[0, :60]).all())
        self.assertEqual(x[0, 60], 0)
        self.assertEqual(x[2, 60], 129)
        self.assertTrue(np.isnan(x[2, 61:121]).all())
        with self.assertRaises(ValueError):
            window_features(series, [130])

    def test_chunked_predictions_match_single_batch_and_roundtrip(self):
        rng = np.random.default_rng(5)
        series = rng.normal(size=(180, 2)).astype(np.float32)
        labels = (series[:, 0] > 0).astype(np.int8)
        x = window_features(series, np.arange(len(series)))
        model = fit_model(x, labels)
        expected = model.predict_proba(x)[:, 1]
        np.testing.assert_array_equal(score_series(model, series, batch_size=37), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ubj"
            model.save_model(path)
            restored = type(model)()
            restored.load_model(path)
            np.testing.assert_array_equal(restored.predict_proba(x)[:, 1], expected)

    def test_threshold_spans_are_inclusive(self):
        self.assertEqual(positive_spans(np.array([.5, .6, .1, .8])), [(0, 1), (3, 3)])
        self.assertEqual(positive_spans(np.zeros(3)), [])


class PostprocessingTests(unittest.TestCase):
    def test_short_gaps_merge_but_longer_gaps_remain(self):
        for gap, expected in ((15, 1), (16, 2), (30, 2)):
            with self.subTest(gap=gap):
                scores = np.r_[np.ones(60), np.zeros(gap), np.ones(60)]
                self.assertEqual(len(positive_spans(majority_vote_fraction(scores))), expected)

    def test_isolated_bursts_and_ties(self):
        for length, expected in ((14, 0), (15, 1)):
            with self.subTest(length=length):
                scores = np.r_[np.zeros(60), np.ones(length), np.zeros(60)]
                self.assertEqual(len(positive_spans(majority_vote_fraction(scores))), expected)

    def test_even_window_alignment_and_clipped_video_edges(self):
        scores = np.zeros(60)
        scores[30] = 0.5
        fractions = majority_vote_fraction(scores)
        np.testing.assert_array_equal(np.flatnonzero(fractions), np.arange(16, 46))
        self.assertEqual(fractions[30], 1 / 30)
        np.testing.assert_array_equal(majority_vote_fraction(np.ones(5)), np.ones(5))
        np.testing.assert_array_equal(majority_vote_fraction(np.zeros(5)), np.zeros(5))
        self.assertEqual(len(majority_vote_fraction([])), 0)

    def test_full_video_exports_processed_spans_and_retains_raw_predictions(self):
        scores = np.r_[np.ones(40), np.zeros(10), np.ones(40)].astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("models.xgboost.train.score_series", return_value=scores):
                result = evaluate_full_videos(None, {"test": np.zeros((90, 2))},
                                              {"test": [(0, 39), (50, 89)]},
                                              output, output / "predictions")
            exported = pd.read_csv(output / "predictions/test.csv")
            self.assertEqual(exported[["StartFrame", "EndFrame"]].values.tolist(), [[0, 89]])
            saved = pd.read_parquet(output / "scores/test_scores.parquet")
            np.testing.assert_array_equal(saved.predicted_circling, scores >= .5)
            self.assertTrue(saved.postprocessed_circling.all())
            self.assertEqual(result["segment_metrics"]["test"]["pred_segments"], 2)
            self.assertEqual(result["postprocessed"]["segment_metrics"]["test"]["pred_segments"], 1)


if __name__ == "__main__":
    unittest.main()
