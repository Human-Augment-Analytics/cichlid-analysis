"""Regression coverage for the pair-motion feature package."""

import unittest

import numpy as np

from feature_engineering.pair_motion import angular_temporal as at
from feature_engineering.pair_motion.feature_groups import ALL_MODEL_FEATURES, FEATURE_GROUPS
from feature_engineering.pair_motion.features_motion import (
    OUTPUT_COLUMNS,
    _orbit_toy,
    build_motion_features,
    run_angular_temporal_checks,
    run_synthetic_checks,
)


class PairMotionTest(unittest.TestCase):
    def test_feature_definitions(self) -> None:
        """Known distances, bearings, rates, and smoothing boundaries stay valid."""
        run_synthetic_checks()

    def test_angular_temporal_definitions(self) -> None:
        """Orbit turns, corotation, rate clipping, and window boundaries on a toy pair."""
        run_angular_temporal_checks()

    def test_model_features_exist_and_are_not_sex_or_ids(self) -> None:
        self.assertEqual(len(ALL_MODEL_FEATURES), len(set(ALL_MODEL_FEATURES)))
        for column in ALL_MODEL_FEATURES:
            self.assertIn(column, OUTPUT_COLUMNS)
            self.assertNotIn("sex", column)
            self.assertNotIn("TrackID", column)
            self.assertFalse(column.endswith("_px"))
        self.assertEqual(set(FEATURE_GROUPS), {
            "distance_proximity", "orientation_facing", "speed_closing",
            "angular_movement", "temporal_consistency",
        })

    def test_windows_are_gap_safe_and_direction_configurable(self) -> None:
        pairs, tracks = _orbit_toy(frames=60)
        features, _ = build_motion_features(pairs, tracks)
        segments = features.groupby(["pair_episode_id", "smooth_segment"], sort=False).ngroup().to_numpy()
        trailing = features.copy()
        at.add_window_features(trailing, segments, window_seconds=(1.0,), direction="trailing")
        centered = features.copy()
        at.add_window_features(centered, segments, window_seconds=(1.0,), direction="centered")
        first = (features["pair_episode_id"] == 1) & (features["FrameNum"] == 0)
        # Trailing sees only itself at the first frame; centered sees the following half second.
        self.assertAlmostEqual(float(trailing.loc[first, "w1s_coverage"].iloc[0]), 1 / 31, places=5)
        self.assertGreater(float(centered.loc[first, "w1s_coverage"].iloc[0]), 0.4)
        with self.assertRaises(ValueError):
            at.add_window_features(features.copy(), segments, direction="sideways")

    def test_unsorted_rows_are_rejected(self) -> None:
        pairs, tracks = _orbit_toy(frames=20)
        features, _ = build_motion_features(pairs, tracks)
        reversed_rows = features.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(RuntimeError):
            at.add_window_features(reversed_rows, np.zeros(len(reversed_rows), dtype=int))


if __name__ == "__main__":
    unittest.main()
