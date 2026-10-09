"""Regression coverage for the pair-motion feature package."""

import unittest

from feature_engineering.pair_motion.features_motion import run_synthetic_checks


class PairMotionTest(unittest.TestCase):
    def test_feature_definitions(self) -> None:
        """Known distances, bearings, rates, and smoothing boundaries stay valid."""
        run_synthetic_checks()


if __name__ == "__main__":
    unittest.main()
