import unittest

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from circling_model import frame_labels
from pose_features import pose_frame_features as frame_features
from pose_change import KEYPOINTS


def poses():
    rows = []
    for frame in range(45):
        for track in (1, 2):
            if frame == 15 or (track == 2 and frame == 20):
                continue
            row = dict(project_id='p', day_label='v', TrackID=track, FrameNum=frame)
            for index, name in enumerate(KEYPOINTS):
                row[f'{name}_x'] = float(index * 3 + frame + track * 100)
                row[f'{name}_y'] = float(track * 20)
            rows.append(row)
    return pl.DataFrame(rows)


class FeaturesTest(unittest.TestCase):
    def test_inclusive_labels(self):
        self.assertEqual(frame_labels(np.arange(8), [(2, 4), (6, 6)]).tolist(), [0, 0, 1, 1, 1, 0, 1, 0])

    def test_missing_frame_and_track_gaps(self):
        features = frame_features(poses(), 0, 44)
        self.assertEqual(features.height, 45)
        self.assertEqual(features['track_count'][15], 0)
        self.assertIsNone(features['Nose_mean'][16])
        self.assertEqual(features['track_count'][20], 1)
        self.assertAlmostEqual(features['Nose_mean'][21], np.log1p(1 / 27), places=6)

    def test_no_future_features(self):
        data = poses()
        before = frame_features(data, 0, 30)
        altered = data.with_columns(pl.when(pl.col('FrameNum') > 30)
                                   .then(pl.col('Nose_x') + 10000).otherwise(pl.col('Nose_x')).alias('Nose_x'))
        assert_frame_equal(before, frame_features(altered, 0, 30))

    def test_single_frame_matches_batch(self):
        data = poses()
        batch = frame_features(data, 0, 44).tail(1)
        subset = frame_features(data.filter(pl.col('FrameNum') >= 14), 44, 44)
        assert_frame_equal(batch, subset, rel_tol=1e-6, abs_tol=1e-6)

    def test_empty_frame(self):
        result = frame_features(poses().head(0), 0, 0)
        self.assertEqual(result.height, 1)
        self.assertEqual(result['track_count'][0], 0)


class PairFeaturesTest(unittest.TestCase):
    def pairs(self):
        from circling_model import PAIR_DISTANCES, PAIR_ANGLES, PAIR_RATES
        return pl.DataFrame([
            dict(project_id='p', day_label='v', FrameNum=0, TrackID_1=1, TrackID_2=track,
                 **{c: distance for c in PAIR_DISTANCES},
                 **{c: angle for c in PAIR_ANGLES},
                 **{c: None for c in PAIR_RATES})
            for track, distance, angle in [(2, 10., np.pi - .01), (3, 30., -np.pi + .01)]
        ], schema_overrides={c: pl.Float64 for c in PAIR_RATES})

    def test_aggregation_and_missing_pairs(self):
        from circling_model import pair_frame_features
        features = pair_frame_features(self.pairs(), 0, 1)
        self.assertEqual(features.width, 53)
        self.assertEqual(features['pair_count'].to_list(), [2., 0.])
        self.assertEqual(features['centroid_distance_px_mean'][0], 20.)
        self.assertEqual(features['centroid_distance_px_min'][0], 10.)
        self.assertEqual(features['centroid_distance_px_max'][0], 30.)
        self.assertIsNone(features['centroid_distance_px_mean'][1])
        self.assertAlmostEqual(features['orbital_rad_sin_mean'][0], 0., places=6)
        self.assertLess(features['orbital_rad_cos_mean'][0], -.99)

    def test_combined_retains_pose_features(self):
        from circling_model import frame_features as combined
        result = combined(poses(), self.pairs(), 0, 44)
        original = frame_features(poses(), 0, 44)
        self.assertEqual(result.width, 103)
        assert_frame_equal(result.select(original.columns), original)
        self.assertEqual(result.height, original.height)


if __name__ == '__main__':
    unittest.main()
