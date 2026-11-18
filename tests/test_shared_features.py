"""Leakage and bookkeeping checks for the leave-one-video-out experiments."""

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from experiments.shared_features import dataset, lovo
from experiments.shared_features.common import positive_spans, scores_to_spans
from feature_engineering.pair_motion.features_motion import _orbit_toy, build_motion_features


class ShortSpansTest(unittest.TestCase):
    def test_positive_spans_are_inclusive(self) -> None:
        self.assertEqual(positive_spans([0, 1, 1, 0, 1]), [(1, 2), (4, 4)])

    def test_min_length_drops_short_runs(self) -> None:
        scores = np.zeros(200)
        scores[10:20] = 1.0
        scores[100:180] = 1.0
        self.assertEqual(len(scores_to_spans(scores, 0.5, min_length=0)), 2)
        kept = scores_to_spans(scores, 0.5, min_length=50)
        self.assertEqual(len(kept), 1)
        self.assertGreaterEqual(kept[0][0], 95)


class InnerBlocksTest(unittest.TestCase):
    def test_inner_training_never_touches_validation_or_buffer(self) -> None:
        frames = np.arange(0, 100_000)
        seen = np.zeros(len(frames), dtype=int)
        for train, valid in lovo.inner_blocks(frames):
            self.assertFalse(np.any(train & valid))
            lo, hi = frames[valid].min(), frames[valid].max()
            near = (frames >= lo - lovo.INNER_BUFFER_FRAMES) & (frames <= hi + lovo.INNER_BUFFER_FRAMES)
            self.assertFalse(np.any(train & near))
            seen += valid
        self.assertTrue(np.all(seen == 1), "every training frame is validated exactly once")


class FoldIsolationTest(unittest.TestCase):
    def _frames(self, test_shift: float) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for video, shift in (("0028_vid", 0.0), ("0031_vid", test_shift)):
            n = 3000
            frame = np.arange(n)
            label = ((frame // 300) % 2).astype(int)
            rows.append(pd.DataFrame({
                "VideoID": video, "FrameNum": frame, "label": label,
                "w4s_close_frac": np.where(label == 1, 0.9, 0.1) + shift,
                "w4s_orbit_turns_abs": np.where(label == 1, 1.0, 0.1) + shift,
            }))
        return pd.concat(rows, ignore_index=True)

    def test_rule_parameters_depend_only_on_the_training_video(self) -> None:
        spans = {"0028_vid": [(300 * i, 300 * i + 299) for i in range(1, 10, 2)]}
        with mock.patch.dict(lovo.VIDEO_FRAMES, {"0028_vid": 3000, "0031_vid": 3000}):
            a = lovo.fit_rule(self._frames(0.0).query("VideoID == '0028_vid'"), spans)
            b = lovo.fit_rule(self._frames(5.0).query("VideoID == '0028_vid'"), spans)
        self.assertEqual(a, b)

    def test_each_video_is_held_out_once_and_never_trained_on(self) -> None:
        videos = ["a", "b", "c"]
        held_out = []
        for train_videos, test_video in lovo.folds(videos):
            self.assertNotIn(test_video, train_videos)
            self.assertEqual(sorted(train_videos + [test_video]), videos)
            held_out.append(test_video)
        self.assertEqual(held_out, videos)

    def test_inner_folds_are_cut_inside_each_training_video(self) -> None:
        table = pd.DataFrame({
            "VideoID": ["a"] * 4000 + ["b"] * 2000,
            "FrameNum": np.r_[np.arange(4000), np.arange(2000)],
        })
        seen = np.zeros(len(table), dtype=int)
        for train, valid in lovo.inner_folds(table):
            self.assertFalse(np.any(train & valid))
            for video in ("a", "b"):
                mask = (table["VideoID"] == video).to_numpy()
                frames = table["FrameNum"].to_numpy()
                self.assertTrue(np.any(valid & mask), "every inner fold validates part of every video")
                lo, hi = frames[valid & mask].min(), frames[valid & mask].max()
                near = mask & (frames >= lo - lovo.INNER_BUFFER_FRAMES) & (frames <= hi + lovo.INNER_BUFFER_FRAMES)
                self.assertFalse(np.any(train & near))
            seen += valid
        self.assertTrue(np.all(seen == 1))

    def test_postprocessing_is_chosen_on_all_training_videos(self) -> None:
        spans = {"a": [(100, 399)], "b": [(500, 799)]}
        table = pd.DataFrame({"VideoID": ["a"] * 1000 + ["b"] * 1000, "FrameNum": np.r_[np.arange(1000), np.arange(1000)]})
        scores = np.zeros(len(table), dtype=np.float32)
        scores[100:400] = 0.9
        scores[1500:1800] = 0.3  # video b's event only clears a low threshold
        with mock.patch.dict(lovo.VIDEO_FRAMES, {"a": 1000, "b": 1000}):
            (threshold, _), key = lovo.choose_postprocess(table, scores, spans)
        self.assertLessEqual(threshold, 0.3)
        self.assertEqual(key[0], 1.0)

    def test_full_length_scores_default_to_zero_without_a_pair(self) -> None:
        out = lovo.full_length_scores(np.array([2, 5]), np.array([0.7, 0.9]), 8)
        self.assertEqual(out.tolist(), [0, 0, np.float32(0.7), 0, 0, np.float32(0.9), 0, 0])


class LabelJoinTest(unittest.TestCase):
    def test_labels_follow_inclusive_spans_and_features_are_label_free(self) -> None:
        pairs, tracks = _orbit_toy(frames=60)
        features, _ = build_motion_features(pairs, tracks)
        self.assertNotIn("label", features.columns)
        features["day_label"] = "0028_vid"
        keys = features[["FrameNum", "TrackID_1", "TrackID_2"]].copy()
        with mock.patch.dict(dataset.VIDEO_FRAMES, {"0028_vid": 2000}):
            table = dataset.build_video_frames(features, keys, "0028_vid", {"0028_vid": [(10, 19)]})
        labelled = table.loc[table["label"] == 1, "FrameNum"]
        self.assertEqual((labelled.min(), labelled.max(), len(labelled)), (10, 19, 10))
        self.assertEqual(table["FrameNum"].is_unique, True)
        for column in dataset.AGGREGATE_COLUMNS:
            self.assertIn(column, table.columns)


if __name__ == "__main__":
    unittest.main()
