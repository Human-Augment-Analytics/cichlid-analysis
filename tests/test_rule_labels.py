import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from models.rule_based.episode_labels import (FPS, label_interval_file, label_intervals,
                                              label_predictions, label_table, video_key)



def series(rates, dist=50.0):
    rate = np.concatenate([np.full(int(seconds * FPS), r, dtype=float) for r, seconds in rates])
    return np.full(len(rate), dist), rate


class LabelIntervalsTest(unittest.TestCase):
    def test_positive_rotation_is_clockwise(self):
        dist, rate = series([(2 * np.pi, 10)])          # one turn per second
        [(direction, cycles)] = label_intervals(dist, rate, [(0, 3 * FPS - 1)])
        self.assertEqual(direction, "CW")
        self.assertAlmostEqual(cycles, 3)

    def test_negative_rotation_is_counterclockwise(self):
        dist, rate = series([(-2 * np.pi, 10)])
        [(direction, cycles)] = label_intervals(dist, rate, [(FPS, 3 * FPS - 1)])
        self.assertEqual(direction, "CCW")
        self.assertAlmostEqual(cycles, 2)

    def test_reversal_counts_both_ways_and_takes_majority(self):
        dist, rate = series([(2 * np.pi, 4), (-2 * np.pi, 8)])
        [(direction, cycles)] = label_intervals(dist, rate, [(0, 12 * FPS - 1)])
        self.assertEqual(direction, "CCW")
        self.assertAlmostEqual(cycles, 12, delta=0.1)

    def test_far_apart_fish_do_not_rotate(self):
        dist, rate = series([(2 * np.pi, 5)], dist=500)
        [(_, cycles)] = label_intervals(dist, rate, [(0, 5 * FPS - 1)])
        self.assertEqual(cycles, 0)

    def test_interval_past_end_of_data(self):
        dist, rate = series([(2 * np.pi, 2)])
        self.assertEqual(label_intervals(dist, rate, [(10 * FPS, 11 * FPS)]), [(None, 0.0)])

    def test_video_key(self):
        for name in ("0031", "0031_vid", "0031_vid.csv", "outputs/predictions/x/0031_vid.csv"):
            self.assertEqual(video_key(name), "0031")


class LabelPredictionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data_dir = root / "data"
        cache = self.data_dir / "rule_detector_cache"
        cache.mkdir(parents=True)
        dist, rate = series([(-2 * np.pi, 10)])
        np.savez(cache / "series_0031.npz", dist_clean=dist, rate_clean=rate, dist_raw=dist, rate_raw=-rate)
        self.model_dir = root / "predictions" / "some_model"
        self.model_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_replaces_existing_labels_and_keeps_other_columns(self):
        table = pd.DataFrame({"Video ID": ["0031_vid"], "Start Frame": [0], "End Frame": [2 * FPS - 1],
                              "score": [0.9], "direction": ["CW"]})
        labelled = label_table(table, self.data_dir, "0031_vid")
        self.assertEqual(list(labelled.columns), ["Video ID", "Start Frame", "End Frame", "score",
                                                  "Direction", "Cycles"])
        self.assertEqual(labelled.Direction.tolist(), ["CCW"])
        self.assertAlmostEqual(labelled.Cycles.item(), 2)
        self.assertEqual(label_table(table, self.data_dir, "0031_vid", cleaned=False).Direction.item(), "CW")

    def test_writes_labelled_copy_of_each_file(self):
        pd.DataFrame({"VideoID": ["0031_vid"], "StartFrame": [0], "EndFrame": [FPS - 1]}).to_csv(
            self.model_dir / "0031_vid.csv", index=False)
        pd.DataFrame(columns=["VideoID", "StartFrame", "EndFrame"]).to_csv(
            self.model_dir / "0028_vid.csv", index=False)   # header only: no data needed
        out = label_predictions(self.model_dir, data_dir=self.data_dir)
        self.assertEqual(out, self.model_dir.with_name("some_model_rule_labels"))
        labelled = pd.read_csv(out / "0031_vid.csv")
        self.assertEqual(labelled[["Direction", "Cycles"]].values.tolist(), [["CCW", 1.0]])
        self.assertEqual(len(pd.read_csv(out / "0028_vid.csv")), 0)
        with self.assertRaises(ValueError):
            label_predictions(self.model_dir, self.model_dir, data_dir=self.data_dir)

    def test_writes_oracle_labels_from_multi_video_interval_file(self):
        source = self.model_dir / "merged.csv"
        pd.DataFrame({"VideoID": ["0031_vid"], "StartFrame": [0],
                      "EndFrame": [2 * FPS - 1]}).to_csv(source, index=False)
        output = self.model_dir / "oracle"
        self.assertEqual(label_interval_file(source, output, self.data_dir), output)
        labelled = pd.read_csv(output / "0031_vid.csv")
        self.assertEqual(labelled[["Direction", "Cycles"]].values.tolist(), [["CCW", 2.0]])


if __name__ == "__main__":
    unittest.main()
