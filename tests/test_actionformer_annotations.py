import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.common import config


class TestFinalizedAnnotations(unittest.TestCase):
    def make_csv(self, directory, name, rows):
        with (Path(directory) / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['VideoID', 'Start Frame', 'End Frame', 'Behavior'])
            writer.writeheader()
            writer.writerows(rows)

    def test_load_and_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_csv(directory, 'circling_annotations_vid0028.csv', [
                {'VideoID': '0028_vid', 'Start Frame': 10, 'End Frame': 20, 'Behavior': 'Circling'},
                {'VideoID': '0028_vid', 'Start Frame': 30, 'End Frame': 40, 'Behavior': 'Other'},
            ])
            with patch.object(config, 'video_frames', return_value={'0028_vid': 100}):
                self.assertEqual(config.read_annotations(directory), {'0028_vid': [(10, 20)]})

    def test_duplicate_recording_id_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for day in ('day5', 'day6'):
                self.make_csv(directory, f'circling_annotations_vid0007-{day}.csv', [
                    {'VideoID': '0007_vid', 'Start Frame': 10, 'End Frame': 20, 'Behavior': 'Circling'},
                ])
            with patch.object(config, 'video_frames', return_value={'0007_vid': 100}):
                with self.assertRaises(SystemExit):
                    config.read_annotations(directory)

    def test_out_of_range_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_csv(directory, 'circling_annotations_vid0028.csv', [
                {'VideoID': '0028_vid', 'Start Frame': 90, 'End Frame': 110, 'Behavior': 'Circling'},
            ])
            with patch.object(config, 'video_frames', return_value={'0028_vid': 100}):
                with self.assertRaises(SystemExit):
                    config.read_annotations(directory)


if __name__ == '__main__':
    unittest.main()
