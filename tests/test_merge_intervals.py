import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from merge_intervals import Interval, OUTPUT_FIELDS, merge_intervals, write_merged_spans


class MergeIntervalsTest(unittest.TestCase):
    def test_overlap_adjacency_and_video_separation(self):
        rows = merge_intervals([
            Interval('0031_vid', 1, 3, '0031-0001', 'rule-based'),
            Interval('0031_vid', 4, 6, 'pred-1', 'predicted'),
            Interval('0031_vid', 6, 9, '0031-0002', 'rule-based'),
            Interval('0031_vid', 11, 12, 'pred-2', 'predicted'),
            Interval('0028_vid', 2, 8, '0028-0001', 'rule-based'),
        ])
        self.assertEqual(list(rows), ['0028_vid', '0031_vid'])
        self.assertEqual([(row['Start Frame'], row['End Frame']) for row in rows['0031_vid']],
                         [(1, 9), (11, 12)])
        first = rows['0031_vid'][0]
        self.assertEqual(first['Source'], 'both')
        self.assertEqual(first['Verified'], '')
        self.assertEqual(first['Rule IDs'], '0031-0001, 0031-0002')
        self.assertEqual(first['Predicted Event IDs'], 'pred-1')
        self.assertEqual((first['Start Time'], first['End Time']),
                         ('00:00:00.033', '00:00:00.333'))
        self.assertEqual(rows['0028_vid'][0]['Source'], 'rule-based')

    def test_writes_one_file_per_video(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rules = root / 'rule-based.tsv'
            rules.write_text('0031-0001\t10\t20\n0028-0001\t10\t20\n')
            predicted = root / '0031_vid_spans_threshold_0.50.csv'
            with predicted.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    'Video ID', 'Event ID', 'Start Frame', 'End Frame',
                ])
                writer.writeheader()
                writer.writerow({'Video ID': '0031_vid', 'Event ID': 'pred-1',
                                 'Start Frame': 21, 'End Frame': 25})

            paths = write_merged_spans(rules, [predicted], root / 'merged')
            self.assertEqual([path.name for path in paths], [
                '0028_vid_merged_spans.csv', '0031_vid_merged_spans.csv',
            ])
            with paths[1].open(newline='') as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames[4], 'Verified')
                self.assertEqual(tuple(reader.fieldnames), OUTPUT_FIELDS)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]['Start Frame'], rows[0]['End Frame']), ('10', '25'))
            self.assertEqual(rows[0]['Source'], 'both')
            self.assertEqual(rows[0]['Verified'], '')

            rows[0]['Verified'] = 'Yes'
            with paths[1].open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            write_merged_spans(rules, [predicted], root / 'merged')
            with paths[1].open(newline='') as stream:
                self.assertEqual(next(csv.DictReader(stream))['Verified'], 'Yes')
