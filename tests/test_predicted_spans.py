import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import polars as pl
from predicted_spans import annotate_spans, find_positive_spans


def detect(labels, offset=0):
    return find_positive_spans(pl.DataFrame({'FrameNum':np.arange(len(labels))+offset,
                                           'circling_score':np.asarray(labels,dtype=float)}))


class SpansTest(unittest.TestCase):
    def test_strict_duration(self):
        self.assertTrue(detect([1]*300).is_empty())
        self.assertEqual(detect([1]*301)['frame_count'][0],301)

    def test_fraction_boundary(self):
        self.assertEqual(detect([1]*271+[0]*30).height,1)
        self.assertTrue(detect([1]*270+[0]*31).is_empty())

    def test_long_run_merges(self):
        result=detect([1]*900,offset=100)
        self.assertEqual(result.select('start_frame','end_frame').row(0),(100,999))
        self.assertEqual(result.height,1)

    def test_disjoint_and_valid(self):
        labels=np.r_[np.ones(500),np.zeros(600),np.ones(700)]
        result=detect(labels)
        self.assertEqual(result.height,2)
        previous=-1
        for r in result.iter_rows(named=True):
            self.assertGreater(r['start_frame'],previous)
            self.assertGreater(r['duration_seconds'],10)
            self.assertGreaterEqual(labels[r['start_frame']:r['end_frame']+1].mean(),.9)
            previous=r['end_frame']

    def test_empty(self):
        self.assertTrue(detect([]).is_empty())
        self.assertTrue(detect([0]*500).is_empty())

    def test_review_metadata(self):
        with TemporaryDirectory() as directory:
            video = Path(directory) / 'sample.mp4'
            pl.DataFrame({'FrameNum': [0, 10, 10, 20, 301],
                          'TrackID': [99, 2, 1, 2, 8]}).write_parquet(video.with_suffix('.parquet'))
            result = annotate_spans(detect([1] * 301, offset=10), video)
            row = result.row(0, named=True)
            self.assertEqual(row['Video ID'], 'sample')
            self.assertEqual(row['Event ID'], 'sample-0000010-0000310')
            self.assertEqual((row['Start Time'], row['End Time']),
                             ('00:00:00.333', '00:00:10.367'))
            self.assertEqual(row['Behavior'], 'Predicted circling')
            self.assertEqual(row['Fish IDs'], '1, 2, 8')
            self.assertIn('100.0% positive frames', row['Confidence / Notes'])
            self.assertEqual(annotate_spans(detect([]), video).height, 0)
