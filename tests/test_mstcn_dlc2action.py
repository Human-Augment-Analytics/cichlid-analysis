import contextlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from evaluation.evaluate import GROUND_TRUTH, evaluate, read_intervals, read_predictions
from models.logistic_regression.postprocess import find_positive_spans
from models.logistic_regression.train import CIRCLING_SPANS_BY_VIDEO, VIDEO_FRAMES
from models.mstcn_dlc2action import predict as mstcn_predict
from models.mstcn_dlc2action import train as mstcn_train
from models.mstcn_dlc2action.dlc_yolo_converter import dlc2action_bridge as br

STRIDE = 2
META = {v: {'n_frames': (n - 1) // STRIDE + 1, 'stride': STRIDE, 'frame_min': 0, 'fps': 15.0}
        for v, n in VIDEO_FRAMES.items()}
N_ROWS = {v: m['n_frames'] for v, m in META.items()}
# the 15 spans as H5 rows, as convert-ann writes them
POSITIVES = {v: [(s // STRIDE, -(-(e + 1) // STRIDE)) for s, e in spans]
             for v, spans in CIRCLING_SPANS_BY_VIDEO.items()}


class SamplingTest(unittest.TestCase):
    def setUp(self):
        self.segments = mstcn_train.split_segments(mstcn_train.sample_segments(POSITIVES, N_ROWS))

    def test_balanced_per_video_and_every_positive_segment_kept(self):
        for video, spans in POSITIVES.items():
            rows = self.segments.filter(pl.col('video') == video)
            self.assertEqual(rows['circling'].sum(), (~rows['circling']).sum())
            positive = sum(e - s for s, e in spans)
            self.assertEqual(rows['positive_rows'].sum(), positive)

    def test_segments_on_the_grid_without_overlap(self):
        for video, rows in self.segments.group_by('video'):
            grid = set(br.segment_spans(N_ROWS[video[0]], 128, 0))
            pairs = list(rows.select('start_row', 'end_row').iter_rows())
            self.assertTrue(set(pairs) <= grid)
            self.assertEqual(len(pairs), len(set(pairs)))

    def test_stratified_80_20_and_deterministic(self):
        test = self.segments.filter(pl.col('split') == 'test')
        self.assertAlmostEqual(test.height / self.segments.height, 0.2, delta=0.02)
        self.assertAlmostEqual(test['circling'].mean(), 0.5, delta=0.05)
        again = mstcn_train.split_segments(mstcn_train.sample_segments(POSITIVES, N_ROWS))
        self.assertTrue(again.equals(self.segments))

    def test_split_file_validation_is_test_and_no_shared_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'repo_split.txt'
            mstcn_train.write_split(self.segments, path, N_ROWS)
            parsed = br.read_split_file(path)
            sidecar = json.loads(Path(str(path) + '.json').read_text())
        self.assertEqual(parsed['val'], parsed['test'])
        self.assertTrue(sidecar['val_is_test'])
        self.assertFalse(set(parsed['train']) & set(parsed['test']))
        self.assertEqual(len(parsed['train']) + len(parsed['test']), self.segments.height)


class AnnotationsTest(unittest.TestCase):
    def test_finalized_annotations_read_like_evaluate(self):
        spans = mstcn_train.read_annotations(GROUND_TRUTH)
        gt = read_intervals(GROUND_TRUTH)
        for video, intervals in spans.items():
            expected = sorted(gt.filter(pl.col('video_id') == video).select('start', 'end').iter_rows())
            self.assertEqual(intervals, expected)

    def test_unset_annotations_stop_with_a_message(self):
        with self.assertRaises(SystemExit):
            mstcn_train.read_annotations('')


class FramesTest(unittest.TestCase):
    def test_rows_map_to_original_frames(self):
        meta = {'stride': 2, 'frame_min': 1}
        np.testing.assert_array_equal(mstcn_train.rows_to_frames(1, 3, meta, 10), [3, 4, 5, 6])
        scores = mstcn_train.frame_scores(np.array([.1, .2, .3]), meta, 8)
        self.assertEqual(scores['FrameNum'].to_list(), list(range(8)))
        np.testing.assert_allclose(scores['circling_score'], [0, .1, .1, .2, .2, .3, .3, 0])


class PredictTest(unittest.TestCase):
    """Fake DLC2Action prediction -> score cache -> spans -> repository evaluation."""

    def test_spans_are_the_logistic_regression_rule_and_evaluate(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, out = Path(tmp) / 'work', Path(tmp) / 'out'
            (work / 'data').mkdir(parents=True)
            (work / 'data' / br.META_FILENAME).write_text(json.dumps(META))
            prediction = work / 'projects' / mstcn_train.PROJECT / 'results' / 'predictions' / 'm_pred'
            prediction.mkdir(parents=True)
            rng = np.random.default_rng(0)
            for video, n in N_ROWS.items():
                p = rng.uniform(0, .4, n)
                for s, e in POSITIVES[video]:
                    p[s:e] = .95
                with open(prediction / f'{video}_m_pred_prediction.pickle', 'wb') as fh:
                    pickle.dump({'classes': {0: 'circling'}, 'min_frames': {'ind0+ind1': 0},
                                 'ind0+ind1': p[None, :]}, fh)
            scores = mstcn_train.cache_scores(work, out / 'scores', name='m')
            # test frames: labels from the 15 spans, scores from the cache
            segments = mstcn_train.split_segments(mstcn_train.sample_segments(POSITIVES, N_ROWS))
            test = mstcn_train.test_frames(segments, META, scores, CIRCLING_SPANS_BY_VIDEO)
            n_test = segments.filter(pl.col('split') == 'test')
            self.assertEqual(test.height, int(((n_test['end_row'] - n_test['start_row']) * STRIDE).sum()))
            self.assertTrue((test.filter(pl.col('circling') == 1)['circling_score'] == .95).all())
            mstcn_predict.predict(out / 'scores', out / 'predictions' / 'm')
            for video in VIDEO_FRAMES:
                scores = pl.read_parquet(out / 'scores' / f'{video}_scores.parquet')
                expected = find_positive_spans(scores, fps=30, threshold=.5, fraction=.9, seconds=7.5)
                got = pl.read_csv(out / 'predictions' / 'm' / f'{video}.csv')
                self.assertEqual(got.columns, ['VideoID', 'StartFrame', 'EndFrame',
                                               'duration_seconds', 'positive_fraction'])
                self.assertEqual(got['StartFrame'].to_list(), expected['start_frame'].to_list())
                self.assertEqual(got['EndFrame'].to_list(), expected['end_frame'].to_list())
            results, _ = evaluate(read_intervals(GROUND_TRUTH), read_predictions(out / 'predictions' / 'm'),
                                  num_frames=VIDEO_FRAMES)
            self.assertEqual(results['video_id'].to_list(), ['0028_vid', '0031_vid', 'overall'])


class PreflightAnnotationsTest(unittest.TestCase):
    """Section 0 of the preflight: FINALIZED_ANNOTATIONS set and in sync."""

    def run_check(self, source, workdir):
        import os
        import preflight_dlc2action as pf
        from dlc2action_circling import Paths
        report = pf.Report()
        old = os.environ.get('MSTCN_FINALIZED_ANNOTATIONS')
        os.environ['MSTCN_FINALIZED_ANNOTATIONS'] = str(source)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pf.check_finalized_annotations(report, Paths(workdir))
        finally:
            if old is None:
                os.environ.pop('MSTCN_FINALIZED_ANNOTATIONS')
            else:
                os.environ['MSTCN_FINALIZED_ANNOTATIONS'] = old
        return report

    def test_set_file_passes_and_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.run_check(GROUND_TRUTH, tmp).errors, 0)
            self.assertEqual(self.run_check(Path(tmp) / 'nope.csv', tmp).errors, 1)


class PreflightWholeFileTest(unittest.TestCase):
    """Section 4 checks the finalized file itself; data/<video>.csv only compared."""

    def check(self, workdir, for_training):
        import argparse
        import preflight_dlc2action as pf
        from dlc2action_circling import Paths
        report = pf.Report()
        spans = mstcn_train.read_annotations(GROUND_TRUTH)
        args = argparse.Namespace(behaviors='circling', min_frames_action=5)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out:
            ann_dir = pf.convert_annotations(report, spans, META, tmp)
            pf.check_annotations(report, Paths(workdir), args, META, ann_dir, for_training)
            converted = {v: pl.read_csv(Path(tmp) / f'{v}.csv') for v in META}
        return report, out.getvalue(), converted

    def test_without_per_video_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'data').mkdir()
            report, text, converted = self.check(tmp, for_training=False)
            self.assertEqual(report.errors, 0)
            self.assertIn('not written yet', text)
            self.assertEqual(self.check(tmp, for_training=True)[0].errors, 1)
        self.assertGreater(converted['0028_vid'].height, 0)

    def test_matching_per_video_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'data').mkdir()
            _, _, converted = self.check(tmp, for_training=False)
            for v, table in converted.items():
                table.write_csv(Path(tmp) / 'data' / f'{v}.csv')
            report, text, _ = self.check(tmp, for_training=True)
        self.assertEqual(report.errors, 0)
        self.assertIn("matches FINALIZED_ANNOTATIONS", text)


class EmbedProbeTest(unittest.TestCase):
    """`embed_probe.py probe`: repository LR on the MS-TCN's split and test frames."""

    def test_probe_scores_the_same_test_frames(self):
        import embed_probe
        with tempfile.TemporaryDirectory() as tmp:
            work, out = Path(tmp) / 'work', Path(tmp) / 'out'
            (work / 'data').mkdir(parents=True)
            out.mkdir()
            (work / 'data' / br.META_FILENAME).write_text(json.dumps(META))
            pl.DataFrame([{'VideoID': v, 'StartFrame': s, 'EndFrame': e}
                          for v, spans in CIRCLING_SPANS_BY_VIDEO.items() for s, e in spans]
                         ).write_csv(work / 'training_spans.csv')
            rng = np.random.default_rng(1)
            for video, n in N_ROWS.items():
                label = np.zeros(n, np.float32)
                for s, e in POSITIVES[video]:
                    label[s:e] = 1
                with open(work / 'data' / f'{video}.csv', 'w') as fh:
                    fh.write('behavior,start,end\n' + ''.join(f'circling,{s},{e}\n' for s, e in POSITIVES[video]))
                x = rng.normal(size=(n, 20)).astype(np.float32)
                x[:, 0] += 3 * label                       # one informative "pair" column
                with open(work / 'data' / f'{video}_pairemb.pickle', 'wb') as fh:
                    pickle.dump({'ind0+ind1': x}, fh)
            segments = mstcn_train.split_segments(mstcn_train.sample_segments(POSITIVES, N_ROWS))
            segments.write_parquet(out / 'sampled_segments.parquet')
            with contextlib.redirect_stdout(io.StringIO()):
                embed_probe.main(['probe', '--workdir', str(work), '--output-dir', str(out),
                                  '--pair-cols', '5'])
            res = pl.read_csv(out / 'probe_pairemb.csv')
        self.assertEqual(res['features'].to_list(), ['pair', 'emb', 'all'])
        test = segments.filter(pl.col('split') == 'test')
        self.assertEqual(set(res['frames'].to_list()),
                         {int(((test['end_row'] - test['start_row']) * STRIDE).sum())})
        ap = dict(zip(res['features'], res['average_precision']))
        self.assertGreater(ap['pair'], 0.9)
        self.assertLess(ap['emb'], ap['pair'])


if __name__ == '__main__':
    unittest.main()