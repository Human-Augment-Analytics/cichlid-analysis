import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

import evaluation.evaluate as evaluate_module
from evaluation import generalizability as gen
from models.common import config
from models.common.frame_model import append_annotated_training_rows, near_frames


class AnnotatedNegativeContextTests(unittest.TestCase):
    def test_near_frames_is_inclusive_and_clipped(self):
        mask = near_frames(20, [0, 10], radius=2)
        self.assertEqual(np.flatnonzero(mask).tolist(), [0, 1, 2, 8, 9, 10, 11, 12])

    def test_added_negatives_keep_clear_of_test_frames(self):
        rows = pl.DataFrame({'video': ['v'] * 10, 'FrameNum': list(range(10)), 'segment': [0] * 5 + [1] * 5,
                             'split': ['train'] * 5 + ['test'] * 5})
        combined, added = append_annotated_training_rows(rows, {'v': 100}, {'v': []}, {'v': [(0, 99)]}, radius=20)
        extra = combined.filter(pl.col('FrameNum') >= 10)
        # Test frames are 5-9, so frames up to 29 share their context and are left out.
        self.assertEqual(extra['FrameNum'].min(), 30)
        self.assertEqual(added, {'v': 70})
        self.assertEqual(set(extra['split']), {'train'})


class FinalizedSpansTests(unittest.TestCase):
    def test_write_is_atomic_and_skips_unchanged_content(self):
        spans = pl.DataFrame({'video_id': ['v'], 'start': [1], 'end': [5]})
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(evaluate_module, 'read_ground_truth', return_value=spans):
            path = Path(directory) / 'finalized_spans.csv'
            config.write_finalized_spans(path)
            self.assertEqual(path.read_text().splitlines(), ['VideoID,StartFrame,EndFrame', 'v,1,5'])
            os.utime(path, (0, 0))
            config.write_finalized_spans(path)
            self.assertEqual(path.stat().st_mtime, 0)  # unchanged content: not rewritten
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['finalized_spans.csv'])


class FoldStateTests(unittest.TestCase):
    def test_training_set_must_exclude_test_and_holdout_videos(self):
        for train in (['a', 'b'], ['b', 'h']):
            with self.subTest(train=train), self.assertRaises(SystemExit):
                gen.run_fold('a', ['a', 'b', 'c'], [], Path('.'), train_videos=train, holdout='h')

    def test_debug_training_sets_use_their_own_folder(self):
        self.assertEqual(gen.fold_dir('a'), gen.FOLDS_DIR / 'a')
        self.assertEqual(gen.fold_dir('a', ['b', 'c'], ['c', 'b']), gen.FOLDS_DIR / 'a')
        debug = gen.fold_dir('a', ['b'], ['b', 'c'])
        self.assertEqual(debug.parent, gen.DEBUG_FOLDS_DIR)
        self.assertTrue(debug.name.startswith('a_1videos_'))

    def test_results_skip_unfinished_models_and_record_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'a'
            (folder / 'predictions' / 'xgboost_turns').mkdir(parents=True)
            (folder / 'predictions' / 'lrcv_turns').mkdir(parents=True)
            for name in ('xgboost_turns', 'lrcv_turns'):
                (folder / 'predictions' / name / 'a.csv').write_text('VideoID,StartFrame,EndFrame\na,1,2\n')
            (folder / 'video_frames.json').write_text(json.dumps({'a': 100}))
            (folder / 'fold.json').write_text(json.dumps({'test_video': 'a', 'train_videos': ['b', 'c'], 'models': {
                # A rerun that failed: no 'predictions' key, although an old CSV is on disk.
                'lrcv_turns': {'annotated_negatives': False, 'train_seconds': 1.0},
                # Recorded before fold.json stored the setting.
                'xgboost_turns': {'train_seconds': 1.0, 'predict_seconds': 1.0,
                                  'predictions': 'predictions/xgboost_turns/a.csv'}}}))
            results = pl.DataFrame({'video_id': ['a', 'overall'], 'f1@50': [0.5, 0.5]})
            with patch.object(evaluate_module, 'read_ground_truth',
                              return_value=pl.DataFrame({'video_id': ['a'], 'start': [1], 'end': [2]})), \
                    patch.object(evaluate_module, 'read_ground_truth_labels',
                                 return_value=pl.DataFrame({'video_id': ['a']})), \
                    patch.object(evaluate_module, 'read_predictions',
                                 return_value=pl.DataFrame({'video_id': ['a'], 'start': [1], 'end': [2]})), \
                    patch.object(evaluate_module, 'evaluate', return_value=(results, None)), \
                    patch.object(gen, 'split_videos', return_value=('h', ['a', 'b', 'c'])):
                rows = gen.fold_results(directory)
        self.assertEqual(rows['model'].to_list(), ['xgboost_turns'])
        self.assertEqual(rows['reviewed_negatives'].to_list(), ['unknown'])
        self.assertEqual(rows['standard_train_set'].to_list(), [True])

    def test_reviewed_negatives_come_from_the_fold_record(self):
        self.assertEqual(gen.reviewed_negatives({'annotated_negatives': True}), 'with')
        self.assertEqual(gen.reviewed_negatives({'annotated_negatives': False}), 'without')
        self.assertEqual(gen.reviewed_negatives({}), 'unknown')

    def test_model_defaults_match_train_clis(self):
        self.assertTrue(gen.MODELS['xgboost']['annotated_negatives'])
        self.assertTrue(gen.MODELS['xgboost_pairmotion']['annotated_negatives'])
        self.assertFalse(gen.MODELS['lrcv_pairmotion']['annotated_negatives'])
        self.assertFalse(gen.MODELS['actionformer_pairmotion'].get('annotated_negatives', False))


if __name__ == '__main__':
    unittest.main()
