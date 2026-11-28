import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import polars as pl

from evaluation.evaluate import evaluate
from evaluation import evaluate_models as helpers


class ModelPerformanceReportTests(unittest.TestCase):
    def test_windows_use_evaluated_frame_counts_for_additional_videos(self):
        truth = pl.DataFrame({'video_id': ['extra_video'], 'start': [0], 'end': [99]})
        results = evaluate(truth, truth, num_frames={'extra_video': 3600})
        with patch.object(helpers, 'video_frames', side_effect=AssertionError('default videos used')):
            windows = helpers.window_report(truth, {'model': results}, {'model': truth})
        self.assertEqual(windows['windows'].to_list(), [3, 3])
        self.assertEqual(windows['f1'].to_list(), [1.0, 1.0])

    def test_heldout_reads_test_and_windowed_xgboost_formats(self):
        metrics = dict(frames=100, positive_frames=20, prevalence=.2,
                       average_precision=.9, roc_auc=.95, precision=.8, recall=.7, f1=.75)
        with TemporaryDirectory() as directory:
            paths = {}
            for name, key in [('mstcn', 'test'), ('xgboost', 'holdout')]:
                paths[name] = Path(directory) / f'{name}.json'
                paths[name].write_text(json.dumps({key: metrics}))
            with patch.object(helpers, 'metrics_path', side_effect=paths.get):
                table = helpers.heldout_metrics(list(paths))
        self.assertEqual(table['model'].to_list(), ['mstcn', 'xgboost'])
        self.assertEqual(table['test_items'].to_list(), [100, 100])

    def test_notebook_runs_for_all_model_families_on_additional_video(self):
        import model_performance_report as report

        names = ['logistic_regression', 'rule_based', 'mstcn_pairmotion', 'xgboost',
                 'xgboost_pose_pair', 'lrcv_pairmotion', 'sartab', 'actionformer_pairmotion']
        truth = pl.DataFrame({'video_id': ['extra_video'], 'start': [10], 'end': [3599],
                              'direction': ['CW'], 'cycles': [2.0]})
        with TemporaryDirectory() as directory:
            models = {}
            for name in names:
                folder = Path(directory) / name
                folder.mkdir()
                # Header-only files must still declare the evaluation scope.
                (folder / 'extra_video.csv').write_text(
                    'VideoID,StartFrame,EndFrame,Direction,Cycles,Score\n'
                    + ('' if name == 'sartab' else 'extra_video,10,3599,CW,2,0.9\n'))
                models[name] = folder
            with (patch('evaluation.evaluate.PREDICTIONS_DIR', Path(directory)),
                  patch('evaluation.evaluate.read_ground_truth', new=lambda: truth.select('video_id', 'start', 'end')),
                  patch('evaluation.evaluate.read_ground_truth_labels', new=lambda *args, **kwargs: truth),
                  patch.object(helpers, 'heldout_metrics', new=lambda names: pl.DataFrame()),
                  patch('video_source.download', side_effect=AssertionError('unexpected download'))):
                _, definitions = report.app.run()
        self.assertEqual(set(definitions['evaluations']), set(names))
        for results, _ in definitions['evaluations'].values():
            self.assertEqual(results['video_id'].to_list(), ['extra_video', 'overall'])
            self.assertEqual(results['num_frames'].to_list(), [3600, 3600])


if __name__ == '__main__':
    unittest.main()
