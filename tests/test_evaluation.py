import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from evaluation.evaluate import evaluate, read_ground_truth, read_intervals, read_predictions
from evaluation.metrics import (average_precision, edit_score, evaluate_attribute_quality,
                                evaluate_labels, evaluate_oracle_labels, evaluate_video, label_spans,
                                map_at_tiou, segment_true_positives, summarize, summarize_labels,
                                temporal_iou)


class EvaluationTest(unittest.TestCase):
    def test_finalized_ground_truth_intervals_are_valid(self):
        ground_truth = read_ground_truth()
        self.assertGreater(len(ground_truth), 0)
        self.assertTrue((ground_truth['start'] >= 0).all())
        self.assertTrue((ground_truth['end'] >= ground_truth['start']).all())

    def test_perfect_prediction(self):
        result = evaluate_video([(10, 19), (40, 49)], [(10, 19), (40, 49)], 100)
        self.assertEqual(result['mof'], 1)
        self.assertEqual(result['edit_score'], 100)
        self.assertEqual(result['frame_f1'], 1)
        self.assertEqual(result['frame_iou'], 1)
        self.assertEqual([result[f'f1@{tau}'] for tau in (10, 25, 50)], [1, 1, 1])

    def test_partial_prediction(self):
        # GT 20 frames, prediction 10 frames of which 5 overlap.
        result = evaluate_video([(0, 9), (20, 29)], [(5, 14)], 40)
        self.assertEqual((result['frame_tp'], result['frame_fp'], result['frame_fn']), (5, 5, 15))
        self.assertAlmostEqual(result['mof'], 20 / 40)
        self.assertAlmostEqual(result['frame_precision'], .5)
        self.assertAlmostEqual(result['frame_recall'], .25)
        self.assertAlmostEqual(result['frame_f1'], 1 / 3)
        # Video-level IoU spans both ground-truth segments: 5 shared of 25 frames.
        self.assertAlmostEqual(result['frame_iou'], 5 / 25)
        # IoU 5/15: a true positive at tau 10 and 25, not 50. One of one predicted
        # and two ground-truth segments gives F1 = 2 * 1 / (1 + 2).
        self.assertEqual([result[f'tp@{tau}'] for tau in (10, 25, 50)], [1, 1, 0])
        self.assertAlmostEqual(result['f1@25'], 2 / 3)
        self.assertEqual(result['f1@50'], 0)

    def test_empty_prediction(self):
        result = evaluate_video([(0, 9)], [], 20)
        self.assertTrue(math.isnan(result['frame_precision']))
        self.assertEqual(result['frame_f1'], 0)
        self.assertEqual(result['frame_iou'], 0)
        self.assertEqual(result['f1@10'], 0)

    def test_frame_iou_ignores_overlapping_duplicates(self):
        result = evaluate_video([(0, 9)], [(0, 4), (3, 9), (10, 14)], 20)
        self.assertAlmostEqual(result['frame_iou'], 10 / 15)

    def test_edit_score_counts_segments(self):
        # GT: bg c bg c bg; prediction: bg c bg -> two deletions over five segments.
        self.assertAlmostEqual(edit_score([(10, 19), (40, 49)], [(10, 49)], 100), 60)

    def test_one_true_positive_per_ground_truth_segment(self):
        # Both predictions overlap the same segment; the second is a false positive.
        self.assertEqual(segment_true_positives([(0, 9)], [(0, 4), (5, 9)], 10), 1)

    def test_iou_threshold_is_inclusive(self):
        # IoU 5/10 is exactly 0.5.
        self.assertEqual(segment_true_positives([(0, 9)], [(0, 4)], 50), 1)
        self.assertEqual(segment_true_positives([(0, 9)], [(0, 3)], 50), 0)

    def test_temporal_iou_uses_half_open_intervals(self):
        self.assertAlmostEqual(temporal_iou((0., 10.), (5., 15.)), 5 / 15)
        self.assertEqual(temporal_iou((0., 5.), (5., 10.)), 0)

    def test_average_precision_is_score_ordered_and_penalizes_duplicates(self):
        gt = [(0., 10.), (20., 30.)]
        predictions = [(20., 30., .7), (0., 10., .9), (0., 10., .8)]
        self.assertAlmostEqual(average_precision(gt, predictions, 50), 5 / 6)
        self.assertTrue(math.isnan(average_precision([], predictions, 50)))
        self.assertEqual(average_precision(gt, [], 50), 0)

    def test_map_averages_tiou_thresholds(self):
        result = map_at_tiou([(0., 10.)], [(0., 6., .9)])
        self.assertEqual(result['map@50'], 1)
        self.assertEqual(result['map@70'], 0)
        self.assertAlmostEqual(result['map_avg'], 6 / 7)

    def test_summary_pools_frames(self):
        rows = [evaluate_video([(0, 9)], [(0, 9)], 10), evaluate_video([(0, 9)], [], 30)]
        overall = summarize(rows)
        self.assertAlmostEqual(overall['mof'], 30 / 40)
        self.assertAlmostEqual(overall['frame_recall'], .5)
        self.assertAlmostEqual(overall['frame_iou'], .5)
        self.assertEqual(overall['edit_score'], (100 + 50) / 2)
        # Pooled: one true positive, one predicted and two ground-truth segments.
        self.assertAlmostEqual(overall['f1@50'], 2 / 3)

    def test_summary_averages_frame_iou(self):
        # IoUs of 1 and 1/3; pooling the frames would give 20/40 instead.
        rows = [evaluate_video([(0, 9)], [(0, 9)], 10), evaluate_video([(0, 29)], [(0, 9)], 30)]
        self.assertAlmostEqual(summarize(rows)['frame_iou'], 2 / 3)

    def test_reads_frame_csvs(self):
        with TemporaryDirectory() as directory:
            frames = Path(directory, 'pred.csv')
            frames.write_text('Video ID,Start Frame,End Frame\n0028_vid,3,7\n')
            times = Path(directory, 'gt.csv')
            times.write_text('VideoID,StartTime,EndTime\n0028_vid,00:00:01,00:00:02.5\n')
            self.assertEqual(read_intervals(frames).rows(), [('0028_vid', 3, 7)])
            with self.assertRaises(ValueError):
                read_intervals(times)

    def test_reads_model_folder(self):
        with TemporaryDirectory() as directory:
            Path(directory, '0028_vid.csv').write_text('VideoID,StartFrame,EndFrame,extra\n0028_vid,3,7,x\n')
            Path(directory, '0031_vid.csv').write_text('VideoID,StartFrame,EndFrame,extra\n')
            self.assertEqual(read_predictions(directory).rows(), [('0028_vid', 3, 7)])
        with TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            read_predictions(directory)

    def test_label_spans_prorates_cycles_and_takes_majority_direction(self):
        events = [(0, 9, 'CW', 2), (10, 29, 'CCW', 4), (30, 39, 'CW', 1)]
        # Span 0-19 holds all of the CW event and half of the CCW one.
        self.assertEqual(label_spans([(0, 19), (50, 59)], events),
                         [(0, 19, 'CW', 4), (50, 59, None, None)])
        self.assertEqual(label_spans([(10, 39)], events), [(10, 39, 'CCW', 5)])

    def test_direction_counts(self):
        gt = [(0, 9, 'CW', 2), (20, 29, 'CCW', 3), (40, 49, None, None), (60, 69, 'CW', 1)]
        # Correct CW, wrong direction on the CCW segment, a prediction on the
        # unlabelled segment (ignored), a spurious CCW and a missed CW segment.
        pred = [(0, 9, 'CW', 3), (20, 29, 'CW', 1), (40, 49, 'CCW', 5), (80, 89, 'CCW', 2)]
        result = evaluate_labels(gt, pred)
        self.assertEqual((result['cw_tp'], result['cw_fp'], result['cw_fn']), (1, 1, 1))
        self.assertEqual((result['ccw_tp'], result['ccw_fp'], result['ccw_fn']), (0, 1, 1))
        self.assertAlmostEqual(result['cw_precision'], .5)
        self.assertAlmostEqual(result['cw_f1'], .5)
        self.assertEqual(result['ccw_f1'], 0)
        self.assertAlmostEqual(result['direction_f1'], .25)
        self.assertAlmostEqual(result['direction_accuracy'], .5)
        # Cycle errors over the two paired segments with known counts: +1 and -2.
        self.assertEqual(result['cycle_pairs'], 2)
        self.assertAlmostEqual(result['cycle_mae'], 1.5)
        self.assertAlmostEqual(result['cycle_rmse'], (5 / 2) ** .5)
        self.assertAlmostEqual(result['cycle_bias'], -.5)
        self.assertEqual((result['gt_cycles'], result['pred_cycles']), (6, 11))

    def test_strict_attribute_metrics_do_not_count_detection_errors(self):
        gt = [(0, 9, 'CW', 2), (20, 29, 'CCW', 3)]
        pred = [(0, 9, 'CW', 2.5), (40, 49, 'CCW', 1)]
        joint = evaluate_labels(gt, pred)
        strict = evaluate_attribute_quality(gt, pred, tau=50)
        self.assertLess(joint['direction_f1'], 1)
        self.assertEqual(strict['attribute_direction_accuracy'], 1)
        self.assertEqual(strict['attribute_direction_balanced_accuracy'], 1)
        self.assertEqual(strict['attribute_direction_coverage'], 1)
        self.assertEqual(strict['attribute_direction_pairs'], 1)
        self.assertEqual(strict['attribute_cycle_mae'], .5)
        self.assertEqual(strict['attribute_cycle_wape'], .25)
        self.assertEqual(strict['attribute_cycle_within_0p5'], 1)

    def test_oracle_metrics_require_exact_ground_truth_boundaries(self):
        gt = [(0, 9, 'CW', 2)]
        result = evaluate_oracle_labels(gt, [(0, 9, 'CW', 2.25)])
        self.assertEqual(result['attribute_direction_accuracy'], 1)
        self.assertEqual(result['attribute_cycle_mae'], .25)
        self.assertEqual(result['attribute_cycle_within_0p5'], 1)
        with self.assertRaises(ValueError):
            evaluate_oracle_labels(gt, [(0, 8, 'CW', 2)])

    def test_unpredicted_labels_are_missing(self):
        result = evaluate_labels([(0, 9, 'CW', 2)], [(0, 9, None, None)])
        self.assertIsNone(result['cw_tp'])
        self.assertIsNone(result['pred_cycles'])
        self.assertTrue(math.isnan(result['direction_f1']))
        self.assertTrue(math.isnan(result['cycle_mae']))
        overall = summarize_labels([result, result])
        self.assertTrue(math.isnan(overall['direction_f1']))
        self.assertTrue(math.isnan(overall['cycle_mae']))

    def test_summary_pools_labels(self):
        rows = [evaluate_labels([(0, 9, 'CW', 2)], [(0, 9, 'CW', 3)]),
                evaluate_labels([(0, 9, 'CCW', 2), (20, 29, 'CW', 2)],
                                [(0, 9, 'CW', 2), (20, 29, 'CW', 6)])]
        overall = summarize_labels(rows)
        self.assertEqual((overall['cw_tp'], overall['cw_fp'], overall['cw_fn']), (2, 1, 0))
        self.assertAlmostEqual(overall['direction_accuracy'], 2 / 3)
        # Errors +1, 0 and +4.
        self.assertAlmostEqual(overall['cycle_mae'], 5 / 3)
        self.assertAlmostEqual(overall['cycle_rmse'], (17 / 3) ** .5)
        self.assertAlmostEqual(overall['cycle_bias'], 5 / 3)

    def test_evaluate_labels_ground_truth_spans_from_events(self):
        gt = pl.DataFrame({'video_id': ['v', 'v'], 'start': [0, 20], 'end': [9, 29]})
        # One annotated event spans both ground-truth spans; they must stay separate.
        gt_labels = pl.DataFrame({'video_id': ['v'], 'start': [0], 'end': [29],
                                  'direction': ['CCW'], 'cycles': [3.]})
        predicted = pl.DataFrame({'video_id': ['v'], 'start': [0], 'end': [9],
                                  'direction': ['CCW'], 'cycles': [1.]})
        results, _ = evaluate(gt, predicted, num_frames={'v': 40}, gt_labels=gt_labels)
        overall = results.row(-1, named=True)
        self.assertEqual((overall['ccw_tp'], overall['ccw_fn']), (1, 1))
        self.assertAlmostEqual(overall['cycle_bias'], 0)

    def test_reads_labels(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, 'labels.csv')
            path.write_text('VideoID,Start Frame,End Frame,Rotation direction,No. of turns/circles\n'
                            '0028_vid,0,9,CW,2.5\n0028_vid,20,29,ccw,\n,,,,\n')
            self.assertEqual(read_intervals(path, labels=True).rows(),
                             [('0028_vid', 0, 9, 'CW', 2.5), ('0028_vid', 20, 29, 'CCW', None)])
            path.write_text('VideoID,StartFrame,EndFrame\n0028_vid,0,9\n')
            self.assertEqual(read_intervals(path, labels=True).rows(), [('0028_vid', 0, 9, None, None)])
            path.write_text('VideoID,StartFrame,EndFrame,Direction\n0028_vid,0,9,left\n')
            with self.assertRaises(ValueError):
                read_intervals(path, labels=True)

    def test_reads_scores_and_filters_behavior(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, 'pred.csv')
            path.write_text('VideoID,StartFrame,EndFrame,Behavior,Score\n'
                            'v,0,9,Circling,0.8\nv,20,29,Chase,0.4\n')
            self.assertEqual(read_intervals(path, scores=True, behavior='circling').rows(),
                             [('v', 0, 9, .8)])
            path.write_text('VideoID,StartFrame,EndFrame,Score\nv,0,9,not-a-score\n')
            with self.assertRaises(ValueError):
                read_intervals(path, scores=True)

    def test_evaluate_reports_map_only_with_scores(self):
        gt = pl.DataFrame({'video_id': ['v'], 'start': [0], 'end': [9]})
        scored = pl.DataFrame({'video_id': ['v'], 'start': [0], 'end': [9], 'score': [.9]})
        results, warnings = evaluate(gt, scored, num_frames={'v': 20})
        self.assertEqual(results.row(-1, named=True)['map_avg'], 1)
        self.assertFalse(warnings)
        unscored = scored.drop('score')
        results, warnings = evaluate(gt, unscored, num_frames={'v': 20})
        self.assertNotIn('map_avg', results.columns)
        self.assertTrue(any('mAP unavailable' in warning for warning in warnings))

    def test_evaluation_scope_comes_from_declared_prediction_videos(self):
        gt = pl.DataFrame({'video_id': ['v', 'w'], 'start': [0, 0], 'end': [9, 9]})
        predicted = pl.DataFrame({'video_id': ['v'], 'start': [0], 'end': [9]})
        results, warnings = evaluate(gt, predicted, num_frames={'v': 20}, videos=['v'])
        self.assertEqual(results['video_id'].to_list(), ['v', 'overall'])
        self.assertTrue(any('excludes' in warning and "'w'" in warning for warning in warnings))


if __name__ == '__main__':
    unittest.main()
