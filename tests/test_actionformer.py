import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import torch

from models.actionformer.model import (build_model, infer_windows, video_item)
from models.actionformer.postprocess import inclusive_intervals, interval_scores, prediction_table
from models.actionformer.predict import feature_chunk, window_starts
from models.actionformer.train import (normalization_stats, normalize, train_model, training_validation_split,
                                       tune_video_postprocess, validation_loss)
from models.actionformer.tuning import (load_inference_settings, proposal_summary, select_inference_settings,
                                        select_oof_inference_settings)
from models.actionformer.vendor.libs.utils import batched_nms


class ActionFormerTests(unittest.TestCase):
    def test_official_model_forward_loss_and_native_inference(self):
        model, config = build_model(4)
        self.assertEqual(config['model']['num_classes'], 1)
        self.assertEqual(config['model']['n_mha_win_size'], 16)
        features = np.random.default_rng(42).normal(size=(32, 4)).astype(np.float32)
        item = video_item(features)
        item['segments'] = torch.tensor([[4.0, 20.0]])
        item['labels'] = torch.tensor([0])
        model.train()
        losses = model([item])
        self.assertTrue(torch.isfinite(losses['final_loss']))
        losses['final_loss'].backward()
        proposals = infer_windows(model, [(0, features)], torch.device('cpu'), min_score=0.0)
        self.assertIsInstance(proposals, list)
        for (start, end), score in proposals:
            self.assertGreaterEqual(end, start)
            self.assertGreaterEqual(score, 0.0)

    def test_portable_hard_and_soft_nms(self):
        segments = torch.tensor([[0.0, 10.0], [1.0, 9.0], [20.0, 30.0]])
        scores = torch.tensor([0.9, 0.8, 0.7])
        labels = torch.zeros(3, dtype=torch.long)
        hard, hard_scores, _ = batched_nms(segments, scores, labels, 0.3, 0.0, 10,
                                           use_soft_nms=False, multiclass=False, voting_thresh=0)
        self.assertEqual(hard.shape[0], 2)
        np.testing.assert_allclose(hard_scores.numpy(), [0.9, 0.7])
        soft, soft_scores, _ = batched_nms(segments, scores, labels, 0.3, 0.0, 10,
                                           use_soft_nms=True, multiclass=True)
        self.assertEqual(soft.shape[0], 3)
        self.assertLess(soft_scores[1], 0.8)
        self.assertGreater(soft_scores[1], 0.0)

    def test_native_interval_conversion_confidence_and_merge(self):
        proposals = [((2.2, 6.1), 0.8), ((8.0, 10.2), 0.6)]
        intervals = inclusive_intervals(proposals, 20, fps=2, merge_seconds=1.0)
        self.assertEqual(intervals.select('StartFrame', 'EndFrame').rows(), [(2, 10)])
        self.assertEqual(intervals['Confidence'][0], 0.8)
        scores = interval_scores(intervals, 20)
        self.assertEqual(scores[2], 0.8)
        self.assertEqual(scores[11], 0.0)
        empty = prediction_table('v', [], 20)
        self.assertEqual(empty.columns, ['VideoID', 'StartFrame', 'EndFrame', 'duration_seconds',
                                        'positive_fraction', 'Confidence'])
        self.assertTrue(empty.is_empty())

    def test_window_coverage_and_training_only_normalization(self):
        self.assertEqual(window_starts(100), [0])
        starts = window_starts(600)
        self.assertEqual(starts[-1], 344)
        self.assertEqual(starts[0], 0)
        train = [np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)]
        mean, std = normalization_stats(train)
        np.testing.assert_allclose(mean, [2.0, 4.0])
        np.testing.assert_allclose(std, [1.0, 1.0])
        normalized = normalize(np.array([[np.nan, np.inf]], dtype=np.float32), mean, std)
        np.testing.assert_allclose(normalized, [[0.0, 0.0]])

    def test_checkpoint_validation_is_split_only_from_training_samples(self):
        samples = [{
            'sample_id': index, 'split': 'train',
            'segments': torch.tensor([[1.0, 2.0]]) if index % 2 == 0 else torch.empty(0, 2),
        } for index in range(10)]
        fitting, validation = training_validation_split(samples, fraction=0.2, seed=42)
        fit_ids = {sample['sample_id'] for sample in fitting}
        validation_ids = {sample['sample_id'] for sample in validation}
        self.assertEqual(len(fitting), 8)
        self.assertEqual(len(validation), 2)
        self.assertFalse(fit_ids & validation_ids)
        self.assertTrue(all(sample['split'] == 'train' for sample in validation))
        self.assertEqual(sum(sample['segments'].shape[0] > 0 for sample in validation), 1)

    def test_training_restores_best_validation_checkpoint_and_stops(self):
        class TinyLossModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, batch):
                target = torch.stack([sample['feats'].reshape(-1)[0] for sample in batch]).mean()
                loss = (self.value - target) ** 2
                return {'final_loss': loss, 'cls_loss': loss, 'reg_loss': loss * 0}

        fitting = [{'feats': torch.tensor([[0.0]])}]
        validation = [{'feats': torch.tensor([[1.0]])}]
        model = TinyLossModel()
        history = train_model(model, fitting, torch.device('cpu'), epochs=5, batch_size=1,
                              learning_rate=0.1, validation_samples=validation, patience=2, min_epochs=1,
                              checkpoint_start=1)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]['is_best'], True)
        self.assertEqual(model.value.item(), 0.0)

    def test_native_proposal_filters_are_selected_on_validation(self):
        from unittest.mock import patch

        sample = {
            'feats': torch.zeros(2, 300),
            'segments': torch.tensor([[60.0, 240.0]]),
        }
        proposals = [((55.0, 245.0), 0.03), ((0.0, 20.0), 0.02)]
        with tempfile.TemporaryDirectory() as directory:
            with patch('models.actionformer.tuning.infer_windows', return_value=proposals):
                settings = select_inference_settings(object(), [sample], torch.device('cpu'), directory)
            self.assertGreaterEqual(settings['min_score'], 0.025)
            self.assertLessEqual(settings['min_seconds'], 4.0)
            self.assertEqual(load_inference_settings(directory), settings)
            self.assertEqual(load_inference_settings(directory, min_score=0.05)['min_score'], 0.05)
            self.assertTrue((Path(directory) / 'postprocess_candidates.csv').is_file())

    def test_oof_full_video_selection_tunes_confidence_and_merging(self):
        proposals = {
            'a': [((100.0, 201.0), 0.8), ((305.0, 401.0), 0.7), ((500.0, 551.0), 0.2)],
            'b': [((50.0, 180.0), 0.2)],
        }
        truth = {'a': [(100, 400)], 'b': []}
        counts = {'a': 600, 'b': 600}
        with tempfile.TemporaryDirectory() as directory:
            settings = select_oof_inference_settings(
                proposals, truth, counts, directory,
                folds=[{'fold': 1, 'validation_videos': ['a']},
                       {'fold': 2, 'validation_videos': ['b']}])
            self.assertGreaterEqual(settings['min_score'], 0.35)
            self.assertEqual(settings['merge_seconds'], 4.0)
            self.assertEqual(load_inference_settings(directory), settings)
            self.assertEqual(load_inference_settings(directory, merge_seconds=1.5)['merge_seconds'], 1.5)
            payload = (Path(directory) / 'postprocess.json').read_text()
            self.assertIn('full-video native proposals', payload)
            self.assertTrue((Path(directory) / 'postprocess_candidates.csv').is_file())

    def test_oof_training_excludes_each_validation_video(self):
        from unittest.mock import patch

        segments = pl.DataFrame({
            'video': ['a', 'a', 'b', 'b', 'c', 'c'],
            'start_row': [0, 1] * 3, 'end_row': [1, 2] * 3,
            'split': ['train'] * 6,
        })
        meta = {video: {'stride': 2, 'frame_min': 0} for video in ('a', 'b', 'c')}
        spans = {'a': [(0, 10)], 'b': [(0, 10)], 'c': []}
        counts = {'a': 300, 'b': 300, 'c': 300}
        prepared = []

        def prepare(fold_segments, fold_meta, *args):
            videos = sorted(fold_segments['video'].unique().to_list())
            prepared.append(videos)
            return ([{'video': video} for video in videos], ['feature'],
                    np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32))

        with tempfile.TemporaryDirectory() as directory:
            with (patch('models.actionformer.train.prepare_samples', side_effect=prepare),
                  patch('models.actionformer.train.build_model', return_value=(object(), {})),
                  patch('models.actionformer.train.train_model', return_value=[{'epoch': 1}]),
                  patch('models.actionformer.predict.decode_video', return_value=[]) as decode,
                  patch('models.actionformer.tuning.select_oof_inference_settings',
                        return_value={'min_score': 0.1, 'min_seconds': 2.0, 'merge_seconds': 1.0}) as select):
                result = tune_video_postprocess(
                    segments, meta, spans, counts, 'workdir', 'pairmotion', ['feature'],
                    torch.device('cpu'), 1, 1, 1e-4, 0.0, 0, directory)
        self.assertEqual(result['merge_seconds'], 1.0)
        self.assertEqual(len(prepared), 3)
        # With three single-video folds, decode calls follow preparation order.
        for fold, fit_videos in enumerate(prepared):
            validation_video = decode.call_args_list[fold].args[1]
            self.assertNotIn(validation_video, fit_videos)
        self.assertEqual(set(select.call_args.args[0]), {'a', 'b', 'c'})

    def test_validation_loss_disables_dropout_and_empty_diagnostics(self):
        class DropoutLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(0.9)
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, batch):
                value = self.dropout(self.weight)
                return {'final_loss': value}

        model = DropoutLoss()
        samples = [{'feats': torch.ones(1, 1)}]
        self.assertEqual(validation_loss(model, samples, 1), 1.0)
        self.assertEqual(validation_loss(model, samples, 1), 1.0)
        summary = proposal_summary([], [], pl.DataFrame())
        self.assertEqual(summary['raw_proposals'], 0)
        self.assertIsNone(summary['confidence_max'])

    def test_old_checkpoint_decoder_is_reopened_for_raw_proposals(self):
        from models.actionformer.predict import load_checkpoint

        _, config = build_model(2)
        config['model']['test_cfg'].update({
            'pre_nms_thresh': 0.001, 'min_score': 0.01, 'duration_thresh': 0.05,
        })
        old_model, _ = build_model(2, test_cfg=config['model']['test_cfg'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save({'config': config, 'model_state_dict': old_model.state_dict()}, path)
            loaded, _ = load_checkpoint(path, torch.device('cpu'))
        self.assertEqual(loaded.test_pre_nms_thresh, 0.0001)
        self.assertEqual(loaded.test_min_score, 0.0)
        self.assertEqual(loaded.test_duration_thresh, 0.0)

    def test_feature_chunk_reads_pairmotion_parquet_with_exclusive_upper_bound(self):
        from unittest.mock import patch

        from models.common.frame_model import BASE_PAIR_COLUMNS

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = np.arange(301, dtype=np.int64)
            base = pl.DataFrame({
                'FrameNum': frames,
                **{name: np.zeros(len(frames)) for name in BASE_PAIR_COLUMNS},
                'pose_feature': frames.astype(np.float64),
            })
            motion = pl.DataFrame({'FrameNum': frames, 'pm_test': frames.astype(np.float64)})
            base_path, motion_path = root / 'base.parquet', root / 'motion.parquet'
            base.write_parquet(base_path)
            motion.write_parquet(motion_path)
            with (patch('models.actionformer.predict.feature_table', return_value=base_path),
                  patch('models.actionformer.predict.pair_motion_cache_file', return_value=motion_path)):
                result = feature_chunk(root, 'v', 'pairmotion', {}, len(frames), 0, 256)
        self.assertEqual(result.height, 256)
        self.assertEqual(result['FrameNum'].to_list(), list(range(256)))
        self.assertNotIn(256, result['FrameNum'])
        self.assertEqual(result.columns, ['FrameNum', 'pose_feature', 'pm_test'])

    def test_native_confidence_is_used_by_localization_report(self):
        from evaluation.tal_report import model_detections
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'v.csv'
            pl.DataFrame({'VideoID': ['v'], 'StartFrame': [2], 'EndFrame': [8],
                          'Confidence': [0.37]}).write_csv(path)
            detections, scored = model_detections('actionformer_pairmotion', directory, ['v'])
        self.assertTrue(scored)
        self.assertEqual(detections, [('v', 2, 8, 0.37)])

    def test_runner_contains_resumable_actionformer_steps(self):
        runner = (Path(__file__).resolve().parents[1] / 'run.ps1').read_text()
        self.assertIn('[switch]$SkipActionFormer', runner)
        self.assertIn('models.actionformer.train', runner)
        self.assertIn('models.actionformer.predict', runner)
        self.assertIn('actionformer train pairmotion epochs=', runner)
        self.assertIn('$NameOf.actionformer.Values', runner)
        self.assertIn('[switch]$SkipPostprocessTuning', runner)
        self.assertIn('@tunedFrameFlags', runner)
        self.assertIn('[double]$ActionFormerMinSeconds', runner)
        self.assertIn('@actionFormerFlags', runner)
        mstcn_train = (Path(__file__).resolve().parents[1] / 'models/mstcn_dlc2action/train.py').read_text()
        self.assertIn("'--preset', 'aug'", mstcn_train)


if __name__ == '__main__':
    unittest.main()
