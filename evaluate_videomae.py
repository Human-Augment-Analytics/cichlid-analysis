"""Re-evaluate saved VideoMAE/MS-TCN predictions and verify checkpoint replay.

No training, threshold selection or test-based model selection is performed.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch

from circling_model import metrics
from dlc2action_segmentation import ROOT, IGNORE, make_windows, new_model, predicted_segments, span_labels
from temporal_metrics import temporal_metrics

METRICS = ('mof', 'edit_score', 'edit_score_with_background', 'segmental_f1_10',
           'segmental_f1_25', 'segmental_f1_50', 'precision', 'recall', 'f1',
           'average_precision', 'roc_auc')


def score(table):
    known = table.filter(pl.col('label') != IGNORE)
    return {**metrics(known['label'].to_numpy(), known['circling_score'].to_numpy()),
            **temporal_metrics(table)}


def evaluate(output_dir=ROOT/'outputs'/'videomae_mstcn', replay=True):
    output_dir = Path(output_dir)
    report = json.loads((output_dir/'metrics.json').read_text())
    predictions = pl.read_parquet(output_dir/'test_predictions.parquet').sort('video', 'FrameNum')
    manifest = pl.read_csv(output_dir/'span_manifest.csv')
    if report['threshold'] != 0.5:
        raise ValueError('This evaluator expects the original fixed 0.5 threshold')
    assert predictions['video'].unique().to_list() == [report['test_video']]
    assert not predictions.select(pl.struct('video', 'FrameNum').is_duplicated().any()).item()
    assert predictions['circling_score'].is_between(0, 1).all()
    assert predictions['predicted_circling'].equals(predictions['circling_score'] >= .5)
    rows = manifest.filter(pl.col('Video ID') == report['test_video']).to_dicts()
    np.testing.assert_array_equal(predictions['label'].to_numpy(),
                                  span_labels(predictions['FrameNum'].to_numpy(), rows))
    recalculated = score(predictions)
    for key, value in recalculated.items():
        if isinstance(value, (float, int)):
            np.testing.assert_allclose(value, report['test'][key], atol=1e-10)
        else:
            assert value == report['test'][key]
    segments = predicted_segments(predictions)
    saved_segments = pl.read_csv(output_dir/'predicted_segments.csv')
    assert segments.select(pl.exclude('mean_score')).equals(saved_segments.select(pl.exclude('mean_score')))
    np.testing.assert_allclose(segments['mean_score'], saved_segments['mean_score'])
    assert segments['frames'].sum() == predictions['predicted_circling'].sum()
    encoder_dir = Path(report['feature_source']['checkpoint'])
    train_clips = pl.read_csv(encoder_dir/'training_clips.csv')
    assert train_clips['video'].unique().to_list() == [report['train_video']]
    assert report['feature_source']['encoder']['encoder_weight_max_change'] > 0
    checks = {'unique_test_frames': predictions.height, 'evaluated_frames': recalculated['frames'],
              'ignored_frames': predictions.height-recalculated['frames'],
              'predicted_segments': segments.height, 'encoder_train_video': report['train_video'],
              'encoder_train_clips': train_clips.height, 'metrics_match_saved_report': True,
              'labels_match_saved_manifest': True, 'segment_export_verified': True}
    if replay:
        torch.set_num_threads(4)
        checkpoint = torch.load(output_dir/'model.pt', weights_only=True, map_location='cpu')
        model = new_model(len(checkpoint['features'])).eval()
        model.load_state_dict(checkpoint['state_dict'])
        preprocessing = joblib.load(output_dir/'preprocessing.joblib')
        folder = Path(report['feature_source']['embedding_directory'])
        max_error = 0.0
        for group in predictions.partition_by('sequence', maintain_order=True):
            with np.load(folder/f"{group['sequence'][0]}.npz") as saved:
                anchors, embeddings = saved['frames'], saved['embeddings']
            frames = group['FrameNum'].to_numpy()
            x = np.stack([np.interp(frames, anchors, embeddings[:, d])
                          for d in range(embeddings.shape[1])], axis=1).astype(np.float32)
            x = preprocessing.transform(x).astype(np.float32)
            windows, _, locations = make_windows([{'x': x, 'y': group['label'].to_numpy()}], report['window_frames'])
            chunks = []
            with torch.inference_mode():
                for batch in windows.split(16):
                    stages, _ = model(batch, [])
                    chunks.append(stages[-1].softmax(1)[:, 1].numpy())
            window_scores = np.concatenate(chunks)
            replayed = np.concatenate([s[:loc[2]] for s, loc in zip(window_scores, locations)])
            expected = group['circling_score'].to_numpy()
            max_error = max(max_error, float(np.max(np.abs(replayed-expected))))
            np.testing.assert_allclose(replayed, expected, atol=1e-5, rtol=1e-5)
        checks['checkpoint_replay_max_abs_error'] = max_error
        checks['checkpoint_replay_verified'] = True
    baseline_dir = ROOT/'outputs'/'dlc2action_test'
    baseline = pl.read_parquet(baseline_dir/'test_predictions.parquet').sort('video', 'FrameNum')
    columns = ['video', 'sequence', 'FrameNum', 'label']
    same_population = predictions.select(columns).equals(baseline.select(columns))
    # Compare only identical labeled frames when the baseline has been rerun with
    # different excerpt buffers. Preserve gaps for temporal scoring.
    common = predictions.select('video', 'FrameNum', 'label').join(
        baseline.select('video', 'FrameNum', pl.col('label').alias('baseline_label')),
        on=['video', 'FrameNum'], how='inner').filter(pl.col('label') == pl.col('baseline_label'))
    keys = common.select('video', 'FrameNum')
    visual_common = predictions.join(keys, on=['video', 'FrameNum'], how='semi')
    baseline_common = baseline.join(keys, on=['video', 'FrameNum'], how='semi')
    baseline_scores = score(baseline_common)
    visual_scores = score(visual_common)
    comparison = pl.DataFrame({'metric': METRICS,
        'units': ['percent']*6 + ['fraction']*5,
        'pose_pair_baseline': [baseline_scores[k] for k in METRICS],
        'videomae_mstcn': [visual_scores[k] for k in METRICS]})
    comparison.write_csv(output_dir/'evaluation_comparison.csv')
    checks['same_evaluation_frames_as_baseline'] = same_population
    checks['comparison_common_frames'] = common.height
    checks['comparison_evaluated_frames'] = visual_scores['frames']
    (output_dir/'evaluation_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    text = ['# VideoMAE → MS-TCN held-out evaluation', '',
        'Trained both stages using 0028_vid only; evaluated buffered merged-span excerpts from 0031_vid.',
        f"{checks['unique_test_frames']:,} predicted frames; {checks['evaluated_frames']:,} evaluated; "
        f"{checks['ignored_frames']:,} ignored; {checks['predicted_segments']} predicted positive segments.", '',
        f"Comparison below uses only {common.height:,} shared frames with identical labels ({visual_scores['frames']:,} evaluated). Full visual-run metrics remain in metrics.json and report.md.", '',
        '| Metric | Units | Pose/pair baseline | VideoMAE → MS-TCN |', '|---|---|---:|---:|']
    for row in comparison.iter_rows(named=True):
        text.append(f"| {row['metric']} | {row['units']} | {row['pose_pair_baseline']:.4f} | {row['videomae_mstcn']:.4f} |")
    text += ['', 'The visual model underperforms the pose/pair baseline in this run, with many missed '
             'circling frames and fragmented/poorly aligned segments. This is not a controlled feature ablation: '
             'the saved runs use different training schedules.', '',
             'Possible contributors (not established causes): small subjects in full-scene 224×224 inputs, '
             'only 256 clips for encoder adaptation, domain shift between the two videos, and sparse embedding '
             'anchors interpolated to frames. Test results were not used to tune the model or threshold.', '',
             report['temporal_metric_policy'], '', report['limitations'], '',
             'Validation: saved labels, metrics, segment export and training-video membership checked. '
             + ('MS-TCN checkpoint predictions reproduced from cached visual embeddings.' if replay else 'Checkpoint replay skipped.')]
    (output_dir/'evaluation.md').write_text('\n'.join(text)+'\n')
    print(comparison)
    print(json.dumps(checks, indent=2))
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs'/'videomae_mstcn')
    parser.add_argument('--skip-replay', action='store_true')
    args = parser.parse_args()
    evaluate(args.output_dir, replay=not args.skip_replay)
