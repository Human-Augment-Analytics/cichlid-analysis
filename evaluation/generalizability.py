"""Leave-one-video-out generalizability folds.

One annotated video is held out with a fixed seed and never touched. Each remaining video
is the test video of one fold: every model trains on the other videos only, then predicts
the unseen test video. `generalizability_report.py` averages the fold scores.

    python -m evaluation.generalizability --list                 # holdout and folds
    python -m evaluation.generalizability                        # every fold, every model
    python -m evaluation.generalizability --fold-index 0 --models xgboost_turns
    python -m evaluation.generalizability --data-dir /scratch/cichlid/data --alias vid6-v18_d18=0018_vid

Each fold runs the normal train/predict CLIs in a subprocess with its own output root
(CIRCLING_OUTPUTS=outputs/generalizability/folds/<test video>), so folds are independent
and can run as a Slurm array (one --fold-index per task).
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
GENERALIZABILITY_DIR = ROOT / 'outputs' / 'generalizability'
FOLDS_DIR = GENERALIZABILITY_DIR / 'folds'
TRAINING_SPANS = ROOT / 'outputs' / 'finalized_spans.csv'
# Annotation video id -> data file stem (download_data.py names), from the annotation file each id
# is in: circling_annotations_vid0002.csv holds vid7-v02_d00, vid0007-day5 holds vid5-v07_d05, etc.
DEFAULT_ALIASES = {'0007_vid': '0007-day6_vid', 'vid5-v07_d05': '0007-day5_vid', 'vid3-v13_d13': '0013_vid',
                   'vid7-v02_d00': '0002_vid', 'vid6-v18_d18': '0018_vid'}

# name -> (features to prepare, train command, commands that turn the test video's inputs into
# outputs/predictions/<name>/<video>.csv). Pose-only models; embedding variants need
# DINOv2/VideoMAE features and MS-TCN/SARTAB only score videos inside their training run.
FRAME = {'xgboost_pose_pair': ('models.xgboost_frame', 'pair'),
         'xgboost_turns': ('models.xgboost_frame', 'pairturns'),
         'xgboost_pairmotion': ('models.xgboost_frame', 'pairmotion'),
         'lrcv_pose_pair': ('models.logistic_regression_cv', 'pair'),
         'lrcv_turns': ('models.logistic_regression_cv', 'pairturns'),
         'lrcv_pairmotion': ('models.logistic_regression_cv', 'pairmotion')}
MODELS = {
    **{name: {'prepare': ['pair'] + (['pairmotion'] if feature == 'pairmotion' else []),
              'train': ['-m', f'{module}.train', '--features', feature],
              'predict': [['-m', 'models.common.score', name],
                          ['-m', f'{module}.predict', '--features', feature]]}
       for name, (module, feature) in FRAME.items()},
    # The centered-window model of models/xgboost (121 frames of 5 closest-pair signals) with its own
    # 70/30 frame split, 0.5 threshold and 30-frame vote; it needs only the pose and pair tables.
    'xgboost': {
        'prepare': [],
        'train': ['-m', 'models.xgboost.train', '--skip-full-video'],
        'predict': [['-m', 'models.xgboost.predict']]},
    'actionformer_pairmotion': {
        'prepare': ['pair', 'pairmotion'],
        'train': ['-m', 'models.actionformer.train', '--features', 'pairmotion'],
        'predict': [['-m', 'models.actionformer.predict', '--features', 'pairmotion']]},
}


def annotated_videos():
    from evaluation.evaluate import read_ground_truth

    return sorted(read_ground_truth()['video_id'].unique().to_list())


def split_videos(videos=None, seed=SEED):
    """(holdout video, cross-validation videos): the holdout is a seeded choice."""
    videos = sorted(videos or annotated_videos())
    holdout = random.Random(seed).choice(videos)
    return holdout, [v for v in videos if v != holdout]


def fold_dir(test_video):
    return FOLDS_DIR / test_video


def link_data(videos, data_dir, aliases, target=GENERALIZABILITY_DIR / 'data'):
    """Symlink each video's inputs under its annotation id so every pipeline finds them."""
    data_dir, target = Path(data_dir).resolve(), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    missing = []
    for video in videos:
        stem = aliases.get(video, video)
        for suffix, required in (('.parquet', True), ('_pairs.parquet', True), ('.mp4', False)):
            source, link = data_dir / f'{stem}{suffix}', target / f'{video}{suffix}'
            if not source.exists():
                missing += [str(source)] if required else []
                continue
            if suffix.endswith('.parquet') and day_labels(source) != {video}:
                relabel(source, link, video)
                continue
            if link.is_symlink() or link.exists():
                if link.resolve() == source:
                    continue
                link.unlink(missing_ok=True)
            try:
                link.symlink_to(source)
            except FileExistsError:  # another array task linked it first
                if link.resolve() != source:
                    raise
    if missing:
        raise SystemExit('Missing pose inputs (pass --data-dir/--alias):\n  ' + '\n  '.join(missing))
    return target


def day_labels(path):
    return set(pl.scan_parquet(path).select(pl.col('day_label').unique()).collect()['day_label'].to_list())


def relabel(source, destination, video):
    """Copy a pose/pairs table with day_label set to the annotation video id.

    The pair-motion features require day_label == video id; downloaded tables carry their file
    stem (both 0007 recordings say 0007_vid). Schema metadata is kept for the pose contract.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    stat = source.stat()
    signature = {'source': str(source), 'size': stat.st_size, 'mtime': stat.st_mtime, 'day_label': video}
    info = destination.with_name(destination.name + '.json')
    if destination.is_file() and not destination.is_symlink() and info.exists() \
            and json.loads(info.read_text()) == signature:
        return
    reader = pq.ParquetFile(source)
    schema = reader.schema_arrow
    index = schema.get_field_index('day_label')
    partial = destination.with_name(f'.{destination.name}.{os.getpid()}.part')
    with pq.ParquetWriter(partial, schema) as writer:
        for batch in reader.iter_batches(batch_size=500_000):
            label = pa.array([video] * batch.num_rows).cast(schema.field(index).type)
            writer.write_batch(batch.set_column(index, schema.field(index), label))
    if destination.is_symlink():
        destination.unlink(missing_ok=True)
    os.replace(partial, destination)  # atomic, so concurrent array tasks never see a partial file
    info.write_text(json.dumps(signature) + '\n')
    print(f'Relabeled {source.name} -> {destination.name} (day_label {video})', flush=True)


def run(args, env, log):
    started = time.time()
    command = [sys.executable, *args]
    print(f'> {" ".join(args)}  [CIRCLING_VIDEOS={env["CIRCLING_VIDEOS"]}]', flush=True)
    with open(log, 'a') as out:
        out.write(f'\n> {" ".join(args)}  [CIRCLING_VIDEOS={env["CIRCLING_VIDEOS"]}]\n')
        out.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{" ".join(args)} failed ({result.returncode}); see {log}')
    return time.time() - started


def run_fold(test_video, cv_videos, models, data_dir, train_videos=None):
    """Train every model on the fold's training videos and predict its test video."""
    from models.common.config import write_finalized_spans

    train_videos = sorted(train_videos or [v for v in cv_videos if v != test_video])
    folder = fold_dir(test_video)
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / 'run.log'
    base = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONUNBUFFERED': '1',
            'CIRCLING_OUTPUTS': str(folder), 'CIRCLING_DATA_DIR': str(data_dir),
            'CIRCLING_GROUND_TRUTH': write_finalized_spans(TRAINING_SPANS),
            'CIRCLING_LABELS': 'ground_truth', 'CIRCLING_HARD_NEGATIVES': '0'}
    train_env = {**base, 'CIRCLING_VIDEOS': ','.join(train_videos)}
    test_env = {**base, 'CIRCLING_VIDEOS': test_video}
    status_path = folder / 'fold.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.update({'test_video': test_video, 'train_videos': train_videos, 'seed': SEED})
    status.setdefault('models', {})

    def save():
        status_path.write_text(json.dumps(status, indent=2) + '\n')

    prepared = set(status.get('prepared', []))
    for name in models:
        spec = MODELS[name]
        for feature in spec['prepare']:
            if feature not in prepared:
                for env in (train_env, test_env):
                    run(['-m', 'models.mstcn_dlc2action.train', '--prepare-only', '--features', feature], env, log)
                prepared.add(feature)
                status['prepared'] = sorted(prepared)
                save()
        entry = status['models'].setdefault(name, {})
        entry['train_seconds'] = round(run(spec['train'], train_env, log), 1)
        save()
        entry['predict_seconds'] = round(sum(run(step, test_env, log) for step in spec['predict']), 1)
        entry['predictions'] = str(Path('predictions') / name / f'{test_video}.csv')
        save()
        print(f'{test_video} · {name}: trained in {entry["train_seconds"]:.0f} s', flush=True)
    return status


def fold_results(folds_dir=FOLDS_DIR):
    """Evaluate each finished fold's predictions on its test video. One row per model and fold."""
    from evaluation.evaluate import evaluate, read_ground_truth, read_ground_truth_labels, read_predictions

    gt, labels = read_ground_truth(), read_ground_truth_labels()
    rows = []
    for status_path in sorted(Path(folds_dir).glob('*/fold.json')):
        status = json.loads(status_path.read_text())
        video, folder = status['test_video'], status_path.parent
        counts = json.loads((folder / 'video_frames.json').read_text()) \
            if (folder / 'video_frames.json').exists() else {}
        from models.logistic_regression.train import VIDEO_FRAMES
        frames = counts.get(video) or VIDEO_FRAMES.get(video)
        for name, entry in status['models'].items():
            path = folder / entry.get('predictions', f'predictions/{name}/{video}.csv')
            if not path.exists() or frames is None:
                continue
            predicted = read_predictions(path.parent, labels=True, scores=True).filter(pl.col('video_id') == video)
            results, _ = evaluate(gt.filter(pl.col('video_id') == video), predicted,
                                  num_frames={video: frames}, gt_labels=labels.filter(pl.col('video_id') == video),
                                  videos=[video])
            rows.append({'model': name, 'test_video': video, 'train_videos': len(status['train_videos']),
                         'predicted_spans': predicted.height, 'train_seconds': entry.get('train_seconds'),
                         'predict_seconds': entry.get('predict_seconds'),
                         'wall_seconds': (entry.get('train_seconds') or 0) + (entry.get('predict_seconds') or 0),
                         **results.filter(pl.col('video_id') == video).drop('video_id').row(0, named=True)})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list', action='store_true', help='print the holdout and fold test videos, then exit')
    parser.add_argument('--fold', action='append', default=[], help='test video(s) to run (default: all folds)')
    parser.add_argument('--fold-index', type=int, action='append', default=[],
                        help='fold number(s) from --list, e.g. $SLURM_ARRAY_TASK_ID')
    parser.add_argument('--models', nargs='+', choices=sorted(MODELS), default=sorted(MODELS))
    parser.add_argument('--data-dir', type=Path, default=None, help='folder with <video>.parquet and _pairs.parquet')
    parser.add_argument('--alias', action='append', default=[], metavar='VIDEO=STEM',
                        help='data file stem for an annotation video id (repeatable)')
    parser.add_argument('--train-videos', nargs='+', default=None,
                        help='debugging: train on these videos instead of the other fold videos')
    args = parser.parse_args()

    holdout, cv_videos = split_videos()
    if args.list:
        print(f'Seed {SEED}: holdout {holdout} (never trained on or evaluated here)')
        for index, video in enumerate(cv_videos):
            print(f'  fold {index}: test {video}, train {len(cv_videos) - 1} others')
        return
    tests = [*args.fold, *[cv_videos[i] for i in args.fold_index]] or cv_videos
    if holdout in tests:
        raise SystemExit(f'{holdout} is the seeded holdout video; it is not a fold')
    from models.common.config import DATA_DIR
    aliases = {**DEFAULT_ALIASES, **dict(item.split('=', 1) for item in args.alias)}
    needed = sorted(set(cv_videos if args.train_videos is None else [*args.train_videos, *tests]))
    data = link_data(needed, args.data_dir or DATA_DIR, aliases)
    for test in tests:
        run_fold(test, cv_videos, args.models, data, args.train_videos)


if __name__ == '__main__':
    main()
