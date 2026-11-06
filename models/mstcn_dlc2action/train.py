import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'scripts')]

from dlc_yolo_converter import dlc2action_bridge as br

from models.common.config import (
    DATA_DIR,
    DINOV2_DIR,
    FINALIZED_ANNOTATIONS,
    OUTPUTS,
    VIDEOMAE_DIR,
    model_name,
    mstcn_workdir,
    read_annotations,
    video_frames,
)
from models.common.report import protocol, segment_counts, write_report
from models.common.segments import (
    LEN_SEGMENT,
    OVERLAP,
    SEED,
    SPLIT,
    STRIDE,
    frame_scores,
    rows_to_frames,
    sample_segments,
    shared_segments,
    split_segments,
    test_frames,
    write_split,
)

NAMES = {'pairemb': 'mstcn_dinov2', 'pairvmae': 'mstcn_videomae', 'pair': 'mstcn_pose_pair', 'pose': 'mstcn_pose'}
NAME = model_name(NAMES['pairemb'])
PROJECT = 'circling_repo'
CANVAS = '1296x972'
FEATURES = {'pose': [], 'pair': ['--pair-features'], 'pairemb': ['--feature-file', 'pairemb'],
            'pairvmae': ['--feature-file', 'pairvmae']}
EMBEDDINGS = {'pairemb': DINOV2_DIR, 'pairvmae': VIDEOMAE_DIR}
DESCRIPTIONS = {'pose': 'male and partner pose', 'pair': 'pose + 17 pair features',
                'pairemb': 'pose + 17 pair features + 32 PCs of DINOv2 ViT-S/14 fish-masked male crops',
                'pairvmae': 'pose + 17 pair features + 32 PCs of VideoMAE clip embeddings of the whole scene'}
WORKDIR = mstcn_workdir()
OUTPUT_DIR = OUTPUTS / NAME

def run_cli(workdir, *args):
    import dlc2action_circling as cli
    code = cli.main(['--workdir', str(workdir), *args])
    if code:
        raise SystemExit(code)

def need(*paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f'missing input file(s): {missing} (set CIRCLING_DATA_DIR)')

def prepare_data(workdir, data_dir, features, videos=None):
    workdir, data_dir = Path(workdir), Path(data_dir)
    videos = sorted(videos or video_frames())
    data = workdir / 'data'
    for p in (data, workdir / 'splits', workdir / 'projects'):
        p.mkdir(parents=True, exist_ok=True)
    meta = br.load_meta(data) if (data / br.META_FILENAME).exists() else {}
    todo = [v for v in videos if v not in meta]
    if todo:
        keypoints = [data_dir / f'{v}.parquet' for v in todo]
        need(*keypoints)
        print(f'Converting pose to H5 (convert-data): {todo}', flush=True)
        run_cli(workdir, 'convert-data', '--stride', str(STRIDE), '--keypoints', *map(str, keypoints))
    if features != 'pose':
        todo = [v for v in videos if not (data / f'{v}{br.PAIR_FEATURE_SUFFIX}').exists()]
        if todo:
            pairs = [data_dir / f'{v}_pairs.parquet' for v in todo]
            need(*pairs)
            print(f'Building pair features (convert-pairs): {todo}', flush=True)
            run_cli(workdir, 'convert-pairs', '--pairs', *map(str, pairs))
    if features in EMBEDDINGS and not all((data / f'{v}_{features}.pickle').exists() for v in videos):
        import embed_probe
        movies = [data_dir / f'{v}.mp4' for v in videos]
        need(*movies)
        emb_dir = EMBEDDINGS[features]
        sources = [x for v, m in zip(videos, movies) for x in ('--video', f'{v}={m}')]
        if features == 'pairemb':
            print('Extracting DINOv2 embeddings for every row (long; resumable)', flush=True)
            embed_probe.main(['extract-full', '--data-dir', str(data), '--out', str(emb_dir), *sources])
        else:
            import videomae_extract
            print('Extracting VideoMAE clip embeddings (long; resumable)', flush=True)
            videomae_extract.main(['--data-dir', str(data), '--out', str(emb_dir), *sources])
        embed_probe.main(['features', '--data-dir', str(data), '--emb-dir', str(emb_dir), '--name', features])

def train(workdir, output_dir, features='pairemb', num_epochs=100, name=None,
          annotations=FINALIZED_ANNOTATIONS, data_dir=DATA_DIR, fresh=False):
    name = name or model_name(NAMES[features])
    workdir, output_dir = Path(workdir), Path(output_dir or OUTPUTS / name)
    spans = read_annotations(annotations)
    prepare_data(workdir, data_dir, features, sorted(spans))
    segments, meta = shared_segments(workdir, spans, output_dir)

    os.environ['MSTCN_FINALIZED_ANNOTATIONS'] = str(Path(annotations).resolve())
    signature = run_signature(workdir, name, features, num_epochs, sorted(spans))
    progress_path = output_dir / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() and not fresh else {}
    if progress.get('signature') != signature or not episode_exists(workdir, name):
        progress = {'signature': signature}

    def done(stage):
        progress[stage] = True
        progress_path.write_text(json.dumps(progress, indent=2) + '\n')

    if progress.get('trained'):
        print(f'{name}: already trained with these settings, skipping training (--fresh to redo)', flush=True)
    else:
        run_cli(workdir, 'train', '--project-name', PROJECT, '--split', SPLIT, '--episode', name,
                '--len-segment', str(LEN_SEGMENT), '--overlap', str(OVERLAP), '--canvas', CANVAS,
                '--num-epochs', str(num_epochs), '--overwrite', *FEATURES[features])
        done('trained')
    prediction = workdir / 'projects' / PROJECT / 'results' / 'predictions' / f'{name}_pred'
    if progress.get('predicted') and all(any(prediction.glob(f'{v}_*prediction.pickle')) for v in spans):
        print(f'{name}: predictions already made, skipping prediction', flush=True)
    else:
        run_cli(workdir, 'predict', '--project-name', PROJECT, '--episode', name,
                '--prediction-name', f'{name}_pred', '--overwrite')
        done('predicted')
    scores = cache_scores(workdir, output_dir / 'scores', name)

    test = test_frames(segments, meta, scores, spans).select('video', 'FrameNum', 'circling', 'circling_score')
    report = protocol(annotations, 'DLC2Action 1.0 MS-TCN3', episode=name, features=features,
                      validation='DLC2Action validation subset = test segments, used for logging only '
                                 '(no early stopping or model selection).',
                      len_segment=LEN_SEGMENT, overlap=OVERLAP, stride=STRIDE, num_epochs=num_epochs,
                      seed=SEED, videos=sorted(spans), segments=segment_counts(segments),
                      train_positive_rows=int(segments.filter(segments['split'] == 'train')['positive_rows'].sum()))
    notes = [report['label_assumption'], report['sampling'], report['split'], report['validation'],
             f'Input features: {DESCRIPTIONS[features]}. Segment length {LEN_SEGMENT} rows, {num_epochs} epochs.']
    return write_report(output_dir, f'Circling MS-TCN (DLC2Action), {features}', 'MS-TCN', test, report, notes)

def file_digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

def run_signature(workdir, name, features, num_epochs, videos):
    data = Path(workdir) / 'data'
    return {'episode': name, 'features': features, 'num_epochs': num_epochs, 'videos': videos,
            'len_segment': LEN_SEGMENT, 'overlap': OVERLAP, 'canvas': CANVAS,
            'labels': file_digest(Path(workdir) / 'training_spans.csv'),
            'split': file_digest(Path(workdir) / 'splits' / f'{SPLIT}.txt'),
            'feature_file': file_digest(data / f'{videos[0]}_{features}.pickle')
            if features in EMBEDDINGS else None}

def episode_exists(workdir, name):
    import dlc2action_circling as cli
    paths = cli.Paths(workdir)
    if not cli.project_is_complete(paths, PROJECT):
        return False
    try:
        return name in cli._project(paths, PROJECT)._episodes().data.index
    except Exception as error:
        print(f'could not read the episodes of {PROJECT}: {error!r}', flush=True)
        return False

def cache_scores(workdir, scores_dir, name=NAME):
    data = Path(workdir) / 'data'
    meta = br.load_meta(data)
    prediction = Path(workdir) / 'projects' / PROJECT / 'results' / 'predictions' / f'{name}_pred'
    probs = br.load_prediction_probs(prediction)
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for video, count in video_frames().items():
        if video not in probs:
            raise SystemExit(f'no prediction for {video} in {prediction}')
        out[video] = frame_scores(probs[video], meta[video], count)
        out[video].write_parquet(scores_dir / f'{video}_scores.parquet')
        files = sorted(prediction.glob(f'{video}_*prediction.pickle'))
        (scores_dir / f'{video}_scores.json').write_text(json.dumps({
            'frames': count, 'episode': name, 'stride': int(meta[video].get('stride', 1)),
            'prediction': {p.name: p.stat().st_size for p in files}}, indent=2) + '\n')
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workdir', type=Path, default=WORKDIR, help='DLC2Action data and project folder')
    parser.add_argument('--output-dir', type=Path, default=None, help='Default outputs/<name>')
    parser.add_argument('--features', choices=sorted(FEATURES), default='pairemb',
                        help=', '.join(f'{k} -> {v}' for k, v in NAMES.items()))
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--prepare-only', action='store_true',
                        help='convert the data and build the feature files, then stop')
    parser.add_argument('--fresh', action='store_true',
                        help='retrain and predict again even when this episode is already done')
    args = parser.parse_args()
    if args.prepare_only:
        prepare_data(args.workdir, DATA_DIR, args.features)
    else:
        train(args.workdir, args.output_dir, args.features, args.num_epochs, fresh=args.fresh)
