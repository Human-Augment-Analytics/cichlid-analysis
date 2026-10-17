
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'scripts')]

from dlc_yolo_converter import dlc2action_bridge as br  # noqa: E402

from models.logistic_regression.train import (  # noqa: E402
    ROOT,
    THRESHOLD,
    VIDEO_FRAMES,
    frame_labels,
    metrics,
)

# ---------------------------------------------------------------- set before running
# Finalized ground truth for the training labels (CSV with VideoID, StartFrame,
# EndFrame), e.g. ROOT / 'ground_truth_annotations' / 'finalized_annotations' / 'merged_spans.csv'
# or a raw Windows path: r'D:\path\to\merged_spans.csv'
FINALIZED_ANNOTATIONS = 'D:\\rGradSchool\\new_chiclid_analysis\\cichlid-analysis\\ground_truth_annotations\\finalized_annotations\\merged_spans.csv'
# Folder with <video>.parquet, <video>_pairs.parquet and <video>.mp4 (videos only
# needed for the DINOv2 embeddings)
DATA_DIR = 'D:\\rGradSchool\\new_chiclid_analysis\\data'
# ----------------------------------------------------------------------------------

NAME = 'mstcn_dinov2'
PROJECT = 'circling_repo'
SPLIT = 'repo_split'
LEN_SEGMENT = 128
OVERLAP = 0
SEED = 42
CANVAS = '1296x972'
STRIDE = 2          
FEATURES = {'pose': [], 'pair': ['--pair-features'], 'pairemb': ['--feature-file', 'pairemb']}
WORKDIR = ROOT / 'outputs' / 'mstcn_dlc2action'
OUTPUT_DIR = ROOT / 'outputs' / NAME



def read_annotations(path):

    if not str(path).strip():
        raise SystemExit('Set FINALIZED_ANNOTATIONS at the top of models/mstcn_dlc2action/train.py')
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f'FINALIZED_ANNOTATIONS: {path} not found')
    rows = pl.read_csv(path, infer_schema=False)
    columns = {re.sub(r'[^a-z]', '', c.lower()): c for c in rows.columns}
    if not {'videoid', 'startframe', 'endframe'} <= columns.keys():
        raise SystemExit(f'{path}: needs VideoID, StartFrame and EndFrame columns')
    spans = {video: [] for video in VIDEO_FRAMES}
    for video, start, end in rows.select(columns['videoid'], columns['startframe'],
                                         columns['endframe']).iter_rows():
        start, end = int(start), int(end)
        if video not in spans:
            print(f'Ignoring {video}: not one of {sorted(VIDEO_FRAMES)}', flush=True)
            continue
        if start < 0 or end < start or end >= VIDEO_FRAMES[video]:
            raise SystemExit(f'{path}: invalid interval {video} {start}-{end}')
        spans[video].append((start, end))
    for video, intervals in spans.items():
        if not intervals:
            raise SystemExit(f'{path}: no circling intervals for {video}')
        spans[video] = sorted(intervals)
    return spans


# --------------------------------------------------------------------------- sampling and split

def sample_segments(positives, n_rows, len_segment=LEN_SEGMENT, seed=SEED):

    rng = np.random.default_rng(seed)
    tables = []
    for video in VIDEO_FRAMES:
        n = n_rows[video]
        positive = np.zeros(n, dtype=bool)
        for start, end in positives[video]:
            positive[start:end] = True
        spans = np.array(br.segment_spans(n, len_segment, OVERLAP), dtype=np.int64)
        counts = np.array([positive[s:e].sum() for s, e in spans], dtype=np.int64)
        with_circling = np.flatnonzero(counts > 0)
        without = np.flatnonzero(counts == 0)
        if len(without) < len(with_circling):
            raise ValueError(f'{video}: not enough segments without circling to sample')
        picked = np.sort(np.concatenate([
            with_circling, rng.choice(without, size=len(with_circling), replace=False)]))
        tables.append(pl.DataFrame({
            'video': video, 'start_row': spans[picked, 0], 'end_row': spans[picked, 1],
            'positive_rows': counts[picked], 'circling': counts[picked] > 0}))
    return pl.concat(tables)


def split_segments(segments, seed=SEED):
    ids = np.arange(segments.height)
    _, test_ids = train_test_split(ids, test_size=0.2, random_state=seed,
                                   stratify=segments['circling'].to_numpy())
    split = np.full(segments.height, 'train', dtype='<U5')
    split[test_ids] = 'test'
    return segments.with_columns(pl.Series('split', split))


def write_split(segments, path, n_rows):
    ids = {k: [br.segment_id(v, s, e) for v, s, e in segments.filter(pl.col('split') == k)
               .select('video', 'start_row', 'end_row').iter_rows()] for k in ('train', 'test')}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for header, key in (('Training videos:', 'train'), ('Validation videos:', 'test'),
                            ('Test videos:', 'test')):
            f.write(header + '\n')
            f.writelines(sid + '\n' for sid in ids[key])
    Path(str(path) + '.json').write_text(json.dumps({
        'scheme': 'repo', 'len_segment': LEN_SEGMENT, 'overlap': OVERLAP, 'seed': SEED,
        'val_is_test': True, 'n_frames': n_rows,
        'sampling': 'every segment with circling plus an equal number without, per video',
        'split': 'stratified random 80/20 train/test, random_state=42'}, indent=2), encoding='utf-8')



def rows_to_frames(start_row, end_row, meta, frame_count):
    stride, first = int(meta.get('stride', 1)), int(meta.get('frame_min', 0))
    return np.arange(start_row * stride + first, min(end_row * stride + first, frame_count))


def frame_scores(probs, meta, frame_count):
    """Per-row probabilities -> one score per original frame 0..frame_count-1."""
    stride, first = int(meta.get('stride', 1)), int(meta.get('frame_min', 0))
    frames = np.arange(frame_count, dtype=np.int64)
    rows = (frames - first) // stride
    valid = (frames >= first) & (rows < len(probs))
    scores = np.zeros(frame_count)
    scores[valid] = probs[rows[valid]]
    return pl.DataFrame({'FrameNum': frames, 'circling_score': scores})



def run_cli(workdir, *args):
    import dlc2action_circling as cli
    code = cli.main(['--workdir', str(workdir), *args])
    if code:
        raise SystemExit(code)


def prepare_data(workdir, data_dir, features):
    data = Path(workdir) / 'data'
    data_dir = Path(data_dir)
    videos = list(VIDEO_FRAMES)

    def need(*paths):
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise SystemExit(f'missing input file(s): {[str(p) for p in missing]} (set DATA_DIR)')

    meta_path = data / br.META_FILENAME
    meta = br.load_meta(data) if meta_path.exists() else {}
    if set(videos) - set(meta):
        keypoints = [data_dir / f'{v}.parquet' for v in videos]
        need(*keypoints)
        print('Converting pose to H5 (convert-data)', flush=True)
        run_cli(workdir, 'convert-data', '--stride', str(STRIDE), '--keypoints', *map(str, keypoints))
    if features in ('pair', 'pairemb') and not all(
            (data / f'{v}{br.PAIR_FEATURE_SUFFIX}').exists() for v in videos):
        pairs = [data_dir / f'{v}_pairs.parquet' for v in videos]
        need(*pairs)
        print('Building pair features (convert-pairs)', flush=True)
        run_cli(workdir, 'convert-pairs', '--pairs', *map(str, pairs))
    if features == 'pairemb' and not all((data / f'{v}_pairemb.pickle').exists() for v in videos):
        import embed_probe
        movies = [data_dir / f'{v}.mp4' for v in videos]
        need(*movies)
        emb_dir = Path(workdir) / 'emb_fish_full'
        print('Extracting DINOv2 embeddings for every row (long; resumable)', flush=True)
        embed_probe.main(['extract-full', '--data-dir', str(data), '--out', str(emb_dir),
                          *[x for v, m in zip(videos, movies) for x in ('--video', f'{v}={m}')]])
        embed_probe.main(['features', '--data-dir', str(data), '--emb-dir', str(emb_dir)])


def train(workdir, output_dir, features='pairemb', num_epochs=100, name=NAME,
          annotations=FINALIZED_ANNOTATIONS, data_dir=DATA_DIR):
    workdir, output_dir = Path(workdir), Path(output_dir)
    data = workdir / 'data'
    spans = read_annotations(annotations)
    for p in (data, workdir / 'splits', workdir / 'projects'):
        p.mkdir(parents=True, exist_ok=True)
    prepare_data(workdir, data_dir, features)
    meta = br.load_meta(data)
    output_dir.mkdir(parents=True, exist_ok=True)

    # labels: every row of the finalized annotations
    spans_csv = workdir / 'training_spans.csv'
    pl.DataFrame([{'VideoID': v, 'StartFrame': s, 'EndFrame': e}
                  for v, intervals in spans.items() for s, e in intervals]
                 ).write_csv(spans_csv)
    run_cli(workdir, 'convert-ann', '--ground-truth', str(spans_csv))

    # sampling and split
    n_rows = {v: int(meta[v]['n_frames']) for v in meta}
    positives = br.read_positive_intervals(data, list(VIDEO_FRAMES))
    segments = split_segments(sample_segments(positives, n_rows))
    write_split(segments, workdir / 'splits' / f'{SPLIT}.txt', n_rows)
    segments.write_parquet(output_dir / 'sampled_segments.parquet')
    counts = segments.group_by('split', 'circling').len().sort('split', 'circling')
    print(f'Sampled segments:\n{counts}', flush=True)

    # one model on the training segments, then every frame of both videos
    os.environ['MSTCN_FINALIZED_ANNOTATIONS'] = str(Path(annotations).resolve())   # for the preflight
    run_cli(workdir, 'train', '--project-name', PROJECT, '--split', SPLIT, '--episode', name,
            '--len-segment', str(LEN_SEGMENT), '--overlap', str(OVERLAP), '--canvas', CANVAS,
            '--num-epochs', str(num_epochs), '--overwrite', *FEATURES[features])
    run_cli(workdir, 'predict', '--project-name', PROJECT, '--episode', name,
            '--prediction-name', f'{name}_pred', '--overwrite')
    scores = cache_scores(workdir, output_dir / 'scores', name)

    # test metrics, as the logistic regression reports them
    test = test_frames(segments, meta, scores, spans)
    y, score = test['circling'].to_numpy(), test['circling_score'].to_numpy()
    result = metrics(y, score)
    write_report(output_dir, segments, result, y, score, features, num_epochs, name, annotations)
    test.write_parquet(output_dir / 'test_predictions.parquet')
    print(json.dumps(result, indent=2), flush=True)


def test_frames(segments, meta, scores, spans):
    tables = []
    for video, start, end in (segments.filter(pl.col('split') == 'test')
                              .select('video', 'start_row', 'end_row').iter_rows()):
        frames = rows_to_frames(start, end, meta[video], VIDEO_FRAMES[video])
        score = scores[video]['circling_score'].to_numpy()[frames]
        tables.append(pl.DataFrame({
            'video': video, 'FrameNum': frames,
            'circling': frame_labels(frames, spans[video]),
            'circling_score': score, 'predicted_circling': score >= THRESHOLD}))
    return pl.concat(tables)


def cache_scores(workdir, scores_dir, name=NAME):
    data = Path(workdir) / 'data'
    meta = br.load_meta(data)
    prediction = Path(workdir) / 'projects' / PROJECT / 'results' / 'predictions' / f'{name}_pred'
    probs = br.load_prediction_probs(prediction)
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for video, count in VIDEO_FRAMES.items():
        if video not in probs:
            raise SystemExit(f'no prediction for {video} in {prediction}')
        out[video] = frame_scores(probs[video], meta[video], count)
        out[video].write_parquet(scores_dir / f'{video}_scores.parquet')
        files = sorted(prediction.glob(f'{video}_*prediction.pickle'))
        (scores_dir / f'{video}_scores.json').write_text(json.dumps({
            'frames': count, 'episode': name, 'stride': int(meta[video].get('stride', 1)),
            'prediction': {p.name: p.stat().st_size for p in files}}, indent=2) + '\n')
    return out


def write_report(output_dir, segments, result, y, score, features, num_epochs, name, annotations):
    from matplotlib.figure import Figure
    from sklearn.metrics import precision_recall_curve

    train_seg = segments.filter(pl.col('split') == 'train')
    report = {
        'label_assumption': f'All frames outside the intervals in {Path(annotations).name} are non-circling.',
        'annotations': str(Path(annotations).resolve()),
        'sampling': 'Every 128-row segment containing circling plus an equal number of segments '
                    'without circling, sampled without replacement within each source video, seed 42.',
        'split': 'Stratified random 80/20 train/test split of the sampled segments after sampling, '
                 'random_state=42. Segments do not overlap.',
        'validation': 'DLC2Action validation subset = test segments, used for logging only '
                      '(no early stopping or model selection).',
        'limitation': 'Neighboring segments and spans can occur in both splits; this is not an '
                      'independent-event or new-video evaluation.',
        'score_warning': 'Scores describe the balanced sampled dataset, not calibrated probabilities '
                         'at full-video prevalence.',
        'threshold': THRESHOLD, 'model': 'DLC2Action 1.0 MS-TCN3', 'episode': name,
        'features': features, 'len_segment': LEN_SEGMENT, 'overlap': OVERLAP,
        'num_epochs': num_epochs,
        'segments': {s: {'with_circling': int(g['circling'].sum()), 'without': int((~g['circling']).sum())}
                     for s, g in ((s, segments.filter(pl.col('split') == s)) for s in ('train', 'test'))},
        'train_positive_rows': int(train_seg['positive_rows'].sum()),
        'test': result,
        'all_negative_baseline': {'accuracy': float((y == 0).mean()), 'recall': 0.0, 'f1': 0.0},
        'constant_score_average_precision': float(y.mean()),
    }
    (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    fig = Figure(figsize=(8, 6), layout='constrained')
    axis = fig.subplots()
    precision, recall, _ = precision_recall_curve(y, score)
    axis.plot(recall, precision, label=f'MS-TCN: AP={result["average_precision"]:.3f}')
    axis.axhline(y.mean(), linestyle=':', label='Constant-score baseline')
    axis.set(xlabel='Recall', ylabel='Precision', title='Circling MS-TCN: sampled-segment test set',
             xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    fig.savefig(output_dir / 'precision_recall.png', dpi=160)
    lines = ['# Circling MS-TCN (DLC2Action)', '', report['label_assumption'], '',
             report['sampling'], '', report['split'], '', report['validation'], '',
             f"Input features: {features}. Segment length {LEN_SEGMENT} rows, {num_epochs} epochs.", '',
             f'Test: {len(y):,} frames ({int(y.sum()):,} circling).', '',
             '| Test metric | Value |', '|---|---:|']
    for key in ('average_precision', 'roc_auc', 'precision', 'recall', 'f1'):
        lines.append(f'| {key} | {result[key]:.4f} |')
    lines += ['', f'Confusion matrix: TN={result["tn"]}, FP={result["fp"]}, FN={result["fn"]}, '
              f'TP={result["tp"]}.', '', report['limitation'], '', report['score_warning'], '']
    (output_dir / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workdir', type=Path, default=WORKDIR, help='DLC2Action data and project folder')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--features', choices=sorted(FEATURES), default='pairemb',
                        help='pairemb = pose + pair features + DINOv2 embeddings (default)')
    parser.add_argument('--num-epochs', type=int, default=100)
    args = parser.parse_args()
    train(args.workdir, args.output_dir, args.features, args.num_epochs)
