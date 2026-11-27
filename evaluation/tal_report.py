import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from evaluation.evaluate import GROUND_TRUTH, read_intervals, read_predictions
from evaluation.evaluate_models import comparable_models, model_videos
from evaluation.metrics import iou, merge
from models.common.config import OUTPUTS, VIDEOS, model_name, mstcn_workdir, video_frames
from models.common.predict import merge_spans
from models.intervals.logistic_regression.postprocess import find_positive_spans
from models.intervals.logistic_regression.predict import FPS

IOUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
SHOWN = [0.1, 0.3, 0.5, 0.7]
ABLATION_SHOWN = [0.1, 0.7]
FIGURES = OUTPUTS / 'figures'
SCORE_DIRS = {'logistic_regression': 'circling_baseline'}
EMBEDDERS = {'dinov2': 'DINOv2', 'videomae': 'VideoMAE'}
INPUTS = ['Pose', 'Pair', 'Turns', 'DINOv2', 'VideoMAE']
SPAN_RULES = [('threshold runs', None, 0.0), ('7.5 s', 7.5, 0.0), ('2 s', 2.0, 0.0), ('4 s', 4.0, 0.0),
              ('2 s', 2.0, 3.0), ('4 s', 4.0, 3.0), ('4 s', 4.0, 1.0), ('5 s', 5.0, 1.0)]
PRETTY = {'rule_based': 'Rule-based (baseline)', 'logistic_regression': 'Logistic regression (baseline)',
          'sartab': 'SARTAB (Lancaster et al. 2024)'}


def base_name(name):
    return re.sub(r'(_15spans|_hn\d+)+$', '', name)


def gt_spans(gt, videos):
    return {v: merge(gt.filter(pl.col('video_id') == v).select('start', 'end').rows()) for v in videos}


def average_precision(truth, detections, threshold):
    total = sum(len(spans) for spans in truth.values())
    if not total or not detections:
        return 0.0
    detections = sorted(detections, key=lambda d: -d[3])
    taken = {v: np.zeros(len(spans), bool) for v, spans in truth.items()}
    hits = np.zeros(len(detections))
    for i, (video, start, end, _) in enumerate(detections):
        spans = truth.get(video, [])
        overlaps = np.array([iou((start, end), span) for span in spans])
        for j in np.argsort(-overlaps) if len(spans) else []:
            if overlaps[j] < threshold:
                break
            if not taken[video][j]:
                taken[video][j] = True
                hits[i] = 1
                break
    recall = np.cumsum(hits) / total
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    precision = np.r_[0, precision, 0]
    recall = np.r_[0, recall, 1]
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    steps = np.flatnonzero(recall[1:] != recall[:-1]) + 1
    return float(np.sum((recall[steps] - recall[steps - 1]) * precision[steps]))


def map_row(truth, detections):
    row = {f'mAP@{t:g}': 100 * average_precision(truth, detections, t) for t in IOUS}
    row['mAP Avg'] = float(np.mean([row[f'mAP@{t:g}'] for t in IOUS]))
    return row


def frame_scores(name, video):
    path = OUTPUTS / SCORE_DIRS.get(name, name) / 'scores' / f'{video}_scores.parquet'
    if path.exists():
        return pl.read_parquet(path)['circling_score'].to_numpy()
    if base_name(name) == 'rule_based':
        from models.common.turns import turn_table
        try:
            return np.nan_to_num(turn_table(video, video_frames([video])[video])['rule_turns_4s'].to_numpy())
        except (OSError, SystemExit) as error:
            print(f'rule_based {video}: no turn scores ({error}); its spans get equal confidence', flush=True)
    return None


def detections_for(spans_by_video, scores_by_video):
    out = []
    for video, spans in spans_by_video.items():
        score = scores_by_video.get(video)
        for start, end in spans:
            confidence = float(score[start:end + 1].mean()) if score is not None else 1.0
            out.append((video, start, end, confidence))
    return out


def model_detections(name, folder, videos):
    predicted = read_predictions(folder).filter(pl.col('video_id').is_in(videos))
    spans = {v: predicted.filter(pl.col('video_id') == v).select('start', 'end').rows() for v in videos}
    # ActionFormer writes native proposal confidence. Preserve it instead of
    # replacing it with a constant or an unrelated frame-score average.
    native = {}
    for path in Path(folder).glob('*.csv'):
        if path.stem not in videos:
            continue
        table = pl.read_csv(path)
        confidence_column = next((c for c in table.columns if c.lower() == 'confidence'), None)
        if confidence_column is not None:
            for row in table.iter_rows(named=True):
                native[(path.stem, row['StartFrame'], row['EndFrame'])] = float(row[confidence_column])
    if native and len(native) == sum(len(items) for items in spans.values()):
        detections = [(v, start, end, native[(v, start, end)])
                      for v, items in spans.items() for start, end in items]
        return detections, True
    scores = {v: frame_scores(name, v) for v in videos}
    return detections_for(spans, scores), all(s is not None for s in scores.values())


def per_video_spread(truth, detections):
    values = [map_row({v: spans}, [d for d in detections if d[0] == v])['mAP Avg'] for v, spans in truth.items()]
    return float(np.std(values)) if len(values) > 1 else float('nan'), dict(zip(truth, values))


def count_parameters(name):
    base = base_name(name)
    if base == 'rule_based':
        return None, 'hand-set thresholds'
    folder = OUTPUTS / ('circling_baseline' if name == 'logistic_regression' else name)
    if base.startswith('mstcn_'):
        checkpoints = sorted((mstcn_workdir() / 'projects' / 'circling_repo' / 'results' / 'model' / name).glob('*.pt'),
                             key=lambda p: p.stat().st_mtime)
        if not checkpoints:
            return None, 'no checkpoint'
        import torch
        state = torch.load(checkpoints[-1], map_location='cpu', weights_only=False)
        state = state.get('model_state_dict', state)
        return int(sum(t.numel() for k, t in state.items() if hasattr(t, 'numel') and 'num_batches_tracked' not in k)), 'weights'
    if base.startswith('actionformer_'):
        path = folder / 'model.pt'
        if not path.exists():
            return None, 'no checkpoint'
        import torch
        state = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
        return int(sum(t.numel() for t in state.values() if hasattr(t, 'numel'))), 'weights'
    path = folder / 'model.joblib'
    if not path.exists():
        return None, 'no model file'
    model = joblib.load(path)['model']
    estimator = model[-1] if hasattr(model, 'steps') else model
    if hasattr(estimator, 'get_booster'):
        return int(len(estimator.get_booster().trees_to_dataframe())), 'tree nodes'
    if hasattr(estimator, 'coef_'):
        return int(estimator.coef_.size + np.size(estimator.intercept_)), 'weights'
    return None, 'unknown'


def format_count(n):
    if n is None:
        return '–'
    for size, unit in ((1e6, 'M'), (1e3, 'k')):
        if n >= size:
            return f'{n / size:.1f}{unit}'
    return str(n)


def uses_embedder(name):
    return any(key in base_name(name) for key in EMBEDDERS)


def comparison_table(gt, models, videos):
    rows, spreads = [], {}
    for name, folder in models.items():
        covered = [v for v in model_videos(folder) if v in videos]
        truth = gt_spans(gt, covered)
        if not any(truth.values()):
            continue
        detections, scored = model_detections(name, folder, covered)
        spread, per_video = per_video_spread(truth, detections)
        params, unit = count_parameters(name)
        rows.append({'Method': PRETTY.get(name, name),
                     'model': name, '#Params': format_count(params) + ('†' if uses_embedder(name) and params else ''),
                     'params': params, 'params_unit': unit, 'videos': len(covered), 'events': len(detections),
                     'scored': scored, **map_row(truth, detections), 'σ_video': spread,
                     **{f'mAP Avg {v}': value for v, value in per_video.items()}})
        spreads[name] = per_video
    table = pl.DataFrame(rows)
    order = {'rule_based': 0, 'logistic_regression': 1}
    return table.with_columns(pl.col('model').map_elements(lambda n: order.get(base_name(n), 2), return_dtype=pl.Int64)
                              .alias('_rank')).sort('_rank', 'mAP Avg').drop('_rank')


def inputs_of(name):
    base = base_name(name)
    parts = {'Pose': True, 'Pair': not base.endswith('_pose'), 'Turns': '_turns' in base,
             'DINOv2': 'dinov2' in base, 'VideoMAE': 'videomae' in base}
    return parts


def this_run(name):
    return name == model_name(base_name(name))


def input_table(main, family):
    rows = []
    for row in main.filter(pl.col('model').map_elements(lambda n: base_name(n).startswith(family) and this_run(n),
                                                         return_dtype=pl.Boolean)).iter_rows(named=True):
        rows.append({**{k: '✓' if v else '✗' for k, v in inputs_of(row['model']).items()}, 'model': row['model'],
                     **{f'mAP@{t:g}': row[f'mAP@{t:g}'] for t in ABLATION_SHOWN}, 'mAP Avg': row['mAP Avg']})
    if not rows:
        return pl.DataFrame()
    table = pl.DataFrame(rows).sort('mAP Avg')
    return table.select([c for c in table.columns if c == 'model' or table[c].dtype != pl.String
                         or (table[c] == '✓').any()])


def threshold_runs(score, threshold=0.5):
    positive = np.r_[False, score >= threshold, False]
    edges = np.flatnonzero(positive[1:] != positive[:-1])
    return list(zip(edges[::2], edges[1::2] - 1))


def span_rule_table(gt, name, videos, threshold=0.5, fraction=0.9):
    scores = {v: frame_scores(name, v) for v in videos}
    if any(s is None for s in scores.values()):
        return pl.DataFrame()
    truth = gt_spans(gt, videos)
    rows = []
    for label, seconds, gap in SPAN_RULES:
        spans = {}
        for video, score in scores.items():
            if seconds is None:
                spans[video] = [(int(s), int(e)) for s, e in threshold_runs(score, threshold)]
                continue
            table = pl.DataFrame({'FrameNum': np.arange(len(score)), 'circling_score': score})
            found = find_positive_spans(table, fps=FPS, threshold=threshold, fraction=fraction, seconds=seconds)
            found = merge_spans(found, table, threshold, gap)
            spans[video] = found.select('start_frame', 'end_frame').rows()
        detections = detections_for(spans, scores)
        rows.append({'Min span': label if seconds is not None else '–',
                     'Merge gap': f'{gap:g} s' if gap else ('–' if seconds is None else '✗'),
                     'Rule': 'threshold runs only' if seconds is None else 'span rule', 'events': len(detections),
                     **{f'mAP@{t:g}': map_row(truth, detections)[f'mAP@{t:g}'] for t in ABLATION_SHOWN},
                     'mAP Avg': map_row(truth, detections)['mAP Avg']})
    return pl.DataFrame(rows)


def hard_negative_table(main):
    rows = []
    for row in main.iter_rows(named=True):
        match = re.search(r'_hn(\d+)$', row['model'])
        if not match:
            continue
        twin = main.filter(pl.col('model') == row['model'][:match.start()])
        for hard, source in (('✗', twin.row(0, named=True) if twin.height else None), (f'{match.group(1)}%', row)):
            if source:
                rows.append({'Model': base_name(source['model']), 'Hard negatives': hard,
                             **{f'mAP@{t:g}': source[f'mAP@{t:g}'] for t in ABLATION_SHOWN},
                             'mAP Avg': source['mAP Avg']})
    return pl.DataFrame(rows)


def bold_best(table, columns):
    best = {c: table[c].max() for c in columns if c in table.columns and table[c].dtype.is_numeric()}
    return [{c: (f'**{v:.1f}**' if c in best and v == best[c] and isinstance(v, float) else
                 (f'{v:.1f}' if isinstance(v, float) else str(v))) for c, v in row.items()}
            for row in table.iter_rows(named=True)]


def to_markdown(table, columns, caption):
    rows = bold_best(table.select(columns), [c for c in columns if c.startswith('mAP')])
    lines = [f'**{caption}**', '', '| ' + ' | '.join(columns) + ' |',
             '|' + '|'.join('---' if c in ('Method', 'Model') else '---:' for c in columns) + '|']
    lines += ['| ' + ' | '.join(row[c] for c in columns) + ' |' for row in rows]
    return '\n'.join(lines) + '\n'


def latex_escape(text):
    text = text.replace('σ_video', 'SIGMAVIDEO')
    return (text.replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
            .replace('#', r'\#').replace('✓', r'\checkmark').replace('✗', r'$\times$').replace('SIGMAVIDEO', r'$\sigma_{\mathrm{video}}$')
            .replace('†', r'$^\dagger$').replace('–', '--'))


def to_latex(table, columns, caption, label):
    map_columns = [c for c in columns if c.startswith('mAP@') or c == 'mAP Avg']
    rows = bold_best(table.select(columns), map_columns)
    head = [c.replace('mAP@', '').replace('mAP Avg', 'Avg') if c in map_columns else c for c in columns]
    first_map = columns.index(map_columns[0]) if map_columns else len(columns)
    spec = ''.join('l' if c in ('Method', 'Model') else 'c' for c in columns)
    title = latex_escape(re.sub(r'^Table \d+: ', '', caption))
    lines = [r'\begin{table}[t]', r'\centering', rf'\caption{{{title}}}', rf'\label{{{label}}}',
             rf'\begin{{tabular}}{{{spec}}}', r'\toprule']
    if map_columns:
        lines.append(' & ' * first_map + rf'\multicolumn{{{len(map_columns)}}}{{c}}{{\textbf{{mAP@IoU (\%)}}}}'
                     + ' & ' * (len(columns) - first_map - len(map_columns)) + r' \\')
        lines.append(rf'\cmidrule(lr){{{first_map + 1}-{first_map + len(map_columns)}}}')
    lines.append(' & '.join(rf'\textbf{{{latex_escape(h)}}}' for h in head) + r' \\')
    lines.append(r'\midrule')
    for row in rows:
        cells = [re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', latex_escape(row[c])) for c in columns]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def to_figure(table, columns, caption, path):
    from matplotlib.figure import Figure
    rows = bold_best(table.select(columns), [c for c in columns if c.startswith('mAP')])
    figure = Figure(figsize=(12.8, 1.6 + 0.48 * (len(rows) + 1)), layout='constrained', facecolor='#fcfcfb')
    axis = figure.subplots()
    axis.axis('off')
    cells = [[row[c].replace('**', '') for c in columns] for row in rows]
    head = [c.replace('mAP@', 'mAP@IoU ').replace('σ_video', 'σ video') for c in columns]
    drawn = axis.table(cellText=cells, colLabels=head, loc='center', cellLoc='center', bbox=[0, 0, 1, 0.92])
    drawn.auto_set_font_size(False)
    drawn.set_fontsize(11)
    widths = [3.2 if c in ('Method', 'Model') else 1.0 for c in columns]
    for (i, j), cell in drawn.get_celld().items():
        cell.set_width(widths[j] / sum(widths))
        cell.set_edgecolor('#e4e3df')
        cell.set_facecolor('#e9e8e4' if i == 0 else ('#fcfcfb' if i % 2 else '#f3f2ef'))
        if i == 0 or (i > 0 and '**' in rows[i - 1][columns[j]]):
            cell.get_text().set_weight('bold')
        if columns[j] in ('Method', 'Model') and i > 0:
            cell.get_text().set_ha('left')
            cell.PAD = 0.02
    import textwrap
    axis.set_title(textwrap.fill(caption, 150), loc='left', fontsize=13)
    figure.savefig(path, dpi=200, facecolor='#fcfcfb')


def write(table, columns, caption, stem, label, output_dir=None):
    if table.is_empty():
        return ''
    output_dir = Path(output_dir or FIGURES)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.write_csv(output_dir / f'{stem}.csv')
    (output_dir / f'{stem}.tex').write_text(to_latex(table, columns, caption, label), encoding='utf-8')
    markdown = to_markdown(table, columns, caption)
    (output_dir / f'{stem}.md').write_text(markdown, encoding='utf-8')
    to_figure(table, columns, caption, output_dir / f'{stem}.png')
    return markdown


def best_learned(main):
    learned = main.filter(~pl.col('model').map_elements(lambda n: base_name(n) in ('rule_based', 'logistic_regression',
                                                                                    'sartab') or not this_run(n),
                                                       return_dtype=pl.Boolean) & pl.col('scored'))
    return learned.sort('mAP Avg', descending=True)['model'][0] if learned.height else None


def build(models=None, ablation_model=None, threshold=0.5, fraction=0.9):
    gt = read_intervals(GROUND_TRUTH)
    videos = sorted(VIDEOS)
    models = comparable_models(models)
    main = comparison_table(gt, models, videos)
    if main.is_empty():
        raise SystemExit(f'no predictions for {videos}')
    shown = [f'mAP@{t:g}' for t in SHOWN]
    ablation = [f'mAP@{t:g}' for t in ABLATION_SHOWN]
    events = sum(len(s) for s in gt_spans(gt, videos).values())
    number = 0

    def next_number():
        nonlocal number
        number += 1
        return number

    sections = [write(main, ['Method', '#Params', *shown, 'mAP Avg', 'σ_video'],
                      f'Table {next_number()}: Circling detection on {len(videos)} videos ({events} events). mAP@IoU averaged over '
                      'IoU 0.1-0.7. σ_video is the spread of Avg mAP across videos. † Not including the frozen '
                      'DINOv2 / VideoMAE encoder.', f'tal_table{number}_methods', 'tab:methods')]
    for family, title in (('xgboost_', 'XGBoost'), ('mstcn_', 'MS-TCN'), ('lrcv_', 'LR-CV')):
        table = input_table(main, family)
        if table.height > 1:
            columns = [c for c in table.columns if c in INPUTS] + ablation + ['mAP Avg']
            sections.append(write(table, columns, f'Table {next_number()}: Comparison of input features ({title}).',
                                  f'tal_table{number}_inputs_{family.rstrip("_")}', f'tab:inputs-{family.rstrip("_")}'))
    target = ablation_model or best_learned(main)
    if target:
        covered = [v for v in model_videos(models[target]) if v in videos]
        table = span_rule_table(gt, target, covered, threshold, fraction)
        sections.append(write(table, ['Min span', 'Merge gap', 'events', *ablation, 'mAP Avg'],
                              f'Table {next_number()}: Comparison of span rules ({target}, same frame scores).',
                              f'tal_table{number}_span_rules', 'tab:span-rules'))
    table = hard_negative_table(main)
    if table.height:
        sections.append(write(table, ['Model', 'Hard negatives', *ablation, 'mAP Avg'],
                              f'Table {next_number()}: Effect of hard negative training segments.', f'tal_table{number}_hard_negatives',
                              'tab:hard-negatives'))
    notes = ('Detections are each model\'s predicted spans (outputs/predictions). ActionFormer uses its native '
             'Confidence; other learned models use mean per-frame score inside the span; rule-based spans use '
             'their mean 4 s turn count. Average precision follows the ActivityNet protocol: detections ranked '
             'by score, each ground-truth event matched at most once. #Params: network weights for ActionFormer, '
             'MS-TCN and linear models, tree nodes for XGBoost.')
    text = '\n'.join(s for s in sections if s) + '\n' + notes + '\n'
    (FIGURES / 'tal_tables.md').write_text(text, encoding='utf-8')
    return main, text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temporal action localization tables (mAP@IoU) for every model')
    parser.add_argument('--ablation-model', default=None, help='model for the span-rule table (default: best learned)')
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--min-fraction', type=float, default=.9)
    args = parser.parse_args()
    _, text = build(ablation_model=args.ablation_model, threshold=args.threshold, fraction=args.min_fraction)
    print(text)
    print(f'LaTeX, Markdown, CSV and PNG for every table in {FIGURES} (tal_table*.*, tal_tables.md)')
