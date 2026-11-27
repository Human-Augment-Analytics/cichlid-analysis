import argparse
import json

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, precision_score, recall_score

from models.common.config import OUTPUTS, model_name, mstcn_workdir, video_frames
from models.common.frame_model import video_features
from models.common.segments import load_meta
from models.intervals.logistic_regression.features import KEYPOINTS

FRAME_MODELS = ['xgboost_pose_pair', 'xgboost_dinov2', 'xgboost_videomae', 'xgboost_turns', 'xgboost_turns_videomae',
                'xgboost_pairmotion', 'xgboost_pairmotion_dinov2', 'xgboost_pairmotion_videomae',
                'lrcv_pose_pair', 'lrcv_dinov2', 'lrcv_videomae', 'lrcv_turns', 'lrcv_turns_videomae',
                'lrcv_pairmotion', 'lrcv_pairmotion_dinov2', 'lrcv_pairmotion_videomae']
MSTCN_MODELS = {'pose': 'mstcn_pose', 'pose + pair': 'mstcn_pose_pair',
                'pose + pair + DINOv2': 'mstcn_dinov2', 'pose + pair + VideoMAE': 'mstcn_videomae',
                'pose + pair-motion windows': 'mstcn_pairmotion',
                'pose + pair-motion windows + DINOv2': 'mstcn_pairmotion_dinov2',
                'pose + pair-motion windows + VideoMAE': 'mstcn_pairmotion_videomae'}
FIGURES = OUTPUTS / 'figures'
MAX_ROWS = 60_000
REPEATS = 5
SEED = 42


def feature_group(name):
    base = name.removesuffix('_trailing30')
    if base in ('rule_start', 'rule_keep'):
        return 'rule-based: start/keep decision'
    if base.startswith(('rule_turns', 'rule_orbit')):
        return 'rule-based: turns and orbit rate'
    if base.startswith(('rule_closest', 'rule_close')):
        return 'rule-based: closest-pair distance'
    if base.startswith('rule_'):
        return 'rule-based: other'
    if base.startswith(('pairemb_', 'pairvmae_')):
        return 'image embedding'
    if base.startswith('pm_w'):
        return 'pair-motion: temporal consistency'
    if base.startswith('pm_') and any(token in base for token in ('orbit', 'heading', 'corotation', 'bend')):
        return 'pair-motion: angular movement'
    if base.startswith('pm_') and any(token in base for token in ('speed', 'closing', 'approach')):
        return 'pair-motion: speed and closing'
    if base.startswith('pm_') and any(token in base for token in ('facing', 'bearing', 'rel_heading')):
        return 'pair-motion: orientation and facing'
    if base.startswith('pm_'):
        return 'pair-motion: distance and availability'
    if base.startswith('centroid_distance'):
        return 'pair: centroid distance'
    if base.startswith('dist_'):
        return 'pair: nose/tail distances'
    if base.startswith('d_orbital'):
        return 'pair: orbital rate (angular displacement)'
    if base.startswith('d_heading'):
        return 'pair: heading rates'
    if base.startswith(('orbital_rad', 'rel_heading', 'bearing')):
        return 'pair: angles'
    if base == 'pair_count':
        return 'pair: count'
    if base.startswith(('bend', 'turn')):
        return 'pose: bend and turn'
    if base == 'track_count':
        return 'pose: track count'
    if base.split('_')[0] in KEYPOINTS:
        return 'pose: keypoint motion'
    return 'other'


def scores(model, x, y):
    probability = model.predict_proba(x)[:, 1]
    predicted = probability >= 0.5
    return {'average_precision': average_precision_score(y, probability),
            'precision': precision_score(y, predicted, zero_division=0),
            'recall': recall_score(y, predicted, zero_division=0)}


def permutation_drops(model, x, y, columns, rng):
    base = scores(model, x, y)
    drops = []
    for column_ids in columns:
        runs = []
        for _ in range(REPEATS):
            shuffled = x.copy()
            order = rng.permutation(len(x))
            shuffled[:, column_ids] = x[order][:, column_ids]
            after = scores(model, shuffled, y)
            runs.append({k: base[k] - after[k] for k in base})
        drops.append({f'drop_{k}': float(np.mean([r[k] for r in runs])) for k in base}
                     | {'drop_average_precision_std': float(np.std([r['average_precision'] for r in runs]))})
    return base, drops


def native_importance(model, features):
    estimator = model[-1] if hasattr(model, 'steps') else model
    if hasattr(estimator, 'get_booster'):
        gain = estimator.get_booster().get_score(importance_type='gain')
        values = np.array([gain.get(f'f{i}', 0.0) for i in range(len(features))])
        return 'xgboost gain share', values / values.sum() if values.sum() else values
    if hasattr(estimator, 'coef_'):
        names = list(model[0].get_feature_names_out(features)) if hasattr(model, 'steps') else list(features)
        coef = dict(zip(names, estimator.coef_[0]))
        return '|standardized coefficient|', np.array([abs(coef.get(f, 0.0)) for f in features])
    return None, np.full(len(features), np.nan)


def frame_model_test_set(name, bundle):
    test = pl.read_parquet(OUTPUTS / name / 'test_predictions.parquet').select('video', 'FrameNum', 'circling')
    videos = sorted(test['video'].unique())
    workdir = mstcn_workdir()
    meta = load_meta(workdir, videos)
    counts = video_frames(videos)
    tables = []
    for (video,), rows in test.group_by('video', maintain_order=True):
        features = video_features(workdir, video, bundle['feature_set'], meta[video], counts[video],
                                  rows['FrameNum'].to_numpy())
        tables.append(rows.join(features, on='FrameNum', how='left'))
    return pl.concat(tables)


def baseline_test_set():
    return pl.read_parquet(OUTPUTS / 'circling_baseline' / 'sampled_frames.parquet').filter(pl.col('split') == 'test')


def importance(name, folder, test, bundle):
    features = bundle['features']
    if test.height > MAX_ROWS:
        test = test.sample(MAX_ROWS, seed=SEED)
    x = test.select(features).to_numpy().astype(np.float64)
    y = test['circling'].to_numpy().astype(np.int8)
    rng = np.random.default_rng(SEED)
    groups = {}
    for i, feature in enumerate(features):
        groups.setdefault(feature_group(feature), []).append(i)
    base, per_feature = permutation_drops(bundle['model'], x, y, [[i] for i in range(len(features))], rng)
    _, per_group = permutation_drops(bundle['model'], x, y, list(groups.values()), rng)
    label, native = native_importance(bundle['model'], features)
    table = pl.DataFrame([{'feature': f, 'group': feature_group(f), 'native': float(native[i]), **per_feature[i]}
                          for i, f in enumerate(features)]).sort('drop_average_precision', descending=True)
    group_table = pl.DataFrame([{'group': g, 'features': len(ids), **per_group[k]}
                                for k, (g, ids) in enumerate(groups.items())]).sort('drop_average_precision',
                                                                                     descending=True)
    table.write_csv(folder / 'feature_importance.csv')
    group_table.write_csv(folder / 'feature_importance_groups.csv')
    (folder / 'feature_importance.json').write_text(json.dumps(
        {'model': name, 'test_frames': int(len(y)), 'baseline': base, 'native': label, 'repeats': REPEATS,
         'method': 'permutation on the held-out test frames; drop = score before minus score after shuffling'},
        indent=2) + '\n')
    plot(name, table, group_table, label)
    return group_table.with_columns(pl.lit(name).alias('model'))


def plot(name, table, group_table, label):
    from matplotlib.figure import Figure
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12.8, 7.2), layout='constrained', facecolor='#fcfcfb')
    left, right = figure.subplots(1, 2, width_ratios=[1, 1.3])
    groups = group_table.sort('drop_average_precision')
    left.barh(groups['group'], groups['drop_average_precision'], color='#2a78d6')
    left.set(title='Feature groups shuffled together', xlabel='Drop in average precision')
    top = table.head(20).sort('drop_average_precision')
    right.barh(top['feature'], top['drop_average_precision'], color='#eb6834')
    right.set(title='Top 20 single features', xlabel='Drop in average precision')
    for axis in (left, right):
        axis.tick_params(labelsize=8)
        axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle(f'{name}: permutation importance on held-out test frames'
                    + (f' (native: {label} in feature_importance.csv)' if label else ''), x=0.01, ha='left')
    figure.savefig(FIGURES / f'feature_importance_{name}.png', dpi=200)


def plot_overview(groups):
    from matplotlib.figure import Figure
    models = groups['model'].unique(maintain_order=True).to_list()
    names = groups.group_by('group').agg(pl.col('drop_average_precision').max()).sort(
        'drop_average_precision', descending=True)['group'].to_list()
    grid = np.full((len(names), len(models)), np.nan)
    for row in groups.iter_rows(named=True):
        grid[names.index(row['group']), models.index(row['model'])] = row['drop_average_precision']
    figure = Figure(figsize=(12.8, 7.2), layout='constrained', facecolor='#fcfcfb')
    axis = figure.subplots()
    axis.imshow(grid, cmap='Blues', aspect='auto', vmin=0)
    for i in range(len(names)):
        for j in range(len(models)):
            if np.isfinite(grid[i, j]):
                axis.text(j, i, f'{grid[i, j]:.3f}', ha='center', va='center', fontsize=8,
                          color='white' if grid[i, j] > 0.6 * np.nanmax(grid) else '#0b0b0b')
    axis.set_xticks(range(len(models)), models, rotation=30, ha='right', fontsize=9)
    axis.set_yticks(range(len(names)), names, fontsize=9)
    axis.set_title('Drop in average precision when a feature group is shuffled (held-out test frames)', loc='left')
    figure.savefig(FIGURES / 'feature_importance_groups.png', dpi=200)
    groups.write_csv(FIGURES / 'feature_importance_groups.csv')


def mstcn_ablation():
    rows = []
    for inputs, base in MSTCN_MODELS.items():
        path = OUTPUTS / model_name(base) / 'metrics.json'
        if path.exists():
            test = json.loads(path.read_text())['test']
            rows.append({'inputs': inputs, 'model': model_name(base), 'average_precision': test['average_precision'],
                         'precision': test['precision'], 'recall': test['recall'], 'f1': test['f1']})
    if rows:
        table = pl.DataFrame(rows)
        FIGURES.mkdir(parents=True, exist_ok=True)
        table.write_csv(FIGURES / 'feature_importance_mstcn_ablation.csv')
        return table


def stale(folder):
    result, model = folder / 'feature_importance.csv', folder / 'model.joblib'
    return not result.exists() or result.stat().st_mtime < model.stat().st_mtime


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Permutation feature importance for every model with features')
    parser.add_argument('models', nargs='*', help='model names (default: logistic_regression and the per-frame models)')
    parser.add_argument('--force', action='store_true', help='recompute even when the model has not changed')
    args = parser.parse_args()
    names = args.models or ['logistic_regression'] + [model_name(n) for n in FRAME_MODELS]
    tables = []
    for name in names:
        folder = OUTPUTS / ('circling_baseline' if name == 'logistic_regression' else name)
        if not (folder / 'model.joblib').exists():
            print(f'{name}: no model.joblib, skipping', flush=True)
            continue
        if not args.force and not stale(folder):
            print(f'{name}: up to date', flush=True)
            tables.append(pl.read_csv(folder / 'feature_importance_groups.csv').with_columns(pl.lit(name).alias('model')))
            continue
        print(f'{name}: permutation importance', flush=True)
        bundle = joblib.load(folder / 'model.joblib')
        test = baseline_test_set() if name == 'logistic_regression' else frame_model_test_set(name, bundle)
        tables.append(importance(name, folder, test, bundle))
    if tables:
        groups = pl.concat(tables, how='diagonal_relaxed')
        plot_overview(groups)
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=4):
            wide = groups.pivot(on='model', index='group', values='drop_average_precision')
            print('\nDrop in held-out average precision when each feature group is shuffled:\n')
            print(wide)
    ablation = mstcn_ablation()
    if ablation is not None:
        with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
            print('\nMS-TCN (inputs are not separable inside DLC2Action, so its importance is the ablation):\n')
            print(ablation)
    print(f'\nPer model: outputs/<model>/feature_importance*.csv · figures: {FIGURES}')
