import argparse

from models.common.frame_model import train_frame_model

NAMES = {'pair': 'xgboost_pose_pair', 'pairemb': 'xgboost_dinov2', 'pairvmae': 'xgboost_videomae'}
PARAMS = dict(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42,
              tree_method='hist', eval_metric='logloss', n_jobs=-1)

def fit(x, y, groups):
    from xgboost import XGBClassifier
    model = XGBClassifier(**PARAMS)
    model.fit(x, y)
    return model, {'hyperparameters': {k: v for k, v in PARAMS.items() if k != 'n_jobs'}}

def train(features='pairemb'):
    return train_frame_model(
        NAMES[features], features, fit, f'Circling XGBoost (per frame), {features}', 'XGBoost',
        'XGBoost with 400 trees, depth 3, learning rate 0.05, seed 42; missing values are handled '
        'by the trees, no imputation. Fixed threshold 0.5; no tuning uses test labels.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Per-frame XGBoost on the shared segment sample')
    parser.add_argument('--features', choices=sorted(NAMES), default='pairemb',
                        help=', '.join(f'{k} -> {v}' for k, v in NAMES.items()))
    train(parser.parse_args().features)
