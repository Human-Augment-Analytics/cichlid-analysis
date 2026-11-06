import argparse

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.common.frame_model import train_frame_model

NAMES = {'pair': 'lrcv_pose_pair', 'pairemb': 'lrcv_dinov2', 'pairvmae': 'lrcv_videomae'}
CS = np.logspace(-4, 2, 13)

def fit(x, y, groups):
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(x, y, groups))
    model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                          StandardScaler(),
                          LogisticRegressionCV(Cs=CS, cv=folds, scoring='average_precision',
                                               max_iter=1000, solver='lbfgs'))
    model.fit(x, y)
    search = model[-1]
    scores = search.scores_[1].mean(axis=0)
    return model, {'C': float(search.C_[0]), 'C_grid': [float(c) for c in CS],
                   'cv_average_precision': [float(s) for s in scores],
                   'cv': 'StratifiedGroupKFold(5, shuffle=True, random_state=42) grouped by segment'}

def train(features='pair'):
    return train_frame_model(
        NAMES[features], features, fit, f'Circling logistic regression, C by CV, {features}',
        'LogisticRegressionCV',
        'The logistic regression pipeline (median imputation with missingness indicators, standard '
        'scaling, L2) with C chosen from 13 values between 1e-4 and 100 by average precision in a '
        '5-fold cross-validation on the training frames grouped by segment. Fixed threshold 0.5; '
        'no tuning uses test labels.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Per-frame logistic regression with C chosen by grouped CV')
    parser.add_argument('--features', choices=sorted(NAMES), default='pair',
                        help=', '.join(f'{k} -> {v}' for k, v in NAMES.items()))
    train(parser.parse_args().features)
