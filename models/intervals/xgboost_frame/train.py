import argparse

from sklearn.model_selection import StratifiedGroupKFold

from models.common.frame_model import train_frame_model

NAMES = {'pair': 'xgboost_pose_pair', 'pairemb': 'xgboost_dinov2', 'pairvmae': 'xgboost_videomae',
         'pairturns': 'xgboost_turns', 'pairturnsvmae': 'xgboost_turns_videomae',
         'pairmotion': 'xgboost_pairmotion', 'pairmotionemb': 'xgboost_pairmotion_dinov2',
         'pairmotionvmae': 'xgboost_pairmotion_videomae'}
PARAMS = dict(n_estimators=1500, max_depth=4, learning_rate=0.03, min_child_weight=5,
              subsample=0.9, colsample_bytree=0.8, reg_lambda=5.0, random_state=42,
              tree_method='hist', eval_metric='aucpr', early_stopping_rounds=60, n_jobs=-1)


def fit(x, y, groups):
    from xgboost import XGBClassifier

    # Model selection uses a segment-grouped slice of the training data only.
    # The repository test segments never influence the number of trees.
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fit_ids, validation_ids = next(folds.split(x, y, groups))
    model = XGBClassifier(**PARAMS)
    model.fit(x[fit_ids], y[fit_ids], eval_set=[(x[validation_ids], y[validation_ids])], verbose=False)
    details = {'hyperparameters': {k: v for k, v in PARAMS.items() if k != 'n_jobs'},
               'validation': 'one StratifiedGroupKFold training-only split, grouped by segment; AUC-PR early stopping',
               'best_iteration': int(model.best_iteration) if model.best_iteration is not None else None,
               'train_frames_for_fit': len(fit_ids), 'validation_frames_for_early_stopping': len(validation_ids)}
    return model, details


def train(features='pairmotion', tune_postprocess=True):
    return train_frame_model(
        NAMES[features], features, fit, f'Circling XGBoost (per frame), {features}', 'XGBoost',
        'XGBoost histogram trees with up to 1500 rounds, depth 4, learning rate 0.03, row/column subsampling, '
        'L2 regularization and 60-round AUC-PR early stopping on a segment-grouped training-only validation '
        'split. Missing values are handled by the trees. Span settings are selected on video-level OOF predictions '
        'when enabled; final test predictions never choose those settings.', tune_postprocess=tune_postprocess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Per-frame XGBoost on the shared segment sample')
    parser.add_argument('--features', choices=sorted(NAMES), default='pairmotion',
                        help=', '.join(f'{k} -> {v}' for k, v in NAMES.items()))
    parser.add_argument('--skip-postprocess-tuning', action='store_true',
                        help='Disable video-level OOF span-rule selection')
    args = parser.parse_args()
    train(args.features, tune_postprocess=not args.skip_postprocess_tuning)
