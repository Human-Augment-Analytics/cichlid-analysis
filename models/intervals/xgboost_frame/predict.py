from models.common.predict import main, predict  # noqa: F401

NAMES = {'pair': 'xgboost_pose_pair', 'pairemb': 'xgboost_dinov2', 'pairvmae': 'xgboost_videomae',
         'pairturns': 'xgboost_turns', 'pairturnsvmae': 'xgboost_turns_videomae',
         'pairmotion': 'xgboost_pairmotion', 'pairmotionemb': 'xgboost_pairmotion_dinov2',
         'pairmotionvmae': 'xgboost_pairmotion_videomae'}

if __name__ == '__main__':
    main(NAMES, 'pairmotion', 'models.intervals.xgboost_frame.train')
