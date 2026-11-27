from models.common.predict import main, predict  # noqa: F401

NAMES = {'pair': 'lrcv_pose_pair', 'pairemb': 'lrcv_dinov2', 'pairvmae': 'lrcv_videomae',
         'pairturns': 'lrcv_turns', 'pairturnsvmae': 'lrcv_turns_videomae',
         'pairmotion': 'lrcv_pairmotion', 'pairmotionemb': 'lrcv_pairmotion_dinov2',
         'pairmotionvmae': 'lrcv_pairmotion_videomae'}

if __name__ == '__main__':
    main(NAMES, 'pairmotion', 'models.intervals.logistic_regression_cv.train')
