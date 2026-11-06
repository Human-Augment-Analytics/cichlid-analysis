from models.common.predict import main, predict

NAMES = {'pair': 'lrcv_pose_pair', 'pairemb': 'lrcv_dinov2', 'pairvmae': 'lrcv_videomae'}

if __name__ == '__main__':
    main(NAMES, 'pair', 'models.logistic_regression_cv.train')
