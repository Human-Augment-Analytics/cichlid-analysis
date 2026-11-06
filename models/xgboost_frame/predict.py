from models.common.predict import main, predict

NAMES = {'pair': 'xgboost_pose_pair', 'pairemb': 'xgboost_dinov2', 'pairvmae': 'xgboost_videomae'}

if __name__ == '__main__':
    main(NAMES, 'pairemb', 'models.xgboost_frame.train')
