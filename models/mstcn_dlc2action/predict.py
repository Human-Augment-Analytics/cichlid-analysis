from models.common.predict import main, predict

NAMES = {'pairemb': 'mstcn_dinov2', 'pairvmae': 'mstcn_videomae', 'pair': 'mstcn_pose_pair', 'pose': 'mstcn_pose'}
NAME = NAMES['pairemb']

if __name__ == '__main__':
    main(NAMES, 'pairemb', 'models.mstcn_dlc2action.train')
