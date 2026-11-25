from models.common.predict import main, predict  # noqa: F401

NAMES = {'pairemb': 'mstcn_dinov2', 'pairvmae': 'mstcn_videomae', 'pair': 'mstcn_pose_pair',
         'pose': 'mstcn_pose', 'pairmotion': 'mstcn_pairmotion',
         'pairmotionemb': 'mstcn_pairmotion_dinov2', 'pairmotionvmae': 'mstcn_pairmotion_videomae'}
NAME = NAMES['pairmotion']

if __name__ == '__main__':
    main(NAMES, 'pairmotion', 'models.mstcn_dlc2action.train')
