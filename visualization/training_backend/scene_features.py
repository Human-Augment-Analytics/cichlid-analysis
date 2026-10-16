"""Scene-level pair aggregation reused from the earlier seed-learning prototype."""
import numpy as np
import pandas as pd
DISTANCES=['centroid_distance_px','dist_nose1_nose2_px','dist_nose1_tailtip2_px','dist_nose2_tailtip1_px','dist_nose1_spine4of2_px','dist_nose2_spine4of1_px']
ANGLES=['orbital_rad','rel_heading_rad','bearing_1_rad','bearing_2_rad']
RATES=['d_heading_1_rad_s','d_heading_2_rad_s','d_orbital_rad_s']
KEYPOINTS=['Nose','LeftEye','RightEye','Head','Spine1','Spine2','Spine3','Spine4','Peduncle','TailTip']


def aggregate_pairs(table):
    """Video-level learning features; periodic angles represented as sine/cosine."""
    numeric=table[DISTANCES+RATES].apply(pd.to_numeric,errors='raise').copy()
    for name in ANGLES:
        a=pd.to_numeric(table[name],errors='raise')
        numeric[name+'_sin']=np.sin(a);numeric[name+'_cos']=np.cos(a)
    numeric=numeric.replace([np.inf,-np.inf],np.nan)
    numeric['FrameNum']=table.FrameNum.to_numpy()
    grouped=numeric.groupby('FrameNum',sort=True)
    agg=grouped.agg(['mean','min','max']);agg.columns=[a+'__'+b for a,b in agg.columns]
    agg['pair_count']=grouped.size();return agg.reset_index()

