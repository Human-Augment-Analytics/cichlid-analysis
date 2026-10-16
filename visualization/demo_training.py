"""Generate synthetic train/test pair tables with explicit positive and negative labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .training_backend.scene_features import DISTANCES,ANGLES,RATES


def main():
    root=Path('outputs/visualization/training_demo_inputs').resolve();root.mkdir(parents=True,exist_ok=True)
    videos=[]
    for n,split in enumerate(['train','test']):
        vid='synthetic_'+split;rows=[];poses=[];rng=np.random.default_rng(42+n)
        for f in range(300):
            positive=80<=f<220;distance=(90 if positive else 240)+rng.normal(0,3);angle=f/30*(1.2 if positive else .05)+n*.2
            r=dict(FrameNum=f,TrackUID_1=1,TrackUID_2=2)
            r.update({c:distance+rng.normal() for c in DISTANCES});r.update({c:angle+rng.normal(0,.01) for c in ANGLES});r.update({c:(1.2 if positive else .05)+rng.normal(0,.01) for c in RATES});rows.append(r)
            for track,sign in [(1,-1),(2,1)]:
                x=320+sign*distance/2*np.cos(angle);y=240+sign*distance/2*np.sin(angle)
                poses.append(dict(FrameNum=f,TrackUID=track,X_center=x,Y_center=y,Nose_x=x+10,Nose_y=y,TailTip_x=x-10,TailTip_y=y))
        pd.DataFrame(rows).to_parquet(root/f'{vid}_pairs.parquet',index=False)
        pd.DataFrame(poses).to_parquet(root/f'{vid}.parquet',index=False)
        pd.DataFrame([dict(VideoID=vid,StartFrame=a,EndFrame=b,ShouldUse=True,Circling=pos,Notes='Invented synthetic training label') for a,b,pos in [(0,79,False),(80,219,True),(220,299,False)]]).to_csv(root/f'{vid}.csv',index=False)
        videos.append(dict(video_id=vid,split=split,pairs=f'{vid}_pairs.parquet',annotations=dict(path=f'{vid}.csv')))
    (root/'training.json').write_text(json.dumps(dict(fps=30,width=640,height=480,videos=videos),indent=2))
    (root/'review.json').write_text(json.dumps(dict(video_id='synthetic_test',poses='synthetic_test.parquet',out='../training_demo_review',fps=30,width=640,height=480,start_frame=0,end_frame=299,pair=['1','2'],synthetic=True,tracks=[dict(name='Invented labels',path='synthetic_test.csv')]),indent=2))
    print(f'Synthetic training configuration: {root}/training.json')

if __name__=='__main__':main()
