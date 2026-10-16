"""Make a reproducible synthetic clip, visibly marked as a demo."""
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from .review import KEYPOINTS, run

def main():
    root=Path('outputs/visualization/synthetic_inputs');root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for f in range(240):
        angle=f/30*1.1
        for track,offset in [(1,0),(2,np.pi)]:
            x=320+65*np.cos(angle+offset);y=240+65*np.sin(angle+offset)
            r=dict(FrameNum=f,TrackUID=track,X_center=x,Y_center=y)
            for i,k in enumerate(KEYPOINTS):r[k+'_x']=x+(i-4)*4;r[k+'_y']=y+(i-4)*2
            rows.append(r)
    pd.DataFrame(rows).to_parquet(root/'poses.parquet',index=False)
    pd.DataFrame([dict(VideoID='synthetic',StartFrame=60,EndFrame=180,Behavior='Circling',Notes='Invented demo labels; not research results')]).to_csv(root/'annotations.csv',index=False)
    pd.DataFrame([dict(VideoID='synthetic',StartFrame=75,EndFrame=205)]).to_csv(root/'predictions.csv',index=False)
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','color=c=0x122330:s=640x480:r=30:d=8','-vf',"drawtext=text='SYNTHETIC DEMO - not real fish':x=20:y=30:fontcolor=white:fontsize=22",'-c:v','libx264','-pix_fmt','yuv420p',str(root/'video.mp4')],check=True)
    cfg=dict(video_id='synthetic',video='video.mp4',poses='poses.parquet',out='../synthetic_demo',width=640,height=480,fps=30,start_frame=0,end_frame=239,pair=['1','2'],synthetic=True,prediction_scope='Synthetic interval example; no trained model',tracks=[dict(name='Demo human labels',path='annotations.csv',verified=True),dict(name='Demo predicted intervals',path='predictions.csv',kind='prediction')])
    path=root/'config.json';path.write_text(json.dumps(cfg,indent=2));run(path)
    experiment=dict(hypothesis='Synthetic example: proximity retains at least 90% of invented positive events.',video_id='synthetic',video='video.mp4',poses='poses.parquet',annotations=dict(path='annotations.csv',verified=True),out='../synthetic_experiment',fps=30,width=640,height=480,start_frame=0,end_frame=239,distance_thresholds_px=[100,160],minimum_durations_s=[1,2],required_event_coverage=.5,review_rule='d160_t2')
    (root/'experiment.json').write_text(json.dumps(experiment,indent=2))

if __name__=='__main__':main()
