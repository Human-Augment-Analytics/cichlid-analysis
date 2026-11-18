"""Create an explicitly synthetic source and render an example results MP4."""
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from .export_video import render,KEYPOINTS


def make_inputs(root,frames=360):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);width,height=640,480;rows=[]
    encoder=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','30','-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p',str(root/'source.mp4')],stdin=subprocess.PIPE)
    try:
        background=np.random.default_rng(42).integers(22,42,(height,width),dtype=np.uint8);background=np.repeat(background[:,:,None],3,axis=2)
        for f in range(frames):
            image=Image.fromarray(background.copy());d=ImageDraw.Draw(image);d.text((12,12),'SYNTHETIC DEMO - invented fish, labels and predictions',fill='white')
            close=60<=f<240;radius=52 if close else 130;angle=(min(f,60)*.1+max(0,min(f,240)-60)*1.15+max(0,f-240)*.1)/30
            for j,track in enumerate([1,2]):
                theta=angle+j*np.pi;x=320+radius*np.cos(theta);y=240+radius*np.sin(theta);heading=theta+np.pi/2
                direction=np.array([np.cos(heading),np.sin(heading)]);normal=np.array([-direction[1],direction[0]])
                # Elliptical silhouette and a triangular tail, purely synthetic.
                points=[tuple(np.array([x,y])+35*np.cos(t)*direction+12*np.sin(t)*normal) for t in np.linspace(0,2*np.pi,28)]
                color=['#c39a68','#6d9ab0'][j];d.polygon(points,fill=color)
                tail=np.array([x,y])-35*direction;d.polygon([tuple(tail),tuple(tail-18*direction+13*normal),tuple(tail-18*direction-13*normal)],fill=color)
                r=dict(project_id='synthetic',day_label='synthetic',FrameNum=f,TrackUID=track,X_center=x,Y_center=y)
                offsets=[30,23,23,19,12,5,-3,-12,-23,-45]
                for k,offset in zip(KEYPOINTS,offsets):r[k+'_x']=x+offset*direction[0];r[k+'_y']=y+offset*direction[1]
                rows.append(r)
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close();encoder.stdin=None
        if encoder.wait()!=0:raise RuntimeError('Could not create synthetic source')
    finally:
        if encoder.poll() is None:encoder.terminate();encoder.wait()
    pd.DataFrame(rows).to_parquet(root/'poses.parquet',index=False)
    pd.DataFrame([dict(VideoID='synthetic',StartFrame=a,EndFrame=b,ShouldUse=True,Circling=pos) for a,b,pos in [(0,59,False),(60,min(239,frames-1),True)] if a<frames and b>=a]+([dict(VideoID='synthetic',StartFrame=240,EndFrame=frames-1,ShouldUse=True,Circling=False)] if frames>240 else [])).to_csv(root/'annotations.csv',index=False)
    pd.DataFrame([dict(VideoID='synthetic',StartFrame=80,EndFrame=min(frames-1,269))]).to_csv(root/'predictions.csv',index=False)
    cfg=dict(title='Synthetic example | demonstrate results presentation',synthetic=True,video_id='synthetic',expected_project_id='synthetic',video='source.mp4',poses='poses.parquet',out='results_demo.mp4',fps=30,start_frame=0,end_frame=frames-1,pair=['1','2'],detector_name='Invented intervals - NOT a trained model',prediction_scope='Synthetic scene-level interval example',predictions=dict(path='predictions.csv'),annotations=dict(path='annotations.csv'))
    config=root/'config.json';config.write_text(json.dumps(cfg,indent=2));return config


def main():
    config=make_inputs('outputs/visualization/export_demo');render(config,overwrite=True)

if __name__=='__main__':main()
