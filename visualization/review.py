"""Model-independent, short-clip review. Run: python -m visualization.review CONFIG."""
import argparse
import hashlib
import html
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from .core import normalize_pose, payload, write_viewer

KEYPOINTS = ['Nose','LeftEye','RightEye','Head','Spine1','Spine2','Spine3','Spine4','Peduncle','TailTip']
MIDLINE = ['Nose','Head','Spine1','Spine2','Spine3','Spine4','Peduncle','TailTip']


def table(path):
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, encoding_errors='replace')


def truth(value):
    return str(value).strip().lower() in {'true','1','1.0','y','yes'}


def frame(value):
    n = float(value)
    if not np.isfinite(n) or n < 0 or n != int(n):
        raise ValueError(f'Invalid frame: {value}')
    return int(n)


def seconds(value):
    parts = str(value).strip().split(':')
    return sum(float(p) * 60**i for i,p in enumerate(reversed(parts)))


def load_intervals(path, video_id, fps, spec):
    """Keep source notes; do not silently apply reviewer suggestions or certify candidates."""
    data = table(path); result=[]; skipped=0
    start_col = spec.get('start', 'StartFrame' if 'StartFrame' in data else 'start_frame' if 'start_frame' in data else 'StartTime')
    end_col = spec.get('end', 'EndFrame' if 'EndFrame' in data else 'end_frame' if 'end_frame' in data else 'EndTime')
    for number,r in enumerate(data.to_dict('records'), 2):
        vid=r.get('VideoID',r.get('video_id',r.get('video',video_id)))
        if pd.notna(vid) and str(vid)!=video_id: continue
        if pd.isna(r.get(start_col)) or pd.isna(r.get(end_col)):
            skipped+=1; continue
        if 'Time' in start_col or spec.get('units')=='seconds':
            a=round(seconds(r[start_col])*fps); b=round(seconds(r[end_col])*fps)-1
        else:
            a=frame(r[start_col]); b=frame(r[end_col])-(1 if spec.get('end_exclusive',False) else 0)
        if b<a: raise ValueError(f'{path}:{number}: end precedes start')
        status='unknown'
        if spec.get('kind','annotation')=='prediction': status='prediction'
        elif 'ShouldUse' in r:
            if truth(r['ShouldUse']): status='circling' if truth(r.get('Circling')) else 'non-circling'
        elif spec.get('verified',False) or truth(r.get('verified',r.get('Verified',False))):
            behavior=str(r.get('Behavior',r.get('behavior','circling'))).strip().lower()
            status='non-circling' if behavior in {'non-circling','noncircling'} else 'circling' if behavior=='circling' else 'unknown'
        notes=' | '.join(f'{k}: {v}' for k,v in r.items() if ('note' in k.lower() or 'comment' in k.lower() or 'direction' in k.lower() or 'circle' in k.lower()) and pd.notna(v) and str(v).strip())
        result.append(dict(start=a,end=b,status=status,event=str(r.get('EventID',r.get('event_id',number))),notes=notes,fish_ids=str(r.get('FishIDs','')) if pd.notna(r.get('FishIDs','')) else ''))
    return result,skipped


def geometry(pose,pair,start,end,fps):
    # Use box centers, not an average over keypoints. IDs select geometry, never sex/behavior.
    frames=np.arange(start,end+1); xy=[]
    for track in pair:
        p=pose[pose.TrackUID.astype(str)==str(track)]
        if p.FrameNum.duplicated().any(): raise ValueError('Duplicate frame/TrackUID; resolve upstream')
        xy.append(p.set_index('FrameNum')[['X_center','Y_center']].reindex(frames).to_numpy(float))
    rel=xy[1]-xy[0]; dist=np.linalg.norm(rel,axis=1); angle=np.arctan2(rel[:,1],rel[:,0]); angle[dist<1e-8]=np.nan
    rate=np.r_[np.nan,np.arctan2(np.sin(np.diff(angle)),np.cos(np.diff(angle)))*fps]
    g=pd.DataFrame(dict(frame=frames,distance_px=dist,bearing_rad=angle,angular_velocity_rad_s=rate))
    window=max(1,round(4*fps))
    close=pd.Series(np.where(np.isfinite(dist),(dist<160).astype(float),np.nan))
    g['close_fraction_4s']=close.rolling(window,min_periods=window).mean()
    return g


def run(config):
    config=Path(config).resolve(); cfg=json.loads(config.read_text()); base=config.parent
    resolve=lambda p: (base/p).resolve()
    out=resolve(cfg['out']); out.mkdir(parents=True,exist_ok=True)
    fps=float(cfg.get('fps',30)); width=int(cfg.get('width',1296)); height=int(cfg.get('height',972))
    if fps<=0: raise ValueError('fps must be positive')
    start=frame(cfg['start_frame']); end=frame(cfg['end_frame'])
    if end<start or end-start>18000: raise ValueError('Review window must contain 1–18001 frames')
    tracks=[]; audit=[]
    for spec in cfg.get('tracks',[]):
        path=resolve(spec['path']); spans,skipped=load_intervals(path,cfg['video_id'],fps,spec)
        tracks.append(dict(name=spec['name'],kind=spec.get('kind','annotation'),spans=[s for s in spans if s['end']>=start and s['start']<=end]))
        audit.append(dict(name=spec['name'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),skipped_incomplete_rows=skipped))
    pose_path=resolve(cfg['poses'])
    # Predicate pushdown avoids loading a whole multi-hour pose table.
    pose=pd.read_parquet(pose_path,filters=[('FrameNum','>=',start),('FrameNum','<=',end)])
    if 'TrackUID' not in pose: raise ValueError('TrackUID required; do not silently substitute another ID namespace')
    if pose.duplicated(['FrameNum','TrackUID']).any(): raise ValueError('Duplicate frame/TrackUID rows')
    pair=[str(p) for p in cfg.get('pair',[])]
    if len(pair)!=2 or pair[0]==pair[1]: raise ValueError('Specify two distinct TrackUID values in pair; inspect --list-tracks first')
    mapping=dict(frame='FrameNum',track='TrackUID',keypoints={k:dict(x=k+'_x',y=k+'_y') for k in KEYPOINTS if k+'_x' in pose and k+'_y' in pose})
    rows=normalize_pose(pose,mapping,start,end)
    feat=geometry(pose,pair,start,end,fps)
    primary=next((t['spans'] for t in tracks if t['kind']=='annotation'),[])
    scores={}
    for spec in cfg.get('scores',[]):
        p=table(resolve(spec['path'])); vc=spec.get('video','video_id')
        if vc in p: p=p[p[vc].astype(str)==cfg['video_id']]
        fc=spec.get('frame','frame'); sc=spec.get('score','score'); p=p[p[fc].between(start,end)]
        if p[fc].duplicated().any(): raise ValueError('Duplicate score frames')
        for f,s in zip(p[fc],p[sc]):
            if pd.isna(s): continue
            if not np.isfinite(s) or not 0<=s<=1: raise ValueError('Score outside [0,1]')
            scores[str(frame(f))]=float(s)
        break # Explicitly one numeric score source; interval tracks can show many models.
    meta=dict(video_id=cfg['video_id'],start=start,end=end,fps=fps,width=width,height=height,pair=pair,confidence=0,skeleton=list(zip(MIDLINE[:-1],MIDLINE[1:])),synthetic=cfg.get('synthetic',False),pose_available=bool(len(rows)),prediction_scope=cfg.get('prediction_scope','External detector; scope must be confirmed with its author'),comparison_tracks=tracks)
    video=resolve(cfg['video']) if cfg.get('video') else None
    if video:
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate','-of','json',str(video)]))['streams'][0]
        n,d=map(float,info['avg_frame_rate'].split('/'))
        if abs(n/d-fps)>.01: raise ValueError('Video average FPS differs from configured tracking FPS')
        meta.update(width=info['width'],height=info['height'])
        print('Extracting exact source frames; late clips may take several minutes.',flush=True)
        vf=f'select=between(n\\,{start}\\,{end}),setpts=N/({fps}*TB),scale=960:-2'
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(video),'-vf',vf,'-frames:v',str(end-start+1),'-an','-r',str(fps),'-c:v','libx264','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'clip.mp4')],check=True)
        count=int(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','csv=p=0',str(out/'clip.mp4')]))
        if count!=end-start+1: raise ValueError('Clip frame count mismatch')
    data=payload(rows,feat,primary,scores,meta);write_viewer(out,data)
    (out/'review_data.json').write_text(json.dumps(data,allow_nan=False))
    feat.to_csv(out/'features.csv',index=False)
    (out/'provenance.json').write_text(json.dumps(dict(config=cfg,inputs=audit,pose_path=str(pose_path),frame_policy='zero-based inclusive frames; timestamp intervals converted as [start,end)',limitations=['IDs and sex are not biological ground truth','Missing or unknown labels are not negatives','Pair geometry is diagnostic; does not establish circling','No model training or accuracy claims']),indent=2))
    print(f'Review ready: {out}/index.html')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');p.add_argument('--list-tracks',action='store_true');a=p.parse_args()
    if a.list_tracks:
        path=Path(a.config).resolve();c=json.loads(path.read_text());q=pd.read_parquet((path.parent/c['poses']).resolve(),filters=[('FrameNum','>=',frame(c['start_frame'])),('FrameNum','<=',frame(c['end_frame']))],columns=['FrameNum','TrackUID'])
        print(q.groupby('TrackUID').FrameNum.agg(['min','max','count']).to_string());return
    run(a.config)

if __name__=='__main__':main()
