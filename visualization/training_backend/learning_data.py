"""Pose → explicit pair features and verified frame labels, independent of DLC2Action."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import load_table,normalize_pose,features,annotations,pair_table

FEATURE_NAMES=['distance_px','angular_velocity_rad_s','bearing_sin','bearing_cos',
               'speed_a_px_s','speed_b_px_s','heading_cos_difference','alignment_a','alignment_b']


def feature_matrix(rows,pair,start,end,cfg):
    fps=cfg['fps'];conf=cfg.get('confidence',.5)
    feat=features(rows,pair,start,end,fps,conf,cfg.get('centroid_keypoints'))
    angle=feat.bearing_rad.to_numpy();columns=[feat.distance_px.to_numpy(),feat.angular_velocity_rad_s.to_numpy(),np.sin(angle),np.cos(angle)]
    good=rows[(rows.confidence>=conf)&np.isfinite(rows.x)&np.isfinite(rows.y)]
    centers=good
    if cfg.get('centroid_keypoints'):centers=centers[centers.keypoint.isin(cfg['centroid_keypoints'])]
    centers=centers.groupby(['frame','track'])[['x','y']].mean()
    headings=[]
    for fish in pair:
        try:xy=centers.xs(str(fish),level='track').reindex(feat.frame).to_numpy()
        except KeyError:xy=np.full((len(feat),2),np.nan)
        columns.append(np.r_[np.nan,np.linalg.norm(np.diff(xy,axis=0),axis=1)*fps])
        def part(name):
            points=good[(good.track==str(fish))&(good.keypoint==name)].set_index('frame')
            return points[['x','y']].reindex(feat.frame).to_numpy()
        head=part(cfg.get('head_keypoint','nose'));tail=part(cfg.get('tail_keypoint','tail'))
        vector=head-tail;h=np.arctan2(vector[:,1],vector[:,0]);h[np.linalg.norm(vector,axis=1)<1e-8]=np.nan
        headings.append(h)
    columns.extend([np.cos(headings[0]-headings[1]),np.cos(headings[0]-angle),np.cos(headings[1]-(angle+np.pi))])
    return np.column_stack(columns).astype(np.float32), np.isfinite(feat.distance_px.to_numpy())&np.isfinite(angle)


def labels_from_spans(frames,spans,assume_background=False):
    """Ignore unknown, conflicts and rejected rows even under assumed-background mode."""
    labels=np.full(len(frames),0 if assume_background else -100,dtype=np.int64)
    pos=np.zeros(len(frames),bool);neg=pos.copy();unknown=pos.copy()
    for s in spans:
        mask=(frames>=s['start'])&(frames<=s['end'])
        if s['status']=='circling':pos|=mask
        elif s['status']=='non-circling':neg|=mask
        else:unknown|=mask
    labels[neg]=0;labels[pos]=1;labels[unknown|(pos&neg)]=-100
    return labels


def signature(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare(manifest_path,out):
    manifest_path=Path(manifest_path).resolve();root=manifest_path.parent
    manifest=json.loads(manifest_path.read_text());out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not manifest.get('videos'):raise ValueError('Manifest requires a nonempty videos list')
    snapshots=[];ids=set();feature_names=None
    for index,entry in enumerate(manifest['videos']):
        vid=str(entry['video_id'])
        if vid in ids:raise ValueError('Each video_id must be unique; assign one split to each original video')
        ids.add(vid)
        if entry['split'] not in ['train','val','test']:raise ValueError('split must be train, val or test')
        input_key='pairs' if entry.get('pairs') else 'poses'
        poses=(root/entry[input_key]).resolve();schema=(root/entry['schema']).resolve()
        cfg=json.loads(schema.read_text())
        if cfg['fps']<=0 or cfg['width']<=0 or cfg['height']<=0:raise ValueError('Invalid FPS/dimensions')
        source=load_table(poses);f=cfg['pairs' if input_key=='pairs' else 'pose']['frame'];frames_source=pd.to_numeric(source[f],errors='raise')
        if not np.isfinite(frames_source).all() or (frames_source%1!=0).any() or frames_source.min()<0:
            raise ValueError('Pose frames must be nonnegative finite integers')
        start=int(entry.get('start',frames_source.min()));end=int(entry.get('end',frames_source.max()))
        if start<0 or end<start:raise ValueError('Invalid preparation range')
        if len(entry['pair'])!=2 or str(entry['pair'][0])==str(entry['pair'][1]):raise ValueError('Select two distinct fish IDs')
        if input_key=='pairs':
            frames=np.sort(frames_source[(frames_source>=start)&(frames_source<=end)].unique()).astype(np.int64)
            x,valid,_=pair_table(source,cfg['pairs'],entry['pair'],start,end,frame_index=frames)
            current_names=cfg['pairs']['features']
        else:
            rows=normalize_pose(source,cfg['pose'],start,end)
            if cfg.get('coordinate_space','pixels')=='normalized':rows.x*=cfg['width'];rows.y*=cfg['height']
            elif cfg.get('coordinate_space','pixels')!='pixels':raise ValueError('Unknown coordinate space')
            x,valid=feature_matrix(rows,entry['pair'],start,end,cfg)
            current_names=FEATURE_NAMES
        if feature_names is None:feature_names=current_names
        elif feature_names!=current_names:raise ValueError('All videos must use identical feature names and order')
        if input_key!='pairs':frames=np.arange(start,end+1)
        spans=[];annotation_path=None
        if entry.get('annotations'):
            annotation_path=(root/entry['annotations']).resolve()
            spans=annotations(load_table(annotation_path),start,end,vid)
        y=labels_from_spans(frames,spans,manifest.get('assume_unannotated_background',False));y[~valid]=-100
        filename=f'video_{index:03d}.npz'
        np.savez_compressed(out/filename,x=x,y=y,frames=frames,valid=valid)
        rec=dict(entry,file=filename,input_type=input_key,input_path=str(poses),schema=str(schema),annotations=str(annotation_path) if annotation_path else None,
                 pose_sha256=signature(poses),schema_sha256=signature(schema),annotation_sha256=signature(annotation_path) if annotation_path else None,
                 supervised_positive=int((y==1).sum()),supervised_negative=int((y==0).sum()),unknown_or_missing=int((y==-100).sum()))
        rec[input_key]=str(poses)
        if input_key=='pairs' and entry.get('poses'):rec['poses']=str((root/entry['poses']).resolve())
        if entry.get('video'):
            import subprocess
            video=(root/entry['video']).resolve()
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate','-of','json',str(video)]))['streams'][0]
            n,d=map(float,info['avg_frame_rate'].split('/'))
            if abs(cfg['fps']-n/d)>1e-5 or cfg['width']!=info['width'] or cfg['height']!=info['height']:
                raise ValueError('Schema FPS/width/height must match source video')
            rec['video']=str(video)
        snapshots.append(rec)
    # Same content under different filenames cannot silently become train and held-out.
    for i,a in enumerate(snapshots):
        for b in snapshots[i+1:]:
            if a['pose_sha256']==b['pose_sha256'] and a['split']!=b['split']:
                raise ValueError('Identical pose files occur across splits; provide independent videos')
    meta=dict(videos=snapshots,feature_names=feature_names,assume_unannotated_background=manifest.get('assume_unannotated_background',False),review=manifest.get('review'))
    (out/'manifest.json').write_text(json.dumps(meta,indent=2));print(f'Prepared {len(snapshots)} videos → {out.resolve()}')
    return meta


def continuous_runs(valid,frames):
    indices=np.flatnonzero(valid)
    if not len(indices):return []
    breaks=np.flatnonzero((np.diff(indices)!=1)|(np.diff(frames[indices])!=1))+1
    return [part for part in np.split(indices,breaks) if len(part)]
