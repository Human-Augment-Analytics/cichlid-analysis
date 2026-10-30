"""Interval adapter shared with the earlier visualization review workflow."""
import numpy as np
import pandas as pd
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
    start_col = spec.get('start', 'StartFrame' if 'StartFrame' in data else 'start_frame' if 'start_frame' in data else 'Start Frame' if 'Start Frame' in data else 'StartTime')
    end_col = spec.get('end', 'EndFrame' if 'EndFrame' in data else 'end_frame' if 'end_frame' in data else 'End Frame' if 'End Frame' in data else 'EndTime')
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
            behavior=str(r.get('Primary label/behavior',r.get('Behavior',r.get('behavior','circling')))).strip().lower()
            status='non-circling' if behavior in {'non-circling','noncircling'} else 'circling' if behavior=='circling' else 'unknown'
        notes=' | '.join(f'{k}: {v}' for k,v in r.items() if ('note' in k.lower() or 'comment' in k.lower() or 'direction' in k.lower() or 'circle' in k.lower()) and pd.notna(v) and str(v).strip())
        result.append(dict(start=a,end=b,status=status,event=str(r.get('EventID',r.get('event_id',number))),notes=notes,fish_ids=str(r.get('FishIDs','')) if pd.notna(r.get('FishIDs','')) else ''))
    return result,skipped

