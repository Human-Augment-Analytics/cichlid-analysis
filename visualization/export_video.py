"""Render a shareable MP4 from existing video, poses, labels and detector outputs.

python -m visualization.export_video visualization/export0031.json
This renderer does not train models or infer poses/behavior.
"""
import argparse
import csv
import itertools
import shutil
from collections import defaultdict
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

KEYPOINTS=['Nose','LeftEye','RightEye','Head','Spine1','Spine2','Spine3','Spine4','Peduncle','TailTip']
MIDLINE=['Nose','Head','Spine1','Spine2','Spine3','Spine4','Peduncle','TailTip']
BG='#0b1620';PANEL='#142a39';WHITE='#e7f1f7';MUTED='#a8bdca';GREEN='#6ae0ba';ORANGE='#ffbb75';PURPLE='#bc9cff'


def integer(value):
    x=float(value)
    if not math.isfinite(x) or x<0 or x!=int(x):raise ValueError(f'Invalid source frame: {value}')
    return int(x)


def truth(value):return str(value).strip().lower() in {'true','1','1.0','yes','y'}


def stamp(frame,fps):
    milliseconds=round(frame/fps*1000);s,ms=divmod(milliseconds,1000);h,s=divmod(s,3600);m,s=divmod(s,60)
    return f'{h:02}:{m:02}:{s:02}.{ms:03}'


def read_intervals(path,video,role,spec):
    """Explicit source policy; unknown/rejected/conflicting labels remain unknown."""
    rows=pd.read_csv(path,encoding_errors='replace').to_dict('records');out=[];skipped=0
    for r in rows:
        vid=r.get('VideoID',r.get('video_id',r.get('video',video)))
        if pd.notna(vid) and str(vid)!=video:continue
        a=r.get('StartFrame',r.get('Start Frame',r.get('start_frame')));b=r.get('EndFrame',r.get('End Frame',r.get('end_frame')))
        if pd.isna(a) or pd.isna(b):skipped+=1;continue
        a=integer(a);b=integer(b)-(1 if spec.get('end_exclusive',False) else 0)
        if b<a:raise ValueError('Interval end precedes start')
        status='prediction' if role=='prediction' else 'unknown'
        if role!='prediction':
            if 'ShouldUse' in r:
                if truth(r['ShouldUse']):status='circling' if truth(r.get('Circling')) else 'non-circling'
            elif truth(r.get('Verified',r.get('verified',spec.get('verified',False)))):
                label=str(r.get('Primary label/behavior',r.get('Behavior',r.get('behavior','')))).strip().lower()
                status='circling' if label=='circling' else 'non-circling' if label in {'non-circling','no-circling'} else 'unknown'
        notes=' | '.join(str(v) for k,v in r.items() if ('note' in k.lower() or 'comment' in k.lower()) and pd.notna(v))
        # Preserve supplied direction/turn information, but never infer it from a status.
        out.append(dict(start=a,end=b,status=status,direction=str(r.get('Rotation direction',r.get('Direction',''))) if pd.notna(r.get('Rotation direction',r.get('Direction',''))) else '',turns=str(r.get('No. of turns/circles',r.get('Rounds',''))) if pd.notna(r.get('No. of turns/circles',r.get('Rounds',''))) else '',notes=notes))
    return out,skipped


def reference_at(spans,frame):
    active={s['status'] for s in spans if s['start']<=frame<=s['end']}
    if active=={'circling'}:return 'CIRCLING'
    if active=={'non-circling'}:return 'NON-CIRCLING'
    return 'UNKNOWN / UNLABELED'


def pair_geometry(pose,pair,start,end,fps):
    frames=np.arange(start,end+1);xy=[]
    for track in pair:
        p=pose[pose.TrackUID.astype(str)==track].set_index('FrameNum')
        xy.append(p[['X_center','Y_center']].reindex(frames).to_numpy(float))
    rel=xy[1]-xy[0];distance=np.linalg.norm(rel,axis=1);angle=np.arctan2(rel[:,1],rel[:,0]);angle[distance<1e-8]=np.nan
    rate=np.r_[np.nan,np.arctan2(np.sin(np.diff(angle)),np.cos(np.diff(angle)))*fps]
    return pd.DataFrame(dict(frame=frames,distance_px=distance,angular_velocity_rad_s=rate))


def select_pair(pose,start,end,predictions):
    """Pick one stable diagnostic pair; this does not identify the circling actors."""
    window=pose[pose.FrameNum.between(start,end)].copy()
    active=np.zeros(len(window),dtype=bool)
    for event in predictions:active|=window.FrameNum.between(event['start'],event['end']).to_numpy()
    use_prediction=bool(active.any())
    selected=window.loc[active] if use_prediction else window
    closest=defaultdict(int);distances=defaultdict(list)
    for _,group in selected.groupby('FrameNum'):
        good=group[np.isfinite(group.X_center)&np.isfinite(group.Y_center)]
        pairs=[]
        for left,right in itertools.combinations(good.to_dict('records'),2):
            pair=tuple(sorted([str(left['TrackUID']),str(right['TrackUID'])]))
            distance=math.hypot(left['X_center']-right['X_center'],left['Y_center']-right['Y_center'])
            pairs.append((distance,pair));distances[pair].append(distance)
        if pairs:closest[min(pairs)[1]]+=1
    if not closest:raise ValueError('No frame contains two valid tracked centers in this excerpt. Choose another range or explicit pair.')
    pair=max(closest,key=lambda p:(closest[p],len(distances[p]),-float(np.median(distances[p])),p))
    return list(pair),dict(method='Most frequently closest valid pair during predicted intervals' if use_prediction else 'Most frequently closest valid pair in excerpt (no predicted frames available)',closest_frames=closest[pair],coobserved_frames=len(distances[pair]),warning='Diagnostic selection only; can select reflections or incorrect identities; inspect video before sharing')


def fonts():
    for path in ['/System/Library/Fonts/Supplemental/Arial.ttf','/Library/Fonts/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).is_file():return {size:ImageFont.truetype(path,size) for size in [14,16,18,20,24,30]}
    return {size:ImageFont.load_default(size=size) for size in [14,16,18,20,24,30]}


def render(config,overwrite=False,check_only=False):
    config=Path(config).resolve();cfg=json.loads(config.read_text());base=config.parent;resolve=lambda p:(base/p).resolve()
    video=resolve(cfg['video']);pose_path=resolve(cfg['poses']);dest=resolve(cfg['out']);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and not overwrite and not check_only:raise FileExistsError(f'{dest} already exists; use --overwrite to replace it')
    if dest.suffix.lower()!='.mp4':raise ValueError('Output must end in .mp4')
    if dest in {video,pose_path}:raise ValueError('Output cannot replace an input')
    required={'video':video,'individual pose Parquet':pose_path,'prediction CSV':resolve(cfg['predictions']['path'])}
    if cfg.get('annotations'):required['annotation CSV']=resolve(cfg['annotations']['path'])
    if cfg.get('scores'):required['frame-score file']=resolve(cfg['scores']['path'])
    missing=[f'{label}: {path}' for label,path in required.items() if not path.is_file()]
    if missing:raise FileNotFoundError('Missing inputs:\n'+'\n'.join(missing)+'\nConfig paths are relative to the JSON file. Place inputs at these paths or edit the config.')
    for command in ['ffmpeg','ffprobe']:
        if not shutil.which(command):raise RuntimeError(f'{command} is not installed or not on PATH')
    fps=float(cfg.get('fps',30));start=integer(cfg['start_frame']);end=integer(cfg['end_frame'])
    if not math.isfinite(fps) or fps<=0 or end<start or end-start>18000:raise ValueError('Use positive FPS and a clip of at most 18001 frames')
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_frames','-of','json',str(video)]))['streams'][0]
    num,den=map(float,info['avg_frame_rate'].split('/'))
    if abs(num/den-fps)>.01:raise ValueError('Configured FPS differs from source average FPS')
    if str(info.get('nb_frames','')).isdigit() and end>=int(info['nb_frames']):raise ValueError('Clip extends beyond source video')
    pose=pd.read_parquet(pose_path,filters=[('FrameNum','>=',max(0,start-1)),('FrameNum','<=',end)])
    for col in ['FrameNum','TrackUID','X_center','Y_center']:
        if col not in pose:raise ValueError(f'Missing pose column {col}')
    if pose.duplicated(['FrameNum','TrackUID']).any():raise ValueError('Duplicate source frame/TrackUID rows; resolve upstream')
    if not np.isfinite(pose.FrameNum).all() or (pose.FrameNum%1!=0).any():raise ValueError('Pose FrameNum must be an integer')
    for col in ['day_label']:
        if col in pose and len(pose) and set(pose[col].dropna().astype(str))!={cfg['video_id']}:raise ValueError('Pose video label does not match configuration')
    if cfg.get('expected_project_id') and ('project_id' not in pose or set(pose.project_id.dropna().astype(str))!={cfg['expected_project_id']}):raise ValueError('Pose project ID does not match expected recording')
    prediction=cfg['predictions'];pred_path=resolve(prediction['path']);pred,skipped_pred=read_intervals(pred_path,cfg['video_id'],'prediction',prediction)
    if cfg.get('pair')=='auto':pair,pair_selection=select_pair(pose,start,end,pred)
    else:
        pair=[str(x) for x in cfg['pair']]
        pair_selection=dict(method='Explicitly configured TrackUIDs',warning='Diagnostic selection only; no behavior attribution')
    if len(pair)!=2 or pair[0]==pair[1]:raise ValueError('Select two distinct TrackUIDs or set pair to auto')
    poses={int(f):g for f,g in pose.groupby('FrameNum')};geometry=pair_geometry(pose,pair,max(0,start-1),end,fps);geometry=geometry[geometry.frame>=start].reset_index(drop=True)
    records={str(x) for x in pose.TrackUID.unique()}
    if not all(track in records for track in pair):raise ValueError('A selected TrackUID is absent from this excerpt; choose valid IDs or set pair to auto')
    reference=[];skipped_ref=0;ref_path=None
    if cfg.get('annotations'):
        ref_path=resolve(cfg['annotations']['path']);reference,skipped_ref=read_intervals(ref_path,cfg['video_id'],'annotation',cfg['annotations'])
    scores={}
    if cfg.get('scores'):
        spec=cfg['scores'];p=resolve(spec['path']);table=pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)
        vc=spec.get('video','video_id');fc=spec.get('frame','frame');sc=spec.get('score','score')
        if vc in table:table=table[table[vc].astype(str)==cfg['video_id']]
        table=table[table[fc].between(start,end)]
        if table[fc].duplicated().any():raise ValueError('Duplicate score frames')
        for f,s in zip(table[fc],table[sc]):
            if pd.isna(s):continue
            if not np.isfinite(s) or not 0<=s<=1:raise ValueError('Frame score must be finite and in [0,1]')
            scores[integer(f)]=float(s)
    plan=dict(video_id=cfg['video_id'],source_video=str(video),pose_file=str(pose_path),project_ids=sorted(pose.project_id.dropna().astype(str).unique().tolist()) if 'project_id' in pose else [],fps=fps,start_frame=start,end_frame=end,source_start_time=stamp(start,fps),source_end_time=stamp(end,fps),output_seconds=(end-start+1)/fps,selected_pair=pair,pair_selection=pair_selection,geometry_frames=int(geometry.distance_px.notna().sum()),predicted_intervals_in_excerpt=sum(s['end']>=start and s['start']<=end for s in pred),reference_intervals_in_excerpt=sum(s['end']>=start and s['start']<=end for s in reference),available_frame_scores=len(scores),output=str(dest))
    print(json.dumps(plan,indent=2),flush=True)
    if check_only:
        print('Input check passed. No video was rendered.',flush=True)
        return plan
    W,H=1440,960;vx,vy,vw,vh=24,114,936,702;F=fonts();font=lambda size:F[size]
    srcw,srch=info['width'],info['height'];scale=min(vw/srcw,vh/srch);dw=int(srcw*scale)//2*2;dh=int(srch*scale)//2*2;ox=vx+(vw-dw)//2;oy=vy+(vh-dh)//2
    static=Image.new('RGB',(W,H),BG);draw=ImageDraw.Draw(static)
    draw.text((24,14),'CICHLID BEHAVIOR | PIPELINE RESULTS',fill=GREEN,font=font(16))
    draw.text((24,40),cfg.get('title','Video, tracking and behavior predictions')[:70],fill=WHITE,font=font(30))
    draw.text((24,83),f"{cfg['video_id']} | {cfg.get('detector_name','External detector')[:55]} | SOURCE FRAMES {start}-{end}",fill=MUTED,font=font(16))
    draw.rounded_rectangle((vx-4,vy-4,vx+vw+4,vy+vh+4),8,fill=PANEL)
    draw.rounded_rectangle((984,110,1416,816),8,fill=PANEL)
    draw.text((1004,128),'BEHAVIOR OUTPUT',fill=MUTED,font=font(16))
    draw.text((1004,294),'SELECTED PAIR GEOMETRY',fill=MUTED,font=font(16))
    draw.text((1004,321),f'TrackUID {pair[0]} / {pair[1]}',fill=WHITE,font=font(16))
    plot_specs=[]
    for name,label,top,color in [('distance_px','Centroid distance (source px)',365,GREEN),('angular_velocity_rad_s','Pair angular velocity (rad/s)',544,ORANGE)]:
        x0,x1=1037,1394;y0,y1=top+36,top+143;vals=geometry[name].to_numpy(float);valid=vals[np.isfinite(vals)];lo=min(0,float(valid.min())) if len(valid) else 0.;hi=max(0,float(valid.max())) if len(valid) else 1.
        if hi==lo:hi=lo+1
        draw.text((1004,top),label,fill=WHITE,font=font(16));draw.text((1004,y0),f'{hi:.1f}',fill=MUTED,font=font(14));draw.text((1004,y1-10),f'{lo:.1f}',fill=MUTED,font=font(14));draw.line((x0,y1,x1,y1),fill='#486071')
        previous=None
        for i,value in enumerate(vals):
            if not np.isfinite(value):previous=None;continue
            point=(x0+(x1-x0)*i/max(1,len(vals)-1),y1-(y1-y0)*(value-lo)/(hi-lo))
            if previous:draw.line((previous,point),fill=color,width=2)
            previous=point
        if not len(valid):draw.text((x0,y0+35),'No valid selected-pair data',fill=MUTED,font=font(14))
        plot_specs.append((x0,x1,y0,y1))
    draw.text((1004,725),'Pair geometry does not assign',fill=MUTED,font=font(16));draw.text((1004,748),'scene-level predictions to this pair.',fill=MUTED,font=font(16));draw.text((1004,777),'Positive angular velocity = clockwise',fill=MUTED,font=font(14))
    draw.text((24,830),'Human reference',fill=WHITE,font=font(16));draw.text((24,868),'Detector interval',fill=WHITE,font=font(16))
    tx0,tx1=200,1416;span=max(1,end-start+1)
    # Render resolved reference states so conflicting/unknown overlaps cannot look accepted.
    resolved=[]
    for f in range(start,end+1):
        status=reference_at(reference,f)
        if resolved and resolved[-1]['status']==status:resolved[-1]['end']=f
        else:resolved.append(dict(start=f,end=f,status=status))
    for yy,spans,role in [(832,resolved,'ref'),(870,pred,'pred')]:
        draw.rectangle((tx0,yy,tx1,yy+20),fill='#304554')
        for s in spans:
            if s['end']<start or s['start']>end:continue
            color=PURPLE if role=='pred' else GREEN if s['status']=='CIRCLING' else '#33785e' if s['status']=='NON-CIRCLING' else '#6c7c87'
            draw.rectangle((tx0+(max(start,s['start'])-start)/span*(tx1-tx0),yy,tx0+(min(end,s['end'])-start+1)/span*(tx1-tx0),yy+20),fill=color)
    footer='Light green: circling | Dark green: reviewed negative | Purple: prediction | Gray: unknown / unlabeled'
    draw.text((24,914),footer,fill=MUTED,font=font(16))
    if cfg.get('synthetic',False):draw.text((1004,54),'SYNTHETIC DEMO',fill=ORANGE,font=font(20))
    meta=[]
    print(f'Rendering {end-start+1} source frames. Exact source-index extraction decodes from the beginning; late clips can take minutes.',flush=True)
    with tempfile.TemporaryDirectory(prefix='cichlid-render-') as temp:
        temp=Path(temp);candidate=dest.with_name(dest.stem+'.rendering.mp4');decerr=(temp/'decode.log').open('wb');encerr=(temp/'encode.log').open('wb')
        vf=f'select=between(n\\,{start}\\,{end}),scale={dw}:{dh}'
        decoder=subprocess.Popen(['ffmpeg','-v','error','-threads','2','-i',str(video),'-vf',vf,'-frames:v',str(end-start+1),'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=decerr)
        encoder=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(candidate)],stdin=subprocess.PIPE,stderr=encerr)
        try:
            for frame in range(start,end+1):
                raw=decoder.stdout.read(dw*dh*3)
                if len(raw)!=dw*dh*3:raise RuntimeError(f'Decoder ended before source frame {frame}; check source bounds')
                image=static.copy();image.paste(Image.frombytes('RGB',(dw,dh),raw),(ox,oy));d=ImageDraw.Draw(image)
                for j,track in enumerate(pair):
                    group=poses.get(frame,pd.DataFrame());fish=group[group.TrackUID.astype(str)==track] if len(group) else group
                    if not len(fish):continue
                    r=fish.iloc[0];color=[ORANGE,'#72c5ff'][j];points={}
                    for k in KEYPOINTS:
                        x,y=r.get(k+'_x'),r.get(k+'_y')
                        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y) and 0<=x<srcw and 0<=y<srch:points[k]=(ox+x/srcw*dw,oy+y/srch*dh)
                    for first,second in zip(MIDLINE[:-1],MIDLINE[1:]):
                        if first in points and second in points:d.line((points[first],points[second]),fill=color,width=3)
                    for point in points.values():x,y=point;d.ellipse((x-3,y-3,x+3,y+3),fill=color)
                    if points:
                        x,y=points.get('Nose',next(iter(points.values())));text=f'Fish {track}';box=d.textbbox((x,y+8),text,font=font(16));d.rectangle(box,fill=BG);d.text((x,y+8),text,fill=color,font=font(16))
                active=any(s['start']<=frame<=s['end'] for s in pred);ref=reference_at(reference,frame);score=scores.get(frame)
                d.rounded_rectangle((1004,160,1396,205),6,fill='#553f78' if active else '#2b4050');d.text((1014,169),'PREDICTED CIRCLING' if active else 'NO PREDICTED INTERVAL',fill=WHITE,font=font(20))
                d.text((1004,220),'Reference: '+ref,fill=GREEN if ref=='CIRCLING' else MUTED,font=font(16));d.text((1004,249),f'Model score (uncalibrated): {score:.3f}' if score is not None else 'Model score: not supplied',fill=PURPLE,font=font(16))
                d.rectangle((vx,vy,vx+vw,vy+35),fill=BG);d.text((vx+10,vy+6),f'{stamp(frame,fps)} | Source frame {frame}',fill=WHITE,font=font(20))
                row=geometry.iloc[frame-start]
                for name,y,color in [('distance_px',vy+vh-53,GREEN),('angular_velocity_rad_s',vy+vh-28,ORANGE)]:
                    value=row[name];text=('Distance: ' if name=='distance_px' else 'Angular velocity: ')+(f'{value:.2f}' if np.isfinite(value) else 'missing')+(' px' if name=='distance_px' else ' rad/s');d.text((vx+10,y),text,fill=color,font=font(18),stroke_width=2,stroke_fill=BG)
                ratio=(frame-start)/max(1,end-start)
                for x0,x1,y0,y1 in plot_specs:x=x0+ratio*(x1-x0);d.line((x,y0,x,y1),fill=WHITE,width=2)
                x=tx0+(frame-start)/span*(tx1-tx0);d.line((x,826,x,895),fill=WHITE,width=2)
                encoder.stdin.write(image.tobytes());meta.append(dict(frame=frame,predicted_interval=active,reference=ref,score=score))
                if (frame-start+1)%300==0:print(f'Rendered {frame-start+1}/{end-start+1} frames',flush=True)
            encoder.stdin.close();encoder.stdin=None
            if decoder.wait()!=0 or encoder.wait()!=0:raise RuntimeError('FFmpeg failed; '+(temp/'decode.log').read_text(errors='replace')+(temp/'encode.log').read_text(errors='replace'))
            count=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','csv=p=0',str(candidate)]))
            if count!=end-start+1:raise RuntimeError('Output frame count differs from source excerpt')
            os.replace(candidate,dest)
        except Exception:
            for process in [decoder,encoder]:
                if process.poll() is None:process.terminate()
            for process in [decoder,encoder]:
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            candidate.unlink(missing_ok=True);raise
        finally:
            decoder.stdout.close();decerr.close();encerr.close()
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    manifest=dict(config=cfg,selected_pair=pair,pair_selection=pair_selection,export_plan=plan,video_path=str(video),pose_path=str(pose_path),score_sha256=sha(resolve(cfg['scores']['path'])) if cfg.get('scores') else None,prediction_sha256=sha(pred_path),annotation_sha256=sha(ref_path) if ref_path else None,skipped_prediction_rows=skipped_pred,skipped_annotation_rows=skipped_ref,rendered_frames=len(meta),frame_policy='Zero-based inclusive source frames, decoded by index; output plays at configured nominal FPS',prediction_scope=cfg.get('prediction_scope','External predictions: scope must be confirmed with producer'),limitations=['No new model trained or inference performed','Selected pair geometry is not prediction attribution','No confidence inferred from interval CSV','Missing/unlabeled reference remains unknown','No audio is included','Direction and circle counts are not inferred by this renderer'])
    dest.with_suffix('.provenance.json').write_text(json.dumps(manifest,indent=2))
    pd.DataFrame(meta).to_csv(dest.with_suffix('.frames.csv'),index=False);geometry.to_csv(dest.with_suffix('.features.csv'),index=False)
    print(f'Shareable MP4 ready: {dest}',flush=True)
    return dest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');p.add_argument('--overwrite',action='store_true');p.add_argument('--check',action='store_true',help='Validate inputs and print export plan without decoding/rendering video');a=p.parse_args()
    try:render(a.config,a.overwrite,a.check)
    except (FileNotFoundError,ValueError,RuntimeError) as exc:p.exit(2,str(exc)+'\n')

if __name__=='__main__':main()
