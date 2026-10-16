"""Test sustained-proximity candidate generation against annotations, without training."""
import argparse
import hashlib
import html
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from .review import frame, load_intervals, run as review


def nearest_pairs(path,start,end):
    """Read only four columns and the scoped rows; accept track- or frame-sorted sources."""
    result={}
    data=pd.read_parquet(path,columns=['FrameNum','TrackUID','X_center','Y_center'],filters=[('FrameNum','>=',start),('FrameNum','<=',end)])
    if not np.isfinite(data.FrameNum).all() or (data.FrameNum%1!=0).any():
        raise ValueError('FrameNum must contain finite integers')
    _nearest(data,start,end,result)
    return result


def _nearest(data,start,end,result):
    data=data[data.FrameNum.between(start,end)]
    for f,g in data.groupby('FrameNum',sort=False):
        if g.TrackUID.duplicated().any():raise ValueError(f'Duplicate TrackUID at frame {f}')
        g=g[np.isfinite(g.X_center)&np.isfinite(g.Y_center)]
        if len(g)<2:continue
        xy=g[['X_center','Y_center']].to_numpy(float)
        i,j=np.triu_indices(len(g),1);d=np.linalg.norm(xy[i]-xy[j],axis=1);k=int(np.argmin(d))
        result[int(f)]=(float(d[k]),str(g.TrackUID.iloc[i[k]]),str(g.TrackUID.iloc[j[k]]))


def intervals(frames):
    values=sorted(set(int(f) for f in frames));out=[]
    for f in values:
        if out and f==out[-1][1]+1:out[-1][1]=f
        else:out.append([f,f])
    return out


def candidates(near,distance,duration,fps):
    minimum=int(np.ceil(duration*fps))
    return [dict(start=a,end=b) for a,b in intervals(f for f,v in near.items() if v[0]<distance) if b-a+1>=minimum]


def union(spans):
    out=[]
    for a,b in sorted((s['start'],s['end']) for s in spans):
        if out and a<=out[-1][1]+1:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out


def overlap(a,b,spans):return sum(max(0,min(b,y)-max(a,x)+1) for x,y in union(spans))


def evaluate(events,found,near,start,end,required):
    positives=[s for s in events if s['status']=='circling'];negative=[s for s in events if s['status']=='non-circling'];unknown=[s for s in events if s['status']=='unknown']
    # Conflicting or explicitly uncertain frames cannot be treated as ground truth.
    conflict=[]
    for p in positives:
        for n in negative:
            a=max(p['start'],n['start']);b=min(p['end'],n['end'])
            if a<=b:conflict.append(dict(start=a,end=b))
    uncertain=unknown+conflict
    if uncertain:
        # Fail rather than silently altering event definitions; user selects one resolved source.
        if any(overlap(s['start'],s['end'],uncertain) for s in positives+negative):
            raise ValueError('Positive/negative labels overlap uncertain or conflicting spans; resolve or select a consistent annotation source')
    results=[]
    for s in positives:
        a=max(start,s['start']);b=min(end,s['end'])
        if a>b:continue
        total=b-a+1;covered=overlap(a,b,found);observed=sum(f in near for f in range(a,b+1))
        category='retained' if covered/total>=required else 'partially_retained' if covered else 'missed'
        results.append(dict(event=s['event'],start=a,end=b,clipped_to_scope=a!=s['start'] or b!=s['end'],coverage=covered/total,observable_fraction=observed/total,category=category,notes=s['notes']))
    candidates_out=[]
    for s in found:
        pos=overlap(s['start'],s['end'],positives);neg=overlap(s['start'],s['end'],negative)
        category='overlaps_positive' if pos else 'overlaps_verified_negative' if neg else 'unverified_candidate'
        candidates_out.append(dict(**s,category=category,positive_frames=pos,verified_negative_frames=neg))
    total=sum(s['end']-s['start']+1 for s in found)
    positive_frames=overlap(start,end,positives)
    retained_positive_frames=sum(overlap(a,b,positives) for a,b in union(found))
    return dict(annotation_events=len(results),retained_events=sum(s['category']=='retained' for s in results),event_coverage_recall=sum(s['category']=='retained' for s in results)/len(results) if results else None,positive_frame_coverage=retained_positive_frames/positive_frames if positive_frames else None,candidate_count=len(found),candidate_frames=total,scope_frames=end-start+1,observable_frames=len(near),review_load_fraction=total/(end-start+1),unverified_candidates=sum(s['category']=='unverified_candidate' for s in candidates_out)),results,candidates_out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('config');parser.add_argument('--clips',type=int,default=0,help='Render at most this many representative short review clips; can be slow');args=parser.parse_args()
    if args.clips<0:parser.error('--clips must be nonnegative')
    config=Path(args.config).resolve();cfg=json.loads(config.read_text());base=config.parent;resolve=lambda p:(base/p).resolve()
    out=resolve(cfg['out']);out.mkdir(parents=True,exist_ok=True)
    start=frame(cfg.get('start_frame',0));end=frame(cfg['end_frame']);fps=float(cfg.get('fps',30));required=float(cfg.get('required_event_coverage',.5))
    if not 0<=float(cfg.get('target_event_recall',.9))<=1:raise ValueError('target_event_recall must be in [0,1]')
    if end<start or fps<=0 or not 0<required<=1:raise ValueError('Invalid scope/FPS/coverage')
    distances=cfg.get('distance_thresholds_px',[120,160,200]);durations=cfg.get('minimum_durations_s',[1,2,4])
    if not distances or not durations or min(distances)<=0 or min(durations)<=0:raise ValueError('Thresholds/durations must be positive')
    spec=cfg['annotations'];path=resolve(spec['path']);events,skipped=load_intervals(path,cfg['video_id'],fps,spec)
    print('Reading scoped pose centers and computing closest-pair distances...',flush=True)
    near=nearest_pairs(resolve(cfg['poses']),start,end)
    summary=[];all_events=[];all_candidates=[]
    for distance,duration in itertools.product(distances,durations):
        name=f'd{distance:g}_t{duration:g}';found=candidates(near,distance,duration,fps)
        metrics,ev,ca=evaluate(events,found,near,start,end,required)
        summary.append(dict(rule=name,distance_px=distance,duration_s=duration,**metrics,target_event_recall_met=(metrics['event_coverage_recall']>=float(cfg.get('target_event_recall',.9))) if metrics['event_coverage_recall'] is not None else None))
        all_events.extend(dict(rule=name,**s) for s in ev);all_candidates.extend(dict(rule=name,**s) for s in ca)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(all_events,columns=['rule','event','start','end','clipped_to_scope','coverage','observable_fraction','category','notes']).to_csv(out/'event_coverage.csv',index=False)
    pd.DataFrame(all_candidates,columns=['rule','start','end','category','positive_frames','verified_negative_frames']).to_csv(out/'candidates.csv',index=False)
    pd.DataFrame([dict(frame=f,distance_px=v[0],fish_a=v[1],fish_b=v[2]) for f,v in sorted(near.items())],columns=['frame','distance_px','fish_a','fish_b']).to_csv(out/'nearest_pairs.csv',index=False)
    # Representative queue for a prespecified rule, not a rule selected on held-out results.
    chosen=cfg.get('review_rule',summary[0]['rule'])
    if chosen not in {s['rule'] for s in summary}:raise ValueError('review_rule not in sweep')
    queue=[]
    for category in ['missed','partially_retained','retained']:
        selected=[s for s in all_events if s['rule']==chosen and s['category']==category]
        if selected:queue.append(selected[0])
    for category in ['unverified_candidate','overlaps_verified_negative']:
        selected=[s for s in all_candidates if s['rule']==chosen and s['category']==category]
        if selected:queue.append(selected[0])
    links=[];manifest=[]
    for i,s in enumerate(queue):
        a=max(start,s['start']-round(2*fps));b=min(end,a+round(cfg.get('max_clip_seconds',20)*fps)-1,s['end']+round(2*fps))
        pairs=[tuple(sorted(v[1:])) for f,v in near.items() if a<=f<=b]
        pair=list(max(set(pairs),key=pairs.count)) if pairs else []
        item=dict(category=s['category'],source_start=s['start'],source_end=s['end'],review_start=a,review_end=b,pair=pair,rendered=False)
        if i<args.clips and len(pair)==2:
            rc=dict(video_id=cfg['video_id'],video=str(resolve(cfg['video'])) if cfg.get('video') else None,poses=str(resolve(cfg['poses'])),out=str(out/f'clip_{i+1}'),fps=fps,width=cfg.get('width',1296),height=cfg.get('height',972),start_frame=a,end_frame=b,pair=pair,prediction_scope='Proximity candidates; not a trained circling detector',tracks=[dict(spec,path=str(path),name='Selected human labels'),dict(name='Proximity candidates',path=str(out/'review_candidates.csv'),kind='prediction')])
            chosen_rows=[dict(video_id=cfg['video_id'],start_frame=c['start'],end_frame=c['end']) for c in all_candidates if c['rule']==chosen]
            pd.DataFrame(chosen_rows,columns=['video_id','start_frame','end_frame']).to_csv(out/'review_candidates.csv',index=False)
            clipcfg=out/f'clip_{i+1}.json';clipcfg.write_text(json.dumps(rc,indent=2));review(clipcfg);item['rendered']=True
            links.append(f'<li><a href="clip_{i+1}/">{html.escape(s["category"])} — frames {a}–{b}</a></li>')
        manifest.append(item)
    (out/'clip_queue.json').write_text(json.dumps(manifest,indent=2))
    report=dict(hypothesis=cfg['hypothesis'],config=cfg,annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),skipped_annotation_rows=skipped,review_rule=chosen,results=summary,limitations=['Closest pair may change each frame; this is scene-level candidate generation','No training and no proof that proximity establishes circling','Unannotated time is unknown, not false positive ground truth','Missing geometry cannot prove absence of circling','Sweep is exploratory; choose on development data then freeze for held-out evaluation','Scope is explicit; no whole-video claim for an excerpt','Representative clips are truncated when needed; full boundaries remain in CSV'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    body='<h1>Hypothesis experiment</h1><p>'+html.escape(cfg['hypothesis'])+'</p><p>Video '+html.escape(cfg['video_id'])+f' · source frames {start}–{end} · required event coverage {required:.0%}</p>'
    body+='<p>Retained = candidate intervals cover the specified fraction of an annotated event. This is candidate coverage, not detector accuracy. Extra candidates are unverified unless they overlap explicit negative labels.</p>'+pd.DataFrame(summary).to_html(index=False,float_format=lambda x:f'{x:.3f}')
    body+='<h2>Representative review clips</h2><ul>'+''.join(links)+'</ul><p>If no clips are linked, rerun with --clips 3. Clip queue and all event results are available below.</p><p>'+ ' · '.join(f'<a href="{f}">{f}</a>' for f in ['summary.csv','event_coverage.csv','candidates.csv','clip_queue.json','report.json'])+'</p><h2>Interpretation limits</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in report['limitations'])+'</ul>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Proximity hypothesis test</title><style>body{font:16px system-ui;margin:32px;max-width:1400px}table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border:1px solid #ccc}a{color:#066}</style>'+body)
    print(f'Experiment ready: {out}/index.html',flush=True)

if __name__=='__main__':main()
