"""Optional DLC2Action MS-TCN training, prediction, and common-viewer integration."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .review import load_intervals, run as review
from .training_backend.scene_features import aggregate_pairs
from .training_backend.learning_data import prepare, signature


def setup(config,out,assume_background=False):
    config=Path(config).resolve();cfg=json.loads(config.read_text());base=config.parent
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    entries=[];audit=[];originals=[]
    for i,rec in enumerate(cfg['videos']):
        if rec['split'] not in {'train','val','test'}:raise ValueError('Explicit train/val/test split required')
        original=(base/rec['pairs']).resolve();originals.append((rec['split'],signature(original)))
        ann=rec.get('annotations');spans=[];skipped=0
        if ann:spans,skipped=load_intervals((base/ann['path']).resolve(),rec['video_id'],float(cfg.get('fps',30)),ann)
        data=pd.read_parquet(original)
        if data.duplicated(['FrameNum','TrackUID_1','TrackUID_2']).any():raise ValueError('Duplicate frame/pair rows')
        # Sparse contexts around ALL annotations (including explicit negatives and unknowns).
        # An unannotated test video instead uses its full available range unless scoped.
        if rec.get('start_frame') is not None:data=data[data.FrameNum>=int(rec['start_frame'])]
        if rec.get('end_frame') is not None:data=data[data.FrameNum<=int(rec['end_frame'])]
        if cfg.get('annotation_context_seconds') is not None and spans:
            pad=round(float(cfg['annotation_context_seconds'])*float(cfg.get('fps',30)))
            if pad<0:raise ValueError('Context seconds must be nonnegative')
            keep=np.zeros(len(data),dtype=bool)
            for s in spans:keep|=data.FrameNum.between(max(0,s['start']-pad),s['end']+pad).to_numpy()
            data=data.loc[keep]
        if not len(data):raise ValueError(f'No feature rows for {rec["video_id"]}')
        feat=aggregate_pairs(data);featurepath=out/f'features_{i}.parquet';feat.to_parquet(featurepath,index=False)
        schema=dict(fps=cfg.get('fps',30),width=cfg.get('width',1296),height=cfg.get('height',972),pairs=dict(frame='FrameNum',features=[c for c in feat if c!='FrameNum']))
        schema_path=out/f'schema_{i}.json';schema_path.write_text(json.dumps(schema,indent=2))
        ap=None
        if ann:
            ap=out/f'annotations_{i}.csv'
            pd.DataFrame([dict(video_id=rec['video_id'],event_id=s['event'],start_frame=s['start'],end_frame=s['end'],Verified=s['status']!='unknown',behavior='non-circling' if s['status']=='non-circling' else 'circling',notes=s['notes']) for s in spans],columns=['video_id','event_id','start_frame','end_frame','Verified','behavior','notes']).to_csv(ap,index=False)
        entries.append(dict(video_id=rec['video_id'],split=rec['split'],pairs=str(featurepath),schema=str(schema_path),annotations=str(ap) if ap else None,pair=['scene','all_pairs']))
        audit.append(dict(video_id=rec['video_id'],original_pair_path=str(original),original_pair_sha256=originals[-1][1],annotation_path=str((base/ann['path']).resolve()) if ann else None,annotation_sha256=signature((base/ann['path']).resolve()) if ann else None,skipped_incomplete_rows=skipped,frames=len(feat)))
    for i,(split,h) in enumerate(originals):
        for other,hh in originals[i+1:]:
            if h==hh and split!=other:raise ValueError('Same original pair file supplied across splits')
    manifest=out/'manifest.json';manifest.write_text(json.dumps(dict(assume_unannotated_background=assume_background,videos=entries),indent=2))
    (out/'setup_provenance.json').write_text(json.dumps(dict(config=cfg,inputs=audit,feature_definition='52 scene-level features: mean/min/max of distances, rates, sin/cos angles, plus pair_count',assume_unannotated_background=assume_background,warning='Annotation-context sampling is not whole-video evaluation; no pair attribution; labels may reflect an older biological definition'),indent=2))
    prepare(manifest,out/'prepared')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('config');a.add_argument('--out',default='outputs/visualization/learning');a.add_argument('--assume-background',action='store_true')
    a=sub.add_parser('train');a.add_argument('--run',default='outputs/visualization/learning');a.add_argument('--epochs',type=int,default=10);a.add_argument('--device',default='cpu')
    a=sub.add_parser('predict');a.add_argument('--run',default='outputs/visualization/learning');a.add_argument('--model');a.add_argument('--split',choices=['train','val','test','all'],default='test');a.add_argument('--device',default='cpu');a.add_argument('--threshold',type=float,default=.5)
    a=sub.add_parser('review');a.add_argument('config',help='Ordinary visualization review config');a.add_argument('--run',default='outputs/visualization/learning');a.add_argument('--out',default='outputs/visualization/learned_review')
    args=p.parse_args()
    if args.command=='prepare':setup(args.config,args.out,args.assume_background);return
    root=Path(args.run).resolve()
    if args.command=='train':
        from .training_backend.learning import train
        train(root/'prepared',root/'model',epochs=args.epochs,device=args.device);return
    if args.command=='predict':
        from .training_backend.learning import predict
        predict(root/'prepared',Path(args.model).resolve() if args.model else root/'model',root/'predictions',split=args.split,device=args.device,threshold=args.threshold);return
    config=Path(args.config).resolve();cfg=json.loads(config.read_text())
    # Convert original config-relative inputs before moving the derived config to run/.
    for key in ['video','poses']:
        if cfg.get(key):cfg[key]=str((config.parent/cfg[key]).resolve())
    for track in cfg.get('tracks',[]):track['path']=str((config.parent/track['path']).resolve())
    predictions=root/'predictions/predictions.csv';segments=root/'predictions/predicted_segments.csv'
    if not predictions.is_file():raise FileNotFoundError('Run predict before learned review')
    selected=pd.read_csv(predictions,usecols=['video_id','frame','score'])
    selected=selected[(selected.video_id==cfg['video_id'])&selected.frame.between(cfg['start_frame'],cfg['end_frame'])]
    if not selected.score.notna().any():raise ValueError('No model scores in this review window; check video ID, split, prepared annotation contexts, and scope')
    cfg['scores']=[dict(path=str(predictions),frame='frame',score='score',video='video_id')]
    cfg.setdefault('tracks',[]).append(dict(name='Learned MS-TCN intervals',path=str(segments),kind='prediction'))
    cfg['prediction_scope']='Learned scene-level circling; selected fish pair is diagnostic only';cfg['out']=str(Path(args.out).resolve())
    derived=root/'learned_review_config.json';derived.write_text(json.dumps(cfg,indent=2));review(derived)

if __name__=='__main__':main()
