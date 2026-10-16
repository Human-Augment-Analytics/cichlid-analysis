"""DLC2Action 1.0 MS-TCN3 model API + an explicit PyTorch training/inference loop."""
import argparse
import importlib.metadata
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from .learning_data import prepare,continuous_runs,signature


def runtime():
    import torch
    if importlib.metadata.version('dlc2action')!='1.0':
        raise RuntimeError('This adapter is tested with dlc2action==1.0; install the pinned version')
    from dlc2action.model.ms_tcn import MS_TCN3
    return torch,MS_TCN3


def load_videos(data,split=None):
    meta=json.loads((Path(data)/'manifest.json').read_text());result=[]
    for rec in meta['videos']:
        if split is not None and rec['split']!=split:continue
        with np.load(Path(data)/rec['file'],allow_pickle=False) as z:
            result.append((rec,{k:z[k].copy() for k in ['x','y','frames','valid']}))
    return meta,result


def fit_preprocessing(videos):
    examples=[z['x'][(z['y']!=-100)&z['valid']] for _,z in videos]
    examples=np.concatenate(examples) if examples else np.empty((0,0))
    if not len(examples):raise ValueError('No verified supervised training frames')
    median=np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0 for col in examples.T])
    filled=np.where(np.isfinite(examples),examples,median)
    mean=filled.mean(0);scale=filled.std(0);scale[scale<1e-6]=1
    return dict(median=median.tolist(),mean=mean.tolist(),scale=scale.tolist())


def transform(x,pre):
    finite=np.isfinite(x);filled=np.where(finite,x,np.array(pre['median']))
    z=(filled-np.array(pre['mean']))/np.array(pre['scale'])
    return np.column_stack([z,(~finite).astype(np.float32)]).astype(np.float32)


def make_model(config,device):
    torch,MS_TCN3=runtime()
    return MS_TCN3(num_f_maps=config['maps'],num_classes=2,exclusive=True,
                   dims={'loaded':torch.Size([config['features']])},num_layers_R=config['layers'],
                   num_R=1,num_layers_PG=config['layers'],dropout_rate=.2,PG_in_FE=True).to(device)


def training_windows(videos,pre,length):
    examples=[]
    for _,z in videos:
        x=transform(z['x'],pre)
        for run in continuous_runs(z['valid'],z['frames']):
            for offset in range(0,len(run),length):
                idx=run[offset:offset+length]
                if (z['y'][idx]!=-100).any():examples.append((x[idx],z['y'][idx]))
    return examples


def train(data,out,epochs=10,window=256,maps=32,layers=4,lr=.001,seed=42,device='cpu',batch_size=4):
    if min(epochs,window,maps,layers,batch_size)<1 or window<8 or lr<=0:
        raise ValueError('Positive hyperparameters required; window must be at least 8')
    torch,_=runtime();torch.set_num_threads(min(4,torch.get_num_threads()))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    meta,videos=load_videos(data,'train');pre=fit_preprocessing(videos)
    labels=np.concatenate([z['y'][z['y']!=-100] for _,z in videos]);counts=np.bincount(labels,minlength=2)
    if (counts==0).any():raise ValueError('Training needs verified circling AND verified non-circling. Annotate negatives or explicitly enable assumed background in manifest.')
    config=dict(features=len(meta['feature_names'])*2,maps=maps,layers=layers,window=window)
    model=make_model(config,device);optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    weights=torch.tensor(len(labels)/(2*counts),dtype=torch.float32,device=device)
    examples=training_windows(videos,pre,window)
    if not examples:raise ValueError('No supervised windows after tracking quality filtering')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);history=[]
    for epoch in range(epochs):
        model.train();order=np.random.permutation(len(examples));total=0;steps=0
        for offset in range(0,len(order),batch_size):
            batch=[examples[i] for i in order[offset:offset+batch_size]]
            bx=np.zeros((len(batch),config['features'],window),np.float32);by=np.full((len(batch),window),-100,np.int64)
            for i,(x,y) in enumerate(batch):bx[i,:,:len(x)]=x.T;by[i,:len(y)]=y
            x=torch.from_numpy(bx).to(device);y=torch.from_numpy(by).to(device)
            logits,_=model(x,[])
            if logits.ndim!=4:raise RuntimeError(f'Unexpected model output {tuple(logits.shape)}')
            loss=sum(torch.nn.functional.cross_entropy(stage,y,weight=weights,ignore_index=-100) for stage in logits)/len(logits)
            if not torch.isfinite(loss):raise RuntimeError('Training loss is not finite')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=loss.item();steps+=1
        history.append(dict(epoch=epoch+1,training_loss=total/steps));print(f"Epoch {epoch+1}/{epochs}: loss {total/steps:.4f}",flush=True)
    # Fixed-epoch checkpoint: no held-out evaluation used for model selection.
    checkpoint=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config,preprocessing=pre,
                    feature_names=meta['feature_names'],dlc2action_version=importlib.metadata.version('dlc2action'),
                    torch_version=str(torch.__version__),training_videos=[r['video_id'] for r,_ in videos],
                    training_pose_hashes=[r['pose_sha256'] for r,_ in videos])
    torch.save(checkpoint,out/'model.pt')
    (out/'history.json').write_text(json.dumps(history,indent=2))
    (out/'training_manifest.json').write_text(json.dumps(meta,indent=2))
    settings=dict(config,epochs=epochs,seed=seed,lr=lr,batch_size=batch_size,device=device,class_counts=counts.tolist(),
                  class_weights=weights.cpu().tolist(),preprocessing=pre,feature_names=meta['feature_names'],
                  dlc2action_version=checkpoint['dlc2action_version'],torch_version=checkpoint['torch_version'],
                  note='Custom PyTorch loop around DLC2Action MS_TCN3; weighted cross-entropy at both stages; no SSL or temporal smoothing loss; fixed last epoch')
    (out/'settings.json').write_text(json.dumps(settings,indent=2));print(f'Saved trained DLC2Action model → {out.resolve()}')


def predict_run(model,x,torch,device,window):
    scores=np.zeros(len(x));weights=np.zeros(len(x));stride=max(1,window//2)
    # Overlap average, independently per continuous tracking run; no gap crossing.
    for start in range(0,len(x),stride):
        end=min(len(x),start+window);batch=np.zeros((1,x.shape[1],window),np.float32);batch[0,:,:end-start]=x[start:end].T
        with torch.no_grad():
            logits,_=model(torch.from_numpy(batch).to(device),[])
            p=logits[-1].softmax(1)[0,1,:end-start].cpu().numpy()
        blend=np.minimum(np.arange(1,end-start+1),np.arange(end-start,0,-1)).astype(float)
        scores[start:end]+=p*blend;weights[start:end]+=blend
        if end==len(x):break
    return scores/weights


def frame_metrics(y,score,threshold):
    mask=(y!=-100)&np.isfinite(score);yy=y[mask];pp=score[mask]>=threshold
    tp=int(((yy==1)&pp).sum());fp=int(((yy==0)&pp).sum());fn=int(((yy==1)&~pp).sum());tn=int(((yy==0)&~pp).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                evaluated_frames=int(mask.sum()),unknown_or_missing_frames=int((~mask).sum()),threshold=threshold,
                scope='Frame-level metrics on supervised labels and valid pair tracking only; not event-level performance')


def predict(data,model_dir,out,split='test',device='cpu',threshold=.5):
    if not 0<=threshold<=1:raise ValueError('Threshold must be in [0,1]')
    torch,_=runtime();torch.set_num_threads(min(4,torch.get_num_threads()))
    ck=torch.load(Path(model_dir)/'model.pt',map_location='cpu',weights_only=True)
    meta,videos=load_videos(data,None if split=='all' else split)
    if not videos:raise ValueError(f'No videos for split {split}')
    if meta['feature_names']!=ck['feature_names']:raise ValueError('Feature definitions do not match checkpoint')
    model=make_model(ck['config'],device);model.load_state_dict(ck['state_dict'],strict=True);model.eval()
    out=Path(out);out.mkdir(parents=True,exist_ok=True);tables=[];segments=[];metrics={}
    for rec,z in videos:
        if rec['split']!='train' and (rec['video_id'] in ck['training_videos'] or rec['pose_sha256'] in ck['training_pose_hashes']):
            raise ValueError('Held-out video overlaps checkpoint training inputs')
        x=transform(z['x'],ck['preprocessing']);score=np.full(len(x),np.nan)
        for idx in continuous_runs(z['valid'],z['frames']):
            score[idx]=predict_run(model,x[idx],torch,device,ck['config']['window'])
        table=pd.DataFrame({'video_id':rec['video_id'],'frame':z['frames'],'score':score,'split':rec['split'],'label':z['y']})
        tables.append(table);metrics[rec['video_id']]=frame_metrics(z['y'],score,threshold)
        metrics[rec['video_id']]['split']=rec['split'];metrics[rec['video_id']]['assumed_background']=meta['assume_unannotated_background']
        for idx in continuous_runs(np.isfinite(score)&(score>=threshold),z['frames']):
            segments.append(dict(video_id=rec['video_id'],start_frame=int(z['frames'][idx[0]]),end_frame=int(z['frames'][idx[-1]]),mean_score=float(score[idx].mean())))
    pd.concat(tables,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(segments,columns=['video_id','start_frame','end_frame','mean_score']).to_csv(out/'predicted_segments.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (out/'prediction_settings.json').write_text(json.dumps(dict(model_sha256=signature(Path(model_dir)/'model.pt'),split=split,threshold=threshold,device=device,mode='Offline noncausal; 50% overlap window averaging; invalid-pair gaps split inference'),indent=2))
    print(f'Predicted {len(videos)} videos → {out.resolve()}')

