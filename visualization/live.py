"""Full video with on-demand pose overlays. python -m visualization.live CONFIG"""
import argparse,json,math,re,subprocess,threading,mimetypes
from collections import OrderedDict
from functools import partial
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .live_intervals import load_intervals,KEYPOINTS,MIDLINE

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,list): return [clean(v) for v in x]
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer): return int(x)
    return x

def byte_range(header,size):
    if not header:return 0,size-1,False
    m=re.fullmatch(r'bytes=(\d*)-(\d*)',header)
    if not m or not any(m.groups()):raise ValueError('Invalid range')
    a,b=m.groups()
    if not a:
        if int(b)<=0:raise ValueError('Invalid suffix')
        return max(0,size-int(b)),size-1,True
    start=int(a);end=min(int(b) if b else size-1,size-1)
    if start>=size or end<start:raise ValueError('Unsatisfiable range')
    return start,end,True

class Recording:
    def __init__(self,config):
        p=Path(config).resolve();self.cfg=json.loads(p.read_text());self.base=p.parent
        self.video=self.resolve(self.cfg['video']);self.poses=self.resolve(self.cfg['poses'])
        for f in (self.video,self.poses):
            if not f.is_file():raise ValueError(f'Missing input: {f}')
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate:format=duration','-of','json',str(self.video)]))
        s=info['streams'][0];self.fps=float(self.cfg.get('fps',30));a,b=map(float,s['avg_frame_rate'].split('/'))
        if not math.isfinite(self.fps) or self.fps<=0 or abs(a/b-self.fps)>.01:raise ValueError('Video average FPS differs from configured FPS')
        self.schema=pq.ParquetFile(self.poses).schema_arrow.names
        missing={'FrameNum','TrackUID','X_center','Y_center'}-set(self.schema)
        if missing:raise ValueError(f'Missing pose columns: {missing}')
        self.keys=[k for k in KEYPOINTS if k+'_x' in self.schema and k+'_y' in self.schema]
        tracks=[]
        for spec in self.cfg.get('tracks',[]):
            spans,skipped=load_intervals(self.resolve(spec['path']),self.cfg['video_id'],self.fps,spec)
            tracks.append(dict(name=spec['name'],kind=spec.get('kind','annotation'),spans=spans,skipped=skipped))
        self.scores={};spec=self.cfg.get('scores')
        if spec:
            f=self.resolve(spec['path']);df=pd.read_parquet(f) if f.suffix=='.parquet' else pd.read_csv(f);vc=spec.get('video','video_id')
            if vc in df:df=df[df[vc].astype(str)==self.cfg['video_id']]
            fc,sc=spec.get('frame','frame'),spec.get('score','score')
            if df[fc].duplicated().any():raise ValueError('Duplicate score frames')
            for f,scr in zip(df[fc],df[sc]):
                if pd.isna(scr):continue
                if not np.isfinite(scr) or not 0<=scr<=1 or not np.isfinite(f) or f<0 or int(f)!=f:raise ValueError('Invalid score or frame')
                self.scores[str(int(f))]=float(scr)
        self.meta=dict(video_id=self.cfg['video_id'],fps=self.fps,width=s['width'],height=s['height'],duration=float(info['format']['duration']),keys=self.keys,skeleton=list(zip(MIDLINE[:-1],MIDLINE[1:])),tracks=tracks,start_frame=int(self.cfg.get('start_frame',0)),chunk_frames=round(self.fps*10))
        self.cache=OrderedDict();self.lock=threading.Lock()
    def resolve(self,p):return (self.base/p).resolve()
    def chunk(self,index):
        if index<0 or index>math.ceil(self.meta['duration']/10):raise ValueError('Chunk out of range')
        with self.lock:
            if index in self.cache:self.cache.move_to_end(index);return self.cache[index]
            start=index*self.meta['chunk_frames'];end=start+self.meta['chunk_frames']-1
            cols=['FrameNum','TrackUID','X_center','Y_center']+[k+s for k in self.keys for s in ('_x','_y')]+[c for c in ('day_label','project_id') if c in self.schema]
            df=pd.read_parquet(self.poses,columns=cols,filters=[('FrameNum','>=',max(0,start-1)),('FrameNum','<=',end)])
            if df.duplicated(['FrameNum','TrackUID']).any():raise ValueError('Duplicate frame/TrackUID')
            if len(df) and 'day_label' in df and set(df.day_label.dropna().astype(str))!={self.cfg['video_id']}:raise ValueError('Pose video ID mismatch')
            expected=self.cfg.get('expected_project_id')
            if expected and len(df) and 'project_id' in df and set(df.project_id.dropna().astype(str))!={expected}:raise ValueError('Pose project mismatch')
            frames={}
            for r in df.to_dict('records'):
                frames.setdefault(str(int(r['FrameNum'])),[]).append(dict(id=str(r['TrackUID']),center=[r['X_center'],r['Y_center']],points=[[r[k+'_x'],r[k+'_y']] for k in self.keys]))
            obj=dict(start=start,end=end,frames=frames,scores={str(f):self.scores[str(f)] for f in range(start,end+1) if str(f) in self.scores})
            raw=json.dumps(clean(obj),allow_nan=False).encode();self.cache[index]=raw
            while len(self.cache)>6:self.cache.popitem(last=False)
            return raw

class Handler(BaseHTTPRequestHandler):
    def __init__(self,*a,recording,**kw):self.rec=recording;super().__init__(*a,**kw)
    def log_message(self,*a):pass
    def do_HEAD(self):self.serve(True)
    def do_GET(self):self.serve(False)
    def serve(self,head=False):
        url=urlparse(self.path)
        try:
            if url.path=='/video':
                size=self.rec.video.stat().st_size
                try:a,b,part=byte_range(self.headers.get('Range'),size)
                except ValueError:
                    self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return
                self.send_response(206 if part else 200);self.send_header('Content-Type',mimetypes.guess_type(str(self.rec.video))[0] or 'application/octet-stream');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(b-a+1))
                if part:self.send_header('Content-Range',f'bytes {a}-{b}/{size}')
                self.end_headers()
                if not head:
                    with self.rec.video.open('rb') as f:
                        f.seek(a);remain=b-a+1
                        while remain:
                            data=f.read(min(remain,1024*1024))
                            if not data:break
                            self.wfile.write(data);remain-=len(data)
                return
            if url.path=='/':data=Path(__file__).with_name('live.html').read_bytes();typ='text/html; charset=utf-8'
            elif url.path=='/api/meta':data=json.dumps(self.rec.meta,allow_nan=False).encode();typ='application/json'
            elif url.path=='/api/chunk':data=self.rec.chunk(int(parse_qs(url.query).get('index',['0'])[0]));typ='application/json'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',typ);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers()
            if not head:self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as e:
            data=json.dumps({'error':str(e)}).encode();self.send_response(400);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers()
            if not head:self.wfile.write(data)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');p.add_argument('--port',type=int,default=8001);a=p.parse_args()
    rec=Recording(a.config);server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,recording=rec));server.daemon_threads=True
    print(f'Open http://localhost:{a.port} — original full video, no rendering. Ctrl+C to stop.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
