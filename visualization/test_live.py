"""Run: python -m unittest visualization.test_live"""
import json,tempfile,threading,unittest,urllib.request,urllib.error
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from .live import Recording,Handler,byte_range
from .live_intervals import load_intervals

class Tests(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual(byte_range('bytes=5-9',100),(5,9,True))
        self.assertEqual(byte_range('bytes=-10',100),(90,99,True))
        self.assertEqual(byte_range('bytes=90-',100),(90,99,True))
        for h in ['bytes=100-','bytes=9-5','bytes=1-2,4-5','bytes=-0']:
            with self.assertRaises(ValueError):byte_range(h,100)
    def test_interval_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.csv';pd.DataFrame([{'VideoID':'v','Start Frame':2,'End Frame':4,'Primary label/behavior':'approx. circling','Verified':True}]).to_csv(p,index=False)
            spans,_=load_intervals(p,'v',30,{})
            self.assertEqual(spans[0]['status'],'unknown')
    def test_chunk_missing_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pose.parquet';pd.DataFrame([dict(FrameNum=f,TrackUID=i,X_center=float(i),Y_center=float('nan'),day_label='v') for f in [299,300,301,599,600] for i in [1,2]]).to_parquet(p)
            r=Recording.__new__(Recording);r.poses=p;r.cfg={'video_id':'v'};r.schema=['FrameNum','TrackUID','X_center','Y_center','day_label'];r.keys=[];r.meta={'duration':30,'chunk_frames':300};r.cache={};r.lock=threading.Lock();r.scores={}
            from collections import OrderedDict
            r.cache=OrderedDict();d=json.loads(r.chunk(1));self.assertIn('299',d['frames']);self.assertNotIn('600',d['frames']);self.assertIsNone(d['frames']['300'][0]['center'][1])
            r.cfg['video_id']='wrong';r.cache.clear()
            with self.assertRaises(ValueError):r.chunk(1)
    def test_http_seek_and_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'v.mp4';p.write_bytes(bytes(range(100)));rec=SimpleNamespace(video=p,meta={'test':True})
            server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,recording=rec));t=threading.Thread(target=server.serve_forever,daemon=True);t.start();url=f'http://127.0.0.1:{server.server_port}'
            try:
                with urllib.request.urlopen(urllib.request.Request(url+'/video',headers={'Range':'bytes=30-39'})) as response:
                    self.assertEqual(response.status,206);self.assertEqual(response.read(),bytes(range(30,40)));self.assertEqual(response.headers['Content-Range'],'bytes 30-39/100')
                with urllib.request.urlopen(urllib.request.Request(url+'/video',method='HEAD')) as response:self.assertEqual(response.headers['Content-Length'],'100');self.assertEqual(response.read(),b'')
                with self.assertRaises(urllib.error.HTTPError) as err:urllib.request.urlopen(urllib.request.Request(url+'/video',headers={'Range':'bytes=100-'}))
                self.assertEqual(err.exception.code,416)
            finally:server.shutdown();server.server_close()
if __name__=='__main__':unittest.main()
