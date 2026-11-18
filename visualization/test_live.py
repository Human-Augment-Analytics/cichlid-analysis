"""Catalog, input isolation, byte-range and remote-proxy integration checks."""
from functools import partial
from http.server import ThreadingHTTPServer
import io,json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .catalog import Catalog,source_url
from .live import App,Handler,byte_range
from .live_intervals import load_intervals

class Fixture:
 def __init__(self,root):
  self.root=Path(root);(self.root/'visualization').mkdir();(self.root/'data').mkdir();self.config=self.root/'visualization/recordings.json';self.pose=self.root/'data/a_vid.parquet'
  pd.DataFrame([dict(FrameNum=f,TrackUID=i,X_center=i*100.,Y_center=np.nan if f==301 else 10.,day_label='a_vid',project_id='projectA',Nose_x=float(i),Nose_y=10.) for f in [0,1,299,300,301,599,600] for i in [1,2]]).to_parquet(self.pose,index=False)
  self.pred=self.root/'outputs/predictions/logistic_regression';self.pred.mkdir(parents=True);pd.DataFrame([dict(VideoID='a_vid',StartFrame=300,EndFrame=500)]).to_csv(self.pred/'a_vid.csv',index=False)
  self.scores=self.root/'outputs/circling_baseline/scores';self.scores.mkdir(parents=True);pd.DataFrame({'FrameNum':[300,301],'circling_score':[.8,np.nan]}).to_parquet(self.scores/'a_vid_scores.parquet')
  fd=self.root/'outputs/pair_motion';fd.mkdir();pd.DataFrame([dict(day_label=vid,project_id=proj,FrameNum=300,TrackUID_1=1,TrackUID_2=2,centroid_distance_px=d,w4s_close_frac=.9) for vid,proj,d in [('a_vid','projectA',42.),('other','projectB',999.)]]).to_parquet(fd/'pair_features_frame.parquet')
  self.raw=dict(repo_root='..',default_recording='a',recordings=[dict(id='a',video_id='a_vid',expected_project_id='projectA',video='../data/a_vid.mp4',poses='../data/a_vid.parquet',video_url='https://example.test/a.mp4',fps=30,width=640,height=480,frame_count=900,discover_predictions=True,tracks=[dict(name='Missing reference',path='../missing.csv')]),dict(id='b',video_id='other',video_url='https://example.test/b.mp4',fps=30,width=640,height=480,frame_count=900)])
  self.save()
 def save(self):self.config.write_text(json.dumps(self.raw))

class Tests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.fx=Fixture(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_byte_ranges(self):
  self.assertEqual(byte_range('bytes=4-8',100),(4,8,True));self.assertEqual(byte_range('bytes=-5',100),(95,99,True));self.assertEqual(byte_range('bytes=90-',100),(90,99,True))
  for h in ['bytes=100-','bytes=9-5','bytes=-0','bytes=1-2,4-5']:
   with self.assertRaises(ValueError):byte_range(h,100)
 def test_catalog_sources(self):
  cat=Catalog(self.fx.config);r=cat.get('a');self.assertEqual(r.media('auto')[0],'dropbox')
  with self.assertRaises(ValueError):r.media('local')
  (self.fx.root/'data/a_vid.mp4').write_bytes(b'test');self.assertEqual(r.media('auto')[0],'local');self.assertEqual(r.media('dropbox')[0],'dropbox');self.assertEqual(len(r.tracks),1);self.assertTrue(r.warnings)
  with self.assertRaises(ValueError):cat.get('unknown')
 def test_chunk_discovery_and_gaps(self):
  d=json.loads(Catalog(self.fx.config).get('a').chunk(1));self.assertIn('299',d['frames']);self.assertNotIn('600',d['frames']);self.assertIsNone(d['frames']['301'][0]['center'][1]);self.assertEqual(d['scores']['score0'],{'300':.8});self.assertEqual(len(d['features']),1);self.assertEqual(d['features'][0]['values']['centroid_distance_px'],42.)
  empty=json.loads(Catalog(self.fx.config).get('b').chunk(1));self.assertEqual(empty['frames'],{});self.assertEqual(empty['scores'],{})
 def test_pose_identity_metadata(self):
  t=pq.read_table(self.fx.pose);pq.write_table(t.replace_schema_metadata({b'project_id':b'wrong'}),self.fx.pose);r=Catalog(self.fx.config).get('a');self.assertFalse(r.layers);self.assertTrue(any('metadata' in w for w in r.warnings))
 def test_duplicate_pose(self):
  df=pd.read_parquet(self.fx.pose);pd.concat([df,df.iloc[:1]]).to_parquet(self.fx.pose,index=False)
  with self.assertRaisesRegex(ValueError,'Duplicate pose'):Catalog(self.fx.config).get('a').chunk(0)
 def test_duplicate_scores_isolated(self):
  pd.DataFrame({'FrameNum':[300,300],'circling_score':[.2,.3]}).to_parquet(self.fx.scores/'a_vid_scores.parquet');d=json.loads(Catalog(self.fx.config).get('a').chunk(1));self.assertEqual(d['scores'],{});self.assertIn('Duplicate score',d['warnings'][0]);self.assertTrue(d['frames'])
 def test_cleaned_layer(self):
  self.fx.raw['recordings'][0]['cleaned_poses']='../data/a_vid.parquet';self.fx.save();r=Catalog(self.fx.config).get('a');self.assertIn('cleaned',r.layers);self.assertIn('300',json.loads(r.chunk(1,'cleaned'))['frames'])
 def test_unknown_recording_does_not_get_seed_models(self):
  self.fx.raw['recordings'][0].pop('discover_predictions');self.fx.save();r=Catalog(self.fx.config).get('a');self.assertFalse(r.tracks);self.assertFalse(r.score_specs);self.assertFalse(r.feature_names)
 def test_unknown_labels_and_url(self):
  p=self.fx.root/'labels.csv';pd.DataFrame([dict(VideoID='a_vid',StartFrame=1,EndFrame=2,ShouldUse=False,Circling=True)]).to_csv(p,index=False);spans,_=load_intervals(p,'a_vid',30,{});self.assertEqual(spans[0]['status'],'unknown');self.assertIn('raw=1',source_url('https://www.dropbox.com/scl/fi/file/video.mp4?dl=0&rlkey=key'))
  with self.assertRaises(ValueError):source_url('file:///etc/passwd')
 def server(self):
  app=App(self.fx.config);server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,app=app));server.daemon_threads=True;threading.Thread(target=server.serve_forever,daemon=True).start();return server,f'http://127.0.0.1:{server.server_port}'
 def test_http_api_and_local_range(self):
  (self.fx.root/'data/a_vid.mp4').write_bytes(bytes(range(100)));server,url=self.server()
  try:
   with urlopen(url+'/api/catalog') as r:self.assertEqual(len(json.load(r)['recordings']),2)
   with urlopen(Request(url+'/video?recording=a&source=local',headers={'Range':'bytes=30-39'})) as r:self.assertEqual(r.status,206);self.assertEqual(r.read(),bytes(range(30,40)))
   with urlopen(Request(url+'/video?recording=a&source=local',method='HEAD')) as r:self.assertEqual(r.headers['Content-Length'],'100');self.assertEqual(r.read(),b'')
   with urlopen(url+'/api/chunk?recording=b&index=1') as r:self.assertEqual(json.load(r)['frames'],{})
   with self.assertRaises(HTTPError):urlopen(url+'/api/chunk?recording=../escape&index=1')
  finally:server.shutdown();server.server_close()
 def test_proxy_ranges_and_ignored_range(self):
  class Upstream(io.BytesIO):
   status=206
   headers={'Content-Type':'video/mp4','Content-Range':'bytes 10-19/100','Content-Length':'10','Accept-Ranges':'bytes'}
  seen=[]
  def upstream(req,timeout):seen.append(req.get_header('Range'));return Upstream(b'0123456789')
  server,url=self.server()
  try:
   with patch('visualization.live.urlopen',upstream):
    with urlopen(Request(url+'/video?recording=a&source=dropbox',headers={'Range':'bytes=10-19'})) as r:self.assertEqual(r.status,206);self.assertEqual(r.read(),b'0123456789')
   self.assertEqual(seen,['bytes=10-19']);Upstream.status=200
   with patch('visualization.live.urlopen',upstream):
    with self.assertRaises(HTTPError) as err:urlopen(Request(url+'/video?recording=a&source=dropbox',headers={'Range':'bytes=10-19'}))
    self.assertEqual(err.exception.code,502)
  finally:server.shutdown();server.server_close()
 def test_legacy_single_config(self):
  self.fx.config.write_text(json.dumps(self.fx.raw['recordings'][0]));cat=Catalog(self.fx.config);self.assertEqual(len(cat.entries),1);self.assertEqual(cat.get().id,'a')

if __name__=='__main__':unittest.main()
