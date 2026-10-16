import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from .review import geometry, load_intervals, run

class ReviewTests(unittest.TestCase):
    def test_missing_and_rejected_are_not_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.csv'
            pd.DataFrame([dict(VideoID='v',StartFrame=1,EndFrame=4,ShouldUse=True,Circling=False,Notes='reviewed negative'),dict(VideoID='v',StartFrame=5,EndFrame=8,ShouldUse=False,Circling=True),dict(VideoID='v',StartFrame=None,EndFrame=None)]).to_csv(p,index=False)
            spans,skipped=load_intervals(p,'v',30,{})
            self.assertEqual([s['status'] for s in spans],['non-circling','unknown']);self.assertEqual(skipped,1)

    def test_timestamp_and_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.csv'
            pd.DataFrame([dict(VideoID='v',StartTime='00:00:01',EndTime='00:00:02',Behavior='Circling',**{'2nd reviewer comments':'split?'})]).to_csv(p,index=False)
            spans,_=load_intervals(p,'v',30,{'verified':True})
            self.assertEqual((spans[0]['start'],spans[0]['end']),(30,59));self.assertIn('split?',spans[0]['notes'])

    def test_no_derivative_across_gap(self):
        p=pd.DataFrame([dict(FrameNum=f,TrackUID=t,X_center=x,Y_center=y) for f,x,y in [(0,1,0),(2,0,1)] for t,x,y in [('a',0,0),('b',x,y)]])
        g=geometry(p,['a','b'],0,2,30)
        self.assertTrue(g.angular_velocity_rad_s.isna().all())
        self.assertEqual(g.distance_px.iloc[0],1)

    def test_real_sources(self):
        root=Path(__file__).resolve().parents[1]
        p=root/'ground_truth_annotations/finalized_annotations/circling_annotations_vid0031.csv'
        spans,skipped=load_intervals(p,'0031_vid',30,{'verified':True})
        self.assertGreater(len(spans),0);self.assertGreater(skipped,0)
        spans,_=load_intervals(root/'outputs/predictions/rule_based/0031_vid.csv','0031_vid',30,{'kind':'prediction'})
        self.assertTrue(all(s['status']=='prediction' for s in spans))

    def test_export_strict_json_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pd.DataFrame([dict(FrameNum=f,TrackUID=t,X_center=x,Y_center=10,Nose_x=x,Nose_y=10) for f in range(3) for t,x in [(1,10),(2,20)]]).to_parquet(root/'pose.parquet')
            cfg=dict(video_id='</script>',poses='pose.parquet',out='review',start_frame=0,end_frame=2,pair=['1','2'],width=100,height=100)
            (root/'config.json').write_text(json.dumps(cfg));run(root/'config.json')
            data=json.loads((root/'review/review_data.json').read_text());self.assertEqual(data['features'][0]['distance_px'],10)
            content=(root/'review/index.html').read_text();self.assertIn('\\u003c/script>',content)

if __name__=='__main__':unittest.main()
