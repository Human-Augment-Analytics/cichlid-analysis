import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from .export_video import reference_at,pair_geometry,read_intervals,render,select_pair
from .demo_export import make_inputs

class ExportVideoTests(unittest.TestCase):
    def test_unknown_and_conflicting_labels(self):
        self.assertEqual(reference_at([],5),'UNKNOWN / UNLABELED')
        spans=[dict(start=0,end=9,status='circling'),dict(start=5,end=6,status='unknown')]
        self.assertEqual(reference_at(spans,3),'CIRCLING');self.assertEqual(reference_at(spans,5),'UNKNOWN / UNLABELED')
        spans.append(dict(start=0,end=9,status='non-circling'));self.assertEqual(reference_at(spans,3),'UNKNOWN / UNLABELED')

    def test_pair_gaps_and_clockwise_rate(self):
        p=pd.DataFrame([dict(FrameNum=f,TrackUID=t,X_center=x,Y_center=y) for f,x,y in [(0,1,0),(1,0,1),(3,-1,0)] for t,x,y in [(1,0,0),(2,x,y)]])
        g=pair_geometry(p,['1','2'],0,3,30)
        self.assertAlmostEqual(g.angular_velocity_rad_s.iloc[1],np.pi/2*30)
        self.assertTrue(np.isnan(g.angular_velocity_rad_s.iloc[3]))

    def test_provisional_or_approximate_not_auto_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.csv';p.write_text('VideoID,Start Frame,End Frame,Primary label/behavior\nv,0,9,approx. circling\n')
            spans,_=read_intervals(p,'v','annotation',{'verified':True})
            self.assertEqual(spans[0]['status'],'unknown')

    def test_auto_pair_uses_prediction_window(self):
        rows=[]
        for f in range(4):
            for track,x in [(1,0),(2,100 if f<2 else 1),(3,1 if f<2 else 100)]:
                rows.append(dict(FrameNum=f,TrackUID=track,X_center=x,Y_center=0))
        pair,info=select_pair(pd.DataFrame(rows),0,3,[dict(start=2,end=3)])
        self.assertEqual(pair,['1','2']);self.assertEqual(info['closest_frames'],2)
        with self.assertRaises(ValueError):select_pair(pd.DataFrame(rows).query('TrackUID==1'),0,3,[])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
    def test_render_source_range_and_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=make_inputs(root,frames=90);cfg=json.loads(p.read_text());cfg.update(start_frame=70,end_frame=85);p.write_text(json.dumps(cfg));out=render(p)
            metadata=json.loads(out.with_suffix('.provenance.json').read_text());self.assertEqual(metadata['rendered_frames'],16)
            cfg['pair']='auto';p.write_text(json.dumps(cfg));plan=render(p,check_only=True);self.assertEqual(plan['selected_pair'],['1','2']);self.assertEqual(plan['output_seconds'],16/30)
            frames=pd.read_csv(out.with_suffix('.frames.csv'));self.assertEqual(frames.frame.tolist(),list(range(70,86)));self.assertFalse(frames.predicted_interval.iloc[0]);self.assertTrue(frames.predicted_interval.iloc[-1]);self.assertTrue((frames.reference=='CIRCLING').all())
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames','-of','json',str(out)]))['streams'][0]
            self.assertEqual((info['width'],info['height']),(1440,960));self.assertEqual(int(info['nb_frames']),16)
            with self.assertRaises(FileExistsError):render(p)

if __name__=='__main__':unittest.main()
