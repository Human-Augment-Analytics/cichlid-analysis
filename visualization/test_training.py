import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from .training import setup
from .training_backend.scene_features import DISTANCES,ANGLES,RATES
from .training_backend.learning_data import labels_from_spans

class TrainingIntegrationTests(unittest.TestCase):
    def test_updated_annotations_and_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[]
            for f in range(12):
                r=dict(FrameNum=f,TrackUID_1=1,TrackUID_2=2)
                r.update({c:float(f+1) for c in DISTANCES+ANGLES+RATES});rows.append(r)
            pd.DataFrame(rows).to_parquet(root/'pairs.parquet')
            pd.DataFrame([dict(VideoID='v',StartFrame=0,EndFrame=5,ShouldUse=True,Circling=True),dict(VideoID='v',StartFrame=6,EndFrame=11,ShouldUse=True,Circling=False),dict(VideoID='v',StartFrame=None,EndFrame=None)]).to_csv(root/'labels.csv',index=False)
            cfg=dict(videos=[dict(video_id='v',split='train',pairs='pairs.parquet',annotations=dict(path='labels.csv'))])
            p=root/'config.json';p.write_text(json.dumps(cfg));setup(p,root/'run')
            z=np.load(root/'run/prepared/video_000.npz');self.assertEqual(z['x'].shape,(12,52));self.assertEqual(set(z['y']),{0,1})
            audit=json.loads((root/'run/setup_provenance.json').read_text());self.assertEqual(audit['inputs'][0]['skipped_incomplete_rows'],1)

    def test_original_source_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r=dict(FrameNum=0,TrackUID_1=1,TrackUID_2=2);r.update({c:1. for c in DISTANCES+ANGLES+RATES});pd.DataFrame([r]).to_parquet(root/'p.parquet')
            p=root/'config.json';p.write_text(json.dumps(dict(videos=[dict(video_id='a',split='train',pairs='p.parquet'),dict(video_id='b',split='test',pairs='p.parquet')])))
            with self.assertRaisesRegex(ValueError,'Same original'):setup(p,root/'run')

    def test_unknown_and_conflict_not_assumed_negative(self):
        y=labels_from_spans(np.arange(3),[dict(start=1,end=1,status='unknown')],True)
        self.assertEqual(y.tolist(),[0,-100,0])

if __name__=='__main__':unittest.main()
