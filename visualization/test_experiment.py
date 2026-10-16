import unittest
import pandas as pd
from .experiment import candidates,evaluate,_nearest

class HypothesisTests(unittest.TestCase):
    def test_duration_and_missing_breaks(self):
        near={f:(100,'1','2') for f in [0,1,3,4,5]}
        self.assertEqual(candidates(near,160,1,3),[dict(start=3,end=5)])

    def test_union_does_not_double_count(self):
        event=dict(start=0,end=9,status='circling',event='a',notes='')
        metrics,ev,ca=evaluate([event],[dict(start=0,end=5),dict(start=4,end=9)],{i:0 for i in range(10)},0,9,.5)
        self.assertEqual(ev[0]['coverage'],1);self.assertEqual(metrics['event_coverage_recall'],1)

    def test_unknown_extra_is_not_false_positive(self):
        metrics,ev,ca=evaluate([], [dict(start=0,end=4)],{},0,9,.5)
        self.assertIsNone(metrics['event_coverage_recall']);self.assertEqual(ca[0]['category'],'unverified_candidate')

    def test_partial_event_and_missing_observability(self):
        event=dict(start=0,end=9,status='circling',event='a',notes='')
        metrics,ev,ca=evaluate([event],[dict(start=0,end=2)],{0:0,1:0},0,9,.5)
        self.assertEqual(ev[0]['category'],'partially_retained');self.assertEqual(ev[0]['observable_fraction'],.2)

    def test_duplicate_and_unsorted_rows(self):
        rows=pd.DataFrame(dict(FrameNum=[1,0,1,0],TrackUID=[1,1,2,2],X_center=[0,0,10,10],Y_center=[0,0,0,0]));near={}
        _nearest(rows,0,1,near);self.assertEqual(len(near),2)
        with self.assertRaises(ValueError):_nearest(pd.concat([rows,rows]),0,1,{})

if __name__=='__main__':unittest.main()
