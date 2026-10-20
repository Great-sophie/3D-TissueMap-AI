"""Run from project root with pytest -q; scripts folder must be on sys.path."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from optimize_instance_separation import watershed_instances, split_merge_screen, summarize
import pandas as pd

def test_two_separate_cells():
    vol=np.zeros((8,32,32),np.float32)
    vol[1:4,4:11,4:11]=.95
    vol[4:7,20:27,20:27]=.95
    lab,stat=watershed_instances(vol,ensure_each_component=True)
    assert np.unique(lab[lab>0]).size>=2
    assert np.count_nonzero(lab)>0
    assert stat['n_seeds']>=2

def test_missing_seed_component_fallback():
    vol=np.zeros((8,32,32),np.float32)
    vol[1,1,1]=.9
    vol[3:6,20:25,20:25]=.9
    lab,s=watershed_instances(vol,ensure_each_component=True)
    assert lab[1,1,1]>0
    assert s['n_added_seeds']>=1

def test_split_merge_sanity():
    ref=np.zeros((2,20,20),np.int32)
    ref[:,2:6,2:6]=1;ref[:,7:11,2:6]=2
    pred=np.zeros_like(ref);pred[:,2:11,2:6]=1
    a=split_merge_screen(ref,pred,minimum_voxels=3)
    assert a['merge_candidate_preds']==1
    assert a['split_candidate_refs']==0
    pred[:,:,:]=0
    pred[:,2:6,2:4]=1;pred[:,2:6,4:6]=2
    a=split_merge_screen(ref,pred,minimum_voxels=3)
    assert a['split_candidate_refs']==1

def test_summary():
    rows=[dict(variant='a',pq=.1,recall=.2,precision=.3,n_predicted=5,n_reference=10,merge_candidate_preds=2,split_candidate_refs=1),dict(variant='b',pq=.2,recall=.3,precision=.4,n_predicted=6,n_reference=10,merge_candidate_preds=1,split_candidate_refs=1)]
    result=summarize(pd.DataFrame(rows))
    assert result[0]['variant']=='b'
