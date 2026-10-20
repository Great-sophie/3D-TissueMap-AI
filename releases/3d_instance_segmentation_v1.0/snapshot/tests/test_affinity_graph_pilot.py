import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_affinity_graph_pilot import graph_from_affinity,sparse_pq

def test_identity_pq():
    a=np.zeros((4,12,12),np.int32)
    a[:,2:6,2:6]=1;a[:,6:10,2:6]=2
    x=sparse_pq(a,a)
    assert x['pq']==1 and x['n_predicted']==2

def test_graph_two_voxel_groups():
    fg=np.zeros((3,4,6),bool)
    fg[:,1:3,1:5]=True
    prob=np.ones((3,*fg.shape),np.float32)
    prob[2,:,:,2]=0.0  # prohibit X edges between x=2 and x=3
    out=graph_from_affinity(fg,prob,0.7)
    assert len(np.unique(out[out>0]))==2
    assert out[1,1,2]!=out[1,1,3]

def test_no_foreground():
    assert graph_from_affinity(np.zeros((3,4,6),bool),np.ones((3,3,4,6)),0.7).sum()==0

def test_empty_predictions():
    a=np.zeros((2,4,4),np.int32);a[:,1:3,1:3]=1
    res=sparse_pq(a,np.zeros_like(a))
    assert res['tp']==0 and res['fn']==1 and res['pq']==0
