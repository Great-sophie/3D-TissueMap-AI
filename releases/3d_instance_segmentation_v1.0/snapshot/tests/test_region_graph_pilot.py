import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from evaluate_affinity_region_graph import block_supervoxels,interface_edges,merge_regions

def test_no_foreground():
    fg=np.zeros((4,6,6),bool)
    sv=block_supervoxels(fg)
    assert sv.max()==0
    assert interface_edges(sv,np.ones((3,*fg.shape),np.float32)).shape==(0,5)

def test_oracle_region_graph_recovers_two_components():
    ref=np.zeros((4,12,12),np.int32)
    ref[:,1:5,1:5]=1
    ref[:,7:11,7:11]=2
    sv=block_supervoxels(ref>0,(2,2,2))
    aff=np.zeros((3,*ref.shape),np.float32)
    for ax in range(3):
        lo=[slice(None)]*3;hi=[slice(None)]*3
        lo[ax]=slice(None,-1);hi[ax]=slice(1,None)
        a,b=ref[tuple(lo)],ref[tuple(hi)]
        aff[ax][tuple(lo)]=np.where((a>0)&(b>0)&(a==b),.95,.05)
    interfaces=interface_edges(sv,aff)
    pred,_=merge_regions(sv,interfaces,0.2,0.35,2000,1)
    assert set(np.unique(pred))=={0,1,2}
    assert np.all(pred[ref==1]==pred[0,1,1])
    assert np.all(pred[ref==2]==pred[0,7,7])
    assert pred[0,1,1]!=pred[0,7,7]

def test_negative_contact_prevents_merge():
    fg=np.zeros((2,4,6),bool)
    fg[:,:,1:5]=True
    sv=block_supervoxels(fg,(2,4,2))
    aff=np.full((3,*fg.shape),0.05,np.float32)
    edges=interface_edges(sv,aff)
    merged,n=merge_regions(sv,edges,0.2,0.35,1000,1)
    assert n==0
    assert (merged>0).sum()==fg.sum()
