from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_region_oracle_connectivity import majority_reference_per_supervoxel,oracle_connected_output,graph_masks,inspect_tile


def test_majority_ref_ignore_background():
    sv=np.array([[[1,1,2,2],[1,1,2,2]]],np.int32)
    gt=np.array([[[1,1,0,0],[1,2,2,2]]],np.int32)
    lab,allpurity,overlap,size=majority_reference_per_supervoxel(sv,gt)
    assert lab.tolist()==[0,1,2]
    assert np.isclose(allpurity[1],.75)
    assert np.isclose(allpurity[2],.5)
    assert np.isclose(overlap[2],1.)


def test_oracle_merges_only_same_ref():
    sv=np.array([[[1,2,3,4]]],np.int32)
    edge=np.array([[1,2,.9,0,2],[2,3,.9,0,2],[3,4,.9,0,2]],float)
    majority=np.array([0,1,1,2,2])
    pred,n=oracle_connected_output(sv,edge,majority,np.ones(3,bool))
    assert n==2
    assert pred[0,0,0]==pred[0,0,1]
    assert pred[0,0,2]==pred[0,0,3]
    assert pred[0,0,1]!=pred[0,0,2]


def test_cutting_true_edge_causes_fragment():
    sv=np.array([[[1,2,3]]],np.int32)
    edge=np.array([[1,2,.9,0,2],[2,3,-.2,.8,2]],float)
    maj=np.array([0,7,7,7])
    masks=graph_masks(edge)
    all_pred,n1=oracle_connected_output(sv,edge,maj,masks['all'])
    filtered_pred,n2=oracle_connected_output(sv,edge,maj,masks['all_three'])
    assert n1==2 and n2==1
    assert len(np.unique(all_pred))==1
    assert len(np.unique(filtered_pred))==2


def test_end_to_end_synthetic_clean_affinity():
    ref=np.zeros((2,8,8),np.int32)
    ref[:,:,:4]=1
    ref[:,:,4:]=2
    p=np.zeros((4,2,8,8),np.float32)
    p[0]=.9
    p[1:]=.95
    # X-faces that cross the reference border are negative.
    p[3,:,:,3]=.05
    metrics,rows=inspect_tile(p,ref,(1,2,2))
    assert metrics['fg_dice']==1
    assert metrics['n_supervoxels']>2
    assert metrics['foreground_6_connected_components']==1
    assert rows[0]['reference_guided_oracle_connected']['pq']==1.
    assert rows[-1]['reference_guided_oracle_connected']['pq']==1.
    assert rows[0]['oracle_different_edges_total']>0
    assert rows[-1]['oracle_different_edges_kept']==0
