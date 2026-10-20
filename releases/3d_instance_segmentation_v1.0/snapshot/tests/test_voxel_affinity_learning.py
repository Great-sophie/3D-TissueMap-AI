import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_voxel_affinity import affinity_targets,conditional_affinity_loss,edge_stats,choose_patch

def test_neighbors_background_is_masked():
    a=np.zeros((4,5,6),dtype=np.uint32)
    a[:,1:3,1:3]=1;a[:,3:5,1:3]=2
    target,valid=affinity_targets(a)
    assert target.shape==(3,4,5,6)
    assert int((valid[0]& (target[0]<.5)).sum())==0
    assert int((valid[1]& (target[1]<.5)).sum())>0
    assert not valid[0,-1].any()
    assert not valid[1,:,-1,:].any()
    assert not valid[2,:,:,-1].any()

def test_loss_has_gradient_and_background_excluded():
    lab=np.zeros((4,8,8),dtype=np.uint32)
    lab[:,1:7,1:4]=1;lab[:,1:7,4:7]=2
    p=torch.zeros((1,4,4,8,8),requires_grad=True)
    loss,d=conditional_affinity_loss(p,lab,.25)
    assert torch.isfinite(loss)
    loss.backward()
    assert p.grad is not None
    assert d['counts']['X']['different']>0

def test_identical_predictions_perfect_edge_accuracy():
    lab=np.zeros((4,6,6),dtype=np.uint32)
    lab[:,1:5,1:3]=1;lab[:,1:5,3:5]=2
    target,valid=affinity_targets(lab)
    res=edge_stats(target,lab)
    assert all(d['balanced_accuracy']==1 for d in res if d['balanced_accuracy'] is not None)
