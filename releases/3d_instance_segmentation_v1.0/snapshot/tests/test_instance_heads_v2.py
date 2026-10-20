import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_instance_heads_v2 import boundary_target,center_target_and_valid,objective,make_model,complete_instance_ids

def two_cells(gap=0):
    lab=np.zeros((8,32,32),dtype=np.int32)
    lab[2:6,8:17,7:13]=1
    lab[2:6,8:17,13+gap:19+gap]=2
    return lab

def test_interface_ignores_isolated_outer_surface():
    isolated=np.zeros((8,32,32),dtype=np.int32)
    isolated[2:6,8:17,7:13]=1
    assert boundary_target(isolated).sum()==0
    touching=two_cells(gap=0)
    interface=boundary_target(touching)
    assert interface.sum()>0
    assert interface[4,11,12]==1
    assert interface[4,11,13]==1
    assert interface[4,8,7]==0

def test_center_peaks_inside_instance_and_masks_truncation():
    lab=two_cells(gap=2)
    lab[:4,0:4,0:4]=3 # reaches crop faces: must not supervise as negative
    heat,valid=center_target_and_valid(lab)
    assert (heat[lab==0]==0).all()
    assert heat.max()<=1
    assert np.any((heat>=.5)&(lab==1))
    assert np.any((heat>=.5)&(lab==2))
    assert (valid[lab==3]==False).all()
    assert 3 not in set(complete_instance_ids(lab))

def test_gradients_for_all_three_heads():
    model=make_model(4)
    logits=model(torch.randn((1,1,8,32,32)))
    total,detail=objective(logits,two_cells(gap=0))
    assert np.isfinite(float(total.detach()))
    assert detail['n_center_target']>0
    assert detail['n_boundary_target']>0
    total.backward()
    assert model.last.weight.grad is not None
    assert (model.last.weight.grad.abs().flatten(1).sum(1)>0).all()

def test_empty_patch_safe():
    model=make_model(4)
    logits=model(torch.randn((1,1,8,32,32)))
    total,_=objective(logits,np.zeros((8,32,32),dtype=np.int32))
    assert torch.isfinite(total)
    total.backward()
