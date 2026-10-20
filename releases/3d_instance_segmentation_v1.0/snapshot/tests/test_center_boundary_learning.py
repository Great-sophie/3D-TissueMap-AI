import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
# run from merged project; existing companion modules already present
from train_center_boundary import boundary_target, center_target, make_model, objective, guided_watershed


def fixture():
    lab=np.zeros((8,32,32),dtype=np.int32)
    lab[2:6,4:12,3:13]=1
    lab[2:6,4:12,13:23]=2
    return lab


def test_boundary_covers_instance_contours():
    lab=fixture()
    b=boundary_target(lab)
    assert b[4,7,12]==1 and b[4,7,13]==1
    assert b[4,7,3]==1
    assert b[4,7,6]==0
    assert np.count_nonzero(b)>0


def test_centers_inside_image_and_no_labels_returns_zero():
    lab=fixture()
    t=center_target(lab)
    assert t.shape==lab.shape and 0<t.max()<=1.0
    assert np.max(center_target(np.zeros_like(lab)))==0


def test_multihead_backward_finite():
    model=make_model(4)
    x=torch.randn((1,1,8,32,32))
    logits=model(x)
    loss,detail=objective(logits,fixture())
    assert logits.shape==(1,3,8,32,32)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_inference_returns_valid_instances():
    lab=fixture()
    shape=lab.shape
    foreground=np.where(lab>0,.9,.1).astype(np.float32)
    boundary=boundary_target(lab)
    centers=center_target(lab)
    pred,stats=guided_watershed(foreground,boundary,centers,center_threshold=.25)
    assert pred.shape==shape
    assert pred.dtype==np.int32
    assert stats['n_seeds']>=1
    assert np.max(pred)>0
