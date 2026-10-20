import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from affinity_fixed_patch_check import select_training_patch
from train_voxel_affinity import affinity_targets


def test_fixed_patch_selection_train_tile():
    image=np.zeros((8,80,80),dtype=np.float32)
    lab=np.zeros((8,80,80),dtype=np.uint32)
    # two adjoining 3D bodies, different IDs along X and Y in central region
    lab[:,18:44,18:32]=1
    lab[:,18:44,32:46]=2
    lab[:,44:58,18:32]=3
    lab[:,44:58,32:46]=4
    lab[0:4,26:35,24:39]=5  # introduce Z changes
    image[lab>0]=.9
    t=[{'y':0,'x':0,'w':80}]
    a,b,location=select_training_patch(image,lab,t,8,48,4,42,500)
    assert location['tile_y']==0
    assert b.shape==(8,48,48)
    assert location['all_axes_have_both_classes']
    _,valid=affinity_targets(b)
    assert all(valid[i].any() for i in range(3))


def test_select_fails_if_all_background():
    image=np.zeros((8,80,80),dtype=np.float32)
    lab=np.zeros((8,80,80),dtype=np.uint32)
    with pytest.raises(RuntimeError):
        select_training_patch(image,lab,[{'y':0,'x':0,'w':80}],8,48,4,42,10)
