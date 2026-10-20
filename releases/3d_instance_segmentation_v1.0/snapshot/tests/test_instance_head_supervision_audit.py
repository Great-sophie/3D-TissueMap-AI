import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_instance_head_supervision import sample_patch_stats,summary

def test_blank_patch_safe():
    image=np.zeros((8,16,16),np.float32)
    mask=np.zeros_like(image,dtype=np.int32)
    d=sample_patch_stats(image,mask)
    assert d['n_instance_ids']==0
    assert d['center_target_voxels_gt_035']==0
    assert d['boundary_target_fraction']==0

def test_center_outside_detection_and_touching_faces():
    image=np.zeros((8,16,16),np.float32)
    mask=np.zeros_like(image,dtype=np.int32)
    mask[2:6,4:10,4:10]=1
    mask[2:6,6:8,6:8]=0
    mask[3:5,11:14,11:14]=2
    result=sample_patch_stats(image,mask)
    assert result['n_instance_ids']==2
    assert result['n_centroids_outside_own_id']>=1
    assert result['center_target_voxels_gt_035']>0
    assert result['boundary_target_fraction']>0

def test_summary():
    d=summary([1,2,3])
    assert d['median']==2 and d['mean']==2
