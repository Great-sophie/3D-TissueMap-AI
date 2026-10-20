import importlib.util
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('sv_audit',HERE/'audit_supervoxel_purity.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_two_touching_ids_block_mixed():
    ref=np.zeros((2,2,4),dtype=np.uint32)
    ref[:, :, :2]=1; ref[:,:,2:]=2
    result=m.audit_one(ref,(2,2,4))
    assert result['n_supervoxels']==1
    assert result['n_mixed_supervoxels']==1
    assert abs(result['minority_reference_voxel_fraction']-0.5)<1e-9


def test_disconnected_ids_in_one_block_are_separated():
    ref=np.zeros((1,3,5),dtype=np.uint32)
    ref[:,1,0]=1;ref[:,1,4]=2
    result=m.audit_one(ref,(1,3,5))
    assert result['n_supervoxels']==2
    assert result['n_mixed_supervoxels']==0
    assert result['minority_reference_voxel_fraction']==0


def test_pure_fine_grained_blocks_are_fragmented():
    ref=np.ones((1,4,4),dtype=np.uint32)
    r=m.audit_one(ref,(1,2,2))
    assert r['n_supervoxels']==4
    assert r['median_supervoxels_per_reference']==4
    assert r['fraction_mixed_supervoxels']==0
