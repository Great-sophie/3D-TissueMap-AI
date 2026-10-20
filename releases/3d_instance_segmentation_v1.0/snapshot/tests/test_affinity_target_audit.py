import importlib.util
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_voxel_affinities.py"
spec=importlib.util.spec_from_file_location("affinity_audit",SCRIPT)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_z_only_inter_instance_boundary():
    vol=np.zeros((3,4,5),dtype=np.uint32)
    vol[0,1:3,1:4]=10
    vol[1,1:3,1:4]=20
    vol[2,1:3,1:4]=20
    z,y,x=[m.axis_affinity_stats(vol,k) for k in range(3)]
    assert z["different_instance_faces"]==6
    assert z["same_instance_faces"]==6
    assert y["different_instance_faces"]==0
    assert x["different_instance_faces"]==0
    zm,ym,xm=m.adjacency_maps(vol,0)
    assert zm.sum()==6
    assert ym.sum()==0 and xm.sum()==0


def test_single_instance_has_only_positive_affinities():
    vol=np.ones((3,4,5),dtype=np.int32)
    for a in range(3):
        res=m.axis_affinity_stats(vol,a)
        assert res["different_instance_faces"]==0
        assert res["same_fraction_among_cell_cell"]==1


def test_invalid_axis_rejected():
    import pytest
    with pytest.raises(ValueError):m.axis_affinity_stats(np.ones((3,4)),0)
