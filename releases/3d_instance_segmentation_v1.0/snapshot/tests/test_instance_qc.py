import importlib.util
import sys
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(root))

def mod(name):
    spec=importlib.util.spec_from_file_location(name,root/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_linkage_comparison():
    m=mod('instance_linkage_qc')
    res=m._comparison([1,2,3],[1,2,3]);assert res['fraction_exact']==1.0
    assert res['pearson'] > .99


def test_instance_split():
    m=mod('instance_watershed_baseline')
    x=np.zeros((8,32,32),np.float32)
    x[1:4,3:11,3:11]=.9
    x[4:7,20:28,20:28]=.9
    for method in ('components','watershed'):
        lab=m.segment(x,method)
        assert len(np.unique(lab[lab>0]))>=2
        met=m.compare_instances(lab,lab)
        assert met['pq']>.999
        assert met['tp']==met['n_reference']


def test_no_foreground():
    m=mod('instance_watershed_baseline')
    arr=np.zeros((8,16,16),np.float32)
    lab=m.segment(arr);assert lab.sum()==0
    met=m.compare_instances(lab,lab)
    assert met['tp']==0
