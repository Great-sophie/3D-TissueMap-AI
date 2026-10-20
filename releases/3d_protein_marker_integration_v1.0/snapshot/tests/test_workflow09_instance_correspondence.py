import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('mod', Path(__file__).resolve().parents[1]/'scripts'/'audit_workflow09_alignment.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_key():
    assert m.source_key('0003_a_b_ac__Ir193.tiff')=='a_b_ac'
    assert m.source_key('a_b_ac.ome.tiff')=='a_b_ac'

def test_count():
    a=np.array([[[0,2],[3,2]],[[2,3],[3,0]]],dtype=np.uint32)
    c=m.area_counts(a)
    assert dict(c)=={2:3,3:3}

def test_area_match():
    a=pd.DataFrame({'label':[2,3],'area':[3,3]})
    obs=pd.Series({2:3,3:3})
    _,s=m.compare_author_area(a,obs,'measured')
    assert s['same_id_and_area_mapping_is_confirmed']

def test_area_mismatch():
    a=pd.DataFrame({'label':[2,3],'area':[3,4]})
    obs=pd.Series({2:3,3:3})
    _,s=m.compare_author_area(a,obs,'full')
    assert not s['same_id_and_area_mapping_is_confirmed']
    assert s['n_shared_but_area_differs']==1

def test_missing_ids():
    a=pd.DataFrame({'label':[2,3],'area':[3,3]})
    obs=pd.Series({2:3,4:3})
    _,s=m.compare_author_area(a,obs,'full')
    assert s['n_shared_ids']==1 and s['n_author_only_ids']==1 and s['n_mask_only_ids']==1

def test_outline_excludes_zero():
    a=np.array([[0,1,1],[0,2,2]])
    b=m.mask_outlines_2d(a)
    assert b[0,1] and b[1,1]
    assert not b[0,0]

def test_overlay_shape():
    a=np.ones((4,5),dtype='uint16')
    b=np.zeros((4,5),dtype='uint32');b[2,2]=1
    c=m.grayscale_overlay(a,b)
    assert c.shape==(4,5,3) and np.isfinite(c).all()
