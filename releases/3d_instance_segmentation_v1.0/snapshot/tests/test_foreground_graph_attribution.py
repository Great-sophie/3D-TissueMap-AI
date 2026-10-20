from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_foreground_graph_attribution import (
    foreground_diagnostics, evaluate_condition, inspect_tile,
)


def make_case():
    reference = np.zeros((2,8,8),np.int32)
    reference[:,:,:4] = 1
    reference[:,:,4:] = 2
    p = np.full((4,2,8,8),0.98,np.float32)
    p[3,:,:,3] = 0.02 # X boundary between two instances
    p[0] = 0.9
    p[0,0,0,0] = 0.1 # one false negative
    # foreground truth everywhere in this synthetic; no false positives
    return p, reference


def test_fg_diagnostics():
    fg=np.array([[[1,1,1,0,0]]],bool)
    ref=np.array([[[1,0,1,1,0]]],np.int32)
    r=foreground_diagnostics(fg,ref)
    assert (r['foreground_tp'],r['foreground_fp'],r['foreground_fn'])==(2,1,1)
    assert r['foreground_dice']==2/3


def test_synthetic_three_conditions():
    p, ref=make_case()
    out=inspect_tile(p,ref,block_zyx=(1,2,2),max_component_voxels=200)
    assert set(out)=={'predicted_foreground','oracle_remove_false_positive_foreground','oracle_exact_reference_foreground'}
    assert out['predicted_foreground']['foreground_fn']==1
    assert out['oracle_remove_false_positive_foreground']['foreground_fn']==1
    assert out['oracle_exact_reference_foreground']['foreground_fn']==0
    assert out['oracle_exact_reference_foreground']['foreground_dice']==1.
    assert out['oracle_exact_reference_foreground']['methods']['oracle_connected_all']['metrics']['pq']==1.


def test_orphan_count_resolves_with_reference_fg():
    ref=np.zeros((2,8,10),np.int32)
    ref[:,:,:4]=1
    ref[:,:,4:8]=2
    p=np.full((4,*ref.shape),.9,np.float32)
    p[3,:,:,3]=.1
    p[0,:,:,8:]=.9 # false-positive stripe with no reference overlap
    result=inspect_tile(p,ref,block_zyx=(1,2,2),max_component_voxels=200)
    raw=result['predicted_foreground']
    cleaned=result['oracle_remove_false_positive_foreground']
    assert raw['foreground_fp']==2*8*2
    assert raw['orphan_supervoxels_without_reference_overlap']>0
    assert cleaned['foreground_fp']==0
    assert cleaned['orphan_supervoxels_without_reference_overlap']==0
    assert raw['methods']['oracle_connected_all']['metrics']['n_predicted']>cleaned['methods']['oracle_connected_all']['metrics']['n_predicted']


def test_reject_invalid_shapes():
    with __import__('pytest').raises(ValueError):
        evaluate_condition(np.ones((2,3,3),bool),np.ones((4,2,3,3)),np.ones((2,3,3),np.int32))
