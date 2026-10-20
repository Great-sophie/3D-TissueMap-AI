"""Tests for fixed full nuclear marker constraints, including integration and tamper resistance."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))

from run_workflow11h_nucleus_constrained_segmentation import protected_markers,segment_slice,run,verify,validate_parent
from run_workflow11g_cadherin_segmentation import segment_slice as segment_original,run as run_original
from test_workflow11g_cadherin_segmentation import fixture_project, sample_images


def test_full_fragment_is_protected_even_when_multiple_fragments_share_id():
    dom=np.ones((32,32),dtype=np.int32)
    nuc=np.zeros_like(dom)
    nuc[5:9,7:10]=21
    nuc[20:24,21:25]=21
    seed, rows=protected_markers(nuc,dom,min_overlap=4)
    assert len(rows)==2
    assert np.array_equal(seed>0,nuc>0)
    assert seed[5,7]!=seed[20,21]
    for row in rows:
        assert int((seed==row['seed_id']).sum())==row['nucleus_fragment_px']


def test_fragment_with_less_than_min_support_not_selected():
    dom=np.ones((20,20),dtype=np.int32)
    nuc=np.zeros_like(dom)
    nuc[1:3,2:4]=3
    nuc[7,8]=4
    seed,rows=protected_markers(nuc,dom,min_overlap=4)
    assert len(rows)==1
    assert int(np.count_nonzero(seed))==4


def test_both_methods_keep_all_nuclear_pixels():
    ck,nuc,cad,_=sample_images()
    domain, seeds, rows, base, assist, _, _=segment_slice(ck,nuc,cad)
    assert len(rows)==2
    assert np.array_equal(base[seeds>0],seeds[seeds>0])
    assert np.array_equal(assist[seeds>0],seeds[seeds>0])
    for row in rows:
        assert int((base==row['seed_id']).sum())>=row['nucleus_fragment_px']
        assert int((assist==row['seed_id']).sum())>=row['nucleus_fragment_px']


def test_area_conservation_and_her2_heldout():
    ck,nuc,cad,her2=sample_images()
    dom,seed,rows,base,assist,q,_=segment_slice(ck,nuc,cad)
    assert np.array_equal(base>0,assist>0)
    assert 'her2' not in segment_slice.__code__.co_varnames
    assert int(np.count_nonzero(base))>0


def test_source_row_identity_and_no_overwrite(tmp_path):
    a=fixture_project(tmp_path)
    run_original(a)
    reference=a.out
    a.reference_run=reference
    a.out=reference.parent/'fixed_v0_2'
    assert validate_parent(a)['status']=='PASS'
    result=run(a)
    assert result['status']=='CREATED'
    assert result['new_containment_violations']==0
    assert verify(a)['status']=='PASS'
    old=pd.read_csv(reference/'section_nuclear_seed_index.csv')
    new=pd.read_csv(a.out/'section_nuclear_seed_index.csv')
    assert old[['z','seed_id','domain_id','source_pseudo_id','nucleus_fragment_px']].equals(
        new[['z','seed_id','domain_id','source_pseudo_id','nucleus_fragment_px']])
    with pytest.raises(FileExistsError):run(a)


def test_parent_tamper_rejected(tmp_path):
    a=fixture_project(tmp_path);run_original(a)
    a.reference_run=a.out;a.out=a.out.parent/'fixed_v0_2';run(a)
    with (a.reference_run/'candidate_boundary_metrics.csv').open('a') as f:f.write('\ncorrupted\n')
    with pytest.raises(RuntimeError):verify(a)


def test_output_tamper_rejected(tmp_path):
    a=fixture_project(tmp_path);run_original(a)
    a.reference_run=a.out;a.out=a.out.parent/'fixed_v0_2';run(a)
    with (a.out/'parent_vs_constrained_geometry.csv').open('a') as f:f.write('\ncorrupted\n')
    with pytest.raises(RuntimeError):verify(a)


def test_invalid_input_cadherin_rejected():
    ck,nuc,cad,_=sample_images()
    cad[2,2]=-1
    with pytest.raises(ValueError,match='non-negative'):segment_slice(ck,nuc,cad)
