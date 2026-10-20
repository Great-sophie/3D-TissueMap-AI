import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_workflow11_shift_controls import (translated_no_wrap,fast_bin,
    summarize_contrast,contrast_values,build_comparison, group_ids, OFFSETS)
from spatial_signal_attribution_core import assign_slice


def test_translate_zero():
    a=np.arange(12).reshape(3,4)
    assert np.array_equal(translated_no_wrap(a,0,0),a)

@pytest.mark.parametrize('dy,dx,expected_y,expected_x',[(1,0,1,0),(-1,0,0,0),(0,2,0,2),(0,-2,0,0),(1,-2,1,0)])
def test_translate_no_wrapping(dy,dx,expected_y,expected_x):
    a=np.zeros((5,6),dtype=int)
    a[1,2]=7
    shifted=translated_no_wrap(a,dy,dx)
    y,x=1+dy,2+dx
    assert shifted[y,x]==7
    assert np.isnan(shifted).any()
    assert np.nansum(shifted)==7


def test_all_off_image_nan():
    assert np.isnan(translated_no_wrap(np.ones((3,4)),5,0)).all()


def test_region_counts_and_means():
    a=np.zeros((19,19),np.uint32)
    a[9,9]=1
    owner=assign_slice(a,1)
    im=np.ones((19,19),np.float64)
    im[9,9]=30
    totals,counts=fast_bin(owner,im,1)
    assert counts[0,1]==1
    assert totals[0,1]==30
    assert totals[1,1]==counts[1,1]
    ids,delta=contrast_values(totals/np.maximum(counts,1),counts,np.array([1]),1,1,0)
    assert ids.tolist()==[1]
    assert delta.tolist()==[-29]


def test_nonfinite_ignored():
    a=np.zeros((19,19),np.uint32)
    a[9,9]=1
    im=np.ones((19,19));im[9,9]=np.nan
    t,c=fast_bin(assign_slice(a,1),im,1)
    assert c[0,1]==0
    assert c[1,1]>0


def test_summary_subset_and_groups():
    d=pd.DataFrame(dict(instance_id=[1,2,3,4],centroid_x_px=[500,550,750,100],
       centroid_y_px=[600,620,650,100],mean_raw_HER2=[10.,0.2,0.1,0.1]))
    groups,thr=group_ids(d)
    assert len(groups['ROI_all'])==2
    assert 1 in groups['ROI_high_selected']
    assert 2 in groups['ROI_other']
    assert len(groups['outside_ROI'])==2


def test_build_comparison():
    rows=[]
    for dy,dx in OFFSETS:
        rows.append(dict(marker='HER2',group='ROI_all',contrast='ring',dy=dy,dx=dx,
                         median_delta=10 if (dy,dx)==(0,0) else 1,
                         fraction_positive=0.9 if (dy,dx)==(0,0) else 0.1,
                         n_paired=20))
    v=build_comparison(rows)
    assert v.iloc[0].n_controls==8
    assert v.iloc[0].observed_median_delta==10
    assert v.iloc[0].n_null_with_median_ge_observed==0


def test_finite_translation_binned_pixel():
    mask=np.zeros((21,21),np.uint32)
    mask[10,10]=1
    image=np.zeros_like(mask,dtype=float);image[10,10]=99
    for dy,dx in OFFSETS:
        shifted=translated_no_wrap(image,dy,dx)
        t,c=fast_bin(assign_slice(mask,1),shifted,1)
        if (dy,dx)==(0,0):assert t[0,1]==99
        else:assert t[0,1]==0


def test_full_offset_synthetic_against_existing_accumulator(tmp_path):
    import tifffile
    from spatial_signal_attribution_core import RegionAccumulator
    from run_workflow11_shift_controls import means_for_marker_all_offsets
    mask=np.zeros((2,64,64),dtype=np.uint32)
    mask[:,20:23,20:23]=1
    mask[:,42:45,42:45]=2
    paths=[]
    acc=RegionAccumulator(max_id=2,markers=('HER2',))
    cache=[]
    for z in range(2):
        a=np.full((64,64),0.5,dtype=np.float32)
        a[16:27,16:27]=4+z
        a[38:49,38:49]=3+z
        imagepath=tmp_path/f'z{z}.tif'
        tifffile.imwrite(imagepath,a)
        paths.append(imagepath)
        own=assign_slice(mask[z],2)
        cache.append((own.selected,own.codes))
        acc.add_plane(own,{'HER2':a})
    means,counts=means_for_marker_all_offsets(mask,paths,2,cache)
    ref_means=acc.stats('HER2')[0]
    assert np.allclose(means[0,:,1:],ref_means[:,1:],equal_nan=True,atol=1e-8)
    assert np.array_equal(counts[0],acc.n_finite['HER2'].reshape(4,3))
    assert means.shape==(9,4,3)
    assert not np.array_equal(means[0,:,1:],means[1,:,1:])


def test_reference_allows_appended_analysis_but_rejects_tampering(tmp_path):
    import json
    from run_workflow11_shift_controls import ensure_run_integrity,sha
    f=tmp_path/'official.csv'
    f.write_text('a,b\n1,2\n')
    h={f.name:sha(f)}
    (tmp_path/'output_sha256.json').write_text(json.dumps(h))
    (tmp_path/'paired_signal_validation.csv').write_text('downstream extra result')
    hashes=ensure_run_integrity(tmp_path)
    assert 'official.csv' in hashes
    assert 'paired_signal_validation.csv' not in hashes
    f.write_text('mutated')
    with pytest.raises(RuntimeError,match='hash changed'):
        ensure_run_integrity(tmp_path)


def test_zero_shift_comparison_matches_reference_table():
    from run_workflow11_shift_controls import check_zero_shift_matches_wf11
    regions=('nuclear_mask','perinuclear_1_2px','perinuclear_3_5px','outer_context_6_10px')
    means=np.zeros((4,3));counts=np.zeros((4,3),dtype=int)
    means[:,1]=[5,8,3,2];counts[:,1]=[10,12,11,9]
    means[:,2]=[2,3,3,1];counts[:,2]=[10,10,8,0]
    long=[]
    for region_idx,region in enumerate(regions):
        for ident in [1,2]:
            c=counts[region_idx,ident]
            long.append(dict(instance_id=ident,marker='HER2',region=region,
                             n_finite_signal_voxels=c,
                             mean_raw_signal=means[region_idx,ident] if c>=10 else np.nan))
    ref=pd.DataFrame(long)
    check_zero_shift_matches_wf11(means,counts,ref,'HER2',np.array([1,2]),10)
    means[1,1]+=0.2
    with pytest.raises(RuntimeError,match='mismatch'):
        check_zero_shift_matches_wf11(means,counts,ref,'HER2',np.array([1,2]),10)
