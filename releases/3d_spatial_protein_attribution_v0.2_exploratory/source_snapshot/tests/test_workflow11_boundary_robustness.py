import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_workflow11_boundary_robustness import (
    competition_boundary,cached_geometry,margin_comparison,
    matched_margin_comparison,micro_comparison,MICRO_OFFSETS,MARGINS,
    ensure_11c_integrity,hashes_for_outputs,
)
from run_workflow11_shift_controls import sha
from spatial_signal_attribution_core import assign_slice


def test_boundary_between_owners():
    a=np.array([[1,1,2],[1,1,2],[0,0,0]],dtype=int)
    b=competition_boundary(a)
    assert b[0,1] and b[0,2]
    assert not b[0,0]
    assert not b[2,1]


def test_boundary_does_not_trigger_on_unlabeled():
    a=np.array([[0,1,0],[0,1,0]],dtype=int)
    assert not competition_boundary(a).any()


def test_boundary_validation():
    with pytest.raises(ValueError):competition_boundary(np.ones((3,4,5)))


def test_single_owner_full_keep():
    m=np.zeros((25,25),dtype=np.uint32)
    m[12,12]=1
    subsets,nb,nall,nkeep=cached_geometry(m,1)
    assert nb==0
    assert len(subsets)==3
    assert nkeep==[nall]*3


def test_multiple_owners_exclusion_and_nuclear_retention():
    m=np.zeros((25,25),dtype=np.uint32)
    m[12,8]=1
    m[12,16]=2
    subsets,nb,nall,nkeep=cached_geometry(m,2)
    assert nb>0 and nkeep[0]>nkeep[1]>=nkeep[2]
    for selected,codes in subsets:
        assert 12*25+8 in selected
        assert 12*25+16 in selected
        assert len(selected)==len(codes)
        assert np.all(codes>=0)


def test_consistency_zero_margin():
    m=np.zeros((28,28),dtype=np.uint32)
    m[6:9,6:9]=1
    m[15:18,15:18]=2
    g,n,_,_=cached_geometry(m,2)
    ref=assign_slice(m,2)
    assert np.array_equal(g[0][0],ref.selected)
    assert np.array_equal(g[0][1],ref.codes)


def test_offsets_symmetric():
    assert len(MICRO_OFFSETS)==13
    assert len(set(MICRO_OFFSETS))==13
    assert MICRO_OFFSETS[0]==(0,0)
    assert set(round(np.hypot(*p)) for p in MICRO_OFFSETS)=={0,1,2,4}


def test_micro_compare():
    rows=[]
    for dy,dx in MICRO_OFFSETS:
        rows.append({'marker':'HER2','group':'ROI_all','contrast':'a','shift_distance_px':float(np.hypot(dy,dx)),
                     'median_delta':4 if (dy,dx)==(0,0) else 1})
    c=micro_comparison(rows)
    assert len(c)==3
    assert (c.n_shift_medians_ge_observed==0).all()
    assert (c.n_controls==4).all()


def test_micro_compare_incomplete_rejected():
    rows=[{'marker':'HER2','group':'ROI_all','contrast':'a','shift_distance_px':0.,'median_delta':1.},
          {'marker':'HER2','group':'ROI_all','contrast':'a','shift_distance_px':1.,'median_delta':0.}]
    with pytest.raises(ValueError, match='Missing microshift'):micro_comparison(rows)


def test_margin_paired_sample_coverage_and_fair_comparison():
    means=np.zeros((3,4,4),dtype=float)
    counts=np.full((3,4,4),12,dtype=int)
    means[:,1,1:4]=5
    means[:,0,1:4]=1
    means[1,1,1:4]=[4,4,4]
    means[2,1,1:4]=[3,3,3]
    counts[2,1,3]=2
    rows=matched_margin_comparison(means,counts,np.array([1,2,3]),'HER2','ROI','1' if False else 1,0,'r1-nucleus',10)
    assert len(rows)==2
    assert rows[1]['n_common_paired']==2
    assert rows[0]['n_common_paired']==3
    assert rows[0]['median_within_instance_change']==-1
    assert rows[1]['median_within_instance_change']==-2


def test_margin_unmatched():
    means=np.zeros((3,4,3),dtype=float)
    counts=np.full((3,4,3),11,dtype=int)
    means[:,1,:]=[5,4,3]
    r=margin_comparison(means,counts,np.array([1,2]),'HER2','ROI',1,0,'test',10)
    assert len(r)==3 and r[0]['n_paired']==2


def test_hash_check_and_extra_files(tmp_path):
    x=tmp_path/'control_summary.json';x.write_text('{}')
    (tmp_path/'output_sha256.json').write_text(json.dumps({x.name:sha(x)}))
    assert len(ensure_11c_integrity(tmp_path))==2
    x.write_text('corrupt')
    with pytest.raises(RuntimeError,match='changed'):ensure_11c_integrity(tmp_path)


def test_stage_hash_digest(tmp_path):
    f=tmp_path/'foo.csv';f.write_text('x\n1\n')
    assert hashes_for_outputs(tmp_path)==1
    assert json.loads((tmp_path/'output_sha256.json').read_text())['foo.csv']==sha(f)


def test_full_synthetic_workflow_zero_shift_and_verify(tmp_path,monkeypatch):
    import tifffile
    from argparse import Namespace
    from spatial_signal_attribution_core import RegionAccumulator, assign_slice
    from run_spatial_signal_attribution import make_long
    import run_workflow11_boundary_robustness as wf

    mask=np.zeros((2,50,50),dtype=np.uint32)
    mask[:,18:22,15:19]=1
    mask[:,28:33,32:36]=2
    maskfile=tmp_path/'mask.tif'
    tifffile.imwrite(maskfile,mask)
    pth={}
    acc=RegionAccumulator(max_id=2)
    for m in ['Ir193','panCK','HER2','CD31','LYVE1']:
        pth[m]=[]
        for z in range(2):
            a=np.ones((50,50),dtype=np.float32)*(2+len(m)/10)
            a[16:25,13:23]=6+z+len(m)/10
            a[26:36,29:39]=10+z+len(m)/10
            img=tmp_path/f'{m}{z}.tiff';tifffile.imwrite(img,a)
            pth[m].append(img)
    for z in range(2):
        own=assign_slice(mask[z],2)
        acc.add_plane(own,{m:tifffile.imread(pth[m][z]) for m in pth})
    old,_=make_long(acc,np.array([1,2]),10)
    wf11=tmp_path/'wf11';wf11.mkdir()
    old.to_csv(wf11/'instance_marker_region_signal.csv',index=False)
    (wf11/'attribution_summary.json').write_text('{}')
    (wf11/'output_sha256.json').write_text(json.dumps({
      'instance_marker_region_signal.csv':sha(wf11/'instance_marker_region_signal.csv'),
      'attribution_summary.json':sha(wf11/'attribution_summary.json')}))
    wf11c=tmp_path/'wf11c';wf11c.mkdir()
    (wf11c/'control_summary.json').write_text('{}')
    (wf11c/'output_sha256.json').write_text(json.dumps({'control_summary.json':sha(wf11c/'control_summary.json')}))
    base=pd.DataFrame({'instance_id':[1,2],'centroid_x_px':[500,510],
                       'centroid_y_px':[505,530],'mean_raw_HER2':[2,7]})
    sources={str(maskfile):sha(maskfile)}
    for paths in pth.values():
        sources.update({str(f):sha(f) for f in paths})
    def fake_manifest(args):
        return ({'mask_shape_assumed_zyx':list(mask.shape)},pth,base,None,
                {f:sha(f) for f in [maskfile]+sum(pth.values(),[])})
    monkeypatch.setattr(wf,'file_manifest',fake_manifest)
    args=Namespace(mask=maskfile,manifest=tmp_path/'unused.json',workflow09=tmp_path/'wf09',
                   workflow11=wf11,workflow11c=wf11c,out=tmp_path/'out',min_pixels=10,
                   markers='HER2,panCK,CD31,LYVE1,Ir193')
    result=wf.build(args)
    assert result['zero_shift_reference']=='PASS'
    assert result['microshift_rows']==520
    assert result['boundary_matched_rows']==80
    assert wf.verify(args)['status']=='PASS'
    with pytest.raises(FileExistsError):wf.build(args)
    with open(args.out/'microshift_contrasts.csv','a') as f:f.write('\ncorruption')
    with pytest.raises(RuntimeError,match='Output changed'):wf.verify(args)
