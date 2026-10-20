"""Synthetic Workflow11E unit + full manifest-backed integration checks."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_spatial_signal_attribution import sha
from run_workflow11e_membrane_feasibility import (
    epithelial_domain,component_bands,shifted_view_from_full,
    compute_slice,compare_shifts,run,verify,check_args
)


def test_panck_contour_does_not_depend_on_her2():
    a=np.zeros((64,64),float)
    yy,xx=np.indices(a.shape)
    a=80*np.exp(-((yy-32)**2+(xx-32)**2)/250)
    d,t=epithelial_domain(a,.80,1.0,40)
    assert (d>0).sum()>150 and d.max()==1
    assert np.isfinite(t)
    edge,core=component_bands(d==1)
    assert edge.sum()>12 and core.sum()>12
    assert not np.any(edge&core)


def test_flat_channel_has_no_domain():
    d,t=epithelial_domain(np.zeros((32,32)),.8,1.0,40)
    assert d.max()==0


def test_negative_signal_or_nan_rejected():
    for a in (np.full((12,12),-1),np.full((12,12),np.nan)):
        with pytest.raises(ValueError):epithelial_domain(a)


@pytest.mark.parametrize('dy,dx',[(0,0),(1,0),(-1,0),(0,2),(0,-2)])
def test_shifted_images_no_wrap(dy,dx):
    a=np.arange(8*9).reshape(8,9)
    roi=(1,8,1,7)
    got=shifted_view_from_full(a,roi,dy,dx)
    for iy in range(6):
        for ix in range(7):
            sy=iy+1-dy;sx=ix+1-dx
            if 0<=sy<8 and 0<=sx<9:
                assert got[iy,ix]==a[sy,sx]
            else:
                assert np.isnan(got[iy,ix])


def test_independent_contour_and_her2_analysis():
    yy,xx=np.indices((64,64))
    ck=100*np.exp(-((yy-32)**2+(xx-32)**2)/350)
    her2=np.zeros_like(ck)
    r=np.sqrt((yy-32)**2+(xx-32)**2)
    her2[(r>=14)&(r<=17)]=100
    ir=np.zeros_like(ck)
    mask=np.zeros(ck.shape,np.uint32)
    mask[28:32,30:34]=1
    dom, rows, shifted, qc=compute_slice(ck,her2,ir,mask,0,(0,64,0,64),.8,1.,20,8)
    assert qc['n_panCK_domains']>=1
    assert len(rows)>0
    assert len(shifted)==len(rows)*9
    assert len(compare_shifts(pd.DataFrame(shifted)))==9
    # HER2 is held out: segmentation unchanged even if HER2 channel is replaced.
    dom2,_,_,_=compute_slice(ck,her2*0,ir,mask,0,(0,64,0,64),.8,1.,20,8)
    assert np.array_equal(dom,dom2)


def fixture_project(tmp_path):
    root=tmp_path/'fake_project'
    data=root/'data';data.mkdir(parents=True)
    w9=root/'runs'/'3d_protein_marker_integration_v1';w9.mkdir(parents=True)
    cfg=root/'configs';cfg.mkdir()
    markers=['Ir193','panCK','HER2','CD31','LYVE1']
    mask=np.zeros((2,48,48),dtype=np.uint32)
    mask[:,20:23,20:23]=1
    mask[:,30:34,29:33]=2
    mpath=data/'mask.tif'
    tifffile.imwrite(mpath,mask)
    yy,xx=np.indices((48,48))
    rad=np.sqrt((yy-24)**2+(xx-24)**2)
    raws={
      'Ir193':(mask[0]>0).astype(float)*50,
      'panCK':100*np.exp(-(rad**2)/240),
      'HER2':5+40*np.exp(-((rad-12)**2)/5),
      'CD31':np.ones((48,48),float),
      'LYVE1':np.ones((48,48),float)*3,
    }
    ref={k:[] for k in markers}
    specs=[];summarkers=[]
    for marker in markers:
        ps=[];sha_m={}
        for z in range(2):
            a=np.clip(raws[marker].round()+z,0,65535).astype(np.uint16)
            path=data/f'{marker}_{z}.tiff'
            tifffile.imwrite(path,a)
            ps.append(str(path.resolve()))
            sha_m[str(path.resolve())]=sha(path)
            ref[marker].append(a)
        specs.append({'name':marker,'mode':'per_section','axes':'YX','files':ps})
        summarkers.append({'name':marker,'files_sha256':sha_m})
    manifest=cfg/'protein_marker_manifest_registered_5channels.json'
    manifest.write_text(json.dumps({'markers':specs,'expected_mask_sha256':sha(mpath)}))
    ids=np.array([1,2]);shape=[2,48,48]
    frame={'instance_id':ids,'centroid_y_px':[21,31.5],'centroid_x_px':[21,30.5],
           'z_span_planes':[2,2],'voxel_count':[int(np.sum(mask==i)) for i in ids]}
    mean={'instance_id':ids}
    for k in markers:
        v=[float(np.stack(ref[k])[mask==i].mean()) for i in ids]
        frame['mean_raw_'+k]=v; mean['mean_raw_'+k]=v
    pd.DataFrame(frame).to_csv(w9/'instance_marker_spatial_table.csv',index=False)
    pd.DataFrame(mean).to_csv(w9/'instance_marker_mean_matrix.csv',index=False)
    summary={'marker_names':markers,'n_instances':2,'markers':summarkers,
             'manifest_sha256':sha(manifest),'source_mask_sha256':sha(mpath),
             'mask_shape_assumed_zyx':shape}
    (w9/'marker_integration_summary.json').write_text(json.dumps(summary))
    names=['marker_integration_summary.json','instance_marker_spatial_table.csv','instance_marker_mean_matrix.csv']
    (w9/'output_file_sha256.json').write_text(json.dumps({n:sha(w9/n) for n in names}))
    return SimpleNamespace(mask=mpath,manifest=manifest,workflow09=w9,
                           out=root/'runs'/'3d_membrane_localization_feasibility_v0_1',
                           roi=(0,48,0,48),sigma=1.0,panck_quantile=.8,
                           min_component_px=20,min_band_px=6,allow_synthetic_count=True)


def test_run_verify_and_no_overwrite(tmp_path):
    a=fixture_project(tmp_path)
    result=run(a)
    assert result['status']=='CREATED' and result['n_testable_domain_sections']>=1
    assert verify(a)['status']=='PASS'
    assert (a.out/'panCK_HER2_domain_contour_overlay.png').is_file()
    with pytest.raises(FileExistsError):run(a)


def test_modified_output_rejected(tmp_path):
    a=fixture_project(tmp_path);run(a)
    f=a.out/'domain_detection_qc.csv'
    f.write_text(f.read_text()+'garbage\n')
    with pytest.raises(ValueError,match='Output hash'):verify(a)


def test_modified_input_rejected(tmp_path):
    a=fixture_project(tmp_path);run(a)
    p=Path(json.loads(a.manifest.read_text())['markers'][0]['files'][0])
    with p.open('ab') as f:f.write(b'oops')
    with pytest.raises(ValueError,match='marker input changed'):verify(a)


def test_invalid_params_rejected(tmp_path):
    a=fixture_project(tmp_path)
    a.panck_quantile=.999
    with pytest.raises(ValueError):check_args(a)
