"""Workflow 11G: synthetic unit and manifest-backed full integration checks."""
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_spatial_signal_attribution import sha
from run_workflow11g_cadherin_segmentation import (
    args_validate,channel_members,verify_order,component_seed_map,segment_slice,
    shared_interface,region_metrics,summary_pairs,run,verify
)


def sample_images():
    yy,xx=np.indices((64,64))
    ck=100*np.exp(-((yy-31)**2+(xx-31)**2)/380)
    nuc=np.zeros((64,64),np.uint32)
    nuc[23:27,23:27]=15; nuc[34:38,35:39]=16
    cad=1+70*np.exp(-((xx-31)**2)/1.7) * np.exp(-((yy-30)**2)/250)
    her2=70*np.exp(-((xx-31)**2)/1.7) * np.exp(-((yy-30)**2)/250)
    return ck,nuc,cad,her2


def test_seed_map_keeps_source_provenance():
    domain=np.ones((12,12),np.int32)
    nuc=np.zeros((12,12),np.int32)
    nuc[2:4,2:4]=4;nuc[6:8,7:9]=4
    seeds,rows=component_seed_map(nuc,domain,4)
    assert len(rows)==2 and len(np.unique(seeds[seeds>0]))==2
    assert all(r['source_pseudo_id']==4 for r in rows)


def test_no_nucleus_domain_has_no_candidate():
    ck,nuc,cad,_=sample_images()
    result=segment_slice(ck,nuc*0,cad)
    assert not result[2] and not np.any(result[3]) and not np.any(result[4])


def test_cadherin_guided_output_and_heldout_invariance():
    ck,nuc,cad,her2=sample_images()
    domain,seedmap,seedrows,baseline,assisted,q,rs=segment_slice(ck,nuc,cad)
    assert len(seedrows)==2
    assert (domain>0).sum()>0
    assert (baseline>0).sum()==(assisted>0).sum()
    assert np.array_equal(baseline[seedmap>0],seedmap[seedmap>0])
    assert np.array_equal(assisted[seedmap>0],seedmap[seedmap>0])
    # Segmentation API has no HER2 argument by design: held-out marker cannot drive boundaries.
    assert 'her2' not in segment_slice.__code__.co_varnames
    old=region_metrics(assisted,her2,cad,seedrows,'cadherin_assisted')
    alt=region_metrics(assisted,her2*0,cad,seedrows,'cadherin_assisted')
    assert len(old)==len(alt)==len(seedrows)
    assert any((a['HER2_interface_mean']!=b['HER2_interface_mean']) for a,b in zip(old,alt))


def test_shared_interface_excludes_background():
    a=np.array([[0,1,1,2,2,0],[0,1,1,2,2,0]])
    x=shared_interface(a)
    assert x[:,2:4].all()
    assert not x[:,0].any() and not x[:,5].any()
    assert not x[:,1].any() and not x[:,4].any()


def test_unnormalized_cadherin_flat_no_failure():
    ck,nuc,_,_=sample_images()
    d,seed,rows,base,ass,q,_=segment_slice(ck,nuc,np.zeros_like(ck))
    assert len(rows)==2 and np.all(base[seed>0]==seed[seed>0])


def test_one_seed_is_design_property_not_independent_metric():
    ck,nuc,cad,her2=sample_images()
    d,markers,rows,b,a,q,_=segment_slice(ck,nuc,cad)
    for method,lab in [('baseline',b),('cadherin',a)]:
        assert np.array_equal(np.unique(lab[lab>0]), np.unique(markers[markers>0]))
        assert len(region_metrics(lab,her2,cad,rows,method))==len(rows)


def fixture_project(tmp_path):
    root=tmp_path/'synthetic_project'
    data=root/'data';data.mkdir(parents=True)
    w9=root/'runs'/'3d_protein_marker_integration_v1';w9.mkdir(parents=True)
    cfg=root/'configs';cfg.mkdir()
    markers=['Ir193','panCK','HER2','CD31','LYVE1']
    ck,nuc,cad,her2=sample_images()
    mask=np.tile(nuc,(16,1,1))
    mpath=data/'mask.tif'
    tifffile.imwrite(mpath,mask,metadata={'axes':'ZYX'})
    raw={'Ir193':(nuc>0).astype(float)*60,'panCK':ck,'HER2':her2,
         'CD31':np.ones_like(ck),'LYVE1':np.ones_like(ck)*3}
    files_by={};specs=[];summary_markers=[];ref={m:[] for m in markers}
    for m in markers:
        files=[];h={}
        for z in range(16):
            f=data/f'{z+1:04d}_synthetic_section_{z+1:02d}__{m}.tiff'
            a=np.clip(np.round(raw[m]+z/100),0,65535).astype(np.uint16)
            tifffile.imwrite(f,a)
            files.append(str(f.resolve()));h[str(f.resolve())]=sha(f);ref[m].append(a)
        files_by[m]=files
        specs.append({'name':m,'mode':'per_section','axes':'YX','files':files})
        summary_markers.append({'name':m,'files_sha256':h})
    yzip=data/'registered_yb174.zip'
    with zipfile.ZipFile(yzip,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for zi in range(16):
            arr=np.clip(cad,0,65535).astype(np.uint16)
            with io.BytesIO() as b:
                tifffile.imwrite(b,arr)
                z.writestr(f'SIMILARITY10_In115/Yb174/{zi+1:04d}_synthetic_section_{zi+1:02d}__Yb174.tiff',b.getvalue())
    manifest=cfg/'registered.json'
    manifest.write_text(json.dumps({'markers':specs,'expected_mask_sha256':sha(mpath)}))
    frame={'instance_id':[15,16],'centroid_y_px':[24.5,35.5],'centroid_x_px':[24.5,36.5],
           'z_span_planes':[16,16],'voxel_count':[int((mask==i).sum()) for i in [15,16]]}
    mean={'instance_id':[15,16]}
    for marker in markers:
        stack=np.stack(ref[marker]); vals=[float(stack[mask==i].mean()) for i in [15,16]]
        mean['mean_raw_'+marker]=vals;frame['mean_raw_'+marker]=vals
    pd.DataFrame(frame).to_csv(w9/'instance_marker_spatial_table.csv',index=False)
    pd.DataFrame(mean).to_csv(w9/'instance_marker_mean_matrix.csv',index=False)
    summary={'marker_names':markers,'n_instances':2,'markers':summary_markers,
             'manifest_sha256':sha(manifest),'source_mask_sha256':sha(mpath),
             'mask_shape_assumed_zyx':[16,64,64]}
    (w9/'marker_integration_summary.json').write_text(json.dumps(summary))
    names=['marker_integration_summary.json','instance_marker_spatial_table.csv','instance_marker_mean_matrix.csv']
    (w9/'output_file_sha256.json').write_text(json.dumps({n:sha(w9/n) for n in names}))
    return SimpleNamespace(mask=mpath,manifest=manifest,workflow09=w9,cadherin_zip=yzip,
        out=root/'runs'/'wf11g',roi=(0,64,0,64),sigma=1.,panck_quantile=.8,
        min_domain_px=40,min_seed_overlap_px=4,cadherin_weight=.75,
        min_interface_px=1,min_core_px=1,allow_synthetic_count=True)


def test_source_zip_section_match(tmp_path):
    a=fixture_project(tmp_path)
    import json
    manifest=json.loads(a.manifest.read_text())
    p={v['name']:[Path(x) for x in v['files']] for v in manifest['markers']}
    verify_order(channel_members(a.cadherin_zip),p)
    with zipfile.ZipFile(a.cadherin_zip) as old:
        items={k:old.read(k) for k in old.namelist()}
    wrong=a.cadherin_zip.with_name('bad_registered.zip')
    with zipfile.ZipFile(wrong,'w') as out:
        for k,v in items.items():
            out.writestr(k.replace('synthetic_section','wrong_section'),v)
    with pytest.raises(ValueError,match='source section mismatch'):
        verify_order(channel_members(wrong),p)


def test_complete_run_verify_provenance_and_no_overwrite(tmp_path):
    a=fixture_project(tmp_path)
    result=run(a)
    assert result['status']=='CREATED' and result['n_z']==16
    assert result['n_seed_fragments']>=16
    assert verify(a)['status']=='PASS'
    table=pd.read_csv(a.out/'heldout_HER2_matched_comparison.csv')
    assert table[['z','seed_id']].duplicated().sum()==0
    assert (a.out/'cadherin_vs_voronoi_heldout_HER2.png').is_file()
    with pytest.raises(FileExistsError):run(a)


def test_modified_output_rejected(tmp_path):
    a=fixture_project(tmp_path);run(a)
    p=a.out/'candidate_boundary_metrics.csv'
    with p.open('a') as f:f.write('\ncorrupted,output\n')
    with pytest.raises(RuntimeError,match='modified result'):verify(a)


def test_modified_yb174_zip_rejected(tmp_path):
    a=fixture_project(tmp_path);run(a)
    with a.cadherin_zip.open('ab') as f:f.write(b'changed')
    with pytest.raises(RuntimeError,match='provenance changed'):verify(a)


def test_invalid_domain_weight(tmp_path):
    a=fixture_project(tmp_path);a.cadherin_weight=1.01
    with pytest.raises(ValueError):args_validate(a)


def test_missing_registered_zip(tmp_path):
    a=fixture_project(tmp_path);a.cadherin_zip.unlink()
    with pytest.raises(FileNotFoundError):args_validate(a)


def test_wrong_verify_params_rejected(tmp_path):
    a=fixture_project(tmp_path);run(a)
    a.cadherin_weight=.35
    with pytest.raises(RuntimeError,match='Verification parameters'):verify(a)


def test_review_queue_is_not_annotation_truth(tmp_path):
    a=fixture_project(tmp_path);run(a)
    d=pd.read_csv(a.out/'manual_boundary_review_queue.csv')
    assert d.manual_annotation_status.eq('PENDING').all()
    assert d.manual_boundary_quality.eq('UNASSESSED').all()


def test_seeded_segmentation_is_local_to_z(tmp_path):
    a=fixture_project(tmp_path);run(a)
    d=pd.read_csv(a.out/'section_nuclear_seed_index.csv')
    assert d.duplicated(['z','seed_id']).sum()==0
    assert d.duplicated(['seed_id']).sum()>0  # resets every Z: NOT tracked 3D cells


def test_output_comparison_reports_heldout_metric(tmp_path):
    a=fixture_project(tmp_path);run(a)
    q=json.loads((a.out/'workflow11g_summary.json').read_text())
    assert q['marker_role']['HER2']=='HELD-OUT READOUT ONLY'
    assert q['n_matched_seed_rows']==len(pd.read_csv(a.out/'heldout_HER2_matched_comparison.csv'))
