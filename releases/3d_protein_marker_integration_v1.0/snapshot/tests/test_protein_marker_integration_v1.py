"""Workflow09 tests are synthetic; no assertions of real specimen registration."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from protein_marker_integration_core import (sha256,marker_discovery,read_manifest,iter_marker_planes,
  aggregate_one,extract,verify_run)


def make_fixture(tmp_path):
    base=tmp_path/'imc';base.mkdir()
    mask=np.zeros((3,5,6),dtype=np.uint32)
    mask[0,0:2,0:2]=2;mask[1,0:2,0:2]=2
    mask[1,3:5,3:6]=10;mask[2,3:5,3:6]=10
    # persistent signal 2=4; 10=12 ; background=0
    marker=np.where(mask==2,4,np.where(mask==10,12,0)).astype(np.float32)
    maskpath=base/'full_mask_test.tif'
    tifffile.imwrite(maskpath,mask,photometric='minisblack')
    markerpath=base/'HER2_signal.tif'
    tifffile.imwrite(markerpath,marker,photometric='minisblack')
    zcyx=np.stack([marker,marker*2],axis=1)
    np.save(base/'multichannel.npy',zcyx.astype(np.float32))
    slices=[]
    for z in range(3):
        p=base/f'panCK_{z:02}.tif'
        tifffile.imwrite(p,marker[z],photometric='minisblack')
        slices.append(p)
    morph=tmp_path/'morph';morph.mkdir()
    spatial=tmp_path/'spatial';spatial.mkdir()
    pd.DataFrame([{'instance_id':2,'voxel_count':8,'z_span_planes':2,'centroid_y_px':.5,'centroid_x_px':.5},
                  {'instance_id':10,'voxel_count':12,'z_span_planes':2,'centroid_y_px':3.5,'centroid_x_px':4.}]).to_csv(morph/'cell_morphometry.csv',index=False)
    pd.DataFrame([{'instance_id':2,'qc_flag_review':False,'touches_z_stack_boundary':True},
                  {'instance_id':10,'qc_flag_review':False,'touches_z_stack_boundary':True}]).to_csv(morph/'cell_morphology_qc.csv',index=False)
    ms={'source_mask_sha256':sha256(maskpath),'shape_assumed_zyx':[3,5,6],'n_instance_ids':2}
    (morph/'morphometry_summary.json').write_text(json.dumps(ms))
    pd.DataFrame([{'instance_id':2,'n_xy_radius_neighbors':1},
                  {'instance_id':10,'n_xy_radius_neighbors':1}]).to_csv(spatial/'instance_spatial_features.csv',index=False)
    ss={'source_mask_sha256':sha256(maskpath),'workflow07_summary_sha256':sha256(morph/'morphometry_summary.json'),
        'shape_assumed_zyx':[3,5,6],'n_instances':2}
    (spatial/'spatial_organization_summary.json').write_text(json.dumps(ss))
    (spatial/'output_file_sha256.json').write_text(json.dumps({n:sha256(spatial/n) for n in
        ('instance_spatial_features.csv','spatial_organization_summary.json')}))
    specs=[{'name':'HER2','mode':'stack','path':str(markerpath),'axes':'ZYX'},
           {'name':'panCK','mode':'per_section','files':[str(p) for p in slices],'axes':'YX'}]
    manifest={'schema_version':1,'mask_path':str(maskpath),'expected_mask_sha256':sha256(maskpath),
              'image_coordinate_status':'unverified_assumed_same_index_grid','markers':specs}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    return {'mask':mask,'mask_path':maskpath,'marker':marker,'marker_path':markerpath,
            'manifest':path,'morph':morph,'spatial':spatial,'base':base,'data':manifest,'slices':slices}


def test_discovery_only_uses_headers(tmp_path):
    f=make_fixture(tmp_path)
    out=tmp_path/'discovery'
    r=marker_discovery(f['base'],out)
    assert r['n_image_files']==6


def test_discovery_refuses_overwrite(tmp_path):
    f=make_fixture(tmp_path);out=tmp_path/'discovery'
    marker_discovery(f['base'],out)
    with pytest.raises(FileExistsError):marker_discovery(f['base'],out)


def test_manifest_needs_explicit_marker(tmp_path):
    f=make_fixture(tmp_path)
    obj=f['data'];obj['markers']=[]
    f['manifest'].write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='No marker'):read_manifest(f['manifest'])


def test_cannot_use_mask_as_marker_image(tmp_path):
    f=make_fixture(tmp_path)
    f['data']['markers'][0]['path']=str(f['mask_path'])
    f['manifest'].write_text(json.dumps(f['data']))
    with pytest.raises(ValueError,match='instance ID mask'):
        extract(f['manifest'],f['morph'],f['spatial'],tmp_path/'res',True,False)


def test_per_section_vs_stack_equal(tmp_path):
    f=make_fixture(tmp_path)
    ids=np.array([2,10]);a=aggregate_one(f['mask'],ids,f['data']['markers'][0],tmp_path)[0]
    b=aggregate_one(f['mask'],ids,f['data']['markers'][1],tmp_path)[0]
    assert np.allclose(a['mean_raw_signal'],[4,12]);assert np.allclose(a['mean_raw_signal'],b['mean_raw_signal'])
    assert np.all(a['n_mask_voxels']==[8,12])


def test_multichannel_zcyx(tmp_path):
    f=make_fixture(tmp_path)
    spec={'name':'CD31','mode':'stack','path':str(f['base']/'multichannel.npy'),
          'axes':'ZCYX','channel_index':1}
    out=aggregate_one(f['mask'],np.array([2,10]),spec,tmp_path)[0]
    assert np.allclose(out.mean_raw_signal,[8,24])


def test_explicit_shape_mismatch(tmp_path):
    f=make_fixture(tmp_path)
    bad=np.zeros((3,5,7),dtype=np.float32)
    np.save(f['base']/'bad.npy',bad)
    spec={'name':'bad','mode':'stack','path':str(f['base']/'bad.npy'),'axes':'ZYX'}
    with pytest.raises(ValueError,match='plane'):
        list(iter_marker_planes(spec,tmp_path,f['mask'].shape))


def test_explicit_channel_required(tmp_path):
    f=make_fixture(tmp_path)
    obj=f['data'];obj['markers'].append({'name':'Ir193','mode':'stack','path':str(f['base']/'multichannel.npy'),'axes':'ZCYX'})
    f['manifest'].write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='channel_index required'):read_manifest(f['manifest'])


def test_explicit_geometry_ack(tmp_path):
    f=make_fixture(tmp_path)
    with pytest.raises(ValueError,match='allow-unverified'):
        extract(f['manifest'],f['morph'],f['spatial'],tmp_path/'res')


def test_upstream_mask_hash_protection(tmp_path):
    f=make_fixture(tmp_path)
    f['data']['expected_mask_sha256']='0'*64
    f['manifest'].write_text(json.dumps(f['data']))
    with pytest.raises(ValueError,match='expected_mask_sha256'):
        extract(f['manifest'],f['morph'],f['spatial'],tmp_path/'res',True,False)


def test_upstream_morph_volumes_protection(tmp_path):
    f=make_fixture(tmp_path)
    p=f['morph']/'cell_morphometry.csv';df=pd.read_csv(p);df.loc[0,'voxel_count']=100;df.to_csv(p,index=False)
    with pytest.raises(ValueError,match='voxel_count'):
        extract(f['manifest'],f['morph'],f['spatial'],tmp_path/'res',True,False)


def test_upstream_spatial_hash_protection(tmp_path):
    f=make_fixture(tmp_path)
    with (f['spatial']/'instance_spatial_features.csv').open('a') as fp:fp.write('\n')
    with pytest.raises(ValueError,match='Workflow08 output changed'):
        extract(f['manifest'],f['morph'],f['spatial'],tmp_path/'res',True,False)


def test_end_to_end_and_verify(tmp_path):
    f=make_fixture(tmp_path)
    out=tmp_path/'result'
    s=extract(f['manifest'],f['morph'],f['spatial'],out,True,False)
    assert s['n_instances']==2 and s['n_markers']==2 and s['n_instance_marker_rows']==4
    long=pd.read_csv(out/'instance_marker_signal.csv')
    assert list(long['mean_raw_signal'])==[4,12,4,12]
    merged=pd.read_csv(out/'instance_marker_spatial_table.csv')
    assert len(merged)==2 and 'mean_raw_HER2' in merged and 'n_xy_radius_neighbors' in merged
    assert verify_run(out,f['manifest'])['n_verified_files']==5
    with pytest.raises(FileExistsError):extract(f['manifest'],f['morph'],f['spatial'],out,True,False)


def test_manifest_changed_after_run(tmp_path):
    f=make_fixture(tmp_path)
    out=tmp_path/'result';extract(f['manifest'],f['morph'],f['spatial'],out,True,False)
    f['manifest'].write_text(f['manifest'].read_text()+' ')
    with pytest.raises(ValueError,match='manifest changed'):verify_run(out,f['manifest'])


def test_output_change_detected(tmp_path):
    f=make_fixture(tmp_path)
    out=tmp_path/'result';extract(f['manifest'],f['morph'],f['spatial'],out,True,False)
    with (out/'instance_marker_signal.csv').open('a') as p:p.write('x')
    with pytest.raises(ValueError,match='modified'):verify_run(out)


def test_nans_handled_without_inf_propagation(tmp_path):
    f=make_fixture(tmp_path)
    x=f['marker'].copy();x[0,0,0]=np.nan
    spec={'name':'HER2','mode':'stack','path':str(f['base']/'nan.npy'),'axes':'ZYX'}
    np.save(f['base']/'nan.npy',x)
    arr,prof,_,_=aggregate_one(f['mask'],np.array([2,10]),spec,tmp_path)
    assert arr.loc[0,'n_finite_signal_voxels']==7
    assert np.isfinite(arr.loc[0,'mean_raw_signal'])


def test_missing_plane_rejected(tmp_path):
    f=make_fixture(tmp_path)
    spec=f['data']['markers'][1].copy();spec['files']=spec['files'][:2]
    with pytest.raises(ValueError,match='expected 3 slice'):
        list(iter_marker_planes(spec,tmp_path,f['mask'].shape))


def test_unchanged_manifest_but_modified_source_marker_rejected(tmp_path):
    f=make_fixture(tmp_path)
    out=tmp_path/'res';extract(f['manifest'],f['morph'],f['spatial'],out,True,False)
    tifffile.imwrite(f['marker_path'],f['marker']+1,photometric='minisblack')
    with pytest.raises(ValueError,match='Source marker changed'):verify_run(out,f['manifest'])


def test_workflow07_file_hash_crosscheck(tmp_path):
    f=make_fixture(tmp_path)
    ss_path=f['spatial']/'spatial_organization_summary.json'
    ss=json.loads(ss_path.read_text())
    ss['workflow07_files_sha256']={'cell_morphometry.csv':sha256(f['morph']/'cell_morphometry.csv')}
    ss_path.write_text(json.dumps(ss))
    # update Workflow08 manifest to reflect new summary
    files=['instance_spatial_features.csv','spatial_organization_summary.json']
    (f['spatial']/'output_file_sha256.json').write_text(json.dumps({n:sha256(f['spatial']/n) for n in files}))
    m=f['morph']/'cell_morphometry.csv'
    df=pd.read_csv(m);df.loc[0,'centroid_y_px']+=.2;df.to_csv(m,index=False)
    with pytest.raises(ValueError,match='Workflow07 upstream file changed'):
        extract(f['manifest'],f['morph'],f['spatial'],tmp_path/'res',True,False)


def test_figure_overlays_saved(tmp_path):
    f=make_fixture(tmp_path)
    out=tmp_path/'res'
    result=extract(f['manifest'],f['morph'],f['spatial'],out,True,True)
    assert len(result['figures'])==2
    assert all((out/'figures'/fname).stat().st_size > 1000 for fname in result['figures'])
    assert verify_run(out)['n_verified_files']==7
