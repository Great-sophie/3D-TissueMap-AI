"""Deterministic, fast tests for Workflow08. Does not require real IMC data."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from spatial_organization_core import (xy_radius_graph, voxel_interface_graph,
    spatial_features, build_spatial, save_run, verify_run, validate_source, sha256)
from run_3d_spatial_organization import main


def make_mask():
    m=np.zeros((3,8,10),dtype=np.uint32)
    m[0,2,2:4]=1
    m[1,2,2]=1
    m[0,2,4]=2
    m[1,2,3:5]=2
    m[2,2,2]=3
    m[2,5,8]=4  # distinct Z plane, no shared Z with 1,2
    return m


def fixture_tables(mask):
    rows=[];profiles=[];qc=[]
    for idx in sorted(int(x) for x in np.unique(mask) if x):
        zyx=np.argwhere(mask==idx)
        z,y,x=zyx.T
        rows.append({'instance_id':idx,'voxel_count':len(z),'centroid_z_index':z.mean(),
                     'centroid_y_px':y.mean(),'centroid_x_px':x.mean(),
                     'z_span_planes':int(z.max()-z.min()+1)})
        for k in sorted(set(z)):
            xy=zyx[z==k][:,1:]
            profiles.append({'instance_id':idx,'z_index':k,'area_px2':len(xy),
                             'centroid_y_px':xy[:,0].mean(),'centroid_x_px':xy[:,1].mean()})
        qc.append({'instance_id':idx,'qc_flag_review':bool(idx==4),
                   'touches_z_stack_boundary':bool(idx==4),'components_6connected':1})
    morph=pd.DataFrame(rows)
    profile=pd.DataFrame(profiles)
    q=pd.DataFrame(qc)
    c=voxel_interface_graph(mask)
    d={i:0 for i in morph.instance_id}
    for r in c.itertuples(index=False):
        d[r.instance_a]+=r.all_face_count_voxel_grid
        d[r.instance_b]+=r.all_face_count_voxel_grid
    morph['other_instance_interface_faces_voxel']=[d[i] for i in morph.instance_id]
    return morph,profile,q


def write_fixture(tmp_path):
    mask=make_mask()
    mask_path=tmp_path/'mask.tif'
    tifffile.imwrite(mask_path,mask,photometric='minisblack',metadata={'axes':'ZYX'})
    m,p,q=fixture_tables(mask)
    md=tmp_path/'morph';md.mkdir()
    m.to_csv(md/'cell_morphometry.csv',index=False)
    p.to_csv(md/'cell_slice_profiles.csv',index=False)
    q.to_csv(md/'cell_morphology_qc.csv',index=False)
    summary={'shape_assumed_zyx':list(mask.shape),'source_mask_sha256':sha256(mask_path),
             'n_instance_ids':len(m),'foreground_voxels':int((mask>0).sum())}
    (md/'morphometry_summary.json').write_text(json.dumps(summary))
    return mask,mask_path,md,m,p,q


def test_xy_pairs_require_same_z():
    mask=make_mask();m,p,q=fixture_tables(mask)
    neighbors=xy_radius_graph(p,10.,2)
    pairs=set(zip(neighbors.instance_a,neighbors.instance_b))
    assert (1,2) in pairs
    assert (1,3) not in pairs and (2,3) not in pairs
    assert (3,4) in pairs  # same z plane, in radius
    rec=neighbors.loc[(neighbors.instance_a==1)&(neighbors.instance_b==2)].iloc[0]
    assert rec.min_xy_plane_centroid_distance_px==pytest.approx(1.5)
    assert rec.shared_z_planes_within_radius==2


def test_xy_radius_noncandidate_has_empty_schema():
    m,p,q=fixture_tables(make_mask())
    result=xy_radius_graph(p,.001,5)
    assert result.empty
    assert {'instance_a','topk_by_either','topk_by_both'}.issubset(result.columns)


def test_radius_limited_top_k_not_unrestricted():
    p=pd.DataFrame({'instance_id':[1,2,3,4],'z_index':[0]*4,'area_px2':[1]*4,
                    'centroid_y_px':[0.]*4,'centroid_x_px':[0.,1.,2.,100.]})
    n=xy_radius_graph(p,2.1,1)
    assert not ((n.instance_b==4)|(n.instance_a==4)).any()
    assert len(n)==3
    assert n.topk_by_either.any() and not n.topk_by_both.all()


def test_contact_xy_and_z_are_distinct():
    c=voxel_interface_graph(make_mask())
    r=c.set_index(['instance_a','instance_b']).loc[(1,2)]
    assert r.x_face_count_voxel_grid==2  # one XY face on each of z=0 and z=1
    assert r.z_face_count_voxel_grid==1
    assert r.xy_face_count_voxel_grid==2
    assert r.interface_type=='mixed_XY_and_Z'
    assert c.all_face_count_voxel_grid.sum()>=2


def test_z_only_interface_not_xy_radius():
    mask=np.zeros((2,3,3),dtype=np.uint32)
    mask[0,1,1]=5;mask[1,1,1]=6
    c=voxel_interface_graph(mask)
    assert len(c)==1
    assert c.iloc[0].interface_type=='Z_only'
    m,p,q=fixture_tables(mask)
    assert xy_radius_graph(p,100,3).empty


def test_no_3d_distance_in_outputs():
    m=make_mask();mor,p,q=fixture_tables(m)
    outputs=build_spatial(m,mor,p,q,10.,2)
    assert len(outputs['instance_centroids.csv'])==4
    f=outputs['instance_spatial_features.csv']
    assert 'nearest_xy_within_radius_px' in f
    assert not any('euclidean_3d' in x or 'distance_um' in x for x in f.columns)
    assert int(f.set_index('instance_id').loc[4,'n_z_face_interface_neighbors'])==0


def test_provenance_reject_bad_morphometry(tmp_path):
    mask,mp,md,m,p,q=write_fixture(tmp_path)
    m.loc[0,'voxel_count']+=1
    with pytest.raises(ValueError,match='voxel counts'):
        validate_source(mask,m,p,q,json.loads((md/'morphometry_summary.json').read_text()),mp)


def test_provenance_reject_bad_slice(tmp_path):
    mask,mp,md,m,p,q=write_fixture(tmp_path)
    p.loc[0,'area_px2']+=1
    with pytest.raises(ValueError,match='Plane 0'):
        validate_source(mask,m,p,q,json.loads((md/'morphometry_summary.json').read_text()),mp)


def test_provenance_reject_wrong_hash(tmp_path):
    mask,mp,md,m,p,q=write_fixture(tmp_path)
    summary=json.loads((md/'morphometry_summary.json').read_text());summary['source_mask_sha256']='0'*64
    with pytest.raises(ValueError,match='SHA-256'):
        validate_source(mask,m,p,q,summary,mp)


def test_full_run_atomic_verify_and_no_overwrite(tmp_path):
    mask,mp,md,m,p,q=write_fixture(tmp_path)
    out=tmp_path/'out'
    report=save_run(mask,mp,md,out,radius_px=10.,knn_k=2,figures=False)
    assert report['workflow07_interface_face_crosscheck']=='PASS'
    assert report['n_instances']==4
    assert set(x.name for x in out.iterdir())=={
       'instance_centroids.csv','spatial_neighbors.csv','instance_contact_graph.csv',
       'instance_spatial_features.csv','spatial_organization_summary.json','output_file_sha256.json'}
    assert verify_run(out,mp)['n_instances']==4
    with pytest.raises(FileExistsError):
        save_run(mask,mp,md,out,figures=False)
    assert not list(tmp_path.glob('.workflow08_staging_*'))


def test_tamper_detection(tmp_path):
    mask,mp,md,m,p,q=write_fixture(tmp_path)
    out=tmp_path/'out'
    save_run(mask,mp,md,out,figures=False)
    with (out/'instance_centroids.csv').open('a') as f:
        f.write('corrupted\n')
    with pytest.raises(ValueError,match='changed'):
        verify_run(out)


def test_cli_end_to_end(tmp_path):
    mask,mp,md,m,p,q=write_fixture(tmp_path)
    out=tmp_path/'cli'
    main(['--mask',str(mp),'--morphometry-dir',str(md),'--out',str(out),
          '--assume-zyx','--xy-radius-px','10','--no-figures'])
    result=main(['--verify-run','--out',str(out),'--mask',str(mp)])
    assert result['n_instances']==4


def test_cli_refuses_implicit_zyx(tmp_path):
    mask,mp,md,m,p,q=write_fixture(tmp_path)
    with pytest.raises(SystemExit,match='assume-zyx'):
        main(['--mask',str(mp),'--morphometry-dir',str(md),'--out',str(tmp_path/'r')])


def test_figures(tmp_path):
    mask,mp,md,m,p,q=write_fixture(tmp_path)
    save_run(mask,mp,md,tmp_path/'fig_run',radius_px=10.,figures=True)
    assert len(list((tmp_path/'fig_run'/'figures').glob('*.png')))==3
    assert verify_run(tmp_path/'fig_run')['n_instances']==4
