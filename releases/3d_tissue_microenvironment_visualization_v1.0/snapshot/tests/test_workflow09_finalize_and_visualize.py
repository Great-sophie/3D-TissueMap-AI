"""Synthetic no-patient-data tests for Workflow09 reproducible analysis/freeze and Workflow10 HTML viewer."""
import importlib.util,json,shutil,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import tifffile

BASE=Path(__file__).resolve().parents[1]

def load(relative):
    path=BASE/relative
    spec=importlib.util.spec_from_file_location(path.stem,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[path.stem]=mod
    spec.loader.exec_module(mod)
    return mod

analysis=load('scripts/analyze_protein_spatial_v1.py')
freezer=load('tools/freeze_workflow09.py')
viewer=load('scripts/build_3d_tissue_viewer.py')

def fake_table(n=90):
    a=np.arange(n)
    return pd.DataFrame({'instance_id':a+1,'centroid_y_px':(a//10)*25.,
                         'centroid_x_px':(a%10)*30.,'z_span_planes':1+a%4,
                         'voxel_count':20+a%30,'n_xy_radius_neighbors':2+a%15,
                         'qc_flag_review':a%4==0,
                         'mean_raw_Ir193':20+a%35,
                         'mean_raw_panCK':a*.014,
                         'mean_raw_HER2':a*.02 + (a%3)*.11,
                         'mean_raw_CD31':a*.008,
                         'mean_raw_LYVE1':(a%7)+ a*.01})

def test_correlations_and_blocks():
    tab=fake_table(120)
    global_values,byblock,summ,strata=analysis.report(tab,sizes=(120,200),min_block_n=5)
    assert global_values[0]['global_rho']>0
    assert {r['block_px'] for r in summ}=={120,200}
    assert len(byblock)>0 and len(strata)>0
    assert all(r['n_blocks']>=0 for r in summ)

def test_analysis_reproducible_and_refuses_overwrite(tmp_path):
    f=tmp_path/'spatial.csv';fake_table().to_csv(f,index=False)
    out=tmp_path/'analysis'
    answer=analysis.run(f,out)
    assert len(answer['global_results'])==6
    assert analysis.verify(f,out)['files_verified']==5
    with pytest.raises(FileExistsError):analysis.run(f,out)
    (out/'block_summary.csv').write_text('corrupted')
    with pytest.raises(RuntimeError,match='Changed'):analysis.verify(f,out)

def test_analysis_detects_input_modification(tmp_path):
    f=tmp_path/'src.csv';fake_table().to_csv(f,index=False)
    out=tmp_path/'out';analysis.run(f,out)
    f.write_text(f.read_text()+'\n')
    with pytest.raises(RuntimeError,match='changed'):analysis.verify(f,out)

def test_analysis_rejects_duplicate_ids_and_missing_fields():
    tab=fake_table()
    tab.loc[3,'instance_id']=tab.loc[2,'instance_id']
    with pytest.raises(ValueError,match='Duplicate'):analysis.report(tab)
    with pytest.raises(ValueError,match='Missing'):analysis.report(fake_table().drop(columns=['mean_raw_HER2']))

def test_roi_densest_and_voxel_sampling():
    arr=np.zeros((4,32,40),dtype=np.uint32)
    arr[1:3,10:18,19:27]=7
    y,x,side,count=viewer.densest_roi(arr,12)
    assert count==128 and y<=10<y+side and x<=19<x+side
    d=fake_table(2);d.loc[0,'instance_id']=7;d.loc[1,'instance_id']=8
    fig,report=viewer.make_roi(arr,d,roi_side=12,max_points=60)
    assert report['visualized_voxels']==60
    assert report['unique_visible_instance_ids']==1
    assert 'index' in str(fig.layout.scene.zaxis.title.text)

def test_centroid_dropdown_has_all_markers():
    df=fake_table(25)
    df['centroid_z_index']=2.
    fig=viewer.make_overview(df)
    assert len(fig.data)==6
    assert len(fig.layout.updatemenus[0].buttons)==6
    assert [d.visible for d in fig.data]==[True,False,False,False,False,False]

def test_end_to_end_offline_viewer_with_validation(tmp_path):
    root=tmp_path
    (root/'runs'/'3d_protein_marker_integration_v1').mkdir(parents=True)
    (root/'runs'/'3d_cell_morphometry_v1').mkdir(parents=True)
    (root/'data').mkdir()
    mask=np.zeros((3,16,16),dtype=np.uint32)
    mask[:,2:7,1:5]=1
    mask[:,9:14,8:15]=2
    mp=root/'data'/'mask.tif';tifffile.imwrite(mp,mask,photometric='minisblack')
    df=fake_table(2);df.loc[0,'instance_id']=1;df.loc[1,'instance_id']=2
    df.loc[0,['centroid_y_px','centroid_x_px']]=[4,3]
    df.loc[1,['centroid_y_px','centroid_x_px']]=[11,10]
    src=root/'runs'/'3d_protein_marker_integration_v1'/'instance_marker_spatial_table.csv'
    df.to_csv(src,index=False)
    mom=root/'runs'/'3d_cell_morphometry_v1'/'cell_morphometry.csv'
    pd.DataFrame({'instance_id':[1,2],'centroid_z_index':[1.,1.]}).to_csv(mom,index=False)
    sp=src.parent/'marker_integration_summary.json'
    sp.write_text(json.dumps({'n_instances':2,'marker_names':viewer.MARKERS,
                              'source_mask_sha256':viewer.sha(mp),
                              'mask_shape_assumed_zyx':[3,16,16]}))
    (src.parent/'output_file_sha256.json').write_text(json.dumps({'instance_marker_spatial_table.csv':viewer.sha(src)}))
    out=root/'viewer'
    r=viewer.make(mp,src,sp,out,roi_side=10,max_points=100)
    assert r['n_instances']==2 and r['roi']['visualized_voxels']<=100
    assert (out/'centroid_marker_viewer.html').stat().st_size>100000
    assert viewer.verify(out,src,mp,sp)['verified_viewer_files']==3
    with pytest.raises(FileExistsError):viewer.make(mp,src,sp,out)
    (out/'roi_voxel_viewer.html').write_text('broken')
    with pytest.raises(RuntimeError,match='changed'):viewer.verify(out,src,mp,sp)

def test_freeze_preflight_refuses_missing_inputs(tmp_path):
    with pytest.raises(RuntimeError,match='unavailable'):
        freezer.preflight(tmp_path)

def test_freeze_cannot_overwrite(tmp_path):
    dest=tmp_path/freezer.REL
    dest.mkdir(parents=True)
    with pytest.raises(FileExistsError):freezer.make(tmp_path,{'source_code_hashes':{},'n_instances':12})

def test_freeze_release_manifest_and_mutation(tmp_path):
    # Minimal synthetic archive; no reliance on user lab data.
    root=tmp_path
    rel='scripts/foo.py'
    (root/'scripts').mkdir();(root/rel).write_text('print(1)\n')
    report={'source_code_hashes':{rel:freezer.sha(root/rel)},'n_instances':2,'n_markers':5,
            'marker_names':['Ir193','panCK','HER2','CD31','LYVE1'],'n_instance_marker_rows':10,
            'marker_geometry_status':'UNVERIFIED','source_mask_sha256':'test',
            'workflow07_summary_sha256':'test','workflow08_summary_sha256':'test',
            'manifest_sha256':'test','workflow09c_outputs':{},'workflow09c_manifest_hash':'test',
            'workflow09d_outputs':{},'workflow09d_manifest_hash':'test'}
    dest=freezer.make(root,report)
    record=json.loads((dest/'release_sha256.json').read_text())
    assert 'snapshot/scripts/foo.py' in record
    assert record==freezer.manifest_release(dest)
    with pytest.raises(FileExistsError):freezer.make(root,report)
    (dest/'snapshot'/rel).write_text('corrupt')
    assert record!=freezer.manifest_release(dest)


def create_full_synthetic_freeze_fixture(root):
    for rel in freezer.REQUIRED:
        f=root/rel;f.parent.mkdir(parents=True,exist_ok=True)
        f.write_text('test source snapshot\n')
    manifest=root/'configs/protein_marker_manifest_registered_5channels.json'
    manifest.write_text('{"schema_version":1}')
    (root/'runs'/'3d_cell_morphometry_v1').mkdir(parents=True)
    (root/'runs'/'3d_spatial_organization_v1').mkdir(parents=True)
    mask=root/'data/inspect/LVILymph/full_mask_final_segmentation_hwatershed_bg500_90%.tif'
    mask.parent.mkdir(parents=True);mask.write_bytes(b'mock mask')
    ms=root/'runs/3d_cell_morphometry_v1/morphometry_summary.json'
    ms.write_text(json.dumps({'source_mask_path':str(mask)}))
    ss=root/'runs/3d_spatial_organization_v1/spatial_organization_summary.json'
    ss.write_text('{"n_instances":12663}')
    out=root/freezer.RUN;out.mkdir(parents=True)
    for name,number in freezer.REQ_TABLES.items():
        with (out/name).open('w') as f:
            f.write('id\n')
            f.writelines('1\n' for _ in range(number))
    source_marker=root/'data/processed/workflow09_five_markers/Ir193/plane1.tif'
    source_marker.parent.mkdir(parents=True);source_marker.write_bytes(b'mock marker')
    summary={'version':'0.9.0-audit-first','n_instances':12663,'n_markers':5,
             'n_instance_marker_rows':63315,'marker_names':freezer.REQUIRED and viewer.MARKERS,
             'no_protein_table_join':True,'source_mask_sha256':freezer.sha(mask),
             'workflow07_summary_sha256':freezer.sha(ms),'workflow08_summary_sha256':freezer.sha(ss),
             'manifest_sha256':freezer.sha(manifest),
             'marker_geometry_status':'EXPLICIT_AXIS_AND_SHAPE_CHECK_ONLY',
             'markers':[{'name':'Ir193','files_sha256':{str(source_marker):freezer.sha(source_marker)}}]}
    (out/'marker_integration_summary.json').write_text(json.dumps(summary))
    h={p.name:freezer.sha(p) for p in out.iterdir()}
    (out/'output_file_sha256.json').write_text(json.dumps(h))
    ana=root/freezer.ANA;ana.mkdir(parents=True)
    a_summary={'n_instances':12663,'source_table_sha256':freezer.sha(out/'instance_marker_spatial_table.csv')}
    (ana/'association_summary.json').write_text(json.dumps(a_summary))
    (ana/'block_summary.csv').write_text('block_px,n_blocks\n120,38\n')
    ah={p.name:freezer.sha(p) for p in ana.iterdir()}
    (ana/'output_sha256.json').write_text(json.dumps(ah))
    return mask,source_marker

def test_full_release_preflight_apply_verify_and_mutated_sources(tmp_path):
    mask,source_marker=create_full_synthetic_freeze_fixture(tmp_path)
    pre=freezer.preflight(tmp_path)
    assert pre['n_instances']==12663 and pre['n_markers']==5
    dest=freezer.make(tmp_path,pre)
    assert freezer.verify(tmp_path)['n_instances']==12663
    with pytest.raises(FileExistsError):freezer.make(tmp_path,pre)
    source_marker.write_bytes(b'mock marker CHANGED')
    with pytest.raises(RuntimeError,match='modified'):
        freezer.verify(tmp_path)
