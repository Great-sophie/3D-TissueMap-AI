"""Workflow11 functional/synthetic tests, no clinical or marker identity claims."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from spatial_signal_attribution_core import MARKERS, REGIONS, RegionAccumulator, assign_slice
from run_spatial_signal_attribution import file_manifest, run, sha, verify, make_long, hotspot_table


def test_region_assignment_distance_thresholds_and_no_double_count():
    m=np.zeros((27,27),dtype=np.uint32)
    m[12,12]=1
    owner=assign_slice(m,1)
    assert owner.regions[12,12]==0
    assert owner.regions[12,13]==1
    assert owner.regions[12,14]==1
    assert owner.regions[12,15]==2
    assert owner.regions[12,17]==2
    assert owner.regions[12,18]==3
    assert owner.regions[12,22]==3
    assert owner.regions[12,23]==255
    assert len(owner.codes)==len(owner.selected)==len(np.unique(owner.selected))
    assert np.array_equal(owner.owners, np.ones_like(owner.owners))


def test_two_nuclei_exclusive_ownership_and_preserve_mask():
    m=np.zeros((28,28),dtype=np.uint32)
    m[12,6]=1;m[12,20]=2
    own=assign_slice(m,2)
    assert len(own.selected)==len(set(own.selected.tolist()))
    decoded_ids=own.codes % 3
    decoded_region=own.codes // 3
    assert set(decoded_ids)=={1,2}
    assert decoded_ids[own.selected.tolist().index(np.ravel_multi_index((12,6),m.shape))]==1
    assert decoded_ids[own.selected.tolist().index(np.ravel_multi_index((12,20),m.shape))]==2
    assert decoded_region[own.selected.tolist().index(np.ravel_multi_index((12,6),m.shape))]==0
    assert decoded_region[own.selected.tolist().index(np.ravel_multi_index((12,20),m.shape))]==0


def test_one_plane_empty_nuclear_mask():
    owner=assign_slice(np.zeros((12,15),dtype=np.uint32),2)
    assert len(owner.selected)==0
    assert np.all(owner.regions==255)


@pytest.mark.parametrize('wrong',[-1,0,11])
def test_bad_radius_order_rejected(wrong):
    with pytest.raises(ValueError):assign_slice(np.ones((8,8),dtype=np.uint32),1,limits=(2,wrong,10))


def test_accumulator_validity_and_nonoverlap():
    m=np.zeros((15,20),np.uint32)
    m[4,4]=1;m[4,16]=2
    own=assign_slice(m,2)
    acc=RegionAccumulator(2)
    imgs={k:np.full(m.shape, float(j+1)) for j,k in enumerate(MARKERS)}
    imgs['Ir193'][4,4]=101
    imgs['Ir193'][4,16]=202
    acc.add_plane(own,imgs)
    mean=acc.stats('Ir193')[0]
    assert mean[0,1]==101 and mean[0,2]==202
    assert mean[1,1]==1 and mean[1,2]==1
    assert acc.count.reshape(4,-1)[0].sum()==2
    assert acc.count.sum()==len(own.codes)


def test_missing_measurements_do_not_create_false_zero():
    m=np.zeros((9,9),np.uint32)
    m[4,4]=1
    own=assign_slice(m,1)
    acc=RegionAccumulator(1)
    a={name:np.ones(m.shape,dtype=float) for name in MARKERS}
    a['HER2'][4,4]=np.nan
    acc.add_plane(own,a)
    assert acc.n_finite['HER2'][1]==0
    assert np.isnan(acc.stats('HER2')[0][0,1])
    assert acc.stats('HER2')[0][1,1]==1


def test_min_support_returns_nan_not_zeros():
    m=np.zeros((9,9),np.uint32);m[4,4]=1
    own=assign_slice(m,1);acc=RegionAccumulator(1)
    acc.add_plane(own,{k:np.ones(m.shape) for k in MARKERS})
    long,wide=make_long(acc,np.array([1]),10)
    assert pd.isna(wide.loc[0,'HER2__nuclear_mask'])
    assert np.isnan(long.loc[(long.marker=='HER2')&(long.region=='nuclear_mask'),'mean_raw_signal'].iloc[0])
    assert wide.loc[0,'HER2__perinuclear_1_2px']==1


def fixture_project(tmp_path):
    root=tmp_path/'project'
    data=root/'data';data.mkdir(parents=True)
    w9=root/'runs'/'3d_protein_marker_integration_v1';w9.mkdir(parents=True)
    config=root/'configs';config.mkdir()
    image_root=data/'processed';image_root.mkdir()
    mask=np.zeros((2,24,28),np.uint32)
    mask[0,7:9,7:9]=1;mask[1,7:9,8:10]=1
    mask[0,10:13,18:20]=2;mask[1,11:14,18:20]=2
    mask[0,19:21,15:17]=3;mask[1,19:21,16:18]=3
    mp=data/'mask.tif';tifffile.imwrite(mp,mask)
    panel=[];specs=[];arrays={}
    for j,k in enumerate(MARKERS):
        zfiles=[];digest={}
        frames=[]
        for z in range(2):
            a=np.full(mask.shape[1:],j+1,dtype=np.uint16)
            # Constant per instance nucleus across Z but marker-specific, so baseline audit is exact.
            a[mask[z]>0]=(100*(j+1) + mask[z][mask[z]>0])
            # Non-nuclear image signal concentrated near one instance.
            a[3:6,4:10]=20*(j+1)
            f=image_root/f'{k}_{z:02d}.tiff';tifffile.imwrite(f,a)
            zfiles.append(str(f.resolve()));digest[str(f.resolve())]=sha(f)
            frames.append(a)
        arrays[k]=np.stack(frames)
        specs.append({'name':k,'mode':'per_section','axes':'YX','files':zfiles})
        panel.append({'name':k,'files_sha256':digest})
    manifest=config/'protein_marker_manifest_registered_5channels.json'
    manifest.write_text(json.dumps({'markers':specs,'expected_mask_sha256':sha(mp)}))
    ids=np.array([1,2,3])
    frame={'instance_id':ids}
    mean={'instance_id':ids}
    y=[];x=[];counts=[]
    for id in ids:
        pos=np.argwhere(mask==id)
        y.append(pos[:,1].mean());x.append(pos[:,2].mean());counts.append(len(pos))
    frame.update({'centroid_y_px':y,'centroid_x_px':x,'z_span_planes':[2,2,2],'voxel_count':counts})
    for k in MARKERS:
        v=[float(arrays[k][mask==i].mean()) for i in ids]
        mean['mean_raw_'+k]=v;frame['mean_raw_'+k]=v
    pd.DataFrame(frame).to_csv(w9/'instance_marker_spatial_table.csv',index=False)
    pd.DataFrame(mean).to_csv(w9/'instance_marker_mean_matrix.csv',index=False)
    summary={'marker_names':list(MARKERS),'n_instances':3,'markers':panel,
             'manifest_sha256':sha(manifest),'source_mask_sha256':sha(mp),'mask_shape_assumed_zyx':list(mask.shape)}
    summ=w9/'marker_integration_summary.json';summ.write_text(json.dumps(summary))
    filenames=['marker_integration_summary.json','instance_marker_spatial_table.csv','instance_marker_mean_matrix.csv']
    (w9/'output_file_sha256.json').write_text(json.dumps({n:sha(w9/n) for n in filenames}))
    args=SimpleNamespace(mask=mp,manifest=manifest,workflow09=w9,
                         out=root/'runs'/'3d_spatial_signal_attribution_v0_1',
                         min_pixels=2,no_figure=True,allow_synthetic_count=True)
    return args


def test_end_to_end_nuclear_reproduction_and_verification(tmp_path):
    args=fixture_project(tmp_path)
    summ,paths,df,ref,digests=file_manifest(args)
    assert len(digests)==6+2*5
    report=run(args)
    assert report['n_instances']==3
    assert report['n_rows']==3*5*4
    wide=pd.read_csv(args.out/'instance_marker_attribution_matrix.csv')
    for k in MARKERS:
        expect=ref.set_index('instance_id')['mean_raw_'+k]
        for _,row in wide.iterrows():
            assert row[k+'__nuclear_mask']==pytest.approx(expect.loc[row['instance_id']])
    assert verify(args)['status']=='PASS'


def test_output_tamper_detected(tmp_path):
    args=fixture_project(tmp_path);run(args)
    f=args.out/'marker_region_qc.csv'
    f.write_text(f.read_text()+'\nX')
    with pytest.raises(RuntimeError,match='Output changed'):verify(args)


def test_input_tamper_detected(tmp_path):
    args=fixture_project(tmp_path);run(args)
    f=json.loads(args.manifest.read_text())['markers'][0]['files'][0]
    with open(f,'ab') as out:out.write(b'X')
    with pytest.raises(ValueError,match='marker input changed'):verify(args)


def test_refuses_to_overwrite(tmp_path):
    args=fixture_project(tmp_path);run(args)
    with pytest.raises(FileExistsError,match='refusing to overwrite'):run(args)


def test_reference_baseline_mismatch_fails(tmp_path):
    args=fixture_project(tmp_path)
    # Also updates output manifest so input provenance check passes: independent baseline QC catches the mismatch.
    f=args.workflow09/'instance_marker_mean_matrix.csv'
    df=pd.read_csv(f);df.loc[0,'mean_raw_HER2']+=10;df.to_csv(f,index=False)
    output=args.workflow09/'output_file_sha256.json';m=json.loads(output.read_text());m[f.name]=sha(f);output.write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='Nuclear baseline mismatch'):run(args)


def test_hotspot_selection_labels_are_explicit():
    df=pd.DataFrame({'instance_id':[1,2,3,4], 'mean_raw_HER2':[1,1,5,100],
            'centroid_x_px':[500,550,650,900],'centroid_y_px':[500,530,600,300]})
    w=pd.DataFrame({'instance_id':[1,2,3,4]})
    for k in MARKERS:
        for region in REGIONS:w[f'{k}__{region}']=[2.0,3.0,4.0,5.0]
    result,meta=hotspot_table(df,w)
    assert meta['n_roi']==3
    assert meta['n_roi_high']==0 # top 1% threshold picks 100 outside ROI
    assert 'ROI_high_by_WF09_nuclear_HER2' in result.group.values


def test_source_summary_mutation_detected(tmp_path):
    args=fixture_project(tmp_path)
    f=args.workflow09/'marker_integration_summary.json'
    f.write_text(f.read_text()+'\n')
    with pytest.raises(ValueError,match='Workflow09 frozen result changed'):file_manifest(args)
