"""Synthetic scientific-invariant tests for Workflow 07.

Run from repo root: python -m pytest -q tests/test_cell_morphometry_v1.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cell_morphometry_core import (validate_mask, summarize_mask, count_exposed_faces,
                                  save_tables, file_sha256)
from run_3d_cell_morphometry import choose_mask, main


def test_single_voxel_center_has_six_background_faces():
    mask = np.zeros((3,5,5), np.uint16)
    mask[1,2,2]=7
    morphology, profile, qc, report = summarize_mask(mask)
    a=morphology.iloc[0]
    assert a.instance_id == 7
    assert a.voxel_count == 1
    assert (a.centroid_z_index,a.centroid_y_px,a.centroid_x_px)==(1,2,2)
    assert (a.exposed_z_faces_voxel,a.exposed_y_faces_voxel,a.exposed_x_faces_voxel)==(2,2,2)
    assert a.background_faces_voxel == 6
    assert a.other_instance_interface_faces_voxel == 0
    assert profile.area_px2.tolist()==[1]
    assert qc.iloc[0].components_6connected == 1
    assert not bool(qc.iloc[0].touches_xy_image_boundary)
    assert report['n_instance_ids'] == 1


def test_adjacent_different_labels_interface_counted_per_cell():
    mask=np.zeros((3,5,5),np.uint16)
    mask[1,2,2]=2
    mask[1,2,3]=9
    morph,_,_,_=summarize_mask(mask)
    assert morph.other_instance_interface_faces_voxel.tolist()==[1,1]
    assert morph.background_faces_voxel.tolist()==[5,5]
    assert morph.exposed_all_faces_voxel.tolist()==[6,6]


def test_two_adjacent_voxels_same_label_surface_ten():
    mask = np.zeros((3,4,4), dtype=np.uint16)
    mask[1,1,1]=8
    mask[1,2,1]=8
    df,_,_,_=summarize_mask(mask)
    assert int(df.iloc[0].exposed_all_faces_voxel)==10
    assert int(df.iloc[0].background_faces_voxel)==10


def test_gaps_multiple_cc_and_global_border():
    mask=np.zeros((4,8,9), np.uint16)
    mask[0,0,0]=1
    mask[2,0,0]=1
    mask[1,4,4]=2
    mask[1,4,6]=2
    morph,_,qc,info=summarize_mask(mask,min_qc_voxels=1)
    r1=morph.set_index('instance_id').loc[1]
    q1=qc.set_index('instance_id').loc[1]
    q2=qc.set_index('instance_id').loc[2]
    assert (r1.z_span_planes,r1.n_slices_present,r1.z_gap_planes)==(3,2,1)
    assert q1.components_6connected==2
    assert q1.components_26connected==2
    assert q1.touches_xy_image_boundary and q1.touches_z_stack_boundary
    assert 'MISSING_INTERMEDIATE_Z' in q1.qc_reasons
    assert q2.components_6connected==2
    assert q2.components_26connected==2
    assert info['n_z_gapped_ids']==1
    assert info['foreground_voxels']==4


def test_diagonal_touch_differentiates_6_and_26():
    mask=np.zeros((3,4,4),np.uint16)
    mask[1,1,1]=55
    mask[1,2,2]=55
    _,_,q,_=summarize_mask(mask,min_qc_voxels=1)
    assert q.iloc[0].components_6connected==2
    assert q.iloc[0].components_26connected==1


def test_z_axis_and_slice_areas_include_adjacent_planes():
    m=np.zeros((4,5,6),np.uint16)
    m[1,2,2:4]=11
    m[2,2,2]=11
    df, profile,_,_=summarize_mask(m)
    row=df.iloc[0]
    assert row.voxel_count==3
    assert row.n_slices_present==2
    assert row.z_span_planes==2
    assert row.z_gap_planes==0
    assert row.centroid_z_index == pytest.approx(4/3)
    assert row.centroid_x_px == pytest.approx(7/3)
    assert row.min_slice_area_px2 == 1
    assert row.max_slice_area_px2 == 2
    assert row.peak_area_z_index == 1
    assert profile.z_index.tolist()==[1,2]
    assert profile.area_px2.tolist()==[2,1]
    assert row.max_adjacent_plane_centroid_shift_xy_px == pytest.approx(.5)


def test_mask_validation():
    for m in (np.zeros((2,3),np.uint16), np.zeros((0,3,3),np.uint16),
              np.zeros((3,4,4),np.uint16), np.ones((3,4,4),np.float32),
              np.ones((3,4,4),np.bool_),
              np.array([[[-1]]],dtype=np.int32),
              np.array([[[2_000_001]]],dtype=np.int64)):
        with pytest.raises(ValueError):validate_mask(m)


def test_touching_all_boundaries_area_and_face_accounting():
    m=np.zeros((2,2,2),np.uint16)
    m[:,:,:]=7
    morph, profile, qc, summary=summarize_mask(m)
    row=morph.iloc[0]
    assert row.voxel_count==8
    assert row.bbox_occupancy_fraction==1
    assert row.exposed_all_faces_voxel==24
    assert row.background_faces_voxel==24
    assert row.other_instance_interface_faces_voxel==0
    assert row.exposed_z_faces_voxel==8
    assert row.exposed_y_faces_voxel==8
    assert row.exposed_x_faces_voxel==8
    assert qc.iloc[0].touches_xy_image_boundary
    assert qc.iloc[0].touches_z_stack_boundary
    assert profile.area_px2.sum()==summary['foreground_voxels']


def test_run_end_to_end_tiff_outputs_and_no_overwrite(tmp_path):
    d=tmp_path/'source';d.mkdir()
    m=np.zeros((4,8,9),np.uint16)
    m[1:3,3:5,4:6]=101
    m[1,1,1]=4
    f=d/'full_mask_example.tif'
    tifffile.imwrite(f,m,photometric='minisblack',metadata={'axes':'ZYX'})
    out=tmp_path/'outputs'
    main(['--data-dir',str(d),'--assume-zyx','--out',str(out)])
    for name in ('cell_morphometry.csv','cell_shape_features.csv',
                 'cell_morphology_qc.csv','cell_slice_profiles.csv','morphometry_summary.json'):
        assert (out/name).is_file()
    p=pd.read_csv(out/'cell_morphometry.csv')
    assert p.instance_id.tolist()==[4,101]
    assert p.voxel_count.tolist()==[1,8]
    summary=json.loads((out/'morphometry_summary.json').read_text())
    assert summary['source_mask_sha256']==file_sha256(f)
    assert summary['n_instance_ids']==2
    assert len(list((out/'figures').glob('*.png')))>=4
    files=json.loads((out/'output_file_sha256.json').read_text())
    assert 'morphometry_summary.json' in files
    for name, value in files.items():
        assert file_sha256(out/name)==value
    with pytest.raises(SystemExit,match='already exists'):
        main(['--data-dir',str(d),'--assume-zyx','--out',str(out)])


def test_refuses_implicit_zyx_and_ambiguous_mask(tmp_path):
    d=tmp_path/'d';d.mkdir()
    tifffile.imwrite(d/'full_mask_a.tif',np.ones((3,4,5),np.uint16),photometric='minisblack')
    tifffile.imwrite(d/'full_mask_b.tif',np.ones((3,4,5),np.uint16),photometric='minisblack')
    with pytest.raises(SystemExit,match='Refusing to infer'):
        main(['--data-dir',str(d),'--out',str(tmp_path/'output')])
    with pytest.raises(ValueError,match='exactly one'):
        choose_mask(d,None)


def test_output_tables_same_instance_id_and_no_physical_columns(tmp_path):
    m=np.zeros((3,6,6),np.uint16)
    m[1,2,2]=17
    d=tmp_path/'data';d.mkdir()
    f=d/'full_mask_valid.tif'
    tifffile.imwrite(f,m,photometric='minisblack')
    out=tmp_path/'out'
    result=save_tables(m,f,out,figures=False)
    for file in ['cell_morphometry.csv','cell_shape_features.csv','cell_morphology_qc.csv']:
        df=pd.read_csv(out/file)
        assert df.instance_id.tolist()==[17]
        assert not any(('um' in name.lower() or 'micron' in name.lower()) for name in df.columns)
    assert result['figure_files']==[]
    assert 'protein' in ' '.join(result['interpretation_limits']).lower()
