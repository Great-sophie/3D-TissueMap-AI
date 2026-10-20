import json
import numpy as np
import pytest
import tifffile
from tissuemap.volume_qc import _bbox_features, run_audit, run_demo, tiff_header, main


def test_exact_um3():
    labels=np.zeros((4,6,6),dtype=np.int32)
    labels[1:3,2:4,3:5]=10
    df=_bbox_features(labels,(2.,.5,.5))
    assert len(df)==1
    assert df.loc[0,'n_voxels']==8
    assert df.loc[0,'volume_um3']==4.
    assert df.loc[0,'centroid_z_um']==3.


def test_spacing_requires_positive(tmp_path):
    im=np.zeros((1,4,8,8),dtype=np.float32)
    with pytest.raises(ValueError,match="Spacing"):
        run_audit(im,None,(-2.,1.,1.),0,tmp_path/'out','test')


def test_mask_shape_catches_registration_error(tmp_path):
    im=np.zeros((1,4,8,8),dtype=np.float32)
    mask=np.zeros((4,8,7),dtype=np.int32)
    with pytest.raises(ValueError,match="Misregistered"):
        run_audit(im,mask,None,0,tmp_path/'out','test')


def test_no_spacing_means_no_physical_volume(tmp_path):
    im=np.random.default_rng(0).random((1,4,8,8)).astype(np.float32)
    mask=np.zeros((4,8,8),dtype=np.int32);mask[1:3,2:4,2:4]=2
    r=run_audit(im,mask,None,0,tmp_path/'out','test')
    assert not r['physical_units_valid']
    assert 'volume_um3' not in (tmp_path/'out'/'cell_morphology_qc.csv').read_text()
    assert (tmp_path/'out'/'orthogonal_slices.png').exists()


def test_tiff_metadata_header_and_axis_caution(tmp_path):
    f=tmp_path/'s.tiff'
    tifffile.imwrite(f,np.ones((4,8,9),dtype=np.uint16),metadata={'axes':'ZYX'})
    r=tiff_header(f)
    assert r['series'][0]['shape']==[4,8,9]
    assert r['series'][0]['axes_metadata']=='ZYX'
    assert 'not a substitute' in r['warning']


def test_full_demo_has_provenance_and_pitfall_plot(tmp_path):
    r=run_demo(tmp_path/'volume_qc')
    assert 'SYNTHETIC' in r['provenance']
    assert r['teaching_demo']['centroid_3D_um']==18
    assert (tmp_path/'volume_qc'/'why_3d_matters.png').exists()
    assert json.loads((tmp_path/'volume_qc'/'volume_qc_report.json').read_text())['instance_count']>0
