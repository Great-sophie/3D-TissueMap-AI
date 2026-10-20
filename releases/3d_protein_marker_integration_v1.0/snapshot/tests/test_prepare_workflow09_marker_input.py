from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pytest
import tifffile

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'prepare_workflow09_marker_input.py'
spec = importlib.util.spec_from_file_location('prepare_workflow09_marker_input', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture(tmp_path, missing=None, badshape=False, badorder=False):
    mask = tmp_path / 'mask.tif'
    m = np.zeros((4, 8, 9), dtype='uint32')
    m[:,1:3,1:4] = 23
    tifffile.imwrite(mask, m, photometric='minisblack')
    zipimages = tmp_path / 'images.zip'
    zipmodel = tmp_path / 'model.zip'
    root = 'case'
    rows = [f'{root}_s1_p9_r{z+1}_a{z+1}_ac.ome.tiff,{z}' for z in range(4)]
    if badorder:
        rows[2] = f'{root}_s1_p9_r7_a7_ac.ome.tiff,2'
    with ZipFile(zipmodel,'w',ZIP_DEFLATED) as z:
        z.writestr('Model/final_3D_stack_order.csv', '\n'.join(rows)+'\n')
    with ZipFile(zipimages,'w',ZIP_DEFLATED) as z:
        for marker, metal in mod.MARKERS.items():
            for k in range(4):
                if (marker, k)==missing:
                    continue
                from io import BytesIO
                out = BytesIO()
                shape=(9,9) if badshape and marker=='HER2' and k==0 else (8,9)
                tifffile.imwrite(out,np.full(shape,k+1,dtype='uint16'),photometric='minisblack')
                filename=f'{k+1:04d}_{root}_s1_p9_r{k+1}_a{k+1}_ac__{metal}.tiff'
                z.writestr(f'SIMILARITY10_In115/{metal}/{filename}',out.getvalue())
    outdata = tmp_path / 'processed' / 'markers'
    manifest = tmp_path / 'configs' / 'five.json'
    return mask,zipimages,zipmodel,outdata,manifest


def test_preflight_no_write(tmp_path):
    m,z,model,out,manifest = fixture(tmp_path)
    audit = mod.prepare(m,z,model,out,manifest,False)
    assert audit['n_selected_tiffs']==20
    assert not out.exists() and not manifest.exists()


def test_prepare_verify_and_schema(tmp_path):
    m,z,model,out,manifest = fixture(tmp_path)
    mod.prepare(m,z,model,out,manifest,True)
    assert mod.verify(m,z,model,out,manifest)==20
    spec=json.loads(manifest.read_text())
    assert spec['schema_version']==1
    assert spec['image_coordinate_status']=='unverified_assumed_same_index_grid'
    assert [r['name'] for r in spec['markers']]==list(mod.MARKERS)
    assert all(len(r['files'])==4 and r['axes']=='YX' and r['mode']=='per_section' for r in spec['markers'])
    assert spec['expected_mask_sha256']==mod.sha256(m)


def test_no_overwrite(tmp_path):
    m,z,model,out,manifest = fixture(tmp_path)
    mod.prepare(m,z,model,out,manifest,True)
    with pytest.raises(FileExistsError):
        mod.prepare(m,z,model,out,manifest,True)


def test_tampered_extracted_file_detected(tmp_path):
    m,z,model,out,manifest = fixture(tmp_path)
    mod.prepare(m,z,model,out,manifest,True)
    image=next((out/'HER2').glob('*.tiff'))
    image.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='missing or changed'):
        mod.verify(m,z,model,out,manifest)


def test_manifest_edit_detected(tmp_path):
    m,z,model,out,manifest = fixture(tmp_path)
    mod.prepare(m,z,model,out,manifest,True)
    doc=json.loads(manifest.read_text())
    doc['markers'][0]['name']='fake'
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='Manifest changed'):
        mod.verify(m,z,model,out,manifest)


def test_missing_slice_rejected(tmp_path):
    m,z,model,out,manifest=fixture(tmp_path,missing=('CD31',2))
    with pytest.raises(ValueError,match='expected 4'):
        mod.prepare(m,z,model,out,manifest,True)
    assert not out.exists() and not manifest.exists()


def test_wrong_shape_rejected(tmp_path):
    m,z,model,out,manifest=fixture(tmp_path,badshape=True)
    with pytest.raises(ValueError,match='shape/axes'):
        mod.prepare(m,z,model,out,manifest,True)


def test_wrong_stack_order_rejected(tmp_path):
    m,z,model,out,manifest=fixture(tmp_path,badorder=True)
    with pytest.raises(ValueError,match='conflicts with author stack order'):
        mod.prepare(m,z,model,out,manifest,True)


def test_mask_change_detected(tmp_path):
    m,z,model,out,manifest=fixture(tmp_path)
    mod.prepare(m,z,model,out,manifest,True)
    with m.open('ab') as f:f.write(b'change')
    with pytest.raises(ValueError,match='Input mask changed'):
        mod.verify(m,z,model,out,manifest)
