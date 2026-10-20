"""Offline synthetic tests for safe Workflow 10 release creation and verification."""
from __future__ import annotations
import importlib.util
import json
import shutil
import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/freeze_workflow10.py'
spec=importlib.util.spec_from_file_location('workflow10_freezer_for_tests',SCRIPT)
fr=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=fr
spec.loader.exec_module(fr)


def make_png():
    def chunk(name, body):
        return struct.pack('!I', len(body))+name+body+struct.pack('!I',zlib.crc32(name+body)&0xffffffff)
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('!2I5B',1,1,8,2,0,0,0))
            +chunk(b'IDAT',zlib.compress(b'\x00\xff\x00\x00'))+chunk(b'IEND',b''))


def write(root,rel,text_or_bytes):
    path=root/rel
    path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(text_or_bytes,bytes): path.write_bytes(text_or_bytes)
    else: path.write_text(text_or_bytes,encoding='utf-8')
    return path


def fixture(root):
    for name in fr.SOURCE_REQUIRED:
        if name.endswith('freeze_workflow10.py'):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SCRIPT,path)
        else:
            write(root,name,'# synthetic script\n')
    mask=write(root,fr.MASK,b'fake mask with stable identity')
    spatial=write(root,fr.SPATIAL_TABLE,'instance_id,mean_raw_HER2\n1,1.0\n')
    protein=write(root,fr.PROTEIN_SUMMARY,json.dumps({'n_instances':12663,'marker_names':list(fr.MARKERS)}))
    morph=write(root,fr.MORPH_TABLE,'instance_id,centroid_z_index\n1,8\n')
    source_lock={'n_instances':12663,'n_markers':5,'input_mask_sha256':fr.sha(mask),
                 'workflow09c_outputs':{fr.SPATIAL_TABLE.as_posix():fr.sha(spatial),
                                        fr.PROTEIN_SUMMARY.as_posix():fr.sha(protein)},
                 'marker_names':list(fr.MARKERS)}
    old_release=root/fr.PREVIOUS_REL
    write(root,fr.PREVIOUS_REL/'provenance_lock.json',json.dumps(source_lock))
    write(root,fr.PREVIOUS_REL/'README_中文.md','synthetic frozen 09 release')
    write(root,fr.PREVIOUS_REL/'release_sha256.json',json.dumps(fr.files_hash(old_release,'release_sha256.json')))
    vs={'n_instances':12663,'z_units':'integer slice index ONLY; physical spacing unverified',
        'roi':{'visualized_voxels':100},'mask_sha256':fr.sha(mask),
        'spatial_table_sha256':fr.sha(spatial),
        'workflow09_summary_sha256':fr.sha(protein)}
    viewer=root/fr.VIEWER
    write(root,fr.VIEWER/'viewer_summary.json',json.dumps(vs))
    for name in fr.VIEWER_OUTPUTS[:2]:
        write(root,fr.VIEWER/name,'<!DOCTYPE html><html><body>Plotly saved offline '+name+'</body></html>'+'Z'*2500)
    write(root,fr.VIEWER/'viewer_sha256.json',json.dumps(fr.files_hash(viewer,'viewer_sha256.json')))
    write(root,fr.QC/fr.QC_OUTPUTS[0],
          'block_x,block_y,n_instances,n_her2_high,her2_high_fraction\n5,4,435,33,0.075862\n')
    write(root,fr.QC/fr.QC_OUTPUTS[1],make_png())
    return root


def test_missing_project_refused(tmp_path):
    with pytest.raises(RuntimeError,match='unavailable'):
        fr.preflight(tmp_path)


def test_fixture_preflight_dimensions(tmp_path):
    fixture(tmp_path)
    lock=fr.preflight(tmp_path)
    assert lock['n_instances']==12663
    assert len(lock['viewers'])==3 and len(lock['hotspot_qc_hashes'])==2
    assert len(lock['source_code_hashes'])==3


def test_freeze_apply_verify_full_archive(tmp_path):
    fixture(tmp_path)
    lock=fr.preflight(tmp_path)
    dest=fr.make(tmp_path,lock)
    report=fr.verify(tmp_path)
    assert report['verified_viewer_html']==2
    assert report['verified_hotspot_qc_artifacts']==2
    assert report['verified_release_files']>=8
    assert (dest/'viewer/centroid_marker_viewer.html').is_file()
    assert (dest/'hotspot_qc/her2_core_roi_registered_overlay.png').is_file()
    with pytest.raises(FileExistsError):
        fr.make(tmp_path,lock)


def test_refuse_mutation_to_frozen_viewer(tmp_path):
    fixture(tmp_path)
    dest=fr.make(tmp_path,fr.preflight(tmp_path))
    (dest/'viewer/roi_voxel_viewer.html').write_text('corrupted')
    with pytest.raises(RuntimeError,match='release files changed'):
        fr.verify(tmp_path)


def test_refuse_mutation_to_active_viewer(tmp_path):
    fixture(tmp_path)
    fr.make(tmp_path,fr.preflight(tmp_path))
    (tmp_path/fr.VIEWER/'centroid_marker_viewer.html').write_text('corrupted')
    with pytest.raises(RuntimeError,match='Viewer SHA-256'):
        fr.verify(tmp_path)


def test_refuse_mutation_to_hotspot_qc(tmp_path):
    fixture(tmp_path)
    fr.make(tmp_path,fr.preflight(tmp_path))
    with (tmp_path/fr.QC/fr.QC_OUTPUTS[0]).open('a') as f:
        f.write('6,6,40,0,0\n')
    with pytest.raises(RuntimeError,match='provenance no longer matches'):
        fr.verify(tmp_path)


def test_refuse_mutation_to_upstream_frozen_09(tmp_path):
    fixture(tmp_path)
    fr.make(tmp_path,fr.preflight(tmp_path))
    (tmp_path/fr.PREVIOUS_REL/'README_中文.md').write_text('changed')
    with pytest.raises(RuntimeError,match='Workflow09 release archive changed'):
        fr.verify(tmp_path)


def test_refuse_mutation_to_active_mask(tmp_path):
    fixture(tmp_path)
    fr.make(tmp_path,fr.preflight(tmp_path))
    (tmp_path/fr.MASK).write_bytes(b'different')
    with pytest.raises(RuntimeError,match='Mask no longer matches'):
        fr.verify(tmp_path)


def test_refuse_bad_hotspot_file(tmp_path):
    fixture(tmp_path)
    (tmp_path/fr.QC/'her2_core_roi_registered_overlay.png').write_bytes(b'not png')
    with pytest.raises(RuntimeError,match='not a PNG'):
        fr.preflight(tmp_path)


def test_refuse_missing_offline_html(tmp_path):
    fixture(tmp_path)
    (tmp_path/fr.VIEWER/'roi_voxel_viewer.html').unlink()
    with pytest.raises(RuntimeError,match='Viewer SHA-256'):
        fr.preflight(tmp_path)


def test_refuse_changed_script(tmp_path):
    fixture(tmp_path)
    fr.make(tmp_path,fr.preflight(tmp_path))
    (tmp_path/'scripts/build_3d_tissue_viewer.py').write_text('changed')
    with pytest.raises(RuntimeError,match='provenance no longer matches'):
        fr.verify(tmp_path)


def test_default_cli_dry_run_no_release(tmp_path,capfd):
    fixture(tmp_path)
    fr.main.__globals__['__name__'] # no action
    import subprocess
    r=subprocess.run([sys.executable,str(SCRIPT),'--project-root',str(tmp_path)],capture_output=True,text=True,check=True)
    assert 'PREFLIGHT PASS' in r.stdout and 'DRY-RUN ONLY' in r.stdout
    assert not (tmp_path/fr.REL).exists()
