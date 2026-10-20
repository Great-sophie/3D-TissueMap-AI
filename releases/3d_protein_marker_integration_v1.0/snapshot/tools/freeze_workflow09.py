#!/usr/bin/env python3
"""Create and verify an immutable Workflow 09 provenance-only release."""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,tempfile
from pathlib import Path

REL=Path('releases/3d_protein_marker_integration_v1.0')
RUN=Path('runs/3d_protein_marker_integration_v1')
ANA=Path('runs/3d_protein_spatial_association_v1')
REQUIRED=['scripts/run_3d_protein_integration.py','scripts/protein_marker_integration_core.py',
          'scripts/analyze_protein_spatial_v1.py','tools/freeze_workflow09.py',
          'configs/protein_marker_manifest_registered_5channels.json',
          'tests/test_protein_marker_integration_v1.py',
          'tests/test_workflow09_finalize_and_visualize.py']
OPTIONAL=['scripts/audit_workflow09_alignment.py','tools/prepare_workflow09_marker_input.py',
          'tests/test_workflow09_instance_correspondence.py',
          'tests/test_prepare_workflow09_marker_input.py',
          'docs/PROTEIN_MARKER_INTEGRATION_研究设计与限制.md']
REQ_TABLES={'instance_marker_signal.csv':63315,'instance_marker_mean_matrix.csv':12663,
            'instance_marker_spatial_table.csv':12663,'marker_z_intensity_summary.csv':80}

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def regular(p):
    if not p.is_file() or p.is_symlink():raise RuntimeError(f'Required ordinary file unavailable: {p}')

def load(p):regular(p);return json.loads(p.read_text(encoding='utf-8'))

def rows(p):
    regular(p)
    with open(p,newline='',encoding='utf-8-sig') as f:
        r=csv.reader(f);next(r)
        return sum(1 for row in r if row)

def check_output(path,hash_name):
    hashes=load(path/hash_name)
    if not isinstance(hashes,dict) or not hashes:raise RuntimeError('Empty result hash manifest')
    actual={str(f.relative_to(path)) for f in path.rglob('*') if f.is_file()}
    if actual != set(hashes)|{hash_name}:raise RuntimeError(f'Unexpected/missing result files: {actual^ (set(hashes)|{hash_name})}')
    for name,digest in hashes.items():
        p=path/name
        regular(p)
        if sha(p)!=digest:raise RuntimeError(f'Output changed: {p}')
    return hashes

def preflight(root):
    if not root.is_dir():raise RuntimeError('Project root missing')
    sources={}
    for rel in REQUIRED:
        f=root/rel;regular(f);sources[rel]=sha(f)
    for rel in OPTIONAL:
        f=root/rel
        if f.is_file() and not f.is_symlink():sources[rel]=sha(f)
    out=root/RUN;ana=root/ANA
    h=check_output(out,'output_file_sha256.json')
    hah=check_output(ana,'output_sha256.json')
    summary=load(out/'marker_integration_summary.json')
    astr=load(ana/'association_summary.json')
    if summary.get('n_instances')!=12663 or summary.get('n_markers')!=5 or summary.get('n_instance_marker_rows')!=63315:
        raise RuntimeError('Unexpected 09C dimensions; inspect before freezing')
    if set(summary.get('marker_names',[]))!={'Ir193','panCK','HER2','CD31','LYVE1'}:
        raise RuntimeError('Unexpected marker panel')
    for file,n in REQ_TABLES.items():
        if file not in h or rows(out/file)!=n:raise RuntimeError('Unexpected CSV rows: '+file)
    if astr['n_instances']!=summary['n_instances'] or astr['source_table_sha256']!=sha(out/'instance_marker_spatial_table.csv'):
        raise RuntimeError('09D input differs from 09C')
    if sha(root/'configs/protein_marker_manifest_registered_5channels.json')!=summary['manifest_sha256']:
        raise RuntimeError('Marker manifest differs from extraction')
    if summary.get('no_protein_table_join') is not True:
        raise RuntimeError('Unexpected author protein table join status')
    # Verify actual upstream source bytes, not just the output's copied hash metadata.
    morphsum=root/'runs/3d_cell_morphometry_v1/morphometry_summary.json'
    spatsum=root/'runs/3d_spatial_organization_v1/spatial_organization_summary.json'
    regular(morphsum);regular(spatsum)
    if sha(morphsum)!=summary['workflow07_summary_sha256'] or sha(spatsum)!=summary['workflow08_summary_sha256']:
        raise RuntimeError('Upstream Workflow07/08 summary changed since Workflow09 extraction')
    morph=load(morphsum)
    source_mask=root/'data/inspect/LVILymph'/Path(morph['source_mask_path']).name
    regular(source_mask)
    if sha(source_mask)!=summary['source_mask_sha256']:
        raise RuntimeError('Full 3D instance mask changed')
    for spec in summary['markers']:
        for filename,digest in spec['files_sha256'].items():
            image=Path(filename)
            regular(image)
            if sha(image)!=digest:
                raise RuntimeError('Marker TIFF modified or wrong source: '+str(image))
    report={'release_type':'exploratory_spatial_proteomics_baseline',
            'n_instances':summary['n_instances'],'n_markers':summary['n_markers'],
            'extraction_core_version':summary.get('version','unknown'),
            'marker_names':summary['marker_names'],'n_signal_rows':summary['n_instance_marker_rows'],
            'geometry_status':summary['marker_geometry_status'],
            'input_mask_sha256':summary['source_mask_sha256'],
            'workflow07_summary_sha256':summary['workflow07_summary_sha256'],
            'workflow08_summary_sha256':summary['workflow08_summary_sha256'],
            'marker_manifest_sha256':summary['manifest_sha256'],
            'source_code_hashes':sources,
            'workflow09c_outputs':{str(RUN/f):d for f,d in h.items()},
            'workflow09c_manifest_hash':sha(out/'output_file_sha256.json'),
            'workflow09d_outputs':{str(ANA/f):d for f,d in hah.items()},
            'workflow09d_manifest_hash':sha(ana/'output_sha256.json'),
            'background_and_id_audit_status':'VISUAL_PLAUSIBILITY_AND_MASK_COORDINATE_CHECKS; not physical registration',
            'author_mean_table_row_alignment':'NOT_VERIFIED; not joined',
            'notes':['16 planes, XY pixels and Z index only; physical Z spacing unknown',
                     'Single specimen; author watershed instances are pseudo-labels',
                     'Intensity restricted to author pseudo-mask voxels, not full cells',
                     'Spatial blocks are not independent replicates']}
    return report

def manifest_release(folder):
    return {str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name!='release_sha256.json'}

def make(root,report):
    path=root/REL
    if path.exists():raise FileExistsError(f'Release exists; refusing overwrite: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.workflow09_release_stage_',dir=path.parent))
    try:
        for rel in report['source_code_hashes']:
            dst=stage/'snapshot'/rel
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(root/rel,dst)
        (stage/'provenance_lock.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        readme='''# Workflow 09 — 3D Protein Marker Integration v1.0

正式状态：**exploratory mask-restricted signal and spatial association baseline**。本次归档来自真实的 Workflow09C/09D 表格及代码 SHA-256 核验。

- 正式脚本仍位于 `scripts/`，归档副本仅用于复现与审计。
- 输出仍位于 `runs/3d_protein_marker_integration_v1/` 和 `runs/3d_protein_spatial_association_v1/`。
- 五通道：Ir193 / panCK / HER2 / CD31 / LYVE1；12,663 个作者三维 watershed 伪实例。
- `XY` 使用像素坐标；`Z` 是切片索引。未验证真实微米间距或独立配准精度。
- 核样掩码中的强度不能定义为完整细胞膜/胞质蛋白表达；不能据此判定阳性/阴性。
- 作者强度表 6,455 行与 mask ID 的逐行顺序没有得到证实；未进行错误合并。
- 当前关联仅为**单标本探索性描述**，不能进行病人层面推断。

验证方式：`python tools/freeze_workflow09.py --verify`。
'''
        (stage/'README_中文.md').write_text(readme)
        with (stage/'script_workflow_index.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['canonical_path','sha256'])
            w.writerows(sorted(report['source_code_hashes'].items()))
        (stage/'release_sha256.json').write_text(json.dumps(manifest_release(stage),indent=2)+'\n')
        stage.rename(path)
    finally:
        if stage.exists():shutil.rmtree(stage)
    return path

def verify(root):
    dest=root/REL
    stored=load(dest/'release_sha256.json')
    actual=manifest_release(dest)
    if stored!=actual:raise RuntimeError('Release archive modified')
    release=load(dest/'provenance_lock.json')
    now=preflight(root)
    if release!=now:raise RuntimeError('Frozen provenance differs from active code/results')
    for path,digest in release['source_code_hashes'].items():
        if sha(dest/'snapshot'/path)!=digest:raise RuntimeError('Frozen source changed: '+path)
    return {'verified_release_files':len(stored),'n_instances':release['n_instances'],
            'n_markers':release['n_markers']}

def main():
    ap=argparse.ArgumentParser()
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--apply',action='store_true')
    mode.add_argument('--verify',action='store_true')
    ap.add_argument('--project-root',type=Path,default=Path('.'))
    a=ap.parse_args(); root=a.project_root.resolve()
    if a.verify:print('PASS:',json.dumps(verify(root),ensure_ascii=False));return
    report=preflight(root)
    print('PREFLIGHT PASS:',report['n_instances'],'instances,',report['n_markers'],'markers,',len(report['source_code_hashes']),'source files')
    if a.apply:
        print('FROZEN:',make(root,report))
    else:print('DRY-RUN ONLY: use --apply after checking existing releases/ folder')
if __name__=='__main__':main()
