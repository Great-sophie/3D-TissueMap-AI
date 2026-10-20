"""Workflow09 v1 (exploratory): explicit IMC marker-image to watershed pseudo-ID aggregation.

All axes are index-based. Image/mask registration is NOT established by matching shapes.
No physical-unit or biological cell-identity claims are supported.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

VERSION = '0.9.0-audit-first'
KNOWN_MARKERS = ('Ir193', 'panCK', 'HER2', 'CD31', 'LYVE1')
ALLOWED_IMAGE_SUFFIXES = {'.tif', '.tiff', '.npy'}
ALLOWED_MODES = {'stack', 'per_section'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _relative_or_abs(base: Path, path: str) -> Path:
    p = Path(path).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {'.tif', '.tiff'}:
        with tifffile.TiffFile(path) as tif:
            if len(tif.series) != 1:
                raise ValueError(f'{path}: expected exactly one TIFF series')
            return tif.asarray()
    if path.suffix.lower() == '.npy':
        return np.load(path, allow_pickle=False)
    raise ValueError(f'Unsupported image type: {path}')


def _inspect_image(path: Path) -> dict:
    out = {'name':path.name, 'suffix':path.suffix.lower(), 'size_bytes':path.stat().st_size,
           'path':str(path)}
    if path.suffix.lower() in {'.tif', '.tiff'}:
        with tifffile.TiffFile(path) as tif:
            out.update(series_count=len(tif.series), axes=str(tif.series[0].axes),
                       shape=list(map(int,tif.series[0].shape)), dtype=str(tif.series[0].dtype))
    elif path.suffix.lower()=='.npy':
        arr=np.load(path,mmap_mode='r',allow_pickle=False)
        out.update(series_count=None, axes='UNKNOWN_NPY', shape=list(map(int,arr.shape)),dtype=str(arr.dtype))
    else:
        raise ValueError('not an image')
    return out


def marker_discovery(data_dir: Path, out: Path, mask_path: Path | None = None,
                     max_files: int = 6000) -> dict:
    """Read TIFF/NPY headers only. Never infer channel labels or registration."""
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if out.exists():
        raise FileExistsError(f'Discovery output exists: {out}')
    candidates = sorted(p for p in data_dir.rglob('*') if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_SUFFIXES)
    if len(candidates) > max_files:
        raise ValueError(f'{len(candidates)} images > max-files={max_files}; restrict --data-dir')
    rows=[]
    for p in candidates:
        try:
            row=_inspect_image(p)
            hint=[m for m in KNOWN_MARKERS if m.casefold() in p.name.casefold()]
            row.update(marker_filename_hints=';'.join(hint),read_error='')
        except Exception as ex:
            row={'name':p.name,'path':str(p),'suffix':p.suffix.lower(),
                 'size_bytes':p.stat().st_size,'marker_filename_hints':'',
                 'read_error':f'{type(ex).__name__}: {ex}'}
        rows.append(row)
    if mask_path is None:
        masks=sorted(data_dir.glob('full_mask*.tif'))+sorted(data_dir.glob('full_mask*.tiff'))
        mask_path=masks[0] if len(masks)==1 else None
    mask_note={}
    if mask_path is not None:
        mask_note = _inspect_image(mask_path)
        mask_note['sha256']=sha256(mask_path)
    report={'version':VERSION,'scan_directory':str(data_dir.resolve()),'n_image_files':len(rows),
            'mask':mask_note,'geometry_status':'UNVERIFIED',
            'marker_channels':'NOT inferred from filenames; explicit channel mapping required',
            'physical_z_spacing':'UNVERIFIED',
            'caution':'Matching shape/order is necessary, but does NOT prove registration or marker identity.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.wf09_discovery_',dir=out.parent))
    try:
        pd.DataFrame(rows,columns=['name','path','suffix','size_bytes','shape','axes','dtype','series_count',
                                   'marker_filename_hints','read_error']).to_csv(stage/'candidate_image_inventory.csv',index=False)
        (stage/'discovery_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        tmpl={'schema_version':1,'mask_path':str(mask_path.resolve()) if mask_path else 'PATH_TO_AUTHOR_MASK.tif',
              'expected_mask_sha256':mask_note.get('sha256',''),
              'image_coordinate_status':'unverified_assumed_same_index_grid',
              'markers':[]}
        (stage/'marker_manifest_template.json').write_text(json.dumps(tmpl,ensure_ascii=False,indent=2),encoding='utf-8')
        stage.rename(out)
    finally:
        if stage.exists():shutil.rmtree(stage)
    return report


def _check_axis(axis: str, mode: str) -> None:
    permitted=({'ZYX','ZCYX','CZYX'} if mode=='stack' else {'YX','CYX','YXC'})
    if axis not in permitted:
        raise ValueError(f'Unsupported {mode} explicit axes {axis!r}; expected one of {sorted(permitted)}')


def _path_list(spec: dict, base: Path) -> list[Path]:
    mode=spec['mode']
    if mode=='stack':
        if ('path' not in spec) or ('files' in spec):
            raise ValueError(f'{spec["name"]}: stack requires path, not files')
        paths=[_relative_or_abs(base,spec['path'])]
    else:
        paths=[_relative_or_abs(base,s) for s in spec.get('files',[])]
        if not paths or ('path' in spec):
            raise ValueError(f'{spec["name"]}: per_section needs explicitly ordered files')
    if len(set(paths))!=len(paths):
        raise ValueError(f'{spec["name"]}: duplicate slice file paths')
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f'{spec["name"]}: missing {p}')
        if p.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            raise ValueError(f'{spec["name"]}: unsupported {p.suffix}')
    return paths


def read_manifest(path: Path) -> tuple[dict,list[dict]]:
    raw=json.loads(path.read_text(encoding='utf8'))
    if raw.get('schema_version')!=1:
        raise ValueError('manifest schema_version must be 1')
    if raw.get('image_coordinate_status')!='unverified_assumed_same_index_grid':
        raise ValueError('image_coordinate_status must acknowledge unverified shared image grid')
    specs=raw.get('markers')
    if not isinstance(specs,list) or not specs:
        raise ValueError('No marker definitions: populate marker_manifest_template.json before extraction')
    names=[]
    for spec in specs:
        name=spec.get('name','')
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_+.-]{1,40}',name):
            raise ValueError('Each marker needs short alphanumeric name (e.g. HER2, Ir193, panCK)')
        names.append(name)
        mode=spec.get('mode')
        if mode not in ALLOWED_MODES:
            raise ValueError(f'{name}: mode must be stack or per_section')
        _check_axis(str(spec.get('axes','')),mode)
        axes=spec['axes']
        if 'C' in axes:
            if not isinstance(spec.get('channel_index'),int) or spec['channel_index']<0:
                raise ValueError(f'{name}: channel_index required for axes containing C')
        elif 'channel_index' in spec:
            raise ValueError(f'{name}: channel_index forbidden for axes without C')
        _path_list(spec,path.parent)
    if len(set(names))!=len(names):
        raise ValueError('Duplicate marker names')
    maskpath=_relative_or_abs(path.parent,raw['mask_path'])
    if not maskpath.is_file():
        raise FileNotFoundError(maskpath)
    return raw,specs


def _get_plane_from_array(arr:np.ndarray, axes:str, z:int, channel:int|None=None) -> np.ndarray:
    if len(axes)!=arr.ndim:
        raise ValueError(f'Array shape {arr.shape} incompatible with explicit axes {axes}')
    slice_selector=[]
    for a in axes:
        slice_selector.append(z if a=='Z' else channel if a=='C' else slice(None))
    if channel is not None and not (0<=channel<arr.shape[axes.index('C')]):
        raise ValueError(f'channel_index={channel} out of bounds in shape {arr.shape}')
    plane=np.asarray(arr[tuple(slice_selector)])
    if plane.ndim!=2 or axes.replace('Z','').replace('C','')!='YX':
        raise ValueError(f'Expected spatial axes YX, got {axes}')
    if not np.issubdtype(plane.dtype,np.number):
        raise ValueError('Marker image values must be numeric')
    return plane


def iter_marker_planes(spec:dict, base:Path, shape:tuple[int,int,int]):
    nz,ny,nx=shape
    paths=_path_list(spec,base)
    axes=spec['axes'];channel=spec.get('channel_index')
    if spec['mode']=='stack':
        arr=_read_image(paths[0])
        if arr.ndim != len(axes) or arr.shape[axes.index('Z')]!=nz:
            raise ValueError(f'{spec["name"]}: Z dimension mismatch, {arr.shape} axes {axes} vs {shape}')
        for z in range(nz):
            plane=_get_plane_from_array(arr,axes,z,channel)
            if plane.shape!=(ny,nx):
                raise ValueError(f'{spec["name"]}: z{z} plane {plane.shape} != {(ny,nx)}')
            yield z,plane
    else:
        if len(paths)!=nz:
            raise ValueError(f'{spec["name"]}: expected {nz} slice files, got {len(paths)}')
        for z,p in enumerate(paths):
            arr=_read_image(p)
            plane=_get_plane_from_array(arr,axes,0,channel)
            if plane.shape!=(ny,nx):
                raise ValueError(f'{spec["name"]}: z{z} plane {plane.shape} != {(ny,nx)}')
            yield z,plane


def _validate_mask(mask:np.ndarray):
    if mask.ndim!=3 or any(d<1 for d in mask.shape):
        raise ValueError('Mask must be nonempty 3D ZYX')
    if not np.issubdtype(mask.dtype,np.integer) or np.issubdtype(mask.dtype,np.bool_):
        raise ValueError('Mask must contain integer instance IDs')
    if np.issubdtype(mask.dtype,np.signedinteger) and np.any(mask<0):
        raise ValueError('Mask has negative IDs')
    uniq=np.unique(mask)
    ids=uniq[uniq>0].astype(np.int64)
    if not len(ids):
        raise ValueError('Mask has no pseudo-instances')
    return ids


def validate_upstream(mask:np.ndarray, mask_path:Path, morph_dir:Path, spatial_dir:Path) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,dict,dict]:
    summ=json.loads((morph_dir/'morphometry_summary.json').read_text(encoding='utf8'))
    spatial_summ=json.loads((spatial_dir/'spatial_organization_summary.json').read_text(encoding='utf8'))
    morph=pd.read_csv(morph_dir/'cell_morphometry.csv')
    qc=pd.read_csv(morph_dir/'cell_morphology_qc.csv')
    spatial=pd.read_csv(spatial_dir/'instance_spatial_features.csv')
    digest=sha256(mask_path)
    for k,s in [('Workflow07',summ),('Workflow08',spatial_summ)]:
        if s.get('source_mask_sha256')!=digest:
            raise ValueError(f'{k} mask SHA-256 does not match Workflow09 mask')
    if spatial_summ.get('workflow07_summary_sha256')!=sha256(morph_dir/'morphometry_summary.json'):
        raise ValueError('Workflow08 used a different Workflow07 summary')
    for fname, expected in spatial_summ.get('workflow07_files_sha256', {}).items():
        source=morph_dir/fname
        if not source.is_file() or sha256(source)!=expected:
            raise ValueError(f'Workflow07 upstream file changed since Workflow08: {fname}')
    ids,counts=np.unique(mask,return_counts=True)
    sel=ids>0;ids=ids[sel].astype(np.int64);counts=counts[sel]
    if not len(ids):raise ValueError('Empty foreground labels')
    if summ.get('n_instance_ids')!=len(ids) or spatial_summ.get('n_instances')!=len(ids):
        raise ValueError('Instance totals differ between Workflow07/08 and mask')
    for frame,name in [(morph,'Workflow07 morphometry'),(qc,'Workflow07 QC'),(spatial,'Workflow08 features')]:
        if 'instance_id' not in frame or frame.instance_id.duplicated().any():
            raise ValueError(f'{name}: missing or duplicate instance_id')
        if set(frame.instance_id.astype(int))!=set(ids):
            raise ValueError(f'{name}: ID set differs from mask')
    check=morph.set_index('instance_id').loc[ids,'voxel_count'].to_numpy(dtype=np.int64)
    if not np.array_equal(check,counts):
        raise ValueError('Workflow07 voxel_count does not match actual mask')
    for s,name in [(summ,'Workflow07'),(spatial_summ,'Workflow08')]:
        if list(s.get('shape_assumed_zyx',[]))!=list(mask.shape):
            raise ValueError(f'{name} shape mismatch')
    if not {'instance_id','qc_flag_review'}.issubset(qc.columns):
        raise ValueError('Missing QC flag')
    if 'n_xy_radius_neighbors' not in spatial:
        raise ValueError('Missing Workflow08 neighborhood counts')
    return morph,qc,spatial,summ,spatial_summ


def aggregate_one(mask:np.ndarray, ids:np.ndarray, spec:dict, base:Path):
    """No signal threshold: raw count/intensity within pseudo-label voxels, zeros included."""
    n=len(ids)
    # labels mapped to dense indices, avoids allocating max_id+1 when IDs are sparse.
    observations=[[] for _ in range(n)]
    finite_count=np.zeros(n,dtype=np.int64)
    total_count=np.zeros(n,dtype=np.int64)
    zero_count=np.zeros(n,dtype=np.int64)
    sum_val=np.zeros(n,dtype=np.float64)
    sum_sq=np.zeros(n,dtype=np.float64)
    per_z=[]
    marker_name=spec['name']
    preview_plane=None
    target_z=int(np.argmax(np.count_nonzero(mask,axis=(1,2))))
    for z,img in iter_marker_planes(spec,base,mask.shape):
        if z==target_z:
            preview_plane=np.asarray(img).copy()
        m=mask[z];valid=m>0
        lab=m[valid].astype(np.int64)
        if not len(lab):
            per_z.append({'marker':marker_name,'z_index':z,'foreground_voxels':0,
                          'finite_foreground_voxels':0,'mean_signal_within_mask':np.nan,
                          'mean_signal_outside_mask':np.nan})
            continue
        idx=np.searchsorted(ids,lab)
        if not np.array_equal(ids[idx],lab):
            raise ValueError('Unexpected label not in instance ID index')
        vals=np.asarray(img[valid],dtype=np.float64)
        total_count+=np.bincount(idx,minlength=n)
        good=np.isfinite(vals)
        idx_good=idx[good]; vals_good=vals[good]
        finite_count+=np.bincount(idx_good,minlength=n)
        sum_val+=np.bincount(idx_good,weights=vals_good,minlength=n)
        sum_sq+=np.bincount(idx_good,weights=vals_good**2,minlength=n)
        zero_count+=np.bincount(idx_good[vals_good==0],minlength=n)
        # Collect per label values (no huge image-stack buffer per marker).
        ordidx=np.argsort(idx_good,kind='stable')
        sorted_ids=idx_good[ordidx];sorted_values=vals_good[ordidx]
        unique,starts,num=np.unique(sorted_ids,return_index=True,return_counts=True)
        for j,start,count in zip(unique,starts,num):
            observations[int(j)].append(sorted_values[start:start+count])
        outer=np.asarray(img[~valid],dtype=np.float64)
        outer=outer[np.isfinite(outer)]
        per_z.append({'marker':marker_name,'z_index':z,'foreground_voxels':len(lab),
                      'finite_foreground_voxels':int(good.sum()),
                      'mean_signal_within_mask':float(vals_good.mean()) if len(vals_good) else np.nan,
                      'mean_signal_outside_mask':float(outer.mean()) if len(outer) else np.nan})
    records=[]
    for k,uid in enumerate(ids):
        count=int(total_count[k]);nf=int(finite_count[k]);s=float(sum_val[k])
        if nf:
            mu=s/nf
            variance=max(0.0,sum_sq[k]/nf-mu**2)
            vec=np.concatenate(observations[k])
            med,q90=np.quantile(vec,[.5,.9])
            vmin,vmax=float(np.min(vec)),float(np.max(vec))
        else:
            mu=variance=med=q90=vmin=vmax=np.nan
        records.append({'instance_id':int(uid),'marker':marker_name,
                        'n_mask_voxels':count,'n_finite_signal_voxels':nf,
                        'finite_signal_fraction':nf/count if count else np.nan,
                        'mean_raw_signal':mu,'median_raw_signal':float(med),
                        'q90_raw_signal':float(q90),'std_raw_signal':float(np.sqrt(variance)),
                        'sum_raw_signal':s if nf else np.nan,
                        'min_raw_signal':vmin,'max_raw_signal':vmax,
                        'fraction_exact_zero_signal':int(zero_count[k])/nf if nf else np.nan})
    if not np.array_equal(total_count,np.unique(mask[mask>0],return_counts=True)[1]):
        raise ValueError('Marker aggregation volume total per label mismatches mask')
    return pd.DataFrame.from_records(records),pd.DataFrame.from_records(per_z),target_z,preview_plane


def _plot_qc(mask:np.ndarray, plane:np.ndarray, marker_name:str,z:int,path:Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_erosion
    finite=np.isfinite(plane)
    x=np.asarray(plane,dtype=np.float64)
    if np.any(finite):
        vmin,vmax=np.percentile(x[finite],[1,99.5])
        if vmax<=vmin:vmax=vmin+1
    else: vmin,vmax=0.,1.
    # downsample only for preview rendering; never resize/shift source images for extraction.
    scale=max(1,int(np.ceil(max(plane.shape)/800)))
    sl=np.s_[::scale,::scale]
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
    axes[0].imshow(x[sl],cmap='gray',vmin=vmin,vmax=vmax,interpolation='nearest')
    axes[0].set_title(f'{marker_name}: source plane z={z} (index)')
    axes[1].imshow(x[sl],cmap='gray',vmin=vmin,vmax=vmax,interpolation='nearest')
    fg=mask[z]>0
    fg_s=fg[sl]
    if fg_s.any():
        boundary=fg_s & ~binary_erosion(fg_s)
        overlay=np.zeros((*fg_s.shape,4),dtype=float)
        overlay[boundary]=[1,.2,.2,.75]
        axes[1].imshow(overlay,interpolation='nearest')
    axes[1].set_title('Red: segmentation foreground boundary (NOT alignment proof)')
    for ax in axes:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Identical pixel grid only; physical registration unverified',fontsize=10)
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


def _write_verified(out:Path, tables:dict[str,pd.DataFrame], report:dict,
                    previews:list[tuple[str,int,np.ndarray]], mask:np.ndarray, figures:bool):
    if out.exists():raise FileExistsError(f'Output exists: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.wf09_stage_',dir=out.parent))
    try:
        for name,frame in tables.items():
            frame.to_csv(stage/name,index=False,float_format='%.10g')
        report['figures']=[]
        if figures:
            fd=stage/'figures';fd.mkdir()
            for name,z,plane in previews:
                fn=f'marker_{name}_mask_overlay_z{z:02}.png'
                _plot_qc(mask,plane,name,z,fd/fn)
                report['figures'].append(fn)
        (stage/'marker_integration_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
        digest={str(p.relative_to(stage)):sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        (stage/'output_file_sha256.json').write_text(json.dumps(digest,indent=2),encoding='utf8')
        if out.exists():raise FileExistsError(f'Concurrent output exists: {out}')
        stage.rename(out)
    finally:
        if stage.exists():shutil.rmtree(stage)


def extract(manifest:Path, morph_dir:Path, spatial_dir:Path, out:Path,
            allow_unverified_geometry:bool=False, figures:bool=True) -> dict:
    if not allow_unverified_geometry:
        raise ValueError('Explicitly pass --allow-unverified-geometry; this is exploratory signal sampling, NOT validated co-registration')
    if out.exists():raise FileExistsError(out)
    raw,specs=read_manifest(manifest)
    mask_path=_relative_or_abs(manifest.parent,raw['mask_path'])
    expected=raw.get('expected_mask_sha256')
    digest=sha256(mask_path)
    if not expected or expected!=digest:
        raise ValueError('expected_mask_sha256 is missing or differs from actual mask')
    mask=_read_image(mask_path)
    ids=_validate_mask(mask)
    morph,qc,spatial,summ,ssumm=validate_upstream(mask,mask_path,morph_dir,spatial_dir)
    # Check Workflow08 computed data against its own SHA-256 manifest when available.
    spatial_manifest=spatial_dir/'output_file_sha256.json'
    if not spatial_manifest.is_file():raise ValueError('Missing Workflow08 output SHA-256 manifest')
    declared=json.loads(spatial_manifest.read_text())
    if 'instance_spatial_features.csv' not in declared or 'spatial_organization_summary.json' not in declared:
        raise ValueError('Workflow08 manifest is incomplete')
    for name,sha in declared.items():
        if not (spatial_dir/name).is_file() or sha256(spatial_dir/name)!=sha:
            raise ValueError(f'Workflow08 output changed: {name}')
    morph=morph[['instance_id','voxel_count','z_span_planes','centroid_y_px','centroid_x_px']]
    qc=qc[['instance_id','qc_flag_review','touches_z_stack_boundary']]
    spatial=spatial[['instance_id','n_xy_radius_neighbors']]
    long=[];profiles=[];previews=[];markerprovenance=[]
    for spec in specs:
        paths=_path_list(spec,manifest.parent)
        # Reject using watershed label array accidentally as protein intensity input.
        if any(p.resolve()==mask_path.resolve() for p in paths):
            raise ValueError(f'{spec["name"]}: refusing to treat instance ID mask as marker intensity')
        tab,profile,z,plane=aggregate_one(mask,ids,spec,manifest.parent)
        long.append(tab); profiles.append(profile);previews.append((spec['name'],z,plane))
        markerprovenance.append({'name':spec['name'],'mode':spec['mode'],'axes':spec['axes'],
                                 'channel_index':spec.get('channel_index'),
                                 'files_sha256':{str(p):sha256(p) for p in paths}})
    long_df=pd.concat(long,ignore_index=True)
    if len(long_df)!=len(ids)*len(specs):raise ValueError('Incomplete instance-marker matrix')
    if not (long_df.groupby('instance_id')['n_mask_voxels'].nunique()==1).all():
        raise ValueError('Different mask voxel counts reported for same instance')
    if long_df.duplicated(['instance_id','marker']).any():raise ValueError('Duplicate instance-marker pair')
    wide=long_df.pivot(index='instance_id',columns='marker',values='mean_raw_signal')
    wide.columns=[f'mean_raw_{x}' for x in wide.columns];wide=wide.reset_index()
    combined=morph.merge(qc,on='instance_id',validate='one_to_one').merge(spatial,on='instance_id',validate='one_to_one').merge(wide,on='instance_id',validate='one_to_one')
    if len(combined)!=len(ids):raise ValueError('Spatial-molecular join lost instance IDs')
    report={'version':VERSION,'n_instances':len(ids),'n_markers':len(specs),
            'n_instance_marker_rows':len(long_df),'marker_names':[s['name'] for s in specs],
            'mask_shape_assumed_zyx':list(mask.shape),'source_mask_sha256':digest,
            'workflow07_summary_sha256':sha256(morph_dir/'morphometry_summary.json'),
            'workflow08_summary_sha256':sha256(spatial_dir/'spatial_organization_summary.json'),
            'workflow08_output_sha256_verified':True,
            'manifest_sha256':sha256(manifest),'markers':markerprovenance,
            'marker_geometry_status':'EXPLICIT_AXIS_AND_SHAPE_CHECK_ONLY; registration not independently verified',
            'reporting_units':'raw marker intensity within watershed pseudo-ID voxels; no calibrated concentrations',
            'no_protein_table_join':True,'software_versions':{'python':sys.version.split()[0],
              'numpy':np.__version__,'pandas':pd.__version__,'tifffile':tifffile.__version__},
            'limitations':['Only one specimen; no biological/clinical inference',
              'Watershed pseudo-ID is NOT ground truth cell identity',
              'Image dimensions and explicit section order do not establish marker-mask registration',
              'Marker identity/channel selection must be checked externally',
              'Mask may describe nuclei, not whole-cell or membrane regions',
              'Raw channel values not cross-marker normalized; do not compare intensities across markers as expression level',
              'No physical Z spacing or cell-contact proof',
              'No per-cell protein expression CSV joined by numeric ID']}
    tables={'instance_marker_signal.csv':long_df,'instance_marker_mean_matrix.csv':wide,
            'instance_marker_spatial_table.csv':combined,
            'marker_z_intensity_summary.csv':pd.concat(profiles,ignore_index=True)}
    _write_verified(out,tables,report,previews,mask,figures)
    return report


def verify_run(out:Path,manifest:Path|None=None)->dict:
    f=out/'output_file_sha256.json'
    if not f.is_file():raise FileNotFoundError(f)
    declared=json.loads(f.read_text(encoding='utf8'))
    actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
    if actual!=set(declared)|{'output_file_sha256.json'}:
        raise ValueError('Files added/missing in Workflow09 results')
    for name,expected in declared.items():
        if sha256(out/name)!=expected:raise ValueError(f'Workflow09 file modified: {name}')
    s=json.loads((out/'marker_integration_summary.json').read_text(encoding='utf8'))
    if manifest is not None:
        if sha256(manifest)!=s['manifest_sha256']:
            raise ValueError('Marker manifest changed since run')
        raw,specs=read_manifest(manifest)
        mask_path=_relative_or_abs(manifest.parent,raw['mask_path'])
        if sha256(mask_path)!=s['source_mask_sha256']:
            raise ValueError('Source mask changed since run')
        for source in s['markers']:
            for pname, expected in source['files_sha256'].items():
                if not Path(pname).is_file() or sha256(Path(pname))!=expected:
                    raise ValueError(f'Source marker changed since run: {pname}')
    return {'n_verified_files':len(declared),'n_instances':s['n_instances'],
            'n_markers':s['n_markers'],'geometry_status':s['marker_geometry_status']}
