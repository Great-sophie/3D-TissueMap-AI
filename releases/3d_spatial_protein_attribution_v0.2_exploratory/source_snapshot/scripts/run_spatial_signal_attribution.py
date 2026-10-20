#!/usr/bin/env python3
"""Workflow11 v0.1 — XY-only, nonoverlapping pseudo-instance protein attribution.

Modes: --check validates inputs; --apply runs once and refuses to overwrite;
--verify checks all output SHA-256 and immutable input provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_signal_attribution_core import MARKERS, REGIONS, RegionAccumulator, assign_slice

VERSION = '0.1.0-exploratory'
DEFAULT = {
    'mask': Path('data/inspect/LVILymph/full_mask_final_segmentation_hwatershed_bg500_90%.tif'),
    'manifest': Path('configs/protein_marker_manifest_registered_5channels.json'),
    'workflow09': Path('runs/3d_protein_marker_integration_v1'),
    'out': Path('runs/3d_spatial_signal_attribution_v0_1'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def file_manifest(args, *, hashes=True):
    mask = args.mask
    if not mask.is_file(): raise FileNotFoundError(mask)
    summary_path = args.workflow09/'marker_integration_summary.json'
    mean_path = args.workflow09/'instance_marker_mean_matrix.csv'
    wf09_outputs = args.workflow09/'output_file_sha256.json'
    spatial_path = args.workflow09/'instance_marker_spatial_table.csv'
    for f in (summary_path, mean_path, spatial_path, args.manifest, wf09_outputs):
        if not f.is_file():raise FileNotFoundError(f)
    summary = json.loads(summary_path.read_text())
    wf09_hashes = json.loads(wf09_outputs.read_text())
    for candidate in (mean_path, spatial_path, summary_path):
        if wf09_hashes.get(candidate.name) != sha(candidate):
            raise ValueError(f'Workflow09 frozen result changed: {candidate}')
    manifest = json.loads(args.manifest.read_text())
    if set(summary.get('marker_names',()))!=set(MARKERS):raise ValueError('Workflow09 marker set differs')
    if summary.get('n_instances')!=12663 and not getattr(args,'allow_synthetic_count',False):
        raise ValueError('Unexpected Workflow09 instance count; real run requires 12663')
    if summary.get('manifest_sha256')!=sha(args.manifest):raise ValueError('Workflow09 manifest changed')
    if summary.get('source_mask_sha256')!=sha(mask):raise ValueError('Workflow09 mask changed')
    if manifest.get('expected_mask_sha256')!=sha(mask):raise ValueError('Manifest/mask hash mismatch')
    if tuple(summary.get('mask_shape_assumed_zyx', ()))!=tuple(manifest.get('expected_shape', summary.get('mask_shape_assumed_zyx', ()))):
        raise ValueError('Shape metadata differs')
    expected = {m['name']:m['files_sha256'] for m in summary.get('markers',[])}
    index = {}
    srcfiles = [mask, args.manifest, summary_path, mean_path, spatial_path, wf09_outputs]
    for channel in manifest.get('markers',[]):
        name = channel.get('name')
        if name not in MARKERS or name in index: raise ValueError(f'Unexpected/duplicate marker {name}')
        if channel.get('mode') != 'per_section' or channel.get('axes') != 'YX':
            raise ValueError(f'Marker {name} must be per_section/YX')
        paths = [Path(s) for s in channel['files']]
        if len(paths) != int(summary['mask_shape_assumed_zyx'][0]):
            raise ValueError(f'Wrong Z planes for {name}')
        expected_paths = {str(Path(k).resolve()):v for k,v in expected[name].items()}
        if {str(p.resolve()) for p in paths} != set(expected_paths) or len(paths)!=len(set(paths)):
            raise ValueError(f'Z image path set mismatch: {name}')
        for path in paths:
            if not path.is_file():raise FileNotFoundError(path)
            if hashes and sha(path) != expected_paths[str(path.resolve())]:
                raise ValueError(f'Workflow09 marker input changed: {path}')
        index[name] = paths
        srcfiles.extend(paths)
    if set(index)!=set(MARKERS):raise ValueError('Missing marker channel(s)')
    df = pd.read_csv(spatial_path)
    ref = pd.read_csv(mean_path)
    expected_cols={'instance_id','centroid_y_px','centroid_x_px','z_span_planes','voxel_count'}|{'mean_raw_'+m for m in MARKERS}
    if not expected_cols.issubset(df.columns):raise ValueError('Workflow09 spatial table missing expected columns')
    if not ({'instance_id'}|{'mean_raw_'+m for m in MARKERS}).issubset(ref.columns):
        raise ValueError('Workflow09 reference matrix missing columns')
    if df.instance_id.isna().any() or df.instance_id.duplicated().any() or ref.instance_id.duplicated().any():
        raise ValueError('Nonunique instance ID')
    if len(df)!=summary['n_instances'] or len(ref)!=len(df) or set(ref.instance_id)!=set(df.instance_id):
        raise ValueError('Workflow09 instance count or ID set mismatch')
    with tifffile.TiffFile(mask) as tif:
        shape = tuple(tif.series[0].shape)
        dtype = tif.series[0].dtype
    if shape != tuple(summary['mask_shape_assumed_zyx']) or len(shape)!=3 or not np.issubdtype(dtype, np.integer):
        raise ValueError('Full mask ZYX shape/dtype mismatch')
    input_hashes = {str(p.resolve()):sha(p) for p in srcfiles} if hashes else {}
    return summary, index, df, ref, input_hashes


def make_long(acc:RegionAccumulator, ids:np.ndarray, min_pixels:int):
    nids=acc.max_id+1
    count=acc.count.reshape(len(REGIONS), nids)
    records=[]
    wides={'instance_id':ids}
    for marker in MARKERS:
        avg,sd,fraczero=acc.stats(marker)
        n=acc.n_finite[marker].reshape(len(REGIONS),nids)
        for ridx,region in enumerate(REGIONS):
            ns=n[ridx,ids]
            vals=np.where(ns>=min_pixels,avg[ridx,ids],np.nan)
            records.append(pd.DataFrame({
                'instance_id':ids,'marker':marker,'region':region,
                'n_assigned_voxels':count[ridx,ids],
                'n_finite_signal_voxels':ns,
                'sufficient_support':ns>=min_pixels,
                'mean_raw_signal':vals,
                'sd_raw_signal':np.where(ns>=min_pixels,sd[ridx,ids],np.nan),
                'fraction_exact_zero':np.where(ns>=min_pixels,fraczero[ridx,ids],np.nan),
            }))
            wides[f'{marker}__{region}']=vals
        wides[f'{marker}__inner_minus_outer']=wides[f'{marker}__perinuclear_1_2px']-wides[f'{marker}__outer_context_6_10px']
        wides[f'{marker}__inner_minus_nuclear']=wides[f'{marker}__perinuclear_1_2px']-wides[f'{marker}__nuclear_mask']
    return pd.concat(records,ignore_index=True), pd.DataFrame(wides)


def validate_nuclear(acc,ref:pd.DataFrame,df:pd.DataFrame):
    ids=df['instance_id'].to_numpy(dtype=int)
    # Expose a hard failure if one Z plane or channel was read out of order.
    n=acc.count.reshape(len(REGIONS),-1)[0,ids]
    refvox=df.set_index('instance_id').loc[ids,'voxel_count'].to_numpy(dtype=np.int64)
    if not np.array_equal(n,refvox):raise RuntimeError('WF11 nuclear counts differ from Workflow09 morphometry')
    reference = ref.set_index('instance_id').reindex(ids)
    maxerrors={}
    for marker in MARKERS:
        actual=acc.stats(marker)[0][0,ids]
        expected=reference['mean_raw_'+marker].to_numpy(dtype=float)
        if not np.allclose(actual,expected,rtol=2e-8,atol=2e-8,equal_nan=False):
            err=float(np.nanmax(np.abs(actual-expected)))
            raise RuntimeError(f'Nuclear baseline mismatch: {marker}, maxabs={err}')
        maxerrors[marker]=float(np.max(np.abs(actual-expected)))
    return maxerrors


def hotspot_table(spatial,wide):
    d=spatial.merge(wide,on='instance_id',how='inner',validate='one_to_one')
    threshold=float(d.mean_raw_HER2.quantile(.99))
    roi=(d.centroid_x_px>=480)&(d.centroid_x_px<720)&(d.centroid_y_px>=480)&(d.centroid_y_px<720)
    hotspot=d.mean_raw_HER2>=threshold
    result=[]
    for marker in MARKERS:
        for region in REGIONS:
            c=f'{marker}__{region}'
            for group, sel in [('ROI_high_by_WF09_nuclear_HER2',roi & hotspot),
                               ('ROI_other',roi & ~hotspot), ('outside_ROI',~roi)]:
                vals=d.loc[sel,c].dropna()
                result.append({'marker':marker,'region':region,'group':group,
                               'n_instances':int(sel.sum()),'n_supported':len(vals),
                               'median_raw':float(vals.median()) if len(vals) else np.nan,
                               'q25_raw':float(vals.quantile(.25)) if len(vals) else np.nan,
                               'q75_raw':float(vals.quantile(.75)) if len(vals) else np.nan})
    return pd.DataFrame(result), {'roi_x':[480,720],'roi_y':[480,720],
                                  'threshold_WF09_nuclear_HER2_P99':threshold,
                                  'n_roi':int(roi.sum()),'n_roi_high':int((roi & hotspot).sum()),
                                  'selection_and_measurement_on_same_specimen':True}


def write_figure(stage, mask, paths, z=5, x0=480,x1=720,y0=480,y1=720):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    z=min(z,mask.shape[0]-1)
    sl=mask[z]
    own=assign_slice(sl,int(mask.max()))
    a=tifffile.imread(paths['HER2'][z])[y0:y1,x0:x1].astype(float)
    reg=own.regions[y0:y1,x0:x1]
    if a.shape != reg.shape or a.size == 0:raise ValueError('Invalid QC ROI')
    fig,ax=plt.subplots(1,2,figsize=(11,5))
    im=np.log1p(a)
    vmax=float(np.quantile(im,.995))
    ax[0].imshow(im,cmap='magma',vmin=0,vmax=max(vmax,1e-6))
    ax[0].set_title(f'Raw registered HER2, z={z}; log1p')
    ax[1].imshow(im,cmap='gray',vmin=0,vmax=max(vmax,1e-6))
    colors=['#4B9CD3','#00C49A','#FFC145','#E06CFF']
    for r,color in enumerate(colors):
        layer=np.ma.masked_where(reg!=r,np.ones(reg.shape))
        ax[1].imshow(layer,cmap=ListedColormap([color]),alpha=.35,vmin=0,vmax=1,
                     interpolation='nearest')
    ax[1].legend(handles=[Patch(color=c,label=n) for c,n in zip(colors,REGIONS)],
                 loc='upper center',bbox_to_anchor=(.5,-.02),ncol=2,fontsize=7)
    ax[1].set_title('Exclusive XY-nearest pseudo-owner regions')
    for axis in ax:axis.axis('off')
    fig.suptitle(f'WF11 attribution geometry; X {x0}:{x1} Y {y0}:{y1}; Z=index, not µm')
    fig.tight_layout()
    fig.savefig(stage/'her2_region_assignment_z05.png',dpi=150,bbox_inches='tight')
    plt.close(fig)


def run(args):
    summary, paths, df, ref, digests = file_manifest(args)
    if args.min_pixels<1:raise ValueError('min-pixels must be >=1')
    if args.out.exists():raise FileExistsError('Output exists; refusing to overwrite '+str(args.out))
    mask=tifffile.imread(args.mask)
    if not np.issubdtype(mask.dtype,np.integer) or mask.ndim!=3:
        raise ValueError('Mask must be ZYX integer')
    ids=df.instance_id.to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(mask[mask>0]), np.sort(ids)):
        raise ValueError('Mask ID set does not match Workflow09 table')
    acc=RegionAccumulator(int(mask.max()))
    for z in range(mask.shape[0]):
        own=assign_slice(mask[z],acc.max_id)
        images={m:tifffile.imread(paths[m][z]) for m in MARKERS}
        for m,a in images.items():
            if a.shape != mask.shape[1:]:raise ValueError(f'{m} image plane z={z} shape differs')
        acc.add_plane(own,images)
        print(f'PROCESSED z={z:02d}/{mask.shape[0]-1:02d}; assigned={len(own.codes):,}',flush=True)
    errors=validate_nuclear(acc,ref,df)
    long,wide=make_long(acc,ids,args.min_pixels)
    hotspot,hotmeta=hotspot_table(df,wide)
    qcs=[]
    for (m,r),g in long.groupby(['marker','region'],sort=True):
        valid=g.mean_raw_signal.dropna()
        qcs.append({'marker':m,'region':r,'instances_with_min_support':len(valid),
                    'n_instances':len(g),'assigned_voxels':int(g.n_assigned_voxels.sum()),
                    'median_instance_mean':float(valid.median()) if len(valid) else np.nan,
                    'q90_instance_mean':float(valid.quantile(.9)) if len(valid) else np.nan,
                    'fraction_instances_with_support':float(len(valid)/len(g))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.workflow11_tmp_',dir=args.out.parent))
    try:
        long.to_csv(stage/'instance_marker_region_signal.csv',index=False)
        wide.to_csv(stage/'instance_marker_attribution_matrix.csv',index=False)
        pd.DataFrame(qcs).to_csv(stage/'marker_region_qc.csv',index=False)
        hotspot.to_csv(stage/'her2_hotspot_region_comparison.csv',index=False)
        # QC uses only X/Y pixel coordinates and is never evidence of a true cell membrane.
        if not args.no_figure:
            write_figure(stage,mask,paths,z=min(5,mask.shape[0]-1),
                         x0=min(480,mask.shape[2]-1),x1=min(720,mask.shape[2]),
                         y0=min(480,mask.shape[1]-1),y1=min(720,mask.shape[1]))
        record={
            'workflow':'11-Spatially Aware Protein Signal Attribution',
            'version':VERSION,'status':'EXPLORATORY','n_instances':len(ids),
            'n_markers':len(MARKERS),'n_long_rows':len(long),'n_z_sections':mask.shape[0],
            'regions':list(REGIONS),'xy_distance_bins_pixels':[[0,0],[1,2],[2,5],[5,10]],
            'boundaries':'0<d<=2, 2<d<=5, 5<d<=10 XY Euclidean pixel distance; nuclear d=0',
            'assignment':'per-Z 2D scipy EDT nearest nuclear pseudo-ID; exclusive ownership; EDT tie breaking',
            'min_finite_region_pixels':args.min_pixels,
            'input_sha256':digests,'workflow09_summary_sha256':sha(args.workflow09/'marker_integration_summary.json'),
            'max_abs_nuclear_baseline_difference':errors,'her2_hotspot_QC':hotmeta,
            'limitations':[
                'One specimen: no independent clinical or biological replication.',
                'Author watershed pseudo-instances may be nuclear-centric; not verified whole cells.',
                'XY nearest-nucleus ownership is a proxy and does not establish true cell borders or membranes.',
                'Shells may contain another cell cytoplasm/processes or extracellular marker signal.',
                'Outer 6–10px ring is local context, NOT validated background.',
                '2D per-section distance only, no Z physical spacing or anisotropic 3D nearest-owner assignment.',
                'Mask/channel alignment remains visually plausible but not independently registered.',
                'Raw intensities are not normalized across marker channels.',
                'HER2-high ROI was selected from these same data; contrasts are exploratory and circular for HER2 nuclear signal.'
            ]}
        (stage/'attribution_summary.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
        outputhash={p.name:sha(p) for p in sorted(stage.iterdir()) if p.is_file()}
        (stage/'output_sha256.json').write_text(json.dumps(outputhash,indent=2)+'\n')
        stage.rename(args.out)
    finally:
        if stage.exists():shutil.rmtree(stage)
    return {'status':'CREATED','n_instances':len(ids),'n_rows':len(long),
            'n_files_verified_by_manifest':len(outputhash),'nuclear_baseline_match':'PASS','output':str(args.out)}


def verify(args):
    if not args.out.is_dir():raise FileNotFoundError('Workflow11 output missing')
    hashes=json.loads((args.out/'output_sha256.json').read_text())
    have={p.name for p in args.out.iterdir() if p.is_file()}
    if have != set(hashes)|{'output_sha256.json'}:raise RuntimeError('Missing/extra output files')
    for name,digest in hashes.items():
        if sha(args.out/name)!=digest:raise RuntimeError('Output changed: '+name)
    record=json.loads((args.out/'attribution_summary.json').read_text())
    summary, paths, df, ref, digests = file_manifest(args)
    if digests!=record['input_sha256']:raise RuntimeError('Input file change detected')
    if len(df)!=record['n_instances'] or len(MARKERS)!=record['n_markers']:
        raise RuntimeError('Instance/channel count changed')
    if len(pd.read_csv(args.out/'instance_marker_region_signal.csv'))!=record['n_long_rows']:
        raise RuntimeError('Long output row count changed')
    return {'status':'PASS','verified_result_files':len(hashes),
            'verified_input_files':len(digests),'instances':record['n_instances'],
            'instance_marker_region_rows':record['n_long_rows']}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    sel=ap.add_mutually_exclusive_group(required=True)
    sel.add_argument('--check',action='store_true')
    sel.add_argument('--apply',action='store_true')
    sel.add_argument('--verify',action='store_true')
    ap.add_argument('--mask',type=Path,default=DEFAULT['mask'])
    ap.add_argument('--manifest',type=Path,default=DEFAULT['manifest'])
    ap.add_argument('--workflow09',type=Path,default=DEFAULT['workflow09'])
    ap.add_argument('--out',type=Path,default=DEFAULT['out'])
    ap.add_argument('--min-pixels',type=int,default=10)
    ap.add_argument('--no-figure',action='store_true')
    a=ap.parse_args(argv)
    if a.check:
        summary, paths, df, ref, digests = file_manifest(a)
        result={'status':'PREFLIGHT_PASS','instances':len(df),'marker_channels':len(paths),
                'z_sections':summary['mask_shape_assumed_zyx'][0],
                'input_sha256_files':len(digests),'output_exists':a.out.exists(),
                'warning':'Dry-run input validation only. No spatial ownership calculated.'}
    else:result=verify(a) if a.verify else run(a)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
