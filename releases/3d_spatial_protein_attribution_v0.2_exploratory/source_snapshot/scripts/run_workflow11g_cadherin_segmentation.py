#!/usr/bin/env python3
"""Workflow 11G: 2D per-section Cadherin-assisted candidate cell-like segmentation.

ROI-limited feasibility. Ir193-associated AUTHOR pseudo-nuclei are marker seeds,
panCK defines tissue domain, Yb174 Cadherin-associated signal defines optional
edge cost. Gd156 HER2 is READOUT ONLY, NEVER affects mask, seeds or tuning.

Not validated single-cell segmentation or physical 3D morphology. Commands:
--check, --apply, --verify (no automatic overwrites).
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.segmentation import watershed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_spatial_signal_attribution import file_manifest, sha
from run_workflow11e_membrane_feasibility import epithelial_domain

VERSION = '0.1.0-exploratory'
Z_SHOW = (0, 5, 10, 15)
DEFAULT_ROI = (480, 720, 480, 720)
METAL = 'Yb174'


def args_validate(a):
    x0,x1,y0,y1=a.roi
    if not (0<=x0<x1<=1215 and 0<=y0<y1<=1205):
        raise ValueError('ROI must lie within 1215x1205 registered slices')
    if not (.5 < a.panck_quantile < .99 and a.sigma > 0 and a.min_domain_px >= 10):
        raise ValueError('Invalid panCK domain parameters')
    if not (0 <= a.cadherin_weight <= 1 and a.min_seed_overlap_px >= 1):
        raise ValueError('Invalid Cadherin weight / seed support')
    if not a.cadherin_zip.is_file(): raise FileNotFoundError(a.cadherin_zip)
    if a.min_interface_px < 1 or a.min_core_px < 1:raise ValueError('Invalid band support minimum')


def channel_members(path):
    with zipfile.ZipFile(path) as z:
        files=sorted(s for s in z.namelist() if re.search(r'__Yb174\.tiff?$',s,re.I))
    if len(files)!=16 or [Path(s).name[:4] for s in files] != [f'{n:04d}' for n in range(1,17)]:
        raise ValueError('Cadherin registered Yb174 must contain 16 indexed TIFFs')
    if len(set(files))!=16:raise ValueError('Repeated member')
    return files


def verify_order(members, paths):
    for z, member in enumerate(members):
        a=Path(member).name.rsplit('__',1)[0]
        for m in ('HER2','panCK','Ir193'):
            b=paths[m][z].name.rsplit('__',1)[0]
            if a!=b:
                raise ValueError(f'Yb174 source section mismatch with {m} at z={z}: {a} != {b}')


def preflight(a,hashes=True):
    args_validate(a)
    # Reuse the WORKFLOW11/09 strong source-integrity verifier (five channels).
    summary,paths,spatial,ref,digests=file_manifest(a,hashes=hashes)
    members=channel_members(a.cadherin_zip)
    verify_order(members,paths)
    if int(summary['mask_shape_assumed_zyx'][0])!=16:
        raise ValueError('Yb174 registered ZIP currently has exactly 16 sections')
    digests=dict(digests)
    digests[str(a.cadherin_zip.resolve())]=sha(a.cadherin_zip) if hashes else None
    return summary,paths,spatial,ref,digests,members


def component_seed_map(nuclei, epithelial_domains, min_overlap=4):
    """One 2D seed per *connected pseudo-nuclear fragment* within a domain.

    The source ID is retained as provenance; seed_id is LOCAL TO Z. Excludes
    fragments with insufficient overlap. Sources may be fragmented, so these
    are not authoritative individual nuclei or unique 3D cell IDs.
    """
    nuclei=np.asarray(nuclei)
    dom=np.asarray(epithelial_domains)
    if nuclei.shape!=dom.shape or nuclei.ndim!=2:raise ValueError('Seed/domain shape mismatch')
    markers=np.zeros(nuclei.shape,np.int32)
    rows=[]
    sid=0
    for d in range(1,int(dom.max())+1):
        component=dom==d
        ids=np.unique(nuclei[component]);ids=ids[ids>0]
        for source_id in ids:
            fragments,n=ndi.label(component&(nuclei==source_id),structure=ndi.generate_binary_structure(2,1))
            for ci in range(1,n+1):
                fragment=fragments==ci
                npx=int(fragment.sum())
                if npx<min_overlap:continue
                # Deepest pixel within fragment; deterministic tie by row-major argmax.
                dist=ndi.distance_transform_edt(fragment)
                flat=int(np.argmax(dist))
                y,x=np.unravel_index(flat, fragment.shape)
                sid+=1
                markers[y,x]=sid
                rows.append(dict(seed_id=sid,domain_id=d,source_pseudo_id=int(source_id),
                                 nucleus_fragment_px=npx,seed_y_local=int(y),seed_x_local=int(x),
                                 n_fragments_for_source_in_domain=int(n)))
    return markers,rows


def normalize_in_mask(a, component, lower=.10, upper=.90):
    v=a[component]
    if not len(v):return np.zeros_like(a,float)
    low,high=np.quantile(v,[lower,upper])
    if high<=low+1e-12:return np.zeros_like(a,float)
    return np.clip((a-low)/(high-low),0,1)


def segment_slice(panck,nuclei,cadherin,quantile=.80,sigma=1.,min_domain_px=40,
                  min_seed_overlap_px=4,cadherin_weight=.75):
    """Return domain, one-to-one seed mapping, Voronoi and candidate labels.

    Cadherin weight and other hyperparameters are fixed without HER2 tuning.
    """
    if not (panck.shape==nuclei.shape==cadherin.shape and panck.ndim==2):
        raise ValueError('All inputs must be matching 2D ROI arrays')
    if not (np.isfinite(cadherin).all() and (cadherin>=0).all()):
        raise ValueError('Cadherin channel must be finite and nonnegative')
    domain,q=epithelial_domain(panck,quantile=quantile,sigma=sigma,min_component_px=min_domain_px)
    markers,seeds=component_seed_map(nuclei,domain,min_overlap=min_seed_overlap_px)
    baseline=np.zeros(domain.shape,np.int32)
    assisted=np.zeros(domain.shape,np.int32)
    cad_smooth=ndi.gaussian_filter(np.log1p(cadherin.astype(float)),sigma=1)
    region_summary=[]
    for d in range(1,int(domain.max())+1):
        component=domain==d
        dseeds=markers*component
        all_ids=np.unique(dseeds);all_ids=all_ids[all_ids>0]
        region_summary.append({'domain_id':d,'domain_px':int(component.sum()),'n_seed_fragments':len(all_ids)})
        if not len(all_ids):continue
        sl=ndi.find_objects(component.astype(np.int8))[0]
        part=component[sl]
        seed=dseeds[sl]
        _,nearest=ndi.distance_transform_edt(seed==0,return_indices=True)
        near=seed[tuple(nearest)]
        nearest_distance=ndi.distance_transform_edt(seed==0)
        localbase=np.where(part,near,0)
        baseline[sl][part]=localbase[part]
        # High Yb174 signal as *cost/ridge*, plus small seed proximity regularizer.
        c=normalize_in_mask(cad_smooth[sl],part)
        dist=normalize_in_mask(nearest_distance,part)
        elevation=cadherin_weight*c+(1-cadherin_weight)*dist
        lab=watershed(elevation,markers=seed,mask=part,connectivity=1).astype(np.int32)
        assisted[sl][part]=lab[part]
    if np.any((assisted>0)!=(baseline>0)):
        raise RuntimeError('Candidate and baseline have different nonempty epithelial coverage')
    if not np.array_equal(np.unique(assisted[assisted>0]),np.unique(markers[markers>0])):
        raise RuntimeError('Candidate segmentation lost seeds')
    if not np.array_equal(assisted[markers>0],markers[markers>0]):
        raise RuntimeError('Candidate segmentation violates marked nuclear seed')
    if not np.array_equal(baseline[markers>0],markers[markers>0]):
        raise RuntimeError('Voronoi baseline violates marked nuclear seed')
    return domain,markers,seeds,baseline,assisted,q,region_summary


def shared_interface(labels):
    """Within-tissue, 4-connected INTER-segment boundaries only (not exterior)."""
    a=np.asarray(labels)
    edge=np.zeros_like(a,dtype=bool)
    pair=(a[:,:-1]!=a[:,1:]) & (a[:,:-1]>0)&(a[:,1:]>0)
    edge[:,:-1]|=pair;edge[:,1:]|=pair
    pair=(a[:-1,:]!=a[1:,:]) & (a[:-1,:]>0)&(a[1:,:]>0)
    edge[:-1,:]|=pair;edge[1:,:]|=pair
    return edge


def region_metrics(labels,her2,cad,seedrows,method,min_interface_px=3,min_core_px=5):
    """Held-out readouts from shared interfaces and eroded candidate interiors.

    Interface is between two positive region IDs, not tissue/background exterior.
    Core is >=3 Euclidean px inside region, with explicit zero padding to
    retain correct distances at cropped bounding boxes and ROI edges.
    """
    edge=shared_interface(labels)
    bounds=ndi.find_objects(labels)
    out=[]
    for row in seedrows:
        sid=row['seed_id']
        sl=bounds[sid-1] if sid-1<len(bounds) else None
        if sl is None:raise RuntimeError('Missing seeded segment')
        seg=(labels[sl]==sid)
        area=int(seg.sum())
        dist=ndi.distance_transform_edt(np.pad(seg,1,constant_values=0))[1:-1,1:-1]
        edgecell=seg&edge[sl]
        core=seg&(dist>=3)
        ne=int(edgecell.sum());nc=int(core.sum())
        stats={'method':method,'seed_id':sid,'domain_id':row['domain_id'],
               'source_pseudo_id':row['source_pseudo_id'],'segment_area_px':area,
               'n_shared_interface_px':ne,'n_core_px':nc,
               'nucleus_fragment_px':row['nucleus_fragment_px'],
               'testable_interface':bool(ne>=min_interface_px and nc>=min_core_px)}
        for name,im in [('HER2',her2[sl]),('Yb174_Cadherin',cad[sl])]:
            b=float(np.mean(im[edgecell])) if ne>=min_interface_px else np.nan
            c=float(np.mean(im[core])) if nc>=min_core_px else np.nan
            stats[f'{name}_interface_mean']=b
            stats[f'{name}_core_mean']=c
            stats[f'{name}_interface_minus_core']=b-c if np.isfinite(b) and np.isfinite(c) else np.nan
        out.append(stats)
    return out


def summary_pairs(metrics):
    d=pd.DataFrame(metrics)
    cols=['z','seed_id','HER2_interface_minus_core','Yb174_Cadherin_interface_minus_core']
    ba=d[d.method=='nuclear_voronoi'].loc[:,cols].rename(columns={
        'HER2_interface_minus_core':'voronoi_HER2_delta',
        'Yb174_Cadherin_interface_minus_core':'voronoi_Cadherin_delta'})
    ca=d[d.method=='cadherin_assisted'].loc[:,cols].rename(columns={
        'HER2_interface_minus_core':'cadherin_HER2_delta',
        'Yb174_Cadherin_interface_minus_core':'cadherin_Cadherin_delta'})
    paired=ba.merge(ca,on=['z','seed_id'],validate='one_to_one')
    paired['HER2_delta_change']=paired.cadherin_HER2_delta-paired.voronoi_HER2_delta
    paired['Cadherin_delta_change']=paired.cadherin_Cadherin_delta-paired.voronoi_Cadherin_delta
    return paired


def figure_panels(path,selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(selected),5,figsize=(19,len(selected)*3.8),squeeze=False)
    for row,(z,cad,ck,ir,her2,seeds,base,assist) in enumerate(selected):
        lim=lambda a:max(float(np.quantile(np.log1p(a),.995)),1e-6)
        images=[(np.log1p(cad),'magma','Yb174 Cadherin'),
                (np.log1p(ck),'Greens','panCK domain'),
                (np.log1p(ir),'Blues','Ir193 + seeds'),
                (np.log1p(her2),'magma','HER2 + Voronoi'),
                (np.log1p(her2),'magma','HER2 + Cadherin')]
        for col,(data,cmap,title) in enumerate(images):
            ax=axes[row,col]
            ax.imshow(data,cmap=cmap,vmin=0,vmax=lim(np.expm1(data)))
            if col==2:
                ys,xs=np.where(seeds>0)
                ax.scatter(xs,ys,s=4,c='red',alpha=.65)
            if col in (3,4):
                interface=shared_interface(base if col==3 else assist)
                ax.contour(interface.astype(float),levels=[.5],colors='cyan',linewidths=.35)
            ax.set_title(f'z={z} | {title}',fontsize=10)
            ax.axis('off')
    fig.suptitle('WF11G candidate boundaries; Yb174 used ONLY for assisted; HER2 held out; Z=index, not µm',fontsize=13)
    fig.tight_layout()
    fig.savefig(path,dpi=140,bbox_inches='tight');plt.close(fig)


def save_atomic(stage,target):
    if target.exists():raise FileExistsError('Output already exists: '+str(target))
    stage.rename(target)


def run(a):
    if a.out.exists():raise FileExistsError('Refusing to overwrite existing result: '+str(a.out))
    summary,paths,spatial,ref,input_hashes,members=preflight(a)
    mask=tifffile.imread(a.mask)
    if mask.ndim!=3 or tuple(mask.shape)!=tuple(summary['mask_shape_assumed_zyx']):
        raise ValueError('Mask shape mismatch')
    x0,x1,y0,y1=a.roi
    roi_shape=(y1-y0,x1-x0)
    labels_assisted=np.zeros((len(mask),*roi_shape),np.uint32)
    labels_base=np.zeros_like(labels_assisted)
    all_seed=[];all_metrics=[];qcs=[];montage=[]
    with zipfile.ZipFile(a.cadherin_zip) as zipf:
        for z in range(len(mask)):
            original={m:tifffile.imread(paths[m][z]) for m in ('panCK','Ir193','HER2')}
            cad=tifffile.imread(io.BytesIO(zipf.read(members[z])))
            if any(im.shape!=mask.shape[1:] for im in list(original.values())+[cad]):
                raise ValueError(f'Channel-to-mask geometry mismatch at z={z}')
            cropped={m:im[y0:y1,x0:x1] for m,im in original.items()}
            cad=cad[y0:y1,x0:x1]
            domain,seeds,rows,base,candidate,q,domains=segment_slice(
                cropped['panCK'],mask[z,y0:y1,x0:x1],cad,
                quantile=a.panck_quantile,sigma=a.sigma,min_domain_px=a.min_domain_px,
                min_seed_overlap_px=a.min_seed_overlap_px,cadherin_weight=a.cadherin_weight)
            labels_assisted[z]=candidate
            labels_base[z]=base
            for s in rows:
                s.update(z=z,seed_y_px=s.pop('seed_y_local')+y0,seed_x_px=s.pop('seed_x_local')+x0)
            all_seed.extend(rows)
            for met,lab in [('nuclear_voronoi',base),('cadherin_assisted',candidate)]:
                each=region_metrics(lab,cropped['HER2'],cad,rows,met,
                                    min_interface_px=a.min_interface_px,min_core_px=a.min_core_px)
                for r in each:r['z']=z
                all_metrics.extend(each)
            qcs.append({'z':z,'n_domains':int(domain.max()),'n_domains_seeded':sum(t['n_seed_fragments']>0 for t in domains),
                        'n_seeds':len(rows),'n_domain_px':int(np.count_nonzero(domain)),
                        'n_candidate_px':int(np.count_nonzero(candidate)),
                        'n_unseeded_domain_px':int(np.count_nonzero((domain>0)&(candidate==0))),
                        'panCK_log_threshold':q})
            if z in Z_SHOW:montage.append((z,cad,cropped['panCK'],cropped['Ir193'],cropped['HER2'],seeds,base,candidate))
            print(f'PROCESSED z={z:02d}/15 | domains={int(domain.max())} seeded={len(rows)}',flush=True)
    if not all_seed:raise RuntimeError('No nuclear seeds in selected epithelial ROI')
    df_metrics=pd.DataFrame(all_metrics)
    pair=summary_pairs(all_metrics)
    valid=pair.dropna(subset=['HER2_delta_change'])
    print(f'MATCHED held-out HER2 interface readouts: {len(valid)}/{len(pair)}',flush=True)
    # Choose nuclei by geometry only, never by HER2 readout, for future manual QC.
    queue=(pd.DataFrame(all_seed)
           .loc[lambda d:d.z.isin(Z_SHOW)]
           .sort_values(['z','nucleus_fragment_px','source_pseudo_id'],ascending=[True,False,True])
           .groupby('z',group_keys=False).head(4).copy())
    queue['manual_boundary_quality']='UNASSESSED'
    queue['manual_annotation_status']='PENDING'
    a.out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.wf11g_staging_',dir=a.out.parent))
    try:
        tifffile.imwrite(stage/'candidate_cadherin_labels_zyx.tif',labels_assisted,metadata={'axes':'ZYX'})
        tifffile.imwrite(stage/'baseline_voronoi_labels_zyx.tif',labels_base,metadata={'axes':'ZYX'})
        pd.DataFrame(all_seed).to_csv(stage/'section_nuclear_seed_index.csv',index=False)
        df_metrics.to_csv(stage/'candidate_boundary_metrics.csv',index=False)
        pair.to_csv(stage/'heldout_HER2_matched_comparison.csv',index=False)
        pd.DataFrame(qcs).to_csv(stage/'section_segmentation_qc.csv',index=False)
        queue.to_csv(stage/'manual_boundary_review_queue.csv',index=False)
        figure_panels(stage/'cadherin_vs_voronoi_heldout_HER2.png',montage)
        record={'workflow':'11G-2D Cadherin-assisted candidate cell-like segmentation',
                'version':VERSION,'roi_xy':list(a.roi),'n_z':len(mask),
                'n_section_seed_fragments':len(all_seed),'n_candidate_metric_rows':len(df_metrics),
                'n_matched_seed_rows':len(pair),'n_valid_HER2_paired_rows':len(valid),
                'matched_HER2_delta_median_change':float(valid.HER2_delta_change.median()) if len(valid) else None,
                'matched_HER2_delta_fraction_positive':float((valid.HER2_delta_change>0).mean()) if len(valid) else None,
                'n_manual_review_queue':len(queue),
                'n_domains_seeded':int(sum(s['n_domains_seeded'] for s in qcs)),
                'marker_role':{'Ir193':'author pseudo-nuclear seeds','panCK':'epithelial tissue domain',
                               'Yb174':'Cadherin-associated boundary cost','HER2':'HELD-OUT READOUT ONLY'},
                'params':{'panck_quantile':a.panck_quantile,'sigma':a.sigma,
                          'min_domain_px':a.min_domain_px,'min_seed_overlap_px':a.min_seed_overlap_px,
                          'cadherin_weight':a.cadherin_weight,
                          'min_interface_px':a.min_interface_px,'min_core_px':a.min_core_px},
                'method_baseline':'Euclidean nearest 2D pseudo-nuclear seed inside each panCK tissue domain',
                'method_assisted':'2D seeded watershed: weighted normalized smoothed log1p Yb174 intensity + seed distance',
                'comparison':'Held-out HER2 shared-intersegment interface versus >=3px region core; no tuning from HER2',
                'input_sha256':input_hashes,
                'workflow11g_script_sha256':sha(Path(__file__)),
                'limitations':[
                    'ROI selected after HER2 hotspot observation: this is exploratory evaluation on SAME specimen',
                    '2D independent slice labels are not linked real 3D cells; seed_id must always be paired with z',
                    'Author pseudo-nuclear IDs/fragments are not manually verified independent nuclei',
                    'One seed per candidate segment is BY DESIGN, not a segmentation accuracy metric',
                    'Cadherin 36/E-Cadherin antibody target must be checked for exact cross-reactivity',
                    'High Yb174 intensity used in model: Yb174 interface enrichment is NOT independent evidence',
                    'HER2 held out from algorithm but used to select this ROI; NOT independent prospective validation',
                    'Any improved HER2 interface signal could reflect shared tissue context, registration, or segmentation bias',
                    'Registered channel geometry is not independently validated; physical Z spacing unknown',
                    'Cell-to-cell boundaries and biological cell ownership require annotation/orthogonal validation',
                    'No clinical, biological, or generalization claims supported from this single specimen'
                ]}
        (stage/'workflow11g_summary.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
        output_sha={p.name:sha(p) for p in stage.iterdir() if p.is_file()}
        (stage/'output_sha256.json').write_text(json.dumps(output_sha,indent=2)+'\n')
        save_atomic(stage,a.out)
    finally:
        if stage.exists():shutil.rmtree(stage)
    return {'status':'CREATED','n_z':len(mask),'n_seed_fragments':len(all_seed),
            'matched_methods':len(pair),'files_hashed':len(output_sha),'output':str(a.out)}


def verify(a):
    _,_,_,_,hashes,_=preflight(a)
    if not a.out.is_dir():raise FileNotFoundError(a.out)
    record=json.loads((a.out/'workflow11g_summary.json').read_text())
    if hashes != record['input_sha256']:raise RuntimeError('Input provenance changed')
    if sha(Path(__file__)) != record['workflow11g_script_sha256']:
        raise RuntimeError('Workflow11G algorithm script changed')
    if list(a.roi)!=record['roi_xy'] or {'panck_quantile':a.panck_quantile,'sigma':a.sigma,
        'min_domain_px':a.min_domain_px,'min_seed_overlap_px':a.min_seed_overlap_px,
        'cadherin_weight':a.cadherin_weight,'min_interface_px':a.min_interface_px,
        'min_core_px':a.min_core_px}!=record['params']:
        raise RuntimeError('Verification parameters differ from original experiment')
    outputhash=json.loads((a.out/'output_sha256.json').read_text())
    if {p.name for p in a.out.iterdir() if p.is_file()} != set(outputhash)|{'output_sha256.json'}:
        raise RuntimeError('Output file set changed')
    for name,d in outputhash.items():
        if sha(a.out/name)!=d:raise RuntimeError('Tampered or modified result: '+name)
    base=tifffile.imread(a.out/'baseline_voronoi_labels_zyx.tif')
    cad=tifffile.imread(a.out/'candidate_cadherin_labels_zyx.tif')
    if base.shape != cad.shape or base.shape[0]!=record['n_z']:
        raise RuntimeError('Output geometry mismatch')
    paired=pd.read_csv(a.out/'heldout_HER2_matched_comparison.csv')
    seeds=pd.read_csv(a.out/'section_nuclear_seed_index.csv')
    if len(paired)!=record['n_matched_seed_rows'] or len(seeds)!=record['n_section_seed_fragments']:
        raise RuntimeError('Output row count changed')
    if paired.duplicated(['z','seed_id']).any() or seeds.duplicated(['z','seed_id']).any():
        raise RuntimeError('Output seed ID inconsistency')
    return {'status':'PASS','output_files_verified':len(outputhash),'inputs_verified':len(hashes),
            'n_z':base.shape[0],'n_section_seed_fragments':len(seeds)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--apply',action='store_true')
    modes.add_argument('--verify',action='store_true')
    p.add_argument('--mask',type=Path,default=Path('data/inspect/LVILymph/full_mask_final_segmentation_hwatershed_bg500_90%.tif'))
    p.add_argument('--manifest',type=Path,default=Path('configs/protein_marker_manifest_registered_5channels.json'))
    p.add_argument('--workflow09',type=Path,default=Path('runs/3d_protein_marker_integration_v1'))
    p.add_argument('--cadherin-zip',type=Path,default=Path('data/inspect/LVILymph/LVILymphBreastCancer_SIMILARITY10_In115.zip'))
    p.add_argument('--roi',type=int,nargs=4,default=DEFAULT_ROI,metavar=('X0','X1','Y0','Y1'))
    p.add_argument('--panck-quantile',type=float,default=.80)
    p.add_argument('--sigma',type=float,default=1.0)
    p.add_argument('--min-domain-px',type=int,default=40)
    p.add_argument('--min-seed-overlap-px',type=int,default=4)
    p.add_argument('--cadherin-weight',type=float,default=.75)
    p.add_argument('--min-interface-px',type=int,default=3)
    p.add_argument('--min-core-px',type=int,default=5)
    p.add_argument('--out',type=Path,default=Path('runs/3d_cadherin_assisted_celllike_segmentation_v0_1'))
    a=p.parse_args(argv)
    if a.check:
        summary,paths,spatial,ref,digests,members=preflight(a)
        result={'status':'PREFLIGHT_PASS','mask_shape':summary['mask_shape_assumed_zyx'],
                'n_wf09_instances':summary['n_instances'],'roi_xy':list(a.roi),
                'registered_Yb174_Z_slices':len(members),'provenance_inputs':len(digests),
                'note':'2D candidate cell-like segmentation; HER2 held out; no real-cell ground truth'}
    elif a.apply:result=run(a)
    else:result=verify(a)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
