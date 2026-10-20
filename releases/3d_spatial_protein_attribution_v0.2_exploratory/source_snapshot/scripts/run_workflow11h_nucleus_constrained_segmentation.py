#!/usr/bin/env python3
"""Workflow 11H: full-nuclear-fragment-constrained resegmentation of WF11G ROI.

An independent, non-overwriting v0.2 experiment. Uses the exact WF11G panCK
ROI domain, Yb174-only contour cost and Ir193-associated author pseudo-nuclear
fragments. Unlike WF11G v0.1 (one pixel per fragment), *every retained nuclear
fragment pixel* is a hard seed marker in BOTH the Voronoi and Cadherin variants.
HER2 is never used to construct candidate boundaries (held-out readout only).
This corrects containment as a design invariant, NOT proof of true cell borders.
"""
from __future__ import annotations
import argparse
import copy
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import run_workflow11g_cadherin_segmentation as prior

VERSION = '0.2.0-fragment-constrained-exploratory'


def protected_markers(nuclear, domain, min_overlap=4):
    """Expand each prior point marker to its entire within-domain 4CC fragment."""
    point_markers, rows = prior.component_seed_map(nuclear, domain, min_overlap)
    full = np.zeros_like(point_markers, dtype=np.int32)
    for r in rows:
        eligible = (domain == r['domain_id']) & (nuclear == r['source_pseudo_id'])
        comp, _ = ndi.label(eligible, structure=ndi.generate_binary_structure(2,1))
        marker_comp = int(comp[r['seed_y_local'], r['seed_x_local']])
        if not marker_comp:
            raise RuntimeError('Prior nuclear point does not belong to source fragment')
        frag = comp == marker_comp
        if int(frag.sum()) != r['nucleus_fragment_px']:
            raise RuntimeError('Full fragment size mismatch')
        if np.any(full[frag] != 0):
            raise RuntimeError('Overlapping protected nuclear fragments')
        full[frag] = r['seed_id']
    if not np.array_equal(np.unique(full[full>0]), np.unique(point_markers[point_markers>0])):
        raise RuntimeError('Incomplete fragment propagation')
    return full, rows


def segment_slice(panck, nuclei, cadherin, quantile=.80, sigma=1.,
                  min_domain_px=40, min_seed_overlap_px=4, cadherin_weight=.75):
    if not (panck.shape == nuclei.shape == cadherin.shape and panck.ndim == 2):
        raise ValueError('Input geometry must be identical 2D ROI slices')
    if not np.isfinite(cadherin).all() or np.any(cadherin < 0):
        raise ValueError('Yb174 must be finite and non-negative')
    domain, q = prior.epithelial_domain(panck, quantile=quantile, sigma=sigma,
                                       min_component_px=min_domain_px)
    markers, rows = protected_markers(nuclei, domain, min_seed_overlap_px)
    baseline = np.zeros_like(markers)
    assisted = np.zeros_like(markers)
    cad_smooth = ndi.gaussian_filter(np.log1p(cadherin.astype(float)), sigma=1)
    domains = []
    for d in range(1, int(domain.max()) + 1):
        component = domain == d
        dseeds = markers * component
        ids = np.unique(dseeds[dseeds > 0])
        domains.append({'domain_id': d, 'domain_px': int(component.sum()), 'n_seed_fragments': len(ids)})
        if not len(ids):
            continue
        sl = ndi.find_objects(component.astype(np.int8))[0]
        part = component[sl]
        seed = dseeds[sl]
        dist, nearest = ndi.distance_transform_edt(seed == 0, return_indices=True)
        near = seed[tuple(nearest)]
        baseline[sl][part] = near[part]
        c = prior.normalize_in_mask(cad_smooth[sl], part)
        dist = prior.normalize_in_mask(dist, part)
        elevation = cadherin_weight*c + (1-cadherin_weight)*dist
        lab = watershed(elevation, markers=seed, mask=part, connectivity=1).astype(np.int32)
        assisted[sl][part] = lab[part]
    if not np.array_equal(baseline > 0, assisted > 0):
        raise RuntimeError('Tissue coverage differs by method')
    for name, lab in [('baseline', baseline), ('cadherin', assisted)]:
        if not np.array_equal(lab[markers > 0], markers[markers > 0]):
            raise RuntimeError(f'{name} has overwritten a retained nuclear fragment')
        if not np.array_equal(np.unique(lab[lab>0]), np.unique(markers[markers>0])):
            raise RuntimeError(f'{name} lost a source fragment')
        counts = np.bincount(lab.ravel(), minlength=int(markers.max())+1)
        for row in rows:
            if counts[row['seed_id']] < row['nucleus_fragment_px']:
                raise RuntimeError(f'{name} segment is smaller than the full nuclear fragment')
    return domain, markers, rows, baseline, assisted, q, domains


def validate_parent(a):
    parent_args = copy.copy(a)
    parent_args.out = a.reference_run
    return prior.verify(parent_args)


def run(a):
    if a.out.exists():
        raise FileExistsError('Refusing to overwrite output: '+str(a.out))
    if a.out.resolve() == a.reference_run.resolve():
        raise ValueError('New and parent output must differ')
    validate_parent(a)
    summary,paths,_,_,input_hashes,members = prior.preflight(a)
    mask=tifffile.imread(a.mask)
    if tuple(mask.shape) != tuple(summary['mask_shape_assumed_zyx']):
        raise ValueError('Mask geometry changed')
    x0,x1,y0,y1 = a.roi
    shape=(len(mask),y1-y0,x1-x0)
    lab_base=np.zeros(shape, dtype=np.uint32)
    lab_cad=np.zeros(shape, dtype=np.uint32)
    all_seeds=[];all_metrics=[];qcs=[];panels=[]
    with zipfile.ZipFile(a.cadherin_zip) as archive:
        for z in range(len(mask)):
            raw={m:tifffile.imread(paths[m][z]) for m in ('panCK','Ir193','HER2')}
            cad=tifffile.imread(io.BytesIO(archive.read(members[z])))
            if any(im.shape != mask.shape[1:] for im in [*raw.values(),cad]):
                raise ValueError('Per-Z image-to-mask shape mismatch')
            crop={m:im[y0:y1,x0:x1] for m,im in raw.items()}
            cad=cad[y0:y1,x0:x1]
            domain,markers,rows,base,assisted,q,domains=segment_slice(
                crop['panCK'],mask[z,y0:y1,x0:x1],cad,
                quantile=a.panck_quantile,sigma=a.sigma,min_domain_px=a.min_domain_px,
                min_seed_overlap_px=a.min_seed_overlap_px,cadherin_weight=a.cadherin_weight)
            lab_base[z]=base;lab_cad[z]=assisted
            for s in rows:
                s.update(z=z,seed_y_px=s.pop('seed_y_local')+y0,seed_x_px=s.pop('seed_x_local')+x0)
            all_seeds.extend(rows)
            for name, lab in [('nuclear_voronoi',base),('cadherin_assisted',assisted)]:
                mets=prior.region_metrics(lab,crop['HER2'],cad,rows,name,
                                           min_interface_px=a.min_interface_px,min_core_px=a.min_core_px)
                for m in mets:m['z']=z
                all_metrics.extend(mets)
            qcs.append(dict(z=z,n_seed_fragments=len(rows),n_domains_seeded=sum(i['n_seed_fragments']>0 for i in domains),
                            n_nuclear_fragment_px=int((markers>0).sum()),n_candidate_px=int((assisted>0).sum()),
                            n_voronoi_px=int((base>0).sum()),panCK_log_threshold=q))
            if z in prior.Z_SHOW:
                # Use seed CENTRES for display, not every fragment pixel.
                centers=np.zeros_like(markers)
                for r in rows:centers[r['seed_y_px']-y0,r['seed_x_px']-x0]=r['seed_id']
                panels.append((z,cad,crop['panCK'],crop['Ir193'],crop['HER2'],centers,base,assisted))
            print(f'PROCESSED z={z:02d}/15 | protected nuclear fragments={len(rows)}',flush=True)
    seed_table=pd.DataFrame(all_seeds)
    old_seed=pd.read_csv(a.reference_run/'section_nuclear_seed_index.csv')
    keys=['z','seed_id','domain_id','source_pseudo_id','nucleus_fragment_px']
    if not seed_table[keys].equals(old_seed[keys]):
        raise RuntimeError('Seed ID/source/domain identity differs from WF11G v0.1; comparison invalid')
    metrics=pd.DataFrame(all_metrics)
    matched=prior.summary_pairs(all_metrics)
    valid=matched.dropna(subset=['HER2_delta_change'])
    old_metrics=pd.read_csv(a.reference_run/'candidate_boundary_metrics.csv')
    geo=metrics[keys[:2]+['method','segment_area_px','nucleus_fragment_px','n_core_px','testable_interface']].merge(
        old_metrics[['z','seed_id','method','segment_area_px','n_core_px','testable_interface']],
        on=['z','seed_id','method'],validate='one_to_one',suffixes=('_constrained','_original'))
    geo['area_ratio_to_original']=geo.segment_area_px_constrained/geo.segment_area_px_original.clip(lower=1)
    geo['original_area_less_than_nucleus']=geo.segment_area_px_original<geo.nucleus_fragment_px
    geo['new_area_less_than_nucleus']=geo.segment_area_px_constrained<geo.nucleus_fragment_px
    if geo.new_area_less_than_nucleus.any():
        raise RuntimeError('The full-fragment constraint was violated')
    if not all(q['n_voronoi_px']==q['n_candidate_px'] for q in qcs):
        raise RuntimeError('Pixel conservation violated')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.wf11h_stage_',dir=a.out.parent))
    try:
        tifffile.imwrite(stage/'candidate_cadherin_labels_zyx.tif',lab_cad,metadata={'axes':'ZYX'})
        tifffile.imwrite(stage/'baseline_voronoi_labels_zyx.tif',lab_base,metadata={'axes':'ZYX'})
        seed_table.to_csv(stage/'section_nuclear_seed_index.csv',index=False)
        metrics.to_csv(stage/'candidate_boundary_metrics.csv',index=False)
        matched.to_csv(stage/'heldout_HER2_matched_comparison.csv',index=False)
        geo.to_csv(stage/'parent_vs_constrained_geometry.csv',index=False)
        pd.DataFrame(qcs).to_csv(stage/'section_segmentation_qc.csv',index=False)
        prior.figure_panels(stage/'cadherin_vs_voronoi_nucleus_constrained.png',panels)
        note=dict(workflow='11H-full-fragment-protected candidate segmentation',version=VERSION,
                  roi_xy=list(a.roi),n_z=len(mask),n_section_seed_fragments=len(all_seeds),
                  n_valid_HER2_paired_rows=len(valid),
                  n_old_nucleus_containment_violations=int(geo.original_area_less_than_nucleus.sum()),
                  n_new_nucleus_containment_violations=int(geo.new_area_less_than_nucleus.sum()),
                  n_new_seed_collapse_to_le_quarter_original=int((geo.area_ratio_to_original<=.25).sum()),
                  input_sha256=input_hashes,
                  parent_reference_run=str(a.reference_run.resolve()),
                  parent_output_sha256=prior.sha(a.reference_run/'output_sha256.json'),
                  script_sha256=prior.sha(Path(__file__)),
                  params={'panck_quantile':a.panck_quantile,'sigma':a.sigma,
                          'min_domain_px':a.min_domain_px,'min_seed_overlap_px':a.min_seed_overlap_px,
                          'cadherin_weight':a.cadherin_weight,'min_interface_px':a.min_interface_px,
                          'min_core_px':a.min_core_px},
                  marker_role={'Ir193':'full pseudo-nuclear fragments protected','panCK':'domain mask',
                               'Yb174':'boundary cost','HER2':'held-out measurement only'},
                  limitations=['By construction each output segment contains its selected pseudo-nuclear fragment; this is not segmentation accuracy',
                               'Ir193 pseudo-nuclei may be inaccurate, fragmented or absent; unselected nuclear objects are not protected',
                               'Comparison retains a HER2-selected ROI, so HER2 readout is exploratory not prospective',
                               'Yb174 Cadherin and HER2 can colocalize without implying correct cell borders',
                               '2D slices only; Z index is not physical depth, no 3D cell identity',
                               'Manual independent cell junction ground truth remains required'])
        (stage/'workflow11h_summary.json').write_text(json.dumps(note,indent=2,ensure_ascii=False)+'\n')
        hashes={p.name:prior.sha(p) for p in stage.iterdir() if p.is_file()}
        (stage/'output_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
        prior.save_atomic(stage,a.out)
    finally:
        if stage.exists():shutil.rmtree(stage)
    return {'status':'CREATED','output':str(a.out),
            'seed_fragments':len(all_seeds),'old_containment_violations':int(geo.original_area_less_than_nucleus.sum()),
            'new_containment_violations':int(geo.new_area_less_than_nucleus.sum()),
            'valid_heldout_pairs':len(valid),'files_hashed':len(hashes)}


def verify(a):
    validate_parent(a)
    _,_,_,_,input_hashes,_=prior.preflight(a)
    if not a.out.is_dir():raise FileNotFoundError(a.out)
    record=json.loads((a.out/'workflow11h_summary.json').read_text())
    if input_hashes != record['input_sha256']:
        raise RuntimeError('Input provenance has changed')
    if prior.sha(a.reference_run/'output_sha256.json') != record['parent_output_sha256']:
        raise RuntimeError('Parent result manifest has changed')
    if prior.sha(Path(__file__)) != record['script_sha256']:
        raise RuntimeError('11H algorithm script has changed')
    if list(a.roi)!=record['roi_xy'] or {'panck_quantile':a.panck_quantile,
            'sigma':a.sigma,'min_domain_px':a.min_domain_px,
            'min_seed_overlap_px':a.min_seed_overlap_px,'cadherin_weight':a.cadherin_weight,
            'min_interface_px':a.min_interface_px,'min_core_px':a.min_core_px}!=record['params']:
        raise RuntimeError('Run parameters have changed')
    hashes=json.loads((a.out/'output_sha256.json').read_text())
    if {p.name for p in a.out.iterdir() if p.is_file()}!=set(hashes)|{'output_sha256.json'}:
        raise RuntimeError('11H output file list changed')
    for name, sha256 in hashes.items():
        if prior.sha(a.out/name)!=sha256:raise RuntimeError('Modified output file: '+name)
    geo=pd.read_csv(a.out/'parent_vs_constrained_geometry.csv')
    if geo.new_area_less_than_nucleus.any():
        raise RuntimeError('New segmentation violates nuclear size lower bound')
    if len(geo)!=record['n_section_seed_fragments']*2:
        raise RuntimeError('Unexpected count of seeded fragments')
    return {'status':'PASS','verified_files':len(hashes),
            'verified_inputs':len(input_hashes),'n_seed_fragments':record['n_section_seed_fragments'],
            'nucleus_containment_violations':int(geo.new_area_less_than_nucleus.sum())}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--apply',action='store_true');mode.add_argument('--verify',action='store_true')
    p.add_argument('--mask',type=Path,default=Path('data/inspect/LVILymph/full_mask_final_segmentation_hwatershed_bg500_90%.tif'))
    p.add_argument('--manifest',type=Path,default=Path('configs/protein_marker_manifest_registered_5channels.json'))
    p.add_argument('--workflow09',type=Path,default=Path('runs/3d_protein_marker_integration_v1'))
    p.add_argument('--cadherin-zip',type=Path,default=Path('data/inspect/LVILymph/LVILymphBreastCancer_SIMILARITY10_In115.zip'))
    p.add_argument('--roi',type=int,nargs=4,default=prior.DEFAULT_ROI,metavar=('X0','X1','Y0','Y1'))
    p.add_argument('--panck-quantile',type=float,default=.8)
    p.add_argument('--sigma',type=float,default=1.0)
    p.add_argument('--min-domain-px',type=int,default=40)
    p.add_argument('--min-seed-overlap-px',type=int,default=4)
    p.add_argument('--cadherin-weight',type=float,default=.75)
    p.add_argument('--min-interface-px',type=int,default=3)
    p.add_argument('--min-core-px',type=int,default=5)
    p.add_argument('--reference-run',type=Path,default=Path('runs/3d_cadherin_assisted_celllike_segmentation_v0_1'))
    p.add_argument('--out',type=Path,default=Path('runs/3d_nucleus_constrained_celllike_segmentation_v0_2'))
    a=p.parse_args(argv)
    if a.check:
        parent=validate_parent(a)
        _,_,_,_,hashes,_=prior.preflight(a)
        result={'status':'PREFLIGHT_PASS','parent_status':parent['status'],
                'roi':list(a.roi),'verified_input_hashes':len(hashes),'new_output':str(a.out)}
    elif a.apply:result=run(a)
    else:result=verify(a)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
