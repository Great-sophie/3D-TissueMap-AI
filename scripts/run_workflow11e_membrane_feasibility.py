#!/usr/bin/env python3
"""Workflow 11E v0.1: HER2 held-out test on panCK-derived epithelial-domain contours.

NOT single-cell segmentation, NOT membrane attribution. panCK is cytoskeletal;
contours are tissue-domain proxies; Z spacing and exact registration unverified.

Modes: --check | --apply | --verify. Safe new output directory (no overwrite).
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
from scipy.ndimage import (gaussian_filter, binary_opening, binary_fill_holes,
                           distance_transform_edt, label, sum_labels)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_spatial_signal_attribution import DEFAULT as WF11_DEFAULT, file_manifest, sha

NAME = 'Workflow11E-panCK-domain-HER2-heldout-v0.1'
Z_DISPLAY = (0, 5, 10, 15)
OFFSETS = ((0, 0), (0, 8), (0, -8), (8, 0), (-8, 0),
           (0, 16), (0, -16), (16, 0), (-16, 0))
REGION_ROI = (480, 720, 480, 720)  # x0,x1,y0,y1


def check_args(a):
    x0, x1, y0, y1 = a.roi
    if not (0 <= x0 < x1 <= 1215 and 0 <= y0 < y1 <= 1205):
        raise ValueError('Invalid fixed XY ROI')
    if a.sigma <= 0 or not .5 < a.panck_quantile < .99:
        raise ValueError('sigma must be positive, panCK quantile between 0.5 and 0.99')
    if a.min_component_px < 10 or a.min_band_px < 1:
        raise ValueError('min size must be positive')


def epithelial_domain(panck: np.ndarray, quantile=.80, sigma=1., min_component_px=40):
    """Build high-panCK *tissue domain*, never individual cell membrane.

    Quantile is computed on this XY ROI for each section, independent of HER2.
    """
    a = np.asarray(panck, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Expected nonnegative finite 2D panCK image')
    f = gaussian_filter(np.log1p(a), sigma=sigma)
    q = float(np.quantile(f, quantile))
    # When all the pixels are identical there is no support for contours.
    if np.ptp(f) < 1e-12:
        empty = np.zeros_like(a, dtype=np.int32)
        return empty, q
    binary = f > q
    binary = binary_opening(binary, structure=np.ones((3, 3), bool))
    binary = binary_fill_holes(binary)
    labs, n = label(binary)
    if n:
        vals = np.bincount(labs.ravel(), minlength=n+1)
        keep = vals >= min_component_px
        keep[0] = False
        labs = label(keep[labs])[0]
    return labs.astype(np.int32), q


def component_bands(component: np.ndarray):
    """Inside-only epithelial-domain contour (<=2 px) and core (>=5 px)."""
    p = np.asarray(component, dtype=bool)
    if p.ndim != 2:
        raise ValueError('2D boolean component needed')
    dist = distance_transform_edt(p)
    return p & (dist <= 2.0), p & (dist >= 5.0)


def shifted_view_from_full(full: np.ndarray, roi: tuple, dy: int, dx: int) -> np.ndarray:
    """Read shifted source coordinates with NaN padding (never circular wrap)."""
    x0, x1, y0, y1 = roi
    h, w = full.shape
    out = np.full((y1-y0, x1-x0), np.nan, np.float64)
    yy, xx = np.indices(out.shape)
    source_y = yy + y0 - dy
    source_x = xx + x0 - dx
    good = (source_y >= 0) & (source_x >= 0) & (source_y < h) & (source_x < w)
    out[good] = full[source_y[good], source_x[good]]
    return out


def finite_stat(a, region):
    values = a[region]
    values = values[np.isfinite(values)]
    return (len(values), float(np.mean(values)) if len(values) else np.nan,
            float(np.median(values)) if len(values) else np.nan)


def compute_slice(panck_full, her2_full, ir193_full, labels_full,
                  z, roi, quantile, sigma, min_component_px, min_band_px):
    x0,x1,y0,y1=roi
    ck=panck_full[y0:y1,x0:x1]
    domains,threshold=epithelial_domain(ck,quantile,sigma,min_component_px)
    mask_roi=labels_full[y0:y1,x0:x1]
    rows=[]
    by_shift=[]
    domain_count=int(domains.max())
    shifted={tuple(o):shifted_view_from_full(her2_full,roi,*o) for o in OFFSETS}
    for ident in range(1,domain_count+1):
        component=domains==ident
        edge,core=component_bands(component)
        if edge.sum()<min_band_px or core.sum()<min_band_px:
            continue
        n_ids=int(np.unique(mask_roi[component & (mask_roi > 0)]).size)
        pedge=finite_stat(ck,edge)
        pcore=finite_stat(ck,core)
        if pedge[0]<min_band_px or pcore[0]<min_band_px:
            continue
        coords=np.where(component)
        for dy,dx in OFFSETS:
            h=shifted[(dy,dx)]
            e=finite_stat(h,edge)
            c=finite_stat(h,core)
            if e[0]<min_band_px or c[0]<min_band_px:
                continue
            by_shift.append(dict(z=z,domain_id=ident,dy=dy,dx=dx,
                                 n_edge=e[0],n_core=c[0],
                                 HER2_edge_mean=e[1],HER2_core_mean=c[1],
                                 HER2_edge_minus_core=e[1]-c[1]))
            if dy==0 and dx==0:
                rows.append(dict(z=z,domain_id=ident,area_px=int(component.sum()),
                                 n_nuclear_pseudo_ids=n_ids,
                                 centroid_x_px=float(np.mean(coords[1])+x0),
                                 centroid_y_px=float(np.mean(coords[0])+y0),
                                 n_edge=e[0],n_core=c[0],
                                 HER2_edge_mean=e[1],HER2_core_mean=c[1],
                                 HER2_edge_minus_core=e[1]-c[1],
                                 panCK_edge_mean=pedge[1],panCK_core_mean=pcore[1]))
    qc=dict(z=z,threshold_log1p_panCK=threshold,n_panCK_domains=domain_count,
            n_testable_domains=len(rows),n_panCK_domain_pixels=int((domains>0).sum()),
            roi_pixels=int(domains.size),n_nuclear_pseudo_ids_in_domain=int(np.unique(mask_roi[(domains>0)&(mask_roi>0)]).size))
    return domains, rows, by_shift, qc


def compare_shifts(df):
    rows=[]
    if df.empty:
        return pd.DataFrame(columns=['dy','dx','n_valid_domains','median_component_HER2_edge_minus_core','fraction_positive'])
    for (dy,dx),grp in df.groupby(['dy','dx'],sort=True):
        v=grp.HER2_edge_minus_core.to_numpy(float)
        rows.append(dict(dy=int(dy),dx=int(dx),n_valid_domains=len(v),
                         median_component_HER2_edge_minus_core=float(np.median(v)),
                         fraction_positive=float(np.mean(v>0))))
    return pd.DataFrame(rows)


def write_montage(dest, planes, roi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_erosion
    if not planes:
        return
    fig,axes=plt.subplots(len(planes),4,figsize=(16,4*len(planes)),squeeze=False)
    for i,(z,ck,her2,ir,dom) in enumerate(planes):
        data=[(np.log1p(ck),'Greens','panCK (domain source)'),
              (np.log1p(her2),'magma','HER2 (held out)'),
              (np.log1p(ir),'Blues','Ir193 nuclei'),
              (None,None,'HER2 + panCK-domain edge')]
        edge=(dom>0)&~binary_erosion(dom>0)
        for j,(img,cmap,title) in enumerate(data):
            ax=axes[i,j]
            if img is not None:
                vmax=max(float(np.quantile(img,.995)),1e-6)
                ax.imshow(img,cmap=cmap,vmin=0,vmax=vmax)
            else:
                h=np.log1p(her2); c=np.log1p(ck)
                hv=max(float(np.quantile(h,.995)),1e-6)
                cv=max(float(np.quantile(c,.995)),1e-6)
                rgb=np.zeros((*h.shape,3),float)
                rgb[...,0]=np.clip(h/hv,0,1)
                rgb[...,1]=np.clip(c/cv,0,1)
                ax.imshow(rgb)
            ax.contour(edge.astype(float),[.5],colors='cyan',linewidths=.5)
            ax.set_title(f'z={z}: {title}')
            ax.axis('off')
    fig.suptitle(f'PANCK-defined epithelial-domain contour; HER2 NOT used to define contour; ROI {roi}; Z=section index',fontsize=13)
    fig.tight_layout()
    fig.savefig(dest,dpi=130,bbox_inches='tight')
    plt.close(fig)


def trusted_file_manifest(a,hashes=True):
    check_args(a)
    summary,paths,spatial,ref,digests=file_manifest(a,hashes=hashes)
    if set(('HER2','panCK','Ir193'))-set(paths):
        raise ValueError('Missing required registered channels')
    return summary,paths,spatial,ref,digests


def run(a):
    if a.out.exists():
        raise FileExistsError('Refusing to overwrite existing viewer/exploratory results: '+str(a.out))
    summary, paths, spatial, ref, hashes=trusted_file_manifest(a)
    mask=tifffile.imread(a.mask)
    if mask.ndim!=3 or tuple(mask.shape)!=tuple(summary['mask_shape_assumed_zyx']):
        raise ValueError('Mismatch ZYX mask')
    rows=[]; shifts=[]; qcs=[]; montage=[]
    x0,x1,y0,y1=a.roi
    for z in range(mask.shape[0]):
        imgs={name:tifffile.imread(paths[name][z]) for name in ('HER2','panCK','Ir193')}
        if any(im.shape!=mask.shape[1:] for im in imgs.values()):
            raise ValueError(f'Channel dimensions differ at z {z}')
        dom,r,sh,qc=compute_slice(imgs['panCK'],imgs['HER2'],imgs['Ir193'],mask[z],z,
                                  a.roi,a.panck_quantile,a.sigma,a.min_component_px,a.min_band_px)
        rows.extend(r);shifts.extend(sh);qcs.append(qc)
        if z in Z_DISPLAY:
            montage.append((z,imgs['panCK'][y0:y1,x0:x1],imgs['HER2'][y0:y1,x0:x1],
                           imgs['Ir193'][y0:y1,x0:x1],dom))
        print(f'z={z:02d}: domains={qc["n_panCK_domains"]} eligible={len(r)}',flush=True)
    per=pd.DataFrame(rows)
    sh=pd.DataFrame(shifts)
    if per.empty or sh.empty:
        raise RuntimeError('No eligible domain: inspect threshold/min_component_px without HER2 tuning')
    compare=compare_shifts(sh)
    shiftbase=sh.loc[(sh.dy==0)&(sh.dx==0)]
    # Validate exact base matches from both independently assembled tables.
    merge=shiftbase.merge(per,on=['z','domain_id'],validate='one_to_one',suffixes=('_shift','_orig'))
    if len(merge)!=len(per) or not np.allclose(merge.HER2_edge_minus_core_shift,merge.HER2_edge_minus_core_orig,atol=1e-10):
        raise RuntimeError('Zero-shift validation failed')
    a.out.parent.mkdir(exist_ok=True,parents=True)
    stage=Path(tempfile.mkdtemp(prefix='.wf11e_stage_',dir=a.out.parent))
    try:
        per.to_csv(stage/'epithelial_domain_HER2_contour.csv',index=False)
        sh.to_csv(stage/'HER2_domain_shift_per_section.csv',index=False)
        compare.to_csv(stage/'HER2_shift_summary.csv',index=False)
        pd.DataFrame(qcs).to_csv(stage/'domain_detection_qc.csv',index=False)
        write_montage(stage/'panCK_HER2_domain_contour_overlay.png',montage,a.roi)
        rec=dict(workflow=NAME,version='0.1.0-exploratory',
                 n_z=len(mask),roi_xy=list(a.roi),panCK_threshold_quantile=a.panck_quantile,
                 per_z_panCK_log1p_gaussian_sigma=a.sigma,
                 panCK_min_component_pixels=a.min_component_px,min_edge_or_core_pixels=a.min_band_px,
                 panCK_mask_binarization='per-ROI per-section smooth log1p panCK top-quantile; opening; fill holes; min-component-area',
                 internal_panCK_edge='panCK-domain inside <=2 XY px from exterior of binary domain',
                 panCK_domain_core='panCK-domain inside >=5 XY px from exterior',
                 HER2_usage='held-out measurement only; HER2 never used to form contours or tune thresholds',
                 shift_offsets_dy_dx=list(map(list,OFFSETS)),
                 number_components_tested=int(len(per)),
                 number_control_records=int(len(sh)),
                 input_sha256=hashes,
                 limitations=[
                   'panCK is cytoskeletal and NOT a cell-membrane marker; detected edges are epithelial TISSUE-domain contours, NOT single-cell membranes',
                   'Threshold fixed a priori but this ROI was selected based on HER2 hotspot on same specimen',
                   'No independent marker-to-mask registration, cell identity or true Z spacing verified',
                   'Shift controls are technical sensitivity tests, not independent biological negative-control samples or p-values',
                   'HER2 could be brighter at panCK domains because of tissue-context correlation, not true membrane localization',
                   'Image intensity is uncalibrated; per-section quantile preprocessing changes selection area',
                   'Only one breast-cancer specimen; no generalizability claim'
                 ])
        (stage/'membrane_feasibility_summary.json').write_text(json.dumps(rec,indent=2,ensure_ascii=False)+'\n')
        output_digests={p.name:sha(p) for p in sorted(stage.iterdir()) if p.is_file()}
        (stage/'output_sha256.json').write_text(json.dumps(output_digests,indent=2)+'\n')
        stage.rename(a.out)
    finally:
        if stage.exists():shutil.rmtree(stage)
    return {'status':'CREATED','n_z':len(mask),'n_testable_domain_sections':len(per),
            'verified_output_files':len(output_digests),'output':str(a.out)}


def verify(a):
    summary,paths,spatial,ref,hashes=trusted_file_manifest(a)
    record=json.loads((a.out/'membrane_feasibility_summary.json').read_text())
    if record['input_sha256']!=hashes:
        raise ValueError('Input provenance changed compared with saved run')
    listed=json.loads((a.out/'output_sha256.json').read_text())
    if set(p.name for p in a.out.iterdir() if p.is_file())!=set(listed)|{'output_sha256.json'}:
        raise ValueError('Output file set differs from saved record')
    for name,digest in listed.items():
        if sha(a.out/name)!=digest:
            raise ValueError(f'Output hash does not match: {name}')
    # Verify enough structure to reject accidental corruption of internal records.
    d=pd.read_csv(a.out/'epithelial_domain_HER2_contour.csv')
    s=pd.read_csv(a.out/'HER2_domain_shift_per_section.csv')
    z=pd.read_csv(a.out/'domain_detection_qc.csv')
    if len(d)!=record['number_components_tested'] or len(s)!=record['number_control_records'] or len(z)!=record['n_z']:
        raise ValueError('Output row count mismatch')
    return {'status':'PASS','verified_files':len(listed),'n_z':len(z),
            'n_testable_domain_sections':len(d),'n_shift_rows':len(s)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--apply',action='store_true')
    modes.add_argument('--verify',action='store_true')
    for name,default in WF11_DEFAULT.items():
        if name in ('mask','manifest','workflow09'):
            p.add_argument('--'+name,type=Path,default=default)
    p.add_argument('--roi',type=int,nargs=4,default=REGION_ROI,metavar=('X0','X1','Y0','Y1'))
    p.add_argument('--out',type=Path,default=Path('runs/3d_membrane_localization_feasibility_v0_1'))
    p.add_argument('--panck-quantile',type=float,default=.80)
    p.add_argument('--sigma',type=float,default=1.0)
    p.add_argument('--min-component-px',type=int,default=40)
    p.add_argument('--min-band-px',type=int,default=12)
    a=p.parse_args(argv)
    if a.check:
        summary,paths,df,ref,hashes=trusted_file_manifest(a)
        result={'status':'PREFLIGHT_PASS','marker_channels':list(sorted(paths)),
                'n_source_files_verified':len(hashes), 'n_instances':len(df),'roi_xy':list(a.roi),
                'note':'panCK-dependent epithelial domain contours; NOT single-cell membrane segmentation'}
    elif a.apply:
        result=run(a)
    else:
        result=verify(a)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
