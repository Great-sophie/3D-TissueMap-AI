"""Pure functions for exploratory 3D-IMC Z-stack alignment/identity review.

No registration transforms applied; no inference, model training, or physical-unit claims.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def overlapping_views(a: np.ndarray, b: np.ndarray, dy: int, dx: int):
    """Return overlapping windows of fixed A and shifted B (B[y-dy,x-dx])."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError('Expected 2D equal-shaped arrays')
    h,w=a.shape
    y0,y1=max(0,dy),min(h,h+dy)
    x0,x1=max(0,dx),min(w,w+dx)
    if y1<=y0 or x1<=x0:
        raise ValueError('No overlap')
    return a[y0:y1,x0:x1],b[y0-dy:y1-dy,x0-dx:x1-dx]


def normalized_corr(a: np.ndarray,b: np.ndarray,dy=0,dx=0) -> float:
    x,y=overlapping_views(np.asarray(a,float),np.asarray(b,float),int(dy),int(dx))
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<64:return float('nan')
    u=x[ok];v=y[ok];u=u-u.mean();v=v-v.mean()
    norm=float(np.linalg.norm(u)*np.linalg.norm(v))
    return float((u@v)/norm) if norm>1e-12 else float('nan')


def alignment_screen(img0: np.ndarray,img1: np.ndarray,max_shift: int=8) -> dict:
    """Explore a small rigid XY offset using smoothed Ir193 signal, not ground truth.

    Full bounded brute search intentionally favors interpretable diagnostics over false
    precision from interpolation. Positive (dy,dx) translates second plane to first.
    """
    if img0.shape!=img1.shape or img0.ndim!=2 or max_shift<0:
        raise ValueError('Expected paired 2D images and nonnegative shift window')
    a=ndi.gaussian_filter(np.nan_to_num(np.asarray(img0,np.float32)),1.1)
    b=ndi.gaussian_filter(np.nan_to_num(np.asarray(img1,np.float32)),1.1)
    baseline=normalized_corr(a,b,0,0)
    best_corr=-2.;best=(0,0)
    for dy in range(-max_shift,max_shift+1):
        for dx in range(-max_shift,max_shift+1):
            c=normalized_corr(a,b,dy,dx)
            if np.isfinite(c) and c>best_corr:
                best_corr=c;best=(dy,dx)
    if best_corr==-2.:best_corr=float('nan')
    return {'shift_dy_to_align_next':int(best[0]),'shift_dx_to_align_next':int(best[1]),
            'zero_shift_corr':float(baseline),'best_shift_corr':float(best_corr),
            'corr_gain':float(best_corr-baseline),
            'peak_at_search_boundary': bool(max_shift>0 and max(abs(best[0]),abs(best[1]))==max_shift),
            'diagnostic_only':'Shift is an intensity-based plausibility indicator, NOT a registration correction'}


def adjacent_mask_screen(m0:np.ndarray,m1:np.ndarray) -> dict:
    if m0.shape!=m1.shape or m0.ndim!=2:raise ValueError('Expected paired Z label planes')
    a=m0>0;b=m1>0;both=a&b
    inter=int(both.sum());den=int(a.sum()+b.sum())
    return {'mask_foreground_dice':float(2*inter/den) if den else None,
            'both_foreground_faces':inter,
            'same_id_face_fraction_both_foreground':float((m0[both]==m1[both]).mean()) if inter else None,
            'n_shared_ids':int(len((set(np.unique(m0))-{0}) & (set(np.unique(m1))-{0}))) }


def id_slice_statistics(reference:np.ndarray) -> pd.DataFrame:
    """Pseudo-ID Z occupancy plus 3D connectivity within tight bounding boxes."""
    if reference.ndim!=3 or reference.dtype.kind not in 'iu':raise ValueError('3D integer instance IDs required')
    entries={}
    for z in range(reference.shape[0]):
        ids,counts=np.unique(reference[z][reference[z]>0],return_counts=True)
        for key,n in zip(ids,counts):
            i=int(key)
            if i not in entries:entries[i]=[]
            entries[i].append((int(z),int(n)))
    boxes=ndi.find_objects(reference)
    six=ndi.generate_binary_structure(3,1)
    twenty_six=ndi.generate_binary_structure(3,3)
    rows=[]
    for i,occ in entries.items():
        zs=[x for x,_ in occ]
        gaps=sum((zs[j]-zs[j-1]>1) for j in range(1,len(zs)))
        # This tests 3D voxel connectivity of AUTHOR IDs, not real-cell validation.
        slices=boxes[i-1];local=reference[slices]==i
        n6=int(ndi.label(local,structure=six)[1]);n26=int(ndi.label(local,structure=twenty_six)[1])
        rows.append({'reference_id':i,'first_z':min(zs),'last_z':max(zs),
                     'n_occupied_slices':len(zs),'n_gaps_in_occupied_slices':gaps,
                     'total_voxels':sum(v for _,v in occ),
                     'n_3d_6_connected_components':n6,
                     'n_3d_26_connected_components':n26,
                     'is_multi_slice':len(zs)>1})
    cols=['reference_id','first_z','last_z','n_occupied_slices','n_gaps_in_occupied_slices',
          'total_voxels','n_3d_6_connected_components','n_3d_26_connected_components','is_multi_slice']
    return pd.DataFrame(rows,columns=cols).sort_values('reference_id').reset_index(drop=True)


def stratified_samples(links:pd.DataFrame,n_per_category=8,seed=42) -> pd.DataFrame:
    expected={'reference_category','geometry_eligible','z','a','b','mean_pz','footprint_iou'}
    if not expected.issubset(links.columns):raise ValueError(f'Missing columns {sorted(expected-set(links.columns))}')
    subset=links.loc[links.geometry_eligible.astype(bool)].copy()
    subset['pz_gate_pass']=subset.mean_pz>=0.45
    rng=np.random.default_rng(seed)
    picks=[]
    for category in ['reliable_same','reliable_different','uncertain']:
        group=subset.loc[subset.reference_category==category]
        # Include both pass and fail gate cases, no cutoffs tuned from reference labels.
        for gate,n in [(True,n_per_category//2),(False,n_per_category-n_per_category//2)]:
            options=group.loc[group.pz_gate_pass==gate]
            count=min(len(options),n)
            if count:
                picks.extend(rng.choice(options.index.to_numpy(),size=count,replace=False).tolist())
        # Fill gaps for groups with missing gate categories.
        if len([j for j in picks if j in group.index])<min(n_per_category,len(group)):
            used=set(picks);unused=group.loc[~group.index.isin(used)]
            count=min(n_per_category-len([j for j in picks if j in group.index]),len(unused))
            if count:picks.extend(rng.choice(unused.index.to_numpy(),size=count,replace=False).tolist())
    out=subset.loc[picks].copy().reset_index(drop=True)
    out.insert(0,'review_id',[f'ZPAIR_{i+1:03d}' for i in range(len(out))])
    out['manual_identity_assessment']='UNREVIEWED'
    out['manual_registration_assessment']='UNREVIEWED'
    out['reviewer_notes']=''
    return out


def pair_crop_center(reference_crop, initial_mask, row):
    z=int(row.z);a=int(row.a);b=int(row.b)
    if z<0 or z+1>=initial_mask.shape[0]:raise ValueError('Invalid adjacent Z index')
    ma=initial_mask[z]==a;mb=initial_mask[z+1]==b
    overlap=ma&mb
    if overlap.any():ys,xs=np.where(overlap)
    elif ma.any() and mb.any():
        ay,ax=np.mean(np.argwhere(ma),axis=0);by,bx=np.mean(np.argwhere(mb),axis=0)
        return (int(round((ay+by)/2)),int(round((ax+bx)/2)))
    else:raise ValueError('Missing a or b component in its plane')
    return int(round(float(np.mean(ys)))),int(round(float(np.mean(xs))))


def crop_bounds(center,shape,halfwidth=40):
    cy,cx=center;h,w=shape
    y0=max(0,min(h-1,cy-halfwidth));y1=min(h,cy+halfwidth+1)
    x0=max(0,min(w-1,cx-halfwidth));x1=min(w,cx+halfwidth+1)
    return y0,y1,x0,x1


def image_mask_enrichment_screen(ir193:np.ndarray,labels:np.ndarray) -> dict:
    """Screen if bright Ir193 pixels enrich inside a cell mask. Not geometric truth."""
    if ir193.shape!=labels.shape or ir193.ndim!=2:raise ValueError('Expected matched 2D plane')
    fg=labels>0;signal=np.asarray(ir193,np.float32)
    bright_signal=signal[np.isfinite(signal) & (signal>0)]
    if bright_signal.size<25:
        return {'bright_ir193_capture_by_mask':None,'mask_foreground_fraction':float(fg.mean()),
                'bright_capture_enrichment_vs_mask_area':None,'warning':'Insufficient bright pixels'}
    threshold=np.percentile(bright_signal,92)
    bright=np.isfinite(signal) &(signal>=threshold)
    coverage=float(fg.mean());capture=float((fg&bright).sum()/max(int(bright.sum()),1))
    return {'bright_ir193_capture_by_mask':capture,
            'mask_foreground_fraction':coverage,
            'bright_capture_enrichment_vs_mask_area':float(capture/coverage) if coverage else None,
            'warning':'Proxy: author cell masks are not nuclear boundaries; cannot certify alignment'}
