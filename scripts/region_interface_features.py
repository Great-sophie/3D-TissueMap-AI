"""Reference-audited voxel-affinity region-interface table.

Independent of torch; only NumPy/Pandas are needed for this feature logic.
Labels are *author watershed pseudo-labels*. Never feed reference metadata
columns to a classifier; ``FEATURE_COLUMNS`` is the leakage-safe whitelist.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'interface_n_faces', 'interface_z_faces', 'interface_y_faces', 'interface_x_faces',
    'interface_z_fraction', 'interface_y_fraction', 'interface_x_fraction',
    'affinity_p_mean', 'affinity_p_std', 'affinity_p10', 'affinity_p50',
    'affinity_p90', 'affinity_logit_mean_weighted', 'affinity_negative_fraction_weighted',
    'affinity_p_mean_z', 'affinity_p_mean_y', 'affinity_p_mean_x',
    'ir193_face_absdiff_mean',
    'region_volume_min', 'region_volume_max', 'region_volume_ratio_min_max',
    'region_ir193_mean_absdiff', 'region_ir193_std_mean',
    'region_bbox_z_extent_min', 'region_bbox_z_extent_max',
    'region_bbox_y_extent_min', 'region_bbox_y_extent_max',
    'region_bbox_x_extent_min', 'region_bbox_x_extent_max',
]


def region_reference_diagnostics(sv: np.ndarray, reference: np.ndarray):
    """For each sv ID: majority reference ID and overlap coverage/purity.

    Background reference=0 is excluded from voting and contributes to low
    coverage. Identifiers never enter FEATURE_COLUMNS.
    """
    sv = np.asarray(sv)
    ref = np.asarray(reference)
    if sv.shape != ref.shape or sv.ndim != 3:
        raise ValueError('Expected aligned supervoxel and reference volumes [Z,Y,X]')
    n = int(sv.max())
    region_size = np.bincount(sv.ravel(), minlength=n+1).astype(np.int64)
    overlap_mask = (sv > 0) & (ref > 0)
    overlap_size = np.bincount(sv[overlap_mask], minlength=n+1).astype(np.int64)
    majority = np.zeros(n+1, np.int64)
    majority_size = np.zeros(n+1, np.int64)
    if overlap_mask.any():
        s = sv[overlap_mask].astype(np.int64)
        r = ref[overlap_mask].astype(np.int64)
        pairs, count = np.unique(np.stack((s,r),axis=1),axis=0,return_counts=True)
        # Deterministic largest overlap, then smallest reference ID as tie break.
        ranking=np.lexsort((pairs[:,1],-count,pairs[:,0]))
        ranked=pairs[ranking]
        _,first=np.unique(ranked[:,0],return_index=True)
        winners=ranking[first]
        chosen=pairs[winners,0].astype(np.intp)
        majority[chosen]=pairs[winners,1]
        majority_size[chosen]=count[winners]
    coverage = overlap_size / np.maximum(region_size, 1)
    purity = majority_size / np.maximum(overlap_size, 1)
    return dict(majority=majority, overlap_size=overlap_size,
                region_size=region_size, coverage=coverage, purity=purity)


def _region_intensity_and_extents(sv: np.ndarray, image: np.ndarray):
    n = int(sv.max())
    a = sv.ravel().astype(np.intp)
    v = image.ravel().astype(np.float64)
    count = np.bincount(a, minlength=n+1).astype(np.int64)
    summ = np.bincount(a, weights=v, minlength=n+1)
    sq = np.bincount(a, weights=v*v, minlength=n+1)
    mean = summ / np.maximum(count, 1)
    std = np.sqrt(np.maximum(0, sq/np.maximum(count,1)-mean**2))
    mins = np.full((3, n+1), np.iinfo(np.int32).max, dtype=np.int32)
    maxs = np.full((3, n+1), -1, dtype=np.int32)
    z, y, x = np.nonzero(sv)
    ids = sv[z,y,x].astype(np.intp)
    for axis, coord in enumerate((z,y,x)):
        np.minimum.at(mins[axis], ids, coord)
        np.maximum.at(maxs[axis], ids, coord)
    sizes = np.where(maxs >= 0, maxs - mins + 1, 0)
    return count, mean, std, sizes


def _quantiles_by_key(keys, values, unique, q):
    if not len(keys):
        return np.empty(0,dtype=np.float64)
    order = np.lexsort((values, keys))
    sorted_k = keys[order]
    sorted_v = values[order]
    distinct, starts, counts = np.unique(sorted_k, return_index=True, return_counts=True)
    if not np.array_equal(distinct, unique):
        raise AssertionError('Interface key sort differs')
    loc = starts + q*(counts-1)
    low = np.floor(loc).astype(np.int64)
    high = np.ceil(loc).astype(np.int64)
    return sorted_v[low] + (loc-low)*(sorted_v[high]-sorted_v[low])


def extract_region_interfaces(sv: np.ndarray, p_same: np.ndarray,
                              image_ir193: np.ndarray, reference: np.ndarray,
                              *, z_weight: float=0.25, min_coverage: float=0.70,
                              min_purity: float=0.90, min_overlap_voxels: int=3,
                              include_z_labels: bool=False,
                              score_threshold: float=0.2,
                              max_negative_fraction: float=0.35,
                              min_faces: int=2):
    """Return (leakage-safe feature dataset, reference label audit).

    A label is reliable only if *both* predicted-foreground regions overlap
    reference IDs with high coverage, purity and minimum overlap voxel count.
    Cross-Z edges remain provisional unless deliberately opted into training.
    """
    sv = np.asarray(sv)
    p_same = np.asarray(p_same)
    image_ir193 = np.asarray(image_ir193)
    reference = np.asarray(reference)
    if (sv.ndim != 3 or p_same.shape != (3,*sv.shape)
        or image_ir193.shape != sv.shape or reference.shape != sv.shape):
        raise ValueError('Need sv/img/ref [Z,Y,X], p_same [3,Z,Y,X]')
    if not np.issubdtype(sv.dtype,np.integer) or np.any(sv<0):
        raise ValueError('Supervoxels must have nonnegative integer labels')
    if z_weight < 0 or not 0 < min_coverage <= 1 or not 0 < min_purity <= 1 or min_overlap_voxels < 1:
        raise ValueError('Invalid label-quality/weight parameters')
    if not np.all(np.isfinite(p_same)) or not np.all(np.isfinite(image_ir193)):
        raise ValueError('Non-finite image or affinity values')
    n = int(sv.max())
    columns = ['sv_a','sv_b', *FEATURE_COLUMNS, 'baseline_static_gate', 'pseudo_label', 'label_status','eligible_for_training']
    audit_cols = ['sv_a','sv_b','majority_id_a','majority_id_b',
                  'reference_coverage_a','reference_coverage_b',
                  'reference_purity_a','reference_purity_b',
                  'reference_overlap_voxels_a','reference_overlap_voxels_b']
    if not n:
        return pd.DataFrame(columns=columns),pd.DataFrame(columns=audit_cols)
    codebase = n+1
    key_list=[]; p_list=[]; weight_list=[]; axis_list=[]; nuclear_list=[]
    for axis in range(3):
        lo=[slice(None)]*3; hi=[slice(None)]*3
        lo[axis]=slice(None,-1); hi[axis]=slice(1,None)
        a=sv[tuple(lo)]; b=sv[tuple(hi)]
        mask=(a>0)&(b>0)&(a!=b)
        if not np.any(mask):
            continue
        aa=a[mask].astype(np.int64); bb=b[mask].astype(np.int64)
        key_list.append(np.minimum(aa,bb)*codebase+np.maximum(aa,bb))
        p_list.append(np.clip(p_same[axis][tuple(lo)][mask].astype(np.float64),1e-5,1-1e-5))
        weight_list.append(np.full(len(aa),z_weight if axis==0 else 1.,np.float64))
        axis_list.append(np.full(len(aa),axis,np.int8))
        nuclear_list.append(np.abs(image_ir193[tuple(lo)][mask].astype(np.float64)
                                   -image_ir193[tuple(hi)][mask].astype(np.float64)))
    if not key_list:
        return pd.DataFrame(columns=columns),pd.DataFrame(columns=audit_cols)
    keys=np.concatenate(key_list)
    p=np.concatenate(p_list)
    w=np.concatenate(weight_list)
    axes=np.concatenate(axis_list)
    nuclear_absdiff=np.concatenate(nuclear_list)
    u,inv=np.unique(keys,return_inverse=True)
    m=len(u)
    a=(u//codebase).astype(np.int32)
    b=(u%codebase).astype(np.int32)
    ct=np.bincount(inv,minlength=m).astype(np.int64)
    eps=1e-10
    pmean=np.bincount(inv,weights=p,minlength=m)/ct
    pstd=np.sqrt(np.maximum(0,np.bincount(inv,weights=p*p,minlength=m)/ct-pmean**2))
    logits=np.log(p/(1-p))
    wsum=np.bincount(inv,weights=w,minlength=m)
    pooled_logits=np.bincount(inv,weights=w*logits,minlength=m)/np.maximum(wsum,eps)
    negfrac=np.bincount(inv,weights=w*(p<0.5),minlength=m)/np.maximum(wsum,eps)
    facecount=np.stack([np.bincount(inv[axes==i],minlength=m) for i in range(3)],axis=1)
    per_axis_mean=[]
    for axis in range(3):
        mask=axes==axis
        counts=facecount[:,axis]
        per_axis_mean.append(np.where(counts>0,
            np.bincount(inv[mask],weights=p[mask],minlength=m)/np.maximum(counts,1),np.nan))
    region_counts,region_mean,region_std,extents=_region_intensity_and_extents(sv,image_ir193)
    minvol=np.minimum(region_counts[a],region_counts[b])
    maxvol=np.maximum(region_counts[a],region_counts[b])
    result = {
        'sv_a':a,'sv_b':b,'interface_n_faces':ct,
        'interface_z_faces':facecount[:,0], 'interface_y_faces':facecount[:,1],
        'interface_x_faces':facecount[:,2],
        'interface_z_fraction':facecount[:,0]/ct,
        'interface_y_fraction':facecount[:,1]/ct,
        'interface_x_fraction':facecount[:,2]/ct,
        'affinity_p_mean':pmean,'affinity_p_std':pstd,
        'affinity_p10':_quantiles_by_key(keys,p,u,0.10),
        'affinity_p50':_quantiles_by_key(keys,p,u,0.50),
        'affinity_p90':_quantiles_by_key(keys,p,u,0.90),
        'affinity_logit_mean_weighted':pooled_logits,
        'affinity_negative_fraction_weighted':negfrac,
        'affinity_p_mean_z':per_axis_mean[0],
        'affinity_p_mean_y':per_axis_mean[1],
        'affinity_p_mean_x':per_axis_mean[2],
        'ir193_face_absdiff_mean':np.bincount(inv,weights=nuclear_absdiff,minlength=m)/ct,
        'region_volume_min':minvol,'region_volume_max':maxvol,
        'region_volume_ratio_min_max':minvol/np.maximum(maxvol,1),
        'region_ir193_mean_absdiff':np.abs(region_mean[a]-region_mean[b]),
        'region_ir193_std_mean':(region_std[a]+region_std[b])/2,
    }
    for i,axis in enumerate('zyx'):
        result[f'region_bbox_{axis}_extent_min']=np.minimum(extents[i,a],extents[i,b])
        result[f'region_bbox_{axis}_extent_max']=np.maximum(extents[i,a],extents[i,b])
    result['baseline_static_gate']=(
        (pooled_logits >= score_threshold) & (negfrac <= max_negative_fraction)
        & (ct >= min_faces)).astype(np.int8)
    label_ref=region_reference_diagnostics(sv,reference)
    majority=label_ref['majority']
    coverage=label_ref['coverage']
    purity=label_ref['purity']
    overlaps=label_ref['overlap_size']
    passed=(coverage>=min_coverage)&(purity>=min_purity)&(overlaps>=min_overlap_voxels)&(majority>0)
    known=passed[a]&passed[b]
    provisional=known & (facecount[:,0]>0)
    status=np.full(m,'unknown_low_reference_support',dtype=object)
    status[known]='reliable_xy'
    status[provisional]='provisional_z'
    labels=np.full(m,-1,np.int8)
    labels[known]=(majority[a[known]]==majority[b[known]]).astype(np.int8)
    eligibility=known & (~provisional | include_z_labels)
    result['pseudo_label']=labels
    result['label_status']=status
    result['eligible_for_training']=eligibility.astype(np.int8)
    df=pd.DataFrame(result)[columns]
    # Reference-derived quantities are emitted ONLY to the audit table.
    audit=pd.DataFrame({
        'sv_a':a,'sv_b':b,
        'majority_id_a':majority[a], 'majority_id_b':majority[b],
        'reference_coverage_a':coverage[a],'reference_coverage_b':coverage[b],
        'reference_purity_a':purity[a],'reference_purity_b':purity[b],
        'reference_overlap_voxels_a':overlaps[a], 'reference_overlap_voxels_b':overlaps[b],
    })[audit_cols]
    if len(df)!=len(audit) or (df['interface_n_faces'].to_numpy()!=facecount.sum(axis=1)).any():
        raise AssertionError('Interface table integrity failure')
    return df,audit
