"""Post-hoc conservative, per-plane, mutual-best Z-link recovery.

Inference is label-free. This is a diagnostic, NOT cross-slice cell truth.
The pixel coordinate alignment and Z spacing in the source stack are provisional.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def propose_z_links(labels: np.ndarray, p_z: np.ndarray, *, min_faces: int = 2,
                    min_overlap: float = 0.35, min_iou: float = 0.15) -> pd.DataFrame:
    """Z-face overlap between predicted instance IDs; rank reciprocal best per plane.

    Overlap fraction = intersection/min(plane areas); IoU = intersection/union
    of the 2D footprints. Already-continuous (a==b) connections compete during
    reciprocal ranking, so an existing self-link cannot be trumped casually.
    """
    labels = np.asarray(labels)
    p_z = np.asarray(p_z)
    if labels.ndim != 3 or p_z.shape != labels.shape:
        raise ValueError('Expected labels and Z P(same) of matching [Z,Y,X] shape')
    if not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0):
        raise ValueError('Nonnegative integer instance mask required')
    if not np.all(np.isfinite(p_z)) or np.any((p_z<0)|(p_z>1)):
        raise ValueError('Invalid Z-affinity probabilities')
    if min_faces<1 or not (0<=min_overlap<=1) or not (0<=min_iou<=1):
        raise ValueError('Invalid Z-link constraints')
    fields=['z','a','b','n_faces','area_a','area_b','overlap_min','footprint_iou',
            'mean_pz','mutual_best','geometry_eligible']
    parts=[]
    for z in range(labels.shape[0]-1):
        a,b=labels[z],labels[z+1]
        ma=(a>0)&(b>0)
        if not ma.any():continue
        aval=a[ma].astype(np.int64)
        bval=b[ma].astype(np.int64)
        ids_a,n_a=np.unique(a[a>0],return_counts=True)
        ids_b,n_b=np.unique(b[b>0],return_counts=True)
        base=int(labels.max())+1
        combo=aval*base+bval
        pair_ids,pair_counts=np.unique(combo,return_counts=True)
        aa=(pair_ids//base).astype(np.int64)
        bb=(pair_ids%base).astype(np.int64)
        area_a=n_a[np.searchsorted(ids_a,aa)]
        area_b=n_b[np.searchsorted(ids_b,bb)]
        n=pair_counts.astype(np.int64)
        means=np.bincount(np.searchsorted(pair_ids,combo),weights=p_z[z][ma],minlength=len(pair_ids))/n
        frame=pd.DataFrame({'z':z,'a':aa,'b':bb,'n_faces':n,'area_a':area_a,
                            'area_b':area_b,'overlap_min':n/np.minimum(area_a,area_b),
                            'footprint_iou':n/(area_a+area_b-n),'mean_pz':means})
        # Prefer highest IoU, and then more supporting voxel faces.
        frame=frame.sort_values(['footprint_iou','n_faces','a','b'],ascending=[False,False,True,True],kind='stable')
        best_a=frame.drop_duplicates('a').set_index('a')['b']
        best_b=frame.drop_duplicates('b').set_index('b')['a']
        frame['mutual_best']=(frame.b.to_numpy()==best_a.reindex(frame.a).to_numpy()) & (frame.a.to_numpy()==best_b.reindex(frame.b).to_numpy())
        frame['geometry_eligible']=(frame['mutual_best']& (frame.a!=frame.b)&
                                    (frame.n_faces>=min_faces)&(frame.overlap_min>=min_overlap)&
                                    (frame.footprint_iou>=min_iou))
        parts.append(frame)
    if not parts:return pd.DataFrame(columns=fields)
    out=pd.concat(parts,ignore_index=True)
    return out.loc[:,fields]


def rescue_z_links(labels:np.ndarray, proposals:pd.DataFrame, *,
                   require_affinity:bool = False, min_mean_pz:float=0.45,
                   max_component_voxels:int=1200) -> tuple[np.ndarray,dict]:
    """Union only prespecified mutual-best neighboring-slice overlaps.

    All candidate proposals refer to original post-RAG labels and are NOT
    recomputed after rescue; this prevents a moving-match target, but transitive
    matches can still occur. A volume cap protects against growing giants.
    """
    labels=np.asarray(labels)
    if labels.ndim!=3 or not np.issubdtype(labels.dtype,np.integer) or (labels<0).any():
        raise ValueError('Expected nonnegative integer 3D labels')
    if not 0<=min_mean_pz<=1 or max_component_voxels<1:
        raise ValueError('Invalid thresholds')
    n=int(labels.max()); parent=np.arange(n+1,dtype=np.int32)
    sizes=np.bincount(labels.ravel(),minlength=n+1).astype(np.int64)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    eligible=proposals.loc[proposals.geometry_eligible].copy()
    n_geom=len(eligible)
    if require_affinity:
        eligible=eligible.loc[eligible.mean_pz>=min_mean_pz]
    eligible=eligible.sort_values(['footprint_iou','n_faces'],ascending=False,kind='stable')
    naccepted=n_cycle=n_volume=0
    for row in eligible.itertuples(index=False):
        a,b=root(int(row.a)),root(int(row.b))
        if a==b:
            n_cycle+=1;continue
        if sizes[a]+sizes[b]>max_component_voxels:
            n_volume+=1;continue
        if sizes[b]>sizes[a]: a,b=b,a
        parent[b]=a;sizes[a]+=sizes[b];naccepted+=1
    mapping=np.zeros(n+1,dtype=np.int32)
    nonzero=np.unique(labels[labels>0]);roots=np.array([root(int(i)) for i in nonzero],np.int32)
    _,ix=np.unique(roots,return_inverse=True)
    mapping[nonzero]=ix+1
    result=mapping[labels]
    summary={'n_all_z_pairs':int(len(proposals)), 'n_mutual_geometry_eligible':int(n_geom),
             'n_after_affinity_gate':int(len(eligible)), 'require_affinity':bool(require_affinity),
             'min_mean_pz':float(min_mean_pz) if require_affinity else None,
             'n_accepted_merges':naccepted,'n_already_connected':n_cycle,
             'n_volume_rejections':n_volume,'n_before':int(len(nonzero)),
             'n_after':int(np.count_nonzero(np.unique(result))),
             'changes_existing_xy_labels_only_via_z_unions':True,
             'never_accesses_reference':True}
    return result,summary
