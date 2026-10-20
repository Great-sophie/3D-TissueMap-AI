"""Reference-guided *diagnostics* for 3D instance masks; no training or tuning.

Classes of unmatched objects are exclusive heuristic descriptions, not proven
biological error causes. Reference is author watershed pseudo-instances.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def overlap_objects(reference: np.ndarray, predicted: np.ndarray, *, iou_threshold: float=0.5,
                    min_overlap_voxels:int=5, min_reference_fraction:float=0.05):
    if reference.ndim != 3 or reference.shape != predicted.shape:
        raise ValueError('reference and predicted must share 3D ZYX shape')
    if not (0 < iou_threshold < 1) or min_overlap_voxels < 1 or not (0 <= min_reference_fraction <= 1):
        raise ValueError('invalid attribution thresholds')
    ref=reference.astype(np.int64,copy=False)
    pred=predicted.astype(np.int64,copy=False)
    if (ref<0).any() or (pred<0).any():raise ValueError('negative instance IDs unsupported')
    rid, rc=np.unique(ref[ref>0],return_counts=True)
    pid, pc=np.unique(pred[pred>0],return_counts=True)
    nr, npred=len(rid),len(pid)
    good=(ref>0)&(pred>0)
    cols=['reference_id','predicted_id','intersection','ref_voxels','pred_voxels','iou','fraction_of_reference','fraction_of_prediction']
    if good.any():
        rr=np.searchsorted(rid,ref[good]).astype(np.int64)
        pp=np.searchsorted(pid,pred[good]).astype(np.int64)
        pair, counts=np.unique(rr*max(1,npred)+pp,return_counts=True)
        ridx=(pair//max(1,npred)).astype(int)
        pidx=(pair%max(1,npred)).astype(int)
        overlap=pd.DataFrame({'reference_id':rid[ridx], 'predicted_id':pid[pidx],
                              'intersection':counts, 'ref_voxels':rc[ridx], 'pred_voxels':pc[pidx]})
        overlap['iou']=overlap.intersection/(overlap.ref_voxels+overlap.pred_voxels-overlap.intersection)
        overlap['fraction_of_reference']=overlap.intersection/overlap.ref_voxels
        overlap['fraction_of_prediction']=overlap.intersection/overlap.pred_voxels
    else:
        overlap=pd.DataFrame(columns=cols)
    # sparse_pq parity: best IoU first, 1:1 greedy matching at >=.5
    matches=[];usedr=set();usedp=set()
    for t in overlap.loc[overlap.iou>=iou_threshold].sort_values('iou',ascending=False).itertuples(index=False):
        r,p=int(t.reference_id),int(t.predicted_id)
        if r not in usedr and p not in usedp:
            matches.append((r,p,float(t.iou)));usedr.add(r);usedp.add(p)
    substantial=overlap[(overlap.intersection>=min_overlap_voxels)&
                       (overlap.fraction_of_reference>=min_reference_fraction)]
    n_splits=substantial.groupby('reference_id').predicted_id.nunique().to_dict()
    all_overlaps_by_pred={int(p):g for p,g in overlap.groupby('predicted_id')}
    bg_voxels=int(np.count_nonzero((ref==0)&(pred>0)))
    objects=[]
    for p,voxels in zip(pid,pc):
        p=int(p);voxels=int(voxels)
        pairs=all_overlaps_by_pred.get(p)
        if pairs is None or len(pairs)==0:
            category='orphan_no_reference_overlap'
            dom_id=0;max_intersection=0;n_significant_refs=0;n_covered_refs=0
        else:
            best=pairs.loc[pairs.intersection.idxmax()]
            dom_id=int(best.reference_id);max_intersection=int(best.intersection)
            n_covered_refs=int(len(pairs))
            n_significant_refs=int(((pairs.intersection>=min_overlap_voxels)&
                                    (pairs.fraction_of_prediction>=.10)).sum())
            if (voxels-int(pairs.intersection.sum()))/voxels>=.5:
                category='background_dominated_overlap'
            elif n_significant_refs>=2:
                category='multi_reference_overlap'
            elif (n_splits.get(dom_id,0)>=2 and max_intersection>=min_overlap_voxels and
                  max_intersection/int(rc[np.searchsorted(rid,dom_id)])>=min_reference_fraction):
                category='fragment_candidate'
            else:
                category='other_unmatched_overlap'
        if p in usedp:category='matched_iou_ge_0_5'
        objects.append({'predicted_id':p, 'voxels':voxels, 'category':category,
                        'reference_overlap_voxels':int(pairs.intersection.sum()) if pairs is not None else 0,
                        'background_voxel_fraction':float((voxels-int(pairs.intersection.sum()))/voxels) if pairs is not None else 1.,
                        'dominant_reference_id':dom_id,'dominant_intersection_voxels':max_intersection,
                        'n_overlapped_reference_ids':n_covered_refs,'n_significant_reference_ids':n_significant_refs,
                        'is_iou_matched':p in usedp})
    object_df=pd.DataFrame(objects,columns=['predicted_id','voxels','category','reference_overlap_voxels',
                                            'background_voxel_fraction','dominant_reference_id',
                                            'dominant_intersection_voxels','n_overlapped_reference_ids',
                                            'n_significant_reference_ids','is_iou_matched'])
    ref_records=[]
    all_by_ref={int(r):g for r,g in overlap.groupby('reference_id')}
    for r,voxels in zip(rid,rc):
        r=int(r); voxels=int(voxels)
        group=all_by_ref.get(r)
        n_sub=int(n_splits.get(r,0))
        ref_records.append({'reference_id':r,'reference_voxels':voxels,'n_predicted_ids_touching':len(group) if group is not None else 0,
                            'n_substantial_predicted_components':n_sub,
                            'is_fragmented_2plus':n_sub>=2,
                            'fraction_reference_covered':float(group.intersection.sum()/voxels) if group is not None else 0,
                            'has_iou_matched_prediction':r in usedr})
    reference_df=pd.DataFrame(ref_records,columns=['reference_id','reference_voxels','n_predicted_ids_touching',
                                                    'n_substantial_predicted_components','is_fragmented_2plus',
                                                    'fraction_reference_covered','has_iou_matched_prediction'])
    if nr==0 and npred==0:
        pq=1.
    else:
        denom=len(matches)+.5*((nr-len(matches))+(npred-len(matches)))
        pq=float(sum(t[2] for t in matches)/denom) if denom else 0.
    classes=object_df.category.value_counts().to_dict() if len(object_df) else {}
    if sum(v for k,v in classes.items() if k!='matched_iou_ge_0_5')!=npred-len(matches):
        raise AssertionError('Object decomposition failed')
    summary={'n_reference':nr,'n_predicted':npred,'pq':pq,'tp':len(matches),'fp':npred-len(matches),'fn':nr-len(matches),
             'unmatched_predicted_exclusive_categories':{k:int(v) for k,v in classes.items() if k!='matched_iou_ge_0_5'},
             'n_reference_with_2plus_substantial_prediction_components':int((reference_df.n_substantial_predicted_components>=2).sum()),
             'reference_extra_component_count_proxy':int(np.maximum(reference_df.n_substantial_predicted_components.to_numpy()-1,0).sum()),
             'n_predicted_foreground_voxels_in_reference_background':bg_voxels,
             'fraction_predicted_foreground_in_reference_background':float(bg_voxels/max(1,int((pred>0).sum()))),
             'n_orphan_objects':int(classes.get('orphan_no_reference_overlap',0)),
             'attribution_thresholds':{'iou':iou_threshold,'min_overlap_voxels':min_overlap_voxels,
                                       'min_reference_fraction':min_reference_fraction,
                                       'mixed_component_min_fraction_prediction':.10,
                                       'background_dominated_fraction':.50}}
    return summary,object_df,reference_df,overlap


def directional_faces(reference:np.ndarray,predicted:np.ndarray):
    """Conditional connectivity diagnostic along true same-ID voxel faces (pseudo reference)."""
    if reference.shape != predicted.shape or reference.ndim!=3:raise ValueError('ZYX shape mismatch')
    rows=[]
    for axis,name in enumerate(('Z','Y','X')):
        lo=[slice(None)]*3;hi=[slice(None)]*3
        lo[axis]=slice(None,-1);hi[axis]=slice(1,None)
        rlo,rhi=reference[tuple(lo)],reference[tuple(hi)]
        plo,phi=predicted[tuple(lo)],predicted[tuple(hi)]
        same_ref=(rlo>0)&(rlo==rhi)
        both_pred_fg=same_ref&(plo>0)&(phi>0)
        connect=both_pred_fg&(plo==phi)
        sr=int(same_ref.sum());bp=int(both_pred_fg.sum());ok=int(connect.sum())
        rows.append({'axis':name,'n_same_reference_voxel_faces':sr,
                     'n_faces_with_both_predicted_foreground':bp,
                     'n_connected_by_same_prediction':ok,
                     'fraction_both_foreground':float(bp/sr) if sr else None,
                     'conditional_same_prediction_rate':float(ok/bp) if bp else None,
                     'conditional_broken_prediction_rate':float((bp-ok)/bp) if bp else None})
    return pd.DataFrame(rows)
