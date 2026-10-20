#!/usr/bin/env python3
"""Reference-guided audit of saved posthoc Z-rescue proposals, with NO tuning.

This script only reads existing output (candidate CSV + three saved segmentation
masks) and the pseudo-reference mask. It does not train or re-run segmentation.
Reference information is used only AFTER inference for evaluation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def component_reference_stats(initial: np.ndarray, reference: np.ndarray, purity=0.80, coverage=0.50):
    if initial.shape != reference.shape or initial.ndim != 3:
        raise ValueError('Both masks must be equal-shape ZYX')
    ii=initial.astype(np.int64,copy=False)
    rr=reference.astype(np.int64,copy=False)
    if ii.min()<0 or rr.min()<0:
        raise ValueError('IDs must be nonnegative')
    n=int(ii.max())
    vols=np.bincount(ii.ravel(),minlength=n+1)
    overlap=(ii>0)&(rr>0)
    rid=np.zeros(n+1,np.int64);pur=np.zeros(n+1,np.float64);cov=np.zeros(n+1,np.float64)
    ref_overlap=np.bincount(ii[overlap],minlength=n+1)
    if np.any(overlap):
        ref_base=int(rr.max())+1
        pair,counts=np.unique(ii[overlap]*ref_base+rr[overlap],return_counts=True)
        iid=pair//ref_base; refid=pair%ref_base
        # np.unique gives iid ordered; select max-overlap reference for each component
        for a in np.unique(iid):
            pos=np.flatnonzero(iid==a)
            idx=pos[np.argmax(counts[pos])]
            rid[a]=refid[idx]
            pur[a]=counts[idx]/ref_overlap[a]
            cov[a]=ref_overlap[a]/vols[a]
    reliable=(rid>0)&(pur>=purity)&(cov>=coverage)
    return rid,pur,cov,reliable


def link_reference_categories(frame: pd.DataFrame, rid: np.ndarray, purity: np.ndarray,
                              coverage: np.ndarray, reliable: np.ndarray) -> pd.DataFrame:
    if not {'a','b','geometry_eligible','mean_pz','z'}.issubset(frame.columns):
        raise ValueError('Candidate CSV lacks essential columns')
    out=frame.copy()
    a=out.a.to_numpy(dtype=np.int64);b=out.b.to_numpy(dtype=np.int64)
    if np.any(a<1)|np.any(b<1)|np.any(a>=len(rid))|np.any(b>=len(rid)):
        raise ValueError('Candidate references unknown component ID')
    ra,rb=reliable[a],reliable[b]
    out['a_ref_majority']=rid[a];out['b_ref_majority']=rid[b]
    out['a_purity']=purity[a];out['b_purity']=purity[b]
    out['a_reference_coverage']=coverage[a];out['b_reference_coverage']=coverage[b]
    out['reference_category']=np.where(ra&rb& (rid[a]==rid[b]),'reliable_same',
                        np.where(ra&rb&(rid[a]!=rid[b]),'reliable_different','uncertain'))
    return out


def matching_map(reference,prediction):
    """Match the existing >=0.5 IoU one-to-one greedy PQ metric for IDs."""
    from instance_error_attribution import overlap_objects
    _,_,_,overlaps=overlap_objects(reference,prediction)
    if len(overlaps)==0:
        return {}
    usedr=set();usedp=set();chosen={}
    for r in overlaps.loc[overlaps.iou>=0.5].sort_values('iou',ascending=False).itertuples(index=False):
        ri=int(r.reference_id);pi=int(r.predicted_id)
        if ri not in usedr and pi not in usedp:
            chosen[pi]=ri; usedr.add(ri); usedp.add(pi)
    return chosen


def descendant_ids(initial,post):
    """Each initial component should map to exactly one post-merge ID."""
    if initial.shape!=post.shape:
        raise ValueError('Mask shape mismatch')
    mask=initial>0
    pairs=np.unique(np.stack((initial[mask],post[mask]),axis=1),axis=0)
    ids,counts=np.unique(pairs[:,0],return_counts=True)
    if not np.all(counts==1) or np.any(pairs[:,1]==0):
        raise ValueError('Output is not a pure union of baseline components')
    return {int(a):int(b) for a,b in pairs}


def matched_preservation(reference,base,post):
    bm=matching_map(reference,base)
    pm=matching_map(reference,post)
    mapping=descendant_ids(base,post)
    preserved=sum(pm.get(mapping[i])==r for i,r in bm.items())
    return {'baseline_matched_instances':len(bm),
            'preserved_same_reference_after_merging':int(preserved),
            'baseline_matches_lost_or_changed':int(len(bm)-preserved),
            'post_matched_instances':len(pm),
            'new_or_recovered_matches_net':int(len(pm)-preserved)}


def summarize_links(table,threshold=.45):
    g=table.loc[table.geometry_eligible.astype(bool)].copy()
    g['pz_gate_pass']=g.mean_pz>=threshold
    details=[]
    for gate,subset in [('all_geometry',g),('passing_pz_gate',g.loc[g.pz_gate_pass]),
                        ('failing_pz_gate',g.loc[~g.pz_gate_pass])]:
        counts=subset.reference_category.value_counts()
        details.append({'set':gate,'n':len(subset),
           'reliable_same':int(counts.get('reliable_same',0)),
           'reliable_different':int(counts.get('reliable_different',0)),
           'uncertain':int(counts.get('uncertain',0)),
           'reliable_different_fraction_of_reliable':float(counts.get('reliable_different',0)/max(1,(counts.get('reliable_same',0)+counts.get('reliable_different',0)))),
           'median_mean_pz':float(subset.mean_pz.median()) if len(subset) else None,
           'median_footprint_iou':float(subset.footprint_iou.median()) if len(subset) else None})
    by_z=(g.groupby(['z','reference_category']).size().unstack(fill_value=0).reset_index())
    return pd.DataFrame(details),by_z


def run(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Output folder is not empty: {args.out}')
    args.out.mkdir(parents=True,exist_ok=True)
    import sys
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str(here))
    from train_segmentation import read_data
    _,reference,_,_=read_data(args.data_dir)
    f= pd.read_csv(args.split_run_dir/'tile_splits.csv')
    rows=f[(f.split==args.split)&(f.y==args.tile_y)&(f.x==args.tile_x)]
    if len(rows)!=1:
        raise ValueError('Tile not unique or absent from requested split')
    t=rows.iloc[0];y,x,w=int(t.y),int(t.x),int(t.w);guard=args.guard
    if guard*2>=w or y+w>reference.shape[1] or x+w>reference.shape[2]:
        raise ValueError('Invalid crop')
    ref=reference[:,y+guard:y+w-guard,x+guard:x+w-guard]
    npzpath=args.rescue_run_dir/f'z_link_masks_{y}_{x}.npz'
    csvpath=args.rescue_run_dir/f'z_link_candidates_{y}_{x}.csv'
    with np.load(npzpath) as saved:
        base=saved['baseline']; geom=saved['geometry']; gated=saved['geometry_pz']
    for mask in (base,geom,gated):
        if mask.shape!=ref.shape:
            raise ValueError('Saved mask and reference shapes mismatch')
    proposals=pd.read_csv(csvpath)
    ids,purity,coverage,reliable=component_reference_stats(base,ref,args.min_purity,args.min_coverage)
    annotated=link_reference_categories(proposals,ids,purity,coverage,reliable)
    summary,by_z=summarize_links(annotated,args.min_mean_pz)
    preservation={name:matched_preservation(ref,base,mask)
                  for name,mask in [('z_overlap_mutual',geom),('z_overlap_mutual_with_pz_gate',gated)]}
    annotated.to_csv(args.out/'z_link_candidates_reference_audit.csv',index=False)
    summary.to_csv(args.out/'z_link_reference_categories.csv',index=False)
    by_z.to_csv(args.out/'z_link_reference_categories_by_plane.csv',index=False)
    record={'scope':'Reference-guided post-hoc diagnostics of saved VAL predictions; NOT a deployable method',
            'reference':'Author watershed pseudo instances; uncertain Z registration and spacing',
            'tile':{'y':y,'x':x,'split':args.split,'shape_zyx':list(ref.shape)},
            'component_reference_purity_threshold':args.min_purity,
            'component_reference_coverage_threshold':args.min_coverage,
            'mean_pz_gate':args.min_mean_pz,
            'n_initial_baseline_components':int(base.max()),
            'link_categories':summary.to_dict(orient='records'),
            'baseline_match_preservation':preservation,
            'limitations':['Reference majority may be unreliable for mixed components',
                           'Link categories are based on initial components; they do not prove final-chain merge correctness',
                           'Z reference agreement depends on unverified registration',
                           'Do not optimize thresholds on repeatedly reused VAL tiles']}
    (args.out/'z_link_safety_summary.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
    print(summary.to_string(index=False))
    print(json.dumps(preservation,indent=2))
    print(f'Saved: {args.out}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--split-run-dir',type=Path,required=True)
    p.add_argument('--rescue-run-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tile-y',type=int,default=384)
    p.add_argument('--tile-x',type=int,default=576)
    p.add_argument('--split',choices=['train','val'],default='val')
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--min-purity',type=float,default=.8)
    p.add_argument('--min-coverage',type=float,default=.5)
    p.add_argument('--min-mean-pz',type=float,default=.45)
    args=p.parse_args()
    run(args)

if __name__=='__main__':main()
