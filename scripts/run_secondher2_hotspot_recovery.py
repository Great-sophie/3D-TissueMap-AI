#!/usr/bin/env python3
"""SecondHer2 held-out-plane 3D HER2 spatial hotspot recovery benchmark.

Predefined exploratory PROXY endpoints (not biological ground truth):
  - Independent signal: fixed held-out middle-Z measurement per pseudo-instance.
  - Spatial context: kNN among 3D pseudo-instance centroids (XYZ) or XY projection.
  - Hotspots: top/bottom decile of *smoothed signed within-cohort clone-rank difference*.
  - Predictors: nearest 1, nearest 2, all other object Z planes; never target Z.
  - Primary comparison: XYZ nearest-2 vs XYZ nearest-1; secondaries: all vs 2,
    and XYZ vs XY neighbor contexts, with both XYZ & XY-defined target references.

No raw OME images are modified or loaded. No algorithm tuning uses held-out values.
Coordinates/masks from author-provided pseudo-instances are not validated cell GT.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

M = ('HER2_3B5', 'HER2_29D8')
METHODS = ('nearest1', 'nearest2', 'all_others')
CONTEXTS = ('XYZ', 'XY_projection')
SRC = {
    'sections': 'runs/secondher2_2d_vs_3d_heldout_v0_1/per_object_per_section_her2.csv',
    'heldout': 'runs/secondher2_2d_vs_3d_heldout_v0_1/heldout_2d_vs_multiplane_predictions.csv',
    'coords': 'runs/secondher2_3d_spatial_discordance_v0_1/3d_her2_object_spatial_coordinates.csv',
}
SEED = 20261019


def is_true(x):
    return pd.Series(x).astype(str).str.lower().eq('true').to_numpy()


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_and_build(paths):
    s, h, c = [pd.read_csv(paths[k]) for k in ('sections', 'heldout', 'coords')]
    need_s = {'source_pseudo_id','z','n_px','HER2_3B5_sum','HER2_29D8_sum'}
    need_h = {'source_pseudo_id','target_z','primary_eligible','retained',
              'HER2_3B5_heldout_mean','HER2_29D8_heldout_mean',
              'HER2_3B5_single_plane_mean','HER2_29D8_single_plane_mean',
              'HER2_3B5_other_planes_mean','HER2_29D8_other_planes_mean'}
    need_c = {'source_pseudo_id','x_um','y_um','z_um'}
    for name, df, required in (('sections',s,need_s),('heldout',h,need_h),('coords',c,need_c)):
        if required-set(df.columns):
            raise ValueError(f'{name} missing columns: {sorted(required-set(df.columns))}')
    if s.duplicated(['source_pseudo_id','z']).any():
        raise ValueError('Duplicate source ID and Z in per-section table')
    if h.source_pseudo_id.duplicated().any() or c.source_pseudo_id.duplicated().any():
        raise ValueError('Duplicate source ID in heldout/centroid table')
    if (s.n_px<=0).any():
        raise ValueError('Non-positive n_px')
    if not np.isfinite(s[['HER2_3B5_sum','HER2_29D8_sum']].to_numpy()).all():
        raise ValueError('Nonfinite HER2 sums')
    if (s[['HER2_3B5_sum','HER2_29D8_sum']]<0).any().any():
        raise ValueError('Negative HER2 sums')
    h = h.set_index('source_pseudo_id')
    records=[]
    for sid,g in s[s.source_pseudo_id.isin(h.index)].groupby('source_pseudo_id',sort=True):
        g=g.sort_values('z').reset_index(drop=True)
        zz=g.z.to_numpy(int)
        if len(g)<4 or np.any(np.diff(zz)!=1):
            raise ValueError(f'Frozen eligible ID={sid} has missing Z or <4 planes')
        info=h.loc[sid]
        target=int(info.target_z)
        if target!=zz[len(zz)//2]:
            raise ValueError(f'Frozen heldout target mismatch ID={sid}')
        held=g.loc[g.z==target]
        if len(held)!=1:
            raise ValueError(f'Target Z missing ID={sid}')
        non=g.loc[g.z!=target].copy()
        # Deterministic Z tie handling (preceding Z first)
        non=non.iloc[np.lexsort((non.z.to_numpy(),np.abs(non.z.to_numpy()-target)))].reset_index(drop=True)
        n2=non.iloc[:2]
        row={'source_pseudo_id':int(sid),'target_z':target,'n_sections':len(g),
             'primary_eligible': bool(is_true([info.primary_eligible])[0]),
             'retained': bool(is_true([info.retained])[0])}
        for m in M:
            k=m+'_sum'
            trg=float(held[k].iloc[0]/held.n_px.iloc[0])
            pred1=float(non[k].iloc[0]/non.n_px.iloc[0])
            pred2=float(n2[k].sum()/n2.n_px.sum())
            predall=float(non[k].sum()/non.n_px.sum())
            for value in (trg,pred1,pred2,predall):
                if not np.isfinite(value) or value<0:
                    raise ValueError(f'Invalid marker estimate ID={sid}, {m}')
            checks=((trg,info[m+'_heldout_mean']),
                    (pred1,info[m+'_single_plane_mean']),
                    (predall,info[m+'_other_planes_mean']))
            if not all(np.isclose(x,y,rtol=1e-6,atol=1e-6) for x,y in checks):
                raise ValueError(f'Frozen intensity reconciliation failed: ID={sid}, {m}')
            row.update({m+'_target':trg,m+'_nearest1':pred1,
                        m+'_nearest2':pred2,m+'_all_others':predall})
        records.append(row)
    pred=pd.DataFrame.from_records(records)
    if len(pred)!=len(h):
        raise ValueError(f'Expected {len(h)} eligible IDs; recovered {len(pred)}')
    pred=pred.merge(c[['source_pseudo_id','x_um','y_um','z_um']],on='source_pseudo_id',how='left',validate='one_to_one')
    if pred[['x_um','y_um','z_um']].isna().any().any():
        raise ValueError('Missing spatial centroids')
    return pred


def neighbours(points, k):
    if len(points)<=k+1:
        raise ValueError('Fewer objects than kNN requires')
    # Query k+2 so a self index is removable even when coordinates tie.
    distances, ix = cKDTree(points).query(points,k=k+2)
    n=len(points)
    near=np.empty((n,k),dtype=np.int32)
    for i in range(n):
        distinct=np.asarray(ix[i])[np.asarray(ix[i])!=i]
        near[i]=distinct[:k]
    return near


def discordance(df, col):
    a=df['HER2_3B5_'+col].rank(method='average',pct=True).to_numpy(float)
    b=df['HER2_29D8_'+col].rank(method='average',pct=True).to_numpy(float)
    return a-b


def smooth(x,near):
    return (x+np.sum(x[near],axis=1))/(near.shape[1]+1)


def topbottom(scores,ids,q):
    n=len(scores)
    nh=int(np.ceil(q*n))
    if nh*2>=n:
        raise ValueError('Not enough objects to define disjoint hotspot tails')
    # Source-ID deterministic tie break, no tuning on test labels.
    ix=np.lexsort((ids,np.asarray(scores,float)))
    lab=np.zeros(n,dtype=np.int8)
    lab[ix[:nh]]=-1
    lab[ix[-nh:]]=1
    return lab


def metric(truth,pred):
    positive=(truth==1); negative=(truth==-1)
    tp1=np.sum(positive & (pred==1)); tp2=np.sum(negative & (pred==-1))
    n1=np.sum(positive); n2=np.sum(negative)
    p1=np.sum(pred==1); p2=np.sum(pred==-1)
    recall1=tp1/n1 if n1 else np.nan
    recall2=tp2/n2 if n2 else np.nan
    f11=2*tp1/(n1+p1) if n1+p1 else np.nan
    f12=2*tp2/(n2+p2) if n2+p2 else np.nan
    j1=tp1/(n1+p1-tp1) if n1+p1-tp1 else np.nan
    j2=tp2/(n2+p2-tp2) if n2+p2-tp2 else np.nan
    return {'hotspot_pos_recall':float(recall1), 'hotspot_neg_recall':float(recall2),
            'hotspot_macro_f1':float((f11+f12)/2),
            'hotspot_macro_jaccard':float((j1+j2)/2),
            'hotspot_count_each_sign':int(n1)}


def conditional_macro_recall(t,p,idx):
    tt=t[idx];pp=p[idx]
    yes=(tt==1); no=(tt==-1)
    if not np.any(yes) or not np.any(no):
        return np.nan
    return float(0.5*(np.mean(pp[yes]==1)+np.mean(pp[no]==-1)))


def block_bootstrap(df,refs,preds,nboot,seed,block_xy,block_z):
    ids=(np.floor(df.x_um.to_numpy()/block_xy).astype(int).astype(str)+':'+
         np.floor(df.y_um.to_numpy()/block_xy).astype(int).astype(str)+':'+
         np.floor(df.z_um.to_numpy()/block_z).astype(int).astype(str))
    codes,uniq=pd.factorize(ids,sort=True)
    groups=[np.flatnonzero(codes==i) for i in range(len(uniq))]
    if len(groups)<15:
        raise ValueError(f'Too few spatial blocks: {len(groups)}')
    rng=np.random.default_rng(seed)
    bidx=[np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
          for _ in range(nboot)]
    rows=[]
    comparisons=[('XYZ_nearest2','XYZ_nearest1'),
                 ('XYZ_all_others','XYZ_nearest2'),
                 ('XYZ_nearest2','XY_projection_nearest2'),
                 ('XYZ_all_others','XY_projection_nearest1')]
    for ref_name,t in refs.items():
        for method_a,method_b in comparisons:
            a,b=preds[method_a],preds[method_b]
            delta=np.array([conditional_macro_recall(t,a,idx)-conditional_macro_recall(t,b,idx)
                            for idx in bidx])
            delta=delta[np.isfinite(delta)]
            observed=conditional_macro_recall(t,a,np.arange(len(t)))-conditional_macro_recall(t,b,np.arange(len(t)))
            if len(delta)<nboot*.95:
                raise RuntimeError('Too many degenerate bootstrap samples')
            rows.append({'reference_geometry':ref_name,'method_a':method_a,'method_b':method_b,
                         'n_objects':len(t),'n_blocks':len(groups),
                         'observed_delta_macro_recall':float(observed),
                         'conditional_block_boot_p025':float(np.quantile(delta,.025)),
                         'conditional_block_boot_p975':float(np.quantile(delta,.975))})
    return pd.DataFrame(rows),len(groups)


def plot_performance(stats,group,out):
    tab=stats[(stats.group==group)&(stats.reference_geometry=='XYZ')]
    order=['XYZ_nearest1','XYZ_nearest2','XYZ_all_others',
           'XY_projection_nearest1','XY_projection_nearest2','XY_projection_all_others']
    tab=tab.set_index('method').reindex(order)
    fig,ax=plt.subplots(figsize=(10,5.2))
    ys=np.arange(len(order))
    ax.barh(ys,tab.hotspot_macro_f1.to_numpy(float))
    ax.set_yticks(ys,[x.replace('_',' ') for x in order])
    ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('Macro F1 of top/bottom 10% hotspots')
    ax.set_title(f'Independent heldout-Z 3D hotspot proxy: {group}')
    ax.grid(alpha=.18,axis='x')
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(out/f'Figure_H1_hotspot_recovery_{group}.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)


def plot_3d(ref,pred,df,group,out):
    xyz=df[['x_um','y_um','z_um']].to_numpy(float)
    # two independent files rather than overlapping/scaling panels
    for name,vals in (('heldout_reference_proxy',ref),('predicted_XYZ_nearest2',pred)):
        fig=plt.figure(figsize=(7.2,6.3))
        ax=fig.add_subplot(111,projection='3d')
        colors=np.where(vals>0,'#c34b43',np.where(vals<0,'#3e7bb3','#b9bdc4'))
        ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=colors,s=5,alpha=.65,rasterized=True,linewidth=0)
        ax.set(xlabel='X (nominal µm)',ylabel='Y (nominal µm)',zlabel='Z (nominal µm)')
        ax.set_title(f'{group}: {name.replace("_"," ")}')
        ax.view_init(elev=18,azim=-60)
        fig.tight_layout()
        for ext in ('png','pdf'):
            fig.savefig(out/f'Figure_H2_{group}_{name}.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)


def run(args):
    root=args.project_root.expanduser().resolve()
    paths={k:root/v for k,v in SRC.items()}
    missing=[str(p) for p in paths.values() if not p.is_file()]
    if missing: raise FileNotFoundError('Required frozen prior outputs missing:\n'+'\n'.join(missing))
    out=(root/args.out).resolve()
    if out.exists(): raise FileExistsError(f'Refusing to overwrite: {out}')
    if args.k<2 or not (0<args.q<=.25) or args.bootstrap<100: raise ValueError('Bad k/q/bootstrap')
    print('[1/5] Verify per-Z signal, fixed heldout middle section and source centroids',flush=True)
    objects=read_and_build(paths)
    out.mkdir(parents=True)
    figdir=out/'figures';figdir.mkdir()
    stats=[];bootstrap_tables=[];object_tables=[];null_rows=[];sensitivity_rows=[]
    cohorts=[('primary_retained_clean',objects[objects.primary_eligible].copy()),
             ('all_excluded',objects[~objects.retained].copy())]
    for group,g in cohorts:
        if len(g)<50: raise ValueError(f'{group} unexpectedly small')
        g=g.sort_values('source_pseudo_id').reset_index(drop=True)
        ids=g.source_pseudo_id.to_numpy(np.int64)
        print(f'[2/5] {group}: {len(g)} fixed source pseudo-instances; create independent heldout-Z proxy maps',flush=True)
        nears={
            'XYZ':neighbours(g[['x_um','y_um','z_um']].to_numpy(float),args.k),
            'XY_projection':neighbours(g[['x_um','y_um']].to_numpy(float),args.k)
        }
        source={key:discordance(g,key) for key in ('target',)+METHODS}
        reference={ref:topbottom(smooth(source['target'],near),ids,args.q)
                   for ref,near in nears.items()}
        scores={f'{ctx}_{method}':smooth(source[method],near)
                for ctx,near in nears.items() for method in METHODS}
        predicted={name:topbottom(value,ids,args.q) for name,value in scores.items()}
        for ref_name,rt in reference.items():
            for method,pr in predicted.items():
                row={'group':group,'reference_geometry':ref_name,'method':method,
                     'n_objects':len(g),'k':args.k,'q_each_sign':args.q,
                     'rank_spearman_with_heldout_score':float(pd.Series(scores[method]).corr(
                         pd.Series(smooth(source['target'],nears[ref_name])), method='spearman'))}
                row.update(metric(rt,pr))
                stats.append(row)
        print(f'[3/5] {group}: spatial-block paired recovery intervals',flush=True)
        bt,nblocks=block_bootstrap(g,reference,predicted,args.bootstrap,SEED+len(g),args.block_xy,args.block_z)
        bt.insert(0,'group',group)
        bootstrap_tables.append(bt)
        # Prespecified kNN scale sensitivity: same independent heldout target for
        # every k, and the same object cohort for all methods. This is descriptive.
        for kval in sorted(set((6, 12, 24, args.k))):
            if kval == args.k:
                sensitivity_near = nears
                sensitivity_ref = reference['XYZ']
                sensitivity_preds = predicted
            else:
                sensitivity_near = {
                    'XYZ': neighbours(g[['x_um','y_um','z_um']].to_numpy(float), kval),
                    'XY_projection': neighbours(g[['x_um','y_um']].to_numpy(float), kval),
                }
                sensitivity_ref = topbottom(smooth(source['target'], sensitivity_near['XYZ']), ids, args.q)
                sensitivity_preds = {
                    f'{ctx}_{method}':topbottom(smooth(source[method], near), ids, args.q)
                    for ctx, near in sensitivity_near.items() for method in METHODS
                }
            for name in ('XYZ_nearest1', 'XYZ_nearest2', 'XYZ_all_others', 'XY_projection_nearest2'):
                srow = metric(sensitivity_ref, sensitivity_preds[name])
                sensitivity_rows.append({
                    'group':group, 'k':kval, 'method':name, 'n_objects':len(g),
                    'reference_geometry':'XYZ',
                    'hotspot_macro_f1':srow['hotspot_macro_f1'],
                    'hotspot_macro_jaccard':srow['hotspot_macro_jaccard']
                })
        ptab=g[['source_pseudo_id','target_z','n_sections','x_um','y_um','z_um']].copy()
        for k,v in source.items(): ptab[f'signed_rank_difference_{k}']=v
        for k,v in reference.items():ptab[f'reference_hotspot_{k}']=v
        for k,v in predicted.items():ptab[f'pred_hotspot_{k}']=v
        ptab.insert(0,'group',group)
        object_tables.append(ptab)
        # Specific descriptive permutation: scramble predictor scores only,
        # leaving heldout spatial reference and object positions fixed.
        rng=np.random.default_rng(SEED+len(g))
        rt=reference['XYZ'];near=nears['XYZ']
        observed=metric(rt,predicted['XYZ_nearest2'])['hotspot_macro_f1']
        null=np.empty(args.permutations,float)
        for i in range(args.permutations):
            shuffled=rng.permutation(source['nearest2'])
            pr=topbottom(smooth(shuffled,near),ids,args.q)
            null[i]=metric(rt,pr)['hotspot_macro_f1']
        null_rows.append({'group':group,'reference_geometry':'XYZ','method':'XYZ_nearest2',
                          'observed_hotspot_macro_f1':observed,
                          'null_mean':float(np.mean(null)),
                          'null_p025':float(np.quantile(null,.025)),
                          'null_p975':float(np.quantile(null,.975)),
                          'permutation_p_ge':float((1+np.count_nonzero(null>=observed))/(len(null)+1)),
                          'n_permutations':len(null),'n_spatial_blocks':nblocks})
        plot_3d(reference['XYZ'],predicted['XYZ_nearest2'],g,group,figdir)
    result=pd.DataFrame(stats)
    boot=pd.concat(bootstrap_tables,ignore_index=True)
    objs=pd.concat(object_tables,ignore_index=True)
    null=pd.DataFrame(null_rows)
    sensitivity=pd.DataFrame(sensitivity_rows)
    for group,_ in cohorts: plot_performance(result,group,figdir)
    print('[4/5] Write result tables and falsifiable protocol',flush=True)
    result.to_csv(out/'hotspot_recovery_summary.csv',index=False)
    boot.to_csv(out/'spatial_block_recovery_intervals.csv',index=False)
    objs.to_csv(out/'per_object_hotspot_labels.csv',index=False)
    null.to_csv(out/'label_shuffle_negative_control.csv',index=False)
    sensitivity.to_csv(out/'neighbourhood_scale_sensitivity.csv',index=False)
    protocol={'name':'SecondHer2 3D hotspot recovery benchmark','version':'v0.1',
              'scope':'one previously explored specimen, fixed ROI, only internal heldout-Z proxy',
              'primary_group':'primary_retained_clean',
              'secondary_group':'all_excluded',
              'target':'frozen middle occupied Z (one target plane per pseudo-instance, NOT same global tissue plane)',
              'reference':'top/bottom q of neighborhood-smoothed signed within-group percentile-rank discordance of HELDOUT targets',
              'reference_status':'proxy spatial readout, NOT ground truth full-3D HER2 hotspot or manual biological annotation',
              'predictors':'nearest 1, nearest 2 and all other Z (weighted by per-section instance pixel counts) excluding target Z',
              'spatial_context':'k nearest other centroid objects + self, equal-weight smoothing, one graph using XYZ and one XY-projection sensitivity control',
              'xy_caveat':'XY projection graph ignores z but is NOT a single histological 2D plane; no strict 2D-slide-comparable recovery is claimed',
              'clone_discordance':'rank(HER2_3B5)-rank(HER2_29D8); ranks computed separately for target and each predictor within the prespecified group',
              'hotspots':'top and bottom q fractions (by signed difference) each; not HER2-positive labels',
              'scale_sensitivity':'additional fixed k=6, 12, 24 neighbor results, NO tuning based on test hotspot labels',
              'metrics':'top/bottom macro-F1, macro-Jaccard, sign-specific recall; stable source-ID ties',
              'bootstrap':'fixed hotspot labels, resample spatial XYZ blocks with replacement; CI conditional on threshold/reference map (no rethresholding)',
              'negative_control':'permute non-target predictor differences across fixed centroids, reconstruct hotspot; descriptive within one specimen',
              'all_planes_pixel_budget':'more input pixels than nearest1/nearest2; not a matched acquisition budget',
              'preexplored_data':'multiple prior comparisons performed on same specimen, not fresh preregistered test',
              'no_leakage_check':'all predictor HER2 intensities exclude held-out reference plane; segmentation came from original 3D pseudo-instance masks with upstream provenance still to verify',
              'not_supported':'clinical HER2 hotspot status, cross-specimen generalization, true 2D slide vs 3D whole-volume superiority',
              'params':{'k':args.k,'q_each_sign':args.q,'bootstrap':args.bootstrap,
                        'permutations':args.permutations,'spatial_block_um':[args.block_xy,args.block_xy,args.block_z],
                        'seed':SEED},
              'source_sha256':{str(p.relative_to(root)):sha256(p) for p in paths.values()}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf8')
    with (out/'RESULTS_README.md').open('w',encoding='utf8') as f:
        f.write('# SecondHer2 heldout-plane spatial hotspot recovery\n\n')
        f.write('**CAUTION:** these are reference-proxy hotspots from a heldout middle plane, not an independent whole-3D biological hotspot truth. One previously explored specimen.\n\n')
        f.write('Hotspots are extreme signed antibody-rank differences, not antibody expression-positive regions.\n\n')
        f.write('## Results — XYZ proxy reference\n\n')
        f.write('```text\n'+result[result.reference_geometry=='XYZ'][['group','method','n_objects','hotspot_macro_f1','hotspot_macro_jaccard','hotspot_pos_recall','hotspot_neg_recall']].round(5).to_string(index=False)+'\n```')
        f.write('\n\n## Block-bootstrap conditional difference (macro hotspot recall)\n\n')
        f.write('```text\n'+boot[boot.reference_geometry=='XYZ'].round(5).to_string(index=False)+'\n```')
        f.write('\n\n## Negative-control label shuffles\n\n')
        f.write('```text\n'+null.round(5).to_string(index=False)+'\n```')
        f.write('\n\n## Interpretation\n\n')
        f.write('The XY-projection sensitivity reference should be inspected alongside XYZ results. A favorable XYZ graph under an XYZ-defined proxy is not independently proven 3D added value. Frozen middle-plane target is a noisy single-plane reference, not the true 3D hotspot. Thresholds are not tuned to maximize performance. No biological replication.\n')
    print('[5/5] Completed',flush=True)
    print('\n=== HELDOUT 3D HOTSPOT RECOVERY: XYZ REFERENCE ===')
    print(result[result.reference_geometry=='XYZ'][['group','method','n_objects','hotspot_macro_f1','hotspot_macro_jaccard','hotspot_pos_recall','hotspot_neg_recall']].round(5).to_string(index=False))
    print('\n=== PRIMARY PAIRED SPATIAL-BLOCK COMPARISONS (XYZ REFERENCE) ===')
    print(boot[(boot.group=='primary_retained_clean')&(boot.reference_geometry=='XYZ')].round(5).to_string(index=False))
    print('\n=== PROJECTION REFERENCE SENSITIVITY ===')
    print(result[(result.group=='primary_retained_clean')&(result.reference_geometry=='XY_projection')][['method','hotspot_macro_f1']].round(5).to_string(index=False))
    print('\n=== NEIGHBORHOOD SCALE SENSITIVITY ===')
    print(sensitivity[sensitivity.group=='primary_retained_clean'].round(5).to_string(index=False))
    print('\n=== SHUFFLE NEGATIVE CONTROL ===')
    print(null.round(5).to_string(index=False))
    print('\nSUCCESS:',out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path('.'))
    ap.add_argument('--out',type=Path,default=Path('runs/secondher2_3d_hotspot_recovery_v0_1'))
    ap.add_argument('--k',type=int,default=12)
    ap.add_argument('--q',type=float,default=.10)
    ap.add_argument('--bootstrap',type=int,default=500)
    ap.add_argument('--permutations',type=int,default=199)
    ap.add_argument('--block-xy',type=float,default=32)
    ap.add_argument('--block-z',type=float,default=24)
    args=ap.parse_args()
    if args.permutations<99 or args.block_xy<=0 or args.block_z<=0:
        ap.error('permutations must be >=99, block dimensions positive')
    run(args)

if __name__=='__main__':
    main()
