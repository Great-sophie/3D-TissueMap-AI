#!/usr/bin/env python3
"""SecondHer2 reliability of HER2 nearest-2 measurements, exploratory v0.1.

No raw TIFF reading. Frozen heldout middle plane is ONLY a supervised training
label in training spatial stripes and an evaluation label in test stripes.
Feature computation and all standardization/imputation use non-target planes.
Geometry sensitivity uses frozen 12E boundaries from other Z planes only.
Fixed 3 X-stripe spatial holdout + 12um guard; no hyperparameter tuning on tests.
An optional hotspot endpoint reuses heldout-Z proxy reference labels strictly for
secondary evaluation, not for feature engineering or model fitting.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

MARKERS = ('HER2_3B5', 'HER2_29D8')
SEED = 20261019
FILES = {
    'per_z':'runs/secondher2_2d_vs_3d_heldout_v0_1/per_object_per_section_her2.csv',
    'heldout':'runs/secondher2_2d_vs_3d_heldout_v0_1/heldout_2d_vs_multiplane_predictions.csv',
    'coords':'runs/secondher2_3d_spatial_discordance_v0_1/3d_her2_object_spatial_coordinates.csv',
    'boundary':'runs/workflow12_secondher2_fixed_boundary_transfer_all92_v0_1/candidate_boundary_metrics.csv',
    'hotspot':'runs/secondher2_3d_hotspot_recovery_v0_1/per_object_hotspot_labels.csv',
}

def istrue(s):
    return s.astype(str).str.lower().eq('true')

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def required(df, cols, name):
    missing=set(cols)-set(df.columns)
    if missing:raise ValueError(f'{name} missing: {sorted(missing)}')

def geometry_table(raw, eligible_targets):
    """Cache per-source per-z geometric agreement; excludes heldout Z by design."""
    req=['source_pseudo_id','z','seed_id','method','segment_area_px','n_core_px','nucleus_fragment_px']
    required(raw,req,'boundary')
    if raw.duplicated(['method','z','seed_id']).any():
        raise ValueError('Duplicate 12E method, z, seed')
    methods=set(raw.method)
    if not {'nuclear_voronoi','cadherin_assisted'}.issubset(methods):
        raise ValueError('Missing required segmentation methods')
    a=raw[raw.method=='cadherin_assisted'][req].copy()
    v=raw[raw.method=='nuclear_voronoi'][req].copy()
    a=a.merge(v, on=['source_pseudo_id','z','seed_id'], validate='one_to_one', suffixes=('_cad','_vor'))
    if len(a)*2 != len(raw):
        raise ValueError('Boundary pair mismatch')
    a['area_ratio']=a.segment_area_px_cad/a.segment_area_px_vor
    a['area_deviation']=np.abs(np.log(a.area_ratio.clip(lower=1e-8)))
    a['shrunk']=a.area_ratio<=.25
    a['coreless_cad']=a.n_core_px_cad==0
    a['fragment_px']=a.nucleus_fragment_px_vor
    # Crucial: group by source ID and exclude this object's target slice, NOT just global Z.
    zmap=eligible_targets[['source_pseudo_id','target_z']]
    a=a.merge(zmap,on='source_pseudo_id',validate='many_to_one')
    a=a.loc[a.z!=a.target_z]
    rows=a.groupby('source_pseudo_id').agg(
        boundary_n_nontarget_fragments=('area_ratio','size'),
        boundary_median_log_area_shift=('area_deviation','median'),
        boundary_fraction_shrunk=('shrunk','mean'),
        boundary_fraction_coreless=('coreless_cad','mean'),
        boundary_median_nucleus_px=('fragment_px','median'),
    ).reset_index()
    return rows

def build_features(perz, held, coords, boundary):
    required(perz, ['source_pseudo_id','z','n_px', *[x+'_sum' for x in MARKERS]], 'per_z')
    required(held, ['source_pseudo_id','target_z','primary_eligible','retained', *[x+'_heldout_mean' for x in MARKERS]], 'heldout')
    required(coords,['source_pseudo_id','x_um','y_um','z_um'],'coords')
    if perz.duplicated(['source_pseudo_id','z']).any() or not held.source_pseudo_id.is_unique or not coords.source_pseudo_id.is_unique:
        raise ValueError('Duplicate object keys')
    if (perz.n_px<=0).any():raise ValueError('Invalid n_px')
    held=held.loc[istrue(held.primary_eligible)].copy()
    if len(held)<100:raise ValueError('Insufficient primary objects')
    index=held.set_index('source_pseudo_id')
    data=[]
    for sid,g in perz.loc[perz.source_pseudo_id.isin(index.index)].groupby('source_pseudo_id',sort=True):
        g=g.sort_values('z')
        z=g.z.to_numpy(dtype=int)
        target=int(index.loc[sid,'target_z'])
        if len(g)<4 or np.any(np.diff(z)!=1) or target!=z[len(z)//2]:
            raise ValueError(f'Invalid frozen middle target for source {sid}')
        hold=g[g.z==target]; other=g[g.z!=target].copy()
        other=other.iloc[np.lexsort((other.z.to_numpy(),np.abs(other.z.to_numpy()-target)))]
        near=other.iloc[:2]
        row={'source_pseudo_id':int(sid),'target_z':target,'n_sections':int(len(g)),
             'log_total_other_px':np.log1p(other.n_px.sum()),
             'log_near2_px':np.log1p(near.n_px.sum()),
             'near2_area_imbalance':abs(np.log(near.n_px.iloc[0]/near.n_px.iloc[1])),
             'non_target_plane_area_cv':float(other.n_px.std(ddof=0)/other.n_px.mean()),
             'n_observed_non_target':len(other)}
        for m in MARKERS:
            c=m+'_sum'
            val=other[c].to_numpy(float)/other.n_px.to_numpy(float)
            if not np.isfinite(val).all() or (val<0).any():raise ValueError('Nonfinite or negative intensity')
            tr=float(hold[c].iloc[0]/hold.n_px.iloc[0])
            check=float(index.loc[sid,m+'_heldout_mean'])
            if not np.isclose(tr,check,rtol=1e-7,atol=1e-7):raise RuntimeError(f'Frozen target mismatch for {sid}/{m}')
            n2=float(near[c].sum()/near.n_px.sum())
            row[m+'_error_nearest2']=abs(np.log1p(n2)-np.log1p(tr))
            row[m+'_variation2']=abs(np.log1p(val[0])-np.log1p(val[1]))
            row[m+'_log_nearest2_mean']=float(np.log1p(n2))
            row[m+'_other_mad_log']=float(np.median(abs(np.log1p(val)-np.median(np.log1p(val)))))
            row[m+'_near_vs_other_shift']=float(abs(np.log1p(n2)-np.log1p(float(other[c].sum()/other.n_px.sum()))))
        data.append(row)
    d=pd.DataFrame(data)
    if len(d)!=len(held):raise RuntimeError(f'Primary cohort mismatch: {len(d)} vs {len(held)}')
    d=d.merge(coords[['source_pseudo_id','x_um','y_um','z_um']],on='source_pseudo_id',validate='one_to_one',how='left')
    if d[['x_um','y_um','z_um']].isna().any().any():raise ValueError('Missing object coordinates')
    geom=geometry_table(boundary,held[['source_pseudo_id','target_z']])
    d=d.merge(geom,on='source_pseudo_id',how='left',validate='one_to_one')
    if not (d[MARKERS[0]+'_error_nearest2']>=0).all():raise ValueError('Bad errors')
    return d

SHARED=['n_sections','log_total_other_px','log_near2_px','near2_area_imbalance','non_target_plane_area_cv','n_observed_non_target']
GEOM=['boundary_n_nontarget_fragments','boundary_median_log_area_shift','boundary_fraction_shrunk','boundary_fraction_coreless','boundary_median_nucleus_px']

def feature_sets(marker):
    channel=[f'{marker}_{p}' for p in ('variation2','log_nearest2_mean','other_mad_log','near_vs_other_shift')]
    other=MARKERS[1] if marker==MARKERS[0] else MARKERS[0]
    extra=[other+'_variation2', other+'_log_nearest2_mean']
    return {'geometry_only':SHARED+GEOM, 'signal_geometry':SHARED+GEOM+channel+extra}

def split_folds(d,guard):
    x=d.x_um.to_numpy(float)
    cuts=np.quantile(x,[1/3,2/3])
    if cuts[1]-cuts[0]<=2*guard:raise ValueError('Not enough X extent for guarded 3 folds')
    assignment=np.searchsorted(cuts,x,side='left')
    for i in range(3):
        test=np.flatnonzero(assignment==i)
        if i==0:train=np.flatnonzero(x>cuts[0]+guard)
        elif i==1:train=np.flatnonzero((x<cuts[0]-guard)|(x>cuts[1]+guard))
        else:train=np.flatnonzero(x<cuts[1]-guard)
        if len(train)<50 or len(test)<40:raise ValueError('Insufficient fold size')
        min_dist=float(np.min(np.abs(x[train,None]-x[None,test])))
        if min_dist<guard-1e-6:raise RuntimeError('Guard violation')
        yield i,train,test,min_dist

def mk_model():
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),
                         LogisticRegression(C=1.0,max_iter=1000,random_state=SEED))

def safe_auc(y,p):
    return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan

def score(d,marker,method):
    y=d['is_high_error'].to_numpy(int)
    risk=d[method+'_risk'].to_numpy(float)
    err=d[marker+'_error_nearest2'].to_numpy(float)
    ix=np.lexsort((d.source_pseudo_id.to_numpy(),risk))
    half=ix[:len(ix)//2]
    return {'marker':marker,'method':method,'n_objects':len(y),
            'high_error_fraction':float(y.mean()),'auroc':safe_auc(y,risk),
            'auprc':float(average_precision_score(y,risk)),
            'brier':float(brier_score_loss(y,np.clip(risk,0,1))) if method!='variation_heuristic' else np.nan,
            'overall_mean_error':float(err.mean()),
            'low_risk_50pct_mean_error':float(err[half].mean()),
            'low_risk_50pct_high_error_fraction':float(y[half].mean()),
            'selective_risk_reduction_fraction':float(1-err[half].mean()/err.mean()),
            'high_risk_50pct_mean_error':float(err[ix[len(ix)//2:]].mean())}

def spatial_boot_diff(d,marker,B,seed,block_xy=32.,block_z=24.):
    rng=np.random.default_rng(seed)
    block=(np.floor(d.x_um.to_numpy()/block_xy).astype(int).astype(str)+':'+
           np.floor(d.y_um.to_numpy()/block_xy).astype(int).astype(str)+':'+
           np.floor(d.z_um.to_numpy()/block_z).astype(int).astype(str))
    codes,names=pd.factorize(block,sort=True)
    ids=[np.flatnonzero(codes==i) for i in range(len(names))]
    if len(ids)<15:raise RuntimeError('Fewer than 15 spatial blocks')
    truth=d.is_high_error.to_numpy(int)
    full=d.signal_geometry_risk.to_numpy(float)
    heuristic=d.variation_heuristic_risk.to_numpy(float)
    geo=d.geometry_only_risk.to_numpy(float)
    vals=[]
    for _ in range(B):
        idx=np.concatenate([ids[i] for i in rng.integers(0,len(ids),size=len(ids))])
        yy=truth[idx]
        if len(np.unique(yy))<2:continue
        vals.append((roc_auc_score(yy,full[idx])-roc_auc_score(yy,heuristic[idx]),
                     roc_auc_score(yy,full[idx])-roc_auc_score(yy,geo[idx])))
    arr=np.asarray(vals)
    if len(arr)<.95*B:raise RuntimeError('Degenerate block bootstrap draws')
    return {'marker':marker,'n_spatial_blocks':len(names),'n_bootstrap':len(arr),
            'delta_auroc_full_minus_variation':safe_auc(truth,full)-safe_auc(truth,heuristic),
            'delta_variation_p025':float(np.quantile(arr[:,0],.025)),
            'delta_variation_p975':float(np.quantile(arr[:,0],.975)),
            'delta_auroc_full_minus_geometry':safe_auc(truth,full)-safe_auc(truth,geo),
            'delta_geometry_p025':float(np.quantile(arr[:,1],.025)),
            'delta_geometry_p975':float(np.quantile(arr[:,1],.975)),
            'caveat':'internal spatial block resampling, one specimen, test labels previously explored'}

def hotspot_analysis(d, hotspot):
    required(hotspot,['source_pseudo_id','group','reference_hotspot_XYZ','pred_hotspot_XYZ_nearest2'],'hotspot')
    h=hotspot[hotspot.group=='primary_retained_clean'].copy()
    if not h.source_pseudo_id.is_unique:raise ValueError('Duplicate hotspot ID')
    p=d.merge(h[['source_pseudo_id','reference_hotspot_XYZ','pred_hotspot_XYZ_nearest2']],
              on='source_pseudo_id',validate='one_to_one',how='left')
    if p.reference_hotspot_XYZ.isna().any():raise RuntimeError('Hotspot primary labels missing')
    # Not used for training, only post hoc secondary check for predicted hotspots.
    predicted=p[p.pred_hotspot_XYZ_nearest2!=0].copy()
    if len(predicted)<20:raise RuntimeError('Too few predicted hotspots')
    for m in MARKERS:
        if m+'_signal_geometry_risk' not in predicted:raise RuntimeError('Missing risk')
    ranks=np.vstack([p[m+'_signal_geometry_risk'].to_numpy(float) for m in MARKERS]).T
    # Normalize two channels on fixed *train-independent* probability scale: both 0..1.
    p['combined_risk']=ranks.mean(axis=1)
    pred=p.loc[p.pred_hotspot_XYZ_nearest2!=0].copy()
    pred['correct_sign']=(pred.pred_hotspot_XYZ_nearest2==pred.reference_hotspot_XYZ).astype(int)
    rows=[]
    for name in ('combined_risk','HER2_3B5_signal_geometry_risk','HER2_29D8_signal_geometry_risk'):
        v=pred[name].to_numpy(float)
        idx=np.lexsort((pred.source_pseudo_id.to_numpy(),v))
        n=max(1,len(pred)//2)
        low=pred.iloc[idx[:n]]
        rows.append({'risk_source':name,'n_predicted_hotspots':len(pred),
                     'precision_all_predicted':float(pred.correct_sign.mean()),
                     'precision_low_risk_half':float(low.correct_sign.mean()),
                     'precision_gain':float(low.correct_sign.mean()-pred.correct_sign.mean()),
                     'kept_fraction':float(n/len(pred)),
                     'caveat':'secondary heldout-Z hotspot PROXY; not independent biological truth'})
    return pd.DataFrame(rows),p

def figure(d,out):
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,m in zip(axes,MARKERS):
        sub=d[d.marker==m].copy()
        for tag,label in [('signal_geometry','Signal + geometry model'),('geometry_only','Geometry only'),('variation_heuristic','Nearest-2 disagreement')]:
            x=sub[tag+'_risk'].to_numpy(float)
            y=sub.is_high_error.to_numpy(int)
            bins=pd.qcut(pd.Series(x).rank(method='first'),q=5,labels=False)
            rate=pd.DataFrame({'bin':bins,'y':y}).groupby('bin').y.mean().to_numpy()
            ax.plot(np.arange(1,len(rate)+1),rate,marker='o',lw=2,label=label)
        ax.set(xlabel='Predicted-risk quintile (within model)',xticks=[1,2,3,4,5],title=m)
        ax.grid(axis='y',alpha=.17)
    axes[0].set_ylabel('Observed high-error fraction')
    axes[1].legend(loc='best',fontsize=8)
    fig.suptitle('Spatially held-out measurement reliability (frozen middle Z)',fontsize=13)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(out/f'Figure_R1_reliability_risk_strata.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)

def run(args):
    root=args.project_root.resolve()
    path={k:root/v for k,v in FILES.items()}
    for k,p in path.items():
        if not p.is_file():raise FileNotFoundError(f'Required prior result missing: {p}')
    out=(root/args.out).resolve()
    if out.exists():raise FileExistsError(f'Refusing to overwrite: {out}')
    if args.bootstrap<100 or args.guard<=0:raise ValueError('Bootstrap>=100 and positive guard required')
    print('[1/5] Load frozen per-plane HER2, pseudo-instance geometry and hotspot labels',flush=True)
    dfs={k:pd.read_csv(v) for k,v in path.items()}
    feat=build_features(dfs['per_z'],dfs['heldout'],dfs['coords'],dfs['boundary'])
    if not feat.source_pseudo_id.is_unique:raise RuntimeError('Duplicate features')
    print(f'Primary retained eligible objects: {len(feat)}',flush=True)
    print('[2/5] Spatial X-stripe CV with 12um guard, train-only high-error threshold',flush=True)
    allpred=[];foldrows=[]
    for fold, train, test, mindist in split_folds(feat,args.guard):
        train_df=feat.iloc[train];test_df=feat.iloc[test]
        for marker in MARKERS:
            key=marker+'_error_nearest2'
            threshold=float(train_df[key].quantile(.75))
            ytrain=(train_df[key].to_numpy(float)>threshold).astype(int)
            ytest=(test_df[key].to_numpy(float)>threshold).astype(int)
            if ytrain.sum()<20 or len(ytrain)-ytrain.sum()<20:raise ValueError('High-error classes insufficient')
            row=test_df[['source_pseudo_id','target_z','x_um','y_um','z_um']].copy()
            row['marker']=marker;row['fold']=fold;row['is_high_error']=ytest
            row['train_q75_error_threshold']=threshold
            row[key]=test_df[key].to_numpy(float)
            row['variation_heuristic_risk']=test_df[marker+'_variation2'].to_numpy(float)
            cols=feature_sets(marker)
            for tag,fields in cols.items():
                mdl=mk_model()
                mdl.fit(train_df[fields],ytrain)
                row[tag+'_risk']=mdl.predict_proba(test_df[fields])[:,1]
            allpred.append(row)
            foldrows.append({'fold':fold,'marker':marker,'n_train':len(train),'n_test':len(test),
                             'train_error_q75':threshold,'high_error_train_fraction':float(ytrain.mean()),
                             'high_error_test_fraction':float(ytest.mean()),
                             'min_x_train_test_um':mindist,'features_signal_geometry':','.join(cols['signal_geometry'])})
    d=pd.concat(allpred,ignore_index=True)
    if d.duplicated(['source_pseudo_id','marker']).any():raise RuntimeError('Duplicate spatial heldout predictions')
    for m in MARKERS:
        if len(d[d.marker==m])!=len(feat):raise RuntimeError('Missing test objects')
    print('[3/5] Compare reliability models with geometry-only and no-training disagreement baseline',flush=True)
    results=pd.DataFrame([score(d[d.marker==m],m,tag)
                          for m in MARKERS for tag in ('variation_heuristic','geometry_only','signal_geometry')])
    boots=pd.DataFrame([spatial_boot_diff(d[d.marker==m],m,args.bootstrap,SEED+i)
                        for i,m in enumerate(MARKERS)])
    print('[4/5] Secondary: does learned measurement risk enrich correct hotspot calls?',flush=True)
    wide=d.pivot(index='source_pseudo_id',columns='marker',values='signal_geometry_risk')
    wide.columns=[m+'_signal_geometry_risk' for m in wide.columns]
    h, hotlabels=hotspot_analysis(wide.reset_index(),dfs['hotspot'])
    out.mkdir(parents=True)
    (out/'figures').mkdir()
    d.to_csv(out/'spatial_heldout_reliability_predictions.csv',index=False)
    results.to_csv(out/'reliability_model_comparison.csv',index=False)
    boots.to_csv(out/'reliability_block_bootstrap.csv',index=False)
    pd.DataFrame(foldrows).to_csv(out/'train_only_fold_thresholds.csv',index=False)
    h.to_csv(out/'hotspot_reliability_transfer.csv',index=False)
    hotlabels.to_csv(out/'hotspot_reliability_per_object.csv',index=False)
    figure(d,out/'figures')
    protocol={'name':'SecondHer2 measurement reliability modeling','status':'exploratory development specimen',
              'fixed_target':'middle Z slice of >=4-contiguous-section pseudo-instance',
              'endpoint':'train fold 75th percentile of abs(log1p(nearest2)-log1p(heldout_Z))',
              'training':'logistic C=1; impute and scale within training fold; 3 guarded X spatial stripes',
              'features':'strictly non-target per-Z protein and 12E geometry rows; target Z excluded from all features',
              'secondary_hotspot':'no hotspot labels used in training; secondary correctness of XYZ nearest2 predicted extreme 10% vs heldout-Z XYZ proxy map',
              'causal_limitations':['single pre-explored specimen','pseudo-instance mask not manual truth',
                    'hotspot graph and quantile ranks computed on full cohort in earlier benchmark',
                    'risk model may or may not improve upon simpler confidence proxies',
                    'no cross-specimen generalization or clinical reliability validation'],
              'folds':3,'guard_um':args.guard,'bootstrap':args.bootstrap,'seed':SEED,
              'source_sha256':{k:digest(p) for k,p in path.items()}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf8')
    with (out/'RESULTS_README.md').open('w',encoding='utf8') as f:
        f.write('# SecondHer2 Reliability Modeling — exploratory single-specimen\n\n')
        f.write('Frozen heldout middle Z signal is the supervised error label, never a predictor; earlier hotspot proxy labels only secondary evaluation.\n\n')
        f.write('## Spatial holdout reliability\n\n```text\n'+results.round(5).to_string(index=False)+'\n```\n\n')
        f.write('## Block uncertainty\n\n```text\n'+boots.round(5).to_string(index=False)+'\n```\n\n')
        f.write('## Hotspot correctness transfer\n\n```text\n'+h.round(5).to_string(index=False)+'\n```\n\n')
        f.write('Not evidence of biological truth, independent specimens, or clinical classification.\n')
    print('\n=== SPATIAL HELDOUT RELIABILITY ===')
    print(results[['marker','method','n_objects','auroc','auprc','overall_mean_error','low_risk_50pct_mean_error','selective_risk_reduction_fraction']].round(5).to_string(index=False))
    print('\n=== RELIABILITY BOOTSTRAP (DELTA AUROC) ===')
    print(boots.drop(columns='caveat').round(5).to_string(index=False))
    print('\n=== SECONDARY HOTSPOT RELIABILITY TRANSFER ===')
    print(h.drop(columns='caveat').round(5).to_string(index=False))
    print('\nSUCCESS:',out)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,default=Path('runs/secondher2_reliability_model_v0_1'))
    p.add_argument('--bootstrap',type=int,default=499)
    p.add_argument('--guard',type=float,default=12.)
    run(p.parse_args())

if __name__=='__main__':main()
