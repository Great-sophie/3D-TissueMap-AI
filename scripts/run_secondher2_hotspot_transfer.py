#!/usr/bin/env python3
"""Frozen reliability-to-hotspot transfer evaluation for SecondHer2 IMC.

All risks are OOF from PREVIOUSLY trained 3-fold spatial-X reliability model.
Heldout middle-Z proxy hotspot labels are used ONLY for evaluation, not model
training or risk calibration. Prespecified primary: mean of two signal-geometry
risks, 50% predicted-hotspot coverage, sign-balanced. Supports descriptive
coverage curves, reconstruction of a no-training hotspot-margin baseline,
spatial-block resampling and sign-balanced random-selection controls.

Exploratory SINGLE SPECIMEN; noisy proxy reference and neighbor graph derived
from this entire specimen; no biological ground truth or external validation.
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

SEED = 20261019
HOTSPOT = Path('runs/secondher2_3d_hotspot_recovery_v0_1')
RELIABILITY = Path('runs/secondher2_reliability_model_v0_1')
DEFAULT_OUTPUT = Path('runs/secondher2_reliability_hotspot_transfer_v0_1')
MARKERS = ('HER2_3B5','HER2_29D8')
COVERAGES = (0.25,0.50,0.75,1.00)


def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def required(df,keys,desc):
    missing=set(keys)-set(df.columns)
    if missing:raise ValueError(f'{desc}: missing columns {sorted(missing)}')


def build(path_hot,path_rel):
    h=pd.read_csv(path_hot)
    r=pd.read_csv(path_rel)
    keys=['source_pseudo_id','group','x_um','y_um','z_um','signed_rank_difference_nearest2',
          'reference_hotspot_XYZ','pred_hotspot_XYZ_nearest2']
    required(h,keys,'hotspot input')
    required(r,['source_pseudo_id','marker','fold','x_um','y_um','z_um','signal_geometry_risk',
                'geometry_only_risk','variation_heuristic_risk'],'OOF reliability input')
    h=h[h.group=='primary_retained_clean'].copy()
    if not h.source_pseudo_id.is_unique:raise ValueError('Duplicate hotspot source IDs')
    if r.duplicated(['source_pseudo_id','marker']).any():raise ValueError('Duplicate risk source/marker')
    if set(r.marker)!=set(MARKERS):raise ValueError('Unexpected marker labels')
    if h.source_pseudo_id.isna().any():raise ValueError('Missing IDs')
    if len(h)<100:raise ValueError('Too few primary objects')
    raw_ids=set(h.source_pseudo_id.astype(int))
    for marker in MARKERS:
        ids=set(r.loc[r.marker==marker,'source_pseudo_id'].astype(int))
        if ids!=raw_ids:raise ValueError(f'OOF risk/hotspot ID mismatch {marker}; missing {len(raw_ids-ids)} extra {len(ids-raw_ids)}')
    # OOF folds and coordinates must agree across both antibody channels
    piv=r.pivot(index='source_pseudo_id',columns='marker',values=['signal_geometry_risk','geometry_only_risk','variation_heuristic_risk'])
    piv.columns=[f'{marker}__{feature}' for feature,marker in piv.columns]
    p=piv.reset_index()
    folds=r.groupby('source_pseudo_id').fold.nunique()
    if not (folds==1).all():raise ValueError('OOF marker fold mismatch')
    p['fold']=p.source_pseudo_id.map(r.groupby('source_pseudo_id').fold.first())
    d=h.merge(p,on='source_pseudo_id',how='inner',validate='one_to_one').sort_values('source_pseudo_id').reset_index(drop=True)
    if len(d)!=len(h):raise ValueError('Merged primary cohort mismatch')
    for v in ('x_um','y_um','z_um'):
        chk=d[['source_pseudo_id',v]].merge(r[r.marker==MARKERS[0]][['source_pseudo_id',v]],on='source_pseudo_id',suffixes=('_h','_r'))
        if not np.allclose(chk[v+'_h'],chk[v+'_r'],atol=1e-7):raise ValueError(f'Coordinate mismatch: {v}')
    riskcols=[x for x in d if x.endswith('__signal_geometry_risk') or x.endswith('__geometry_only_risk') or x.endswith('__variation_heuristic_risk')]
    if not np.isfinite(d[riskcols].to_numpy(float)).all():raise ValueError('Non-finite OOF risks')
    if not np.isin(d.reference_hotspot_XYZ,[-1,0,1]).all() or not np.isin(d.pred_hotspot_XYZ_nearest2,[-1,0,1]).all():
        raise ValueError('Hotspot class values must be -1/0/1')
    nh=(d.pred_hotspot_XYZ_nearest2!=0).sum()
    if nh<30:raise ValueError('Not enough predicted hotspots')
    d['correct_hotspot_sign']=(d.pred_hotspot_XYZ_nearest2==d.reference_hotspot_XYZ).astype(int)
    d['risk_combined_mean']=(d['HER2_3B5__signal_geometry_risk']+d['HER2_29D8__signal_geometry_risk'])/2
    d['risk_combined_max']=np.maximum(d['HER2_3B5__signal_geometry_risk'],d['HER2_29D8__signal_geometry_risk'])
    # Geometry-only combined scores on common probability scale.
    d['risk_geometry_mean']=(d['HER2_3B5__geometry_only_risk']+d['HER2_29D8__geometry_only_risk'])/2
    for m in MARKERS:
        d['risk_'+m+'_full']=d[m+'__signal_geometry_risk']
        d['risk_'+m+'_variation']=d[m+'__variation_heuristic_risk']
    return d


def reference_margin(d,k,q):
    """Rebuild frozen nearest2 XYZ hotspot score and exact top/bottom classes.

    Margin is distance from nearest2 hot-tail threshold, NOT target labels.
    Risk = -margin, so low risk denotes a prediction deeper within a tail.
    """
    xyz=d[['x_um','y_um','z_um']].to_numpy(float)
    ids=d.source_pseudo_id.to_numpy(np.int64)
    n=len(d)
    if k+2>=n:raise ValueError('Neighborhood k too large')
    _,idx=cKDTree(xyz).query(xyz,k=k+2)
    near=np.empty((n,k),dtype=int)
    for i in range(n):
        others=np.asarray(idx[i])[np.asarray(idx[i])!=i]
        if len(others)<k:raise ValueError('Failed to find distinct kNN indices')
        near[i]=others[:k]
    signed=d.signed_rank_difference_nearest2.to_numpy(float)
    if not np.isfinite(signed).all():raise ValueError('Non-finite nearest2 signed rank difference')
    smooth=(signed+np.sum(signed[near],axis=1))/(k+1)
    nh=int(np.ceil(q*n))
    ix=np.lexsort((ids,smooth))
    pred=np.zeros(n,dtype=np.int8)
    pred[ix[:nh]]=-1;pred[ix[-nh:]]=1
    saved=d.pred_hotspot_XYZ_nearest2.to_numpy(dtype=np.int8)
    if not np.array_equal(pred,saved):
        mismatches=int(np.sum(pred!=saved))
        raise ValueError(f'Nearest2 class reproduction failed ({mismatches} objects). Check k/q in frozen hotspot protocol.')
    cutoff_neg=np.max(smooth[ix[:nh]])
    cutoff_pos=np.min(smooth[ix[-nh:]])
    margin=np.where(pred==1,smooth-cutoff_pos,np.where(pred==-1,cutoff_neg-smooth,np.nan))
    if np.any(margin[pred!=0] < -1e-10):raise ValueError('Invalid confidence margin')
    return -margin


def sign_balanced_indices(d,score,coverage):
    """Select low-risk predictions independently within each predicted sign."""
    chosen=[]
    for sgn in (-1,1):
        idx=np.flatnonzero(d.pred_hotspot_XYZ_nearest2.to_numpy(int)==sgn)
        if not len(idx):raise ValueError('Missing predicted hotspot sign')
        n=max(1,int(round(coverage*len(idx))))
        order=np.lexsort((d.source_pseudo_id.to_numpy()[idx],score[idx]))
        chosen.extend(idx[order[:n]])
    return np.asarray(chosen,dtype=int)


def precision(d,score,coverage):
    ix=sign_balanced_indices(d,score,coverage)
    return float(d.correct_hotspot_sign.to_numpy(float)[ix].mean()),len(ix)


def random_precision(d,coverage,rng):
    ix=[]
    sign=d.pred_hotspot_XYZ_nearest2.to_numpy(int)
    for sgn in (-1,1):
        src=np.flatnonzero(sign==sgn)
        n=max(1,int(round(len(src)*coverage)))
        ix.extend(rng.choice(src,size=n,replace=False))
    return float(d.correct_hotspot_sign.to_numpy(float)[ix].mean())


def spatial_block_indices(d,bxy,bz):
    code=(np.floor(d.x_um.to_numpy()/bxy).astype(int).astype(str)+':'+
          np.floor(d.y_um.to_numpy()/bxy).astype(int).astype(str)+':'+
          np.floor(d.z_um.to_numpy()/bz).astype(int).astype(str))
    ix,uniq=pd.factorize(code,sort=True)
    groups=[np.flatnonzero(ix==j) for j in range(len(uniq))]
    return groups,len(uniq)


def do_bootstrap(d,methods,coverage,nboot,seed,bxy,bz):
    """Conditional block resampling, with risk ranking reselected each replicate."""
    groups,nblocks=spatial_block_indices(d,bxy,bz)
    if nblocks<15:raise ValueError('Fewer than 15 hotspot spatial blocks')
    rng=np.random.default_rng(seed)
    draws={m:[] for m in methods}
    for _ in range(nboot):
        ix=np.concatenate([groups[j] for j in rng.integers(0,nblocks,size=nblocks)])
        dd=d.iloc[ix].reset_index(drop=True)
        if not set((-1,1)).issubset(set(dd.pred_hotspot_XYZ_nearest2)):
            continue
        baseline=float(dd.correct_hotspot_sign.mean())
        margin_p,_=precision(dd,dd['risk_hotspot_margin'].to_numpy(float),coverage)
        for method in methods:
            p,_=precision(dd,dd[method].to_numpy(float),coverage)
            draws[method].append((p-baseline,p-margin_p,p))
    out={m:np.asarray(v,dtype=float) for m,v in draws.items()}
    if any(len(v)<nboot*.95 for v in out.values()):raise ValueError('Too many degenerate block bootstrap draws')
    return out,nblocks


def run(args):
    root=args.project_root.expanduser().resolve()
    prior_hot=root/HOTSPOT
    prior_rel=root/RELIABILITY
    out=(root/args.out).resolve()
    if out.exists():raise FileExistsError(f'Refusing to overwrite: {out}')
    if args.bootstrap<100 or args.random_permutations<99 or args.block_xy<=0 or args.block_z<=0:
        raise ValueError('Bootstrap>=100, random permutations>=99, spatial blocks positive')
    hotspot_path=prior_hot/'per_object_hotspot_labels.csv'
    risk_path=prior_rel/'spatial_heldout_reliability_predictions.csv'
    hotspot_protocol=prior_hot/'protocol.json'
    risk_protocol=prior_rel/'protocol.json'
    for p in (hotspot_path,risk_path,hotspot_protocol,risk_protocol):
        if not p.is_file():raise FileNotFoundError(f'Missing frozen input: {p}')
    hotproto=json.loads(hotspot_protocol.read_text())
    params=hotproto.get('params',{})
    k=int(params.get('k',12));q=float(params.get('q_each_sign',.10))
    if not (k>=2 and 0<q<=.25):raise ValueError('Unsupported frozen k/q')
    print('[1/5] Read previously frozen OOF risk scores and heldout hotspot proxy labels',flush=True)
    d=build(hotspot_path,risk_path)
    d['risk_hotspot_margin']=reference_margin(d,k,q)
    all_hot=d[d.pred_hotspot_XYZ_nearest2!=0].copy().reset_index(drop=True)
    n_hot=len(all_hot)
    if n_hot<20:raise ValueError('Insufficient candidate hotspots')
    if len(all_hot.source_pseudo_id.unique())!=n_hot:raise RuntimeError('Duplicate candidate hotspots')
    methods=['risk_combined_mean','risk_combined_max','risk_HER2_3B5_full','risk_HER2_29D8_full',
             'risk_geometry_mean','risk_HER2_3B5_variation','risk_HER2_29D8_variation','risk_hotspot_margin']
    for c in methods:
        if not np.isfinite(all_hot[c]).all():raise ValueError(f'Nonfinite risk {c}')
    print(f'  {len(d)} eligible pseudo-objects, {n_hot} predicted hotspots; heldout proxy precision={all_hot.correct_hotspot_sign.mean():.5f}',flush=True)
    print(f'[2/5] Frozen sign-balanced risk-coverage curves (25/50/75/100%)',flush=True)
    rows=[];random_rows=[]
    for ci,coverage in enumerate(COVERAGES):
        rng=np.random.default_rng(SEED+10+ci)
        null=np.array([random_precision(all_hot,coverage,rng) for _ in range(args.random_permutations)])
        random_rows.append({'coverage':coverage,'n_random_draws':len(null),'null_mean':float(null.mean()),
                            'null_p025':float(np.quantile(null,.025)),'null_p975':float(np.quantile(null,.975))})
        for method in methods:
            p,n=precision(all_hot,all_hot[method].to_numpy(float),coverage)
            rows.append({'risk_method':method,'coverage':coverage,'n_kept':n,'n_candidate_hotspots':n_hot,
                         'precision':p,'precision_all':float(all_hot.correct_hotspot_sign.mean()),
                         'gain_vs_all':p-float(all_hot.correct_hotspot_sign.mean()),
                         'p_random_ge_observed':float((1+np.sum(null>=p))/(len(null)+1)),
                         'gain_vs_hotspot_margin':np.nan})
    result=pd.DataFrame(rows)
    for coverage in COVERAGES:
        ref=float(result.loc[(result.risk_method=='risk_hotspot_margin')&(result.coverage==coverage),'precision'].iloc[0])
        result.loc[result.coverage==coverage,'gain_vs_hotspot_margin']=result.loc[result.coverage==coverage,'precision']-ref
    print('[3/5] Conditional XYZ spatial-block uncertainty, reselecting low-risk predicted hotspots in each resample',flush=True)
    primary_methods=['risk_combined_mean','risk_HER2_3B5_full','risk_HER2_29D8_full','risk_geometry_mean','risk_hotspot_margin']
    bt_rows=[]
    for ci,coverage in enumerate((.25,.50,.75)):
        samples,nblocks=do_bootstrap(all_hot,primary_methods,coverage,args.bootstrap,SEED+100+ci,args.block_xy,args.block_z)
        for method,bs in samples.items():
            r=result[(result.risk_method==method)&(result.coverage==coverage)].iloc[0]
            bt_rows.append({'risk_method':method,'coverage':coverage,'n_spatial_blocks':nblocks,
                            'n_bootstrap_valid':len(bs),'observed_gain_vs_all':float(r.gain_vs_all),
                            'boot_gain_vs_all_p025':float(np.quantile(bs[:,0],.025)),
                            'boot_gain_vs_all_p975':float(np.quantile(bs[:,0],.975)),
                            'observed_gain_vs_hotspot_margin':float(r.gain_vs_hotspot_margin),
                            'boot_gain_vs_margin_p025':float(np.quantile(bs[:,1],.025)),
                            'boot_gain_vs_margin_p975':float(np.quantile(bs[:,1],.975)),
                            'boot_selected_precision_p025':float(np.quantile(bs[:,2],.025)),
                            'boot_selected_precision_p975':float(np.quantile(bs[:,2],.975))})
    boots=pd.DataFrame(bt_rows)
    print('[4/5] Create reporting tables, audit manifest and two figures',flush=True)
    out.mkdir(parents=True)
    figdir=out/'figures';figdir.mkdir()
    result.to_csv(out/'risk_coverage_comparison.csv',index=False)
    boots.to_csv(out/'conditional_spatial_block_bootstrap.csv',index=False)
    pd.DataFrame(random_rows).to_csv(out/'sign_balanced_random_selection_null.csv',index=False)
    all_hot[['source_pseudo_id','fold','x_um','y_um','z_um','pred_hotspot_XYZ_nearest2','reference_hotspot_XYZ','correct_hotspot_sign',*methods]].to_csv(out/'candidate_hotspots_frozen_risks.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.7,5.3))
    for m in ['risk_combined_mean','risk_HER2_3B5_full','risk_HER2_29D8_full','risk_geometry_mean','risk_hotspot_margin']:
        sub=result[result.risk_method==m].sort_values('coverage')
        ax.plot(sub.coverage.to_numpy(float)*100,sub.precision.to_numpy(float),marker='o',lw=1.8,label=m.replace('risk_','').replace('_',' '))
    ax.axhline(float(all_hot.correct_hotspot_sign.mean()),linestyle='--',color='black',lw=1.1,label='All hotspots / random expectation')
    ax.set(xlabel='Predicted hotspot coverage retained (%)',ylabel='Precision against heldout-Z proxy',ylim=(0,1),
           title='Frozen OOF reliability -> heldout proxy hotspot correctness')
    ax.grid(alpha=.18)
    ax.legend(loc='best',fontsize=8,ncol=2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(figdir/f'Figure_R2_hotspot_risk_coverage.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)
    b50=boots[boots.coverage==.50].copy()
    fig,ax=plt.subplots(figsize=(8.7,4.9))
    labels=b50.risk_method.str.replace('risk_','',regex=False).str.replace('_',' ',regex=False).tolist()
    xs=np.arange(len(b50))
    y=b50.observed_gain_vs_all.to_numpy(float)
    low=b50.boot_gain_vs_all_p025.to_numpy(float);high=b50.boot_gain_vs_all_p975.to_numpy(float)
    ax.errorbar(xs,y,yerr=[y-low,high-y],fmt='o',capsize=3)
    ax.axhline(0,color='black',lw=1,linestyle='--')
    ax.set(xticks=xs,xticklabels=labels,ylabel='Precision gain vs all predicted hotspots',
           title='50% coverage: exploratory spatial block bootstrap')
    ax.tick_params(axis='x',rotation=20)
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(figdir/f'Figure_R3_hotspot_transfer_block_intervals.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)
    primary=result[(result.risk_method=='risk_combined_mean')&(result.coverage==.50)].iloc[0]
    primary_ci=boots[(boots.risk_method=='risk_combined_mean')&(boots.coverage==.50)].iloc[0]
    decision = ('PROVISIONAL SUPPORT on this proxy only' if
                primary_ci.boot_gain_vs_all_p025>0 and primary_ci.boot_gain_vs_margin_p025>0
                else 'NOT ESTABLISHED: combined-risk improvement lacks positive lower bounds against both baselines')
    manifest={'name':'SecondHer2 reliability-to-hotspot transfer','version':'v0.1','status':'exploratory post-hoc secondary endpoint',
              'unit':'candidate hotspots (pseudo-instances), from one previously explored tissue volume',
              'primary_analysis':'Mean of HER2 3B5 and HER2 29D8 frozen out-of-fold signal+geometry risks; sign-balanced 50% selection',
              'reference':'Fixed heldout middle-Z XYZ kNN (k from frozen hotspot protocol) antibody discordance proxy; NOT biological ground truth',
              'selection':'Within positive and negative predicted hotspot classes separately; rank risks and keep fixed fractions',
              'baseline_margin':'Negative signed distance from nearest2 score to frozen top/bottom q cutoff using only predictor data',
              'uncertainty':'Conditional 3D spatial block bootstrap, risk-based low-risk selection reselected each resample; NOT independent biological CI',
              'random_test':'Sign-balanced random choice of predicted hotspots at matched coverage; previously explored sample, descriptive only',
              'primary_decision':decision,
              'no_additional_training':True,'no_target_feature_leakage':'No heldout hotspot labels or heldout-Z intensity used in risk scoring; supplied frozen OOF probabilities',
              'important_limitation':['Hotspot reference uses original whole-cohort kNN graph and ranks','Risk models and prior methods already explored same specimen',
                                      'Hotspots are antibody signed-rank discordance, not clinically validated HER2 status',
                                      'Marker-specific selected-result wins cannot replace the prespecified combined-risk primary analysis',
                                      'No cross-specimen validation, no patient-level inferential claim'],
              'params':{'coverage':[float(x) for x in COVERAGES],'bootstrap':args.bootstrap,
                        'random_permutations':args.random_permutations,'spatial_block_um':[args.block_xy,args.block_xy,args.block_z],
                        'k':k,'q_each_sign':q,'seed':SEED},
              'source_sha256':{str(p.relative_to(root)):sha256(p) for p in (hotspot_path,risk_path,hotspot_protocol,risk_protocol)}}
    (out/'protocol.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
    with (out/'RESULTS_README.md').open('w',encoding='utf8') as f:
        f.write('# SecondHer2 Risk-to-Hotspot Transfer — exploratory\n\n')
        f.write('**Single previously explored specimen. Frozen OOF measurement-error risk, heldout-Z proxy hotspot correctness only.**\n\n')
        f.write('## Primary: combined risk at 50% sign-balanced coverage\n\n')
        f.write(f'All-candidate precision: {float(primary.precision_all):.5f}; selected precision: {float(primary.precision):.5f}; gain: {float(primary.gain_vs_all):+.5f}.\n\n')
        f.write(f'Block-bootstrap gain vs all: [{primary_ci.boot_gain_vs_all_p025:.5f}, {primary_ci.boot_gain_vs_all_p975:.5f}].\n\n')
        f.write(f'Gain vs marker-independent hotspot margin: {float(primary.gain_vs_hotspot_margin):+.5f}; conditional bootstrap: [{primary_ci.boot_gain_vs_margin_p025:.5f}, {primary_ci.boot_gain_vs_margin_p975:.5f}].\n\n')
        f.write('## Primary decision\n\n'+decision+'\n\n')
        f.write('## All risk sources and coverages\n\n```text\n'+result.round(5).to_string(index=False)+'\n```\n\n')
        f.write('## Conditional block resampling\n\n```text\n'+boots.round(5).to_string(index=False)+'\n```\n\n')
        f.write('A positive marker-specific association alone does NOT establish a broadly reliable HER2 discordance hotspot selector.\n')
    print('[5/5] Completed',flush=True)
    print('\n=== PRIMARY COMBINED RISK (50% coverage, sign-balanced) ===')
    print(result[(result.risk_method=='risk_combined_mean')&(result.coverage==.50)].round(5).to_string(index=False))
    print('\n=== RISK COVERAGE (select methods) ===')
    keep=('risk_combined_mean','risk_HER2_3B5_full','risk_HER2_29D8_full','risk_hotspot_margin','risk_geometry_mean')
    print(result[result.risk_method.isin(keep)][['risk_method','coverage','n_kept','precision','gain_vs_all','gain_vs_hotspot_margin','p_random_ge_observed']].round(5).to_string(index=False))
    print('\n=== SPATIAL BLOCK BOOTSTRAP (50% coverage) ===')
    print(boots[boots.coverage==.50][['risk_method','observed_gain_vs_all','boot_gain_vs_all_p025','boot_gain_vs_all_p975','observed_gain_vs_hotspot_margin','boot_gain_vs_margin_p025','boot_gain_vs_margin_p975']].round(5).to_string(index=False))
    print('\n=== PRE-SPECIFIED PRIMARY DECISION ===')
    print(decision)
    print('\nSUCCESS:',out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path('.'))
    ap.add_argument('--out',type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument('--bootstrap',type=int,default=499)
    ap.add_argument('--random-permutations',type=int,default=999)
    ap.add_argument('--block-xy',type=float,default=32.)
    ap.add_argument('--block-z',type=float,default=24.)
    run(ap.parse_args())

if __name__=='__main__':main()
