#!/usr/bin/env python3
"""Frozen, post-exploratory source-only feature ablations for portable IMC reliability.

Uses existing CSVs and functions from run_portable_reliability_transfer.py;
never fits on LVILymph. One untrained baseline and five variants of the same logistic regression:
variation (no training), geometry, signal_only, full,
full_no_absolute_intensity and signal_no_absolute_intensity.
This is an explanatory ablation; LVILymph was already inspected, not a fresh blind test.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

SEED = 20261019
TARGETS = ('Gd156','Cd112','Eu151','Yb174','Er168')
SOURCE_MARKERS = ('HER2_3B5','HER2_29D8')
METHODS = ('variation','geometry','signal_only','full','full_no_absolute_intensity','signal_no_absolute_intensity')
CONTRASTS = (
 ('full','variation'),
 ('full','signal_only'),
 ('full','full_no_absolute_intensity'),
 ('full_no_absolute_intensity','signal_no_absolute_intensity'),
 ('full_no_absolute_intensity','variation'),
)

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def load_portable(root):
    path=root/'scripts'/'run_portable_reliability_transfer.py'
    if not path.is_file(): raise FileNotFoundError(f'Required original script is missing: {path}')
    spec=importlib.util.spec_from_file_location('frozen_portable_module',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    if tuple(module.SOURCE_MARKERS)!=SOURCE_MARKERS or tuple(module.TARGET_MARKERS)!=TARGETS:
        raise RuntimeError('Unexpected source/target panel in baseline script')
    return module,path

def featurizations(port):
    geo=list(port.GEO_COLS)
    sig=list(port.SIGNAL)
    if sig != ['variation2','log_nearest2_mean','other_mad_log','near_vs_other_shift']:
        raise RuntimeError('Original feature schema changed, abort to preserve frozen protocol')
    if len(geo)!=6: raise RuntimeError('Unexpected geometry feature count')
    feats={
       'geometry':geo,
       'signal_only':sig,
       'full':geo+sig,
       'full_no_absolute_intensity':geo+[x for x in sig if x!='log_nearest2_mean'],
       'signal_no_absolute_intensity':[x for x in sig if x!='log_nearest2_mean'],
    }
    if feats['full']!=list(port.FULL_COLS):raise RuntimeError('Full feature schema changed')
    return feats

def score_risks(d, risks, source_threshold):
    err=d['error'].to_numpy(float)
    # Reproduce the original target within-volume high-error proxy; evaluation only.
    q75=float(d['error'].quantile(.75))
    y_rel=(err>q75).astype(int)
    y_fixed=(err>source_threshold).astype(int)
    if len(np.unique(y_rel))!=2: raise RuntimeError('Degenerate target q75 labels')
    out=[]
    for tag, p in risks.items():
        p=np.asarray(p,float)
        if not np.isfinite(p).all(): raise RuntimeError(f'Invalid scores for {tag}')
        ix=np.lexsort((d.pseudo_id.to_numpy(int),p))[:len(d)//2]
        rho=spearmanr(p,err).statistic
        if not np.isfinite(rho): raise RuntimeError(f'Undefined risk-error rho: {tag}')
        out.append({
            'method':tag,'n_objects':int(len(d)),
            'n_target_q75_positives':int(y_rel.sum()),
            'target_high_error_q75_threshold':q75,
            'source_q75_threshold':source_threshold,
            'n_above_source_q75':int(y_fixed.sum()),
            'external_auroc_target_q75':float(roc_auc_score(y_rel,p)),
            'external_auprc_target_q75':float(average_precision_score(y_rel,p)),
            'external_auroc_source_fixed_q75':float(roc_auc_score(y_fixed,p)) if len(np.unique(y_fixed))==2 else np.nan,
            'external_auprc_source_fixed_q75':float(average_precision_score(y_fixed,p)) if len(np.unique(y_fixed))==2 else np.nan,
            'risk_error_spearman':float(rho),
            'overall_mean_error':float(err.mean()),
            'low_risk_half_mean_error':float(err[ix].mean()),
            'selective_risk_reduction_fraction':float(1.-err[ix].mean()/err.mean()),
        })
    return out

def paired_block_ci(d, risk, n_boot, seed, source_threshold):
    y=(d.error.to_numpy(float)>float(d.error.quantile(.75))).astype(int)
    yf=(d.error.to_numpy(float)>source_threshold).astype(int)
    blocks=pd.factorize(d.block.astype(str),sort=True)[0]
    groups=[np.flatnonzero(blocks==i) for i in range(blocks.max()+1)]
    if len(groups)<15:raise ValueError('Insufficient spatial blocks for bootstrap')
    rng=np.random.default_rng(seed)
    rank={k:np.asarray(v,float) for k,v in risk.items()}
    # Same bootstrap indices across contrasts (paired). Retain the fixed source threshold.
    samples=[]
    for _ in range(n_boot):
        ix=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        if len(np.unique(y[ix]))<2:continue
        samples.append(ix)
    if len(samples)<int(.95*n_boot):raise RuntimeError('Too many degenerate bootstrap draws')
    # Evaluate each model once per common resample; differences remain paired.
    needed=sorted({method for contrast in CONTRASTS for method in contrast})
    auc_bs={method:np.array([roc_auc_score(y[ix],rank[method][ix]) for ix in samples]) for method in needed}
    source_valid=[ix for ix in samples if len(np.unique(yf[ix]))==2]
    auc_source_bs={method:np.array([roc_auc_score(yf[ix],rank[method][ix]) for ix in source_valid])
                   for method in needed} if len(np.unique(yf))==2 else {}
    rows=[]
    for name_a,name_b in CONTRASTS:
        a,b=rank[name_a],rank[name_b]
        delta=roc_auc_score(y,a)-roc_auc_score(y,b)
        bs=auc_bs[name_a]-auc_bs[name_b]
        row={'contrast':f'{name_a}-minus-{name_b}',
             'delta_auroc_target_q75':float(delta),
             'lower_025':float(np.quantile(bs,.025)),
             'upper_975':float(np.quantile(bs,.975)),
             'n_blocks':len(groups),'bootstrap_n':len(samples),
             'label':'target_within_volume_upper_quartile_proxy'}
        # Fixed source absolute-error threshold as a secondary sensitivity;
        # same pretrained source threshold, no target recalibration.
        if len(np.unique(yf))==2:
            row['delta_auroc_source_fixed_q75']=float(roc_auc_score(yf,a)-roc_auc_score(yf,b))
            v=auc_source_bs[name_a]-auc_source_bs[name_b]
            row['source_fixed_lower_025']=float(np.quantile(v,.025)) if len(v)>.9*len(samples) else np.nan
            row['source_fixed_upper_975']=float(np.quantile(v,.975)) if len(v)>.9*len(samples) else np.nan
        else:
            row.update(delta_auroc_source_fixed_q75=np.nan,source_fixed_lower_025=np.nan,source_fixed_upper_975=np.nan)
        rows.append(row)
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path('.'))
    ap.add_argument('--out',type=Path,default=Path('runs/lvilymph_portable_reliability_ablation_v0_1'))
    ap.add_argument('--bootstrap',type=int,default=499)
    args=ap.parse_args()
    if args.bootstrap<100:raise ValueError('Require >=100 bootstrap draws')
    root=args.project_root.resolve();out=(root/args.out).resolve()
    if out.exists(): raise FileExistsError(f'Refusing to overwrite: {out}')
    port,portpath=load_portable(root)
    fs=featurizations(port)
    baseline_dir=root/'runs/lvilymph_portable_reliability_transfer_v0_1'
    previous=baseline_dir/'external_reliability_summary.csv'
    if not previous.exists():raise FileNotFoundError(f'Frozen baseline summary needed: {previous}')
    frozen=pd.read_csv(previous)
    print('[1/5] Reconstruct frozen SecondHer2 features and LVILymph non-target signals',flush=True)
    prior=port.get_prior_module(root)
    src,inputs=port.build_source(root,prior)
    sources={m:port.canonicalize_source(src,m) for m in SOURCE_MARKERS}
    targets={m:port.build_target(root,m)[0] for m in TARGETS}
    print('  Source pseudo-objects:',len(src),'target pseudo-objects:',len(targets[TARGETS[0]]),flush=True)
    print('[2/5] Source spatial OOF: frozen LogisticRegression(C=1), no re-tuning',flush=True)
    source_rows=[];source_thresholds={};source_models={};source_oof=[]
    for marker,ss in sources.items():
        fold_preds=[]
        for fold,tr,te,sep in prior.split_folds(ss,12.0):
            trn=ss.iloc[tr];tst=ss.iloc[te]
            thr=float(trn.error.quantile(.75)); ytr=(trn.error.to_numpy(float)>thr).astype(int)
            record=pd.DataFrame({'marker':marker,'fold':fold,'source_pseudo_id':tst.source_pseudo_id.to_numpy(int),
                                 'y':(tst.error.to_numpy(float)>thr).astype(int),'variation_risk':tst.variation2.to_numpy(float)})
            for tag,cols in fs.items():
                mod=port.mk_model();mod.fit(trn[cols],ytr)
                record[tag+'_risk']=mod.predict_proba(tst[cols])[:,1]
            fold_preds.append(record)
        oof=pd.concat(fold_preds,ignore_index=True)
        if len(oof)!=len(ss) or oof.source_pseudo_id.duplicated().any():raise RuntimeError('Bad source OOF partition')
        source_oof.append(oof)
        for tag in METHODS:
            score=oof[tag+'_risk']
            source_rows.append({'source_model':marker,'method':tag,'n_objects':len(oof),
                                'source_spatial_oof_auroc':float(roc_auc_score(oof.y,score)),
                                'source_spatial_oof_auprc':float(average_precision_score(oof.y,score))})
        source_thresholds[marker]=float(ss.error.quantile(.75)); ys=(ss.error>source_thresholds[marker]).astype(int)
        for tag,cols in fs.items():
            mod=port.mk_model();mod.fit(ss[cols],ys)
            source_models[(marker,tag)]=mod
        print(' ',marker,': full source OOF AUROC =',f"{source_rows[-3]['source_spatial_oof_auroc']:.4f}",flush=True)
    print('[3/5] Frozen target predictions on five markers (no fitting on LVILymph)',flush=True)
    metrics=[];bootstrap=[];riskrows=[]
    for sm in SOURCE_MARKERS:
        for ti,tm in enumerate(TARGETS):
            d=targets[tm]
            risks={'variation':d.variation2.to_numpy(float)}
            for name,cols in fs.items():
                risks[name]=source_models[(sm,name)].predict_proba(d[cols])[:,1]
            for m in score_risks(d,risks,source_thresholds[sm]):
                m.update(source_model=sm,target_marker=tm,analysis_role='predeclared_primary' if (sm,tm)==('HER2_3B5','Gd156') else 'secondary_exploratory')
                metrics.append(m)
            for entry in paired_block_ci(d,risks,args.bootstrap,SEED+17*TARGETS.index(tm)+(0 if sm=='HER2_3B5' else 101),source_thresholds[sm]):
                entry.update(source_model=sm,target_marker=tm)
                bootstrap.append(entry)
            p=d[['pseudo_id','block','error']].copy()
            p['source_model']=sm;p['target_marker']=tm
            for key,arr in risks.items():p[key+'_risk']=arr
            riskrows.append(p)
        print(' ',sm,': completed',len(TARGETS),'target markers',flush=True)
    print('[4/5] Verify unchanged original full scores and persist auditable output',flush=True)
    summary=pd.DataFrame(metrics); ci=pd.DataFrame(bootstrap); oo=pd.DataFrame(source_rows)
    baseline_tags={'full':'full','variation':'variation','geometry':'geometry'}
    for sm in SOURCE_MARKERS:
        for tm in TARGETS:
            for tag in baseline_tags:
                old=frozen.query('source_model==@sm and target_marker==@tm and method==@tag')
                new=summary.query('source_model==@sm and target_marker==@tm and method==@tag')
                if len(old)!=1 or len(new)!=1: raise RuntimeError(f'Original baseline match missing: {sm}/{tm}/{tag}')
                x=float(old.external_auroc_target_q75.iloc[0]);y=float(new.external_auroc_target_q75.iloc[0])
                if not np.isclose(x,y,rtol=1e-10,atol=1e-10):
                    raise RuntimeError(f'Frozen baseline reproduction failed {sm}/{tm}/{tag}: {x} vs {y}')
    source_old=pd.read_csv(baseline_dir/'source_portable_spatial_oof.csv')
    for sm in SOURCE_MARKERS:
        for tag in ('variation','geometry','full'):
            x=float(source_old.query('marker==@sm and method==@tag').source_spatial_oof_auroc.iloc[0])
            y=float(oo.query('source_model==@sm and method==@tag').source_spatial_oof_auroc.iloc[0])
            if not np.isclose(x,y,atol=1e-10,rtol=1e-10):raise RuntimeError('Source baseline OOF reproduction failed')
    out.mkdir(parents=True,exist_ok=False)
    summary.to_csv(out/'external_ablation_summary.csv',index=False)
    ci.to_csv(out/'external_paired_spatial_block_intervals.csv',index=False)
    oo.to_csv(out/'source_ablation_spatial_oof.csv',index=False)
    pd.concat(source_oof,ignore_index=True).to_csv(out/'source_ablation_oof_predictions.csv',index=False)
    pd.concat(riskrows,ignore_index=True).to_csv(out/'external_ablation_object_scores.csv',index=False)
    deps={'portable_script':portpath,'baseline_summary':previous,'source_per_z':root/port.SRC_REL['per_z'],
          'source_heldout':root/port.SRC_REL['heldout'],'source_coords':root/port.SRC_REL['coords'],
          'target_heldout':root/port.TGT_REL['heldout'],'target_per_z':root/port.TGT_REL['per_z']}
    audit={'version':'portable_reliability_ablation_v0_1','methods':list(METHODS),
           'feature_sets':fs,'primary_source':'HER2_3B5','primary_target':'Gd156 (c-erbB- tentative)',
           'source_only_fit':True,'logistic_regression_C':1.0,'source_thresholds':source_thresholds,
           'target_labels':'within-target upper quartile and fixed source q75, both evaluation only',
           'n_bootstrap':args.bootstrap,'seed':SEED,
           'baseline_verified_identical':True,
           'interpretation':{'full_vs_signal_only':'added geometry beyond all signal features',
               'full_vs_full_no_absolute_intensity':'contribution/dependence on raw absolute mean intensity',
               'full_no_absolute_intensity_vs_signal_no_absolute_intensity':'geometry beyond scale-free signal features'},
           'limitations':['post-exploration ablation using a target volume already inspected; not blind validation',
             'source and target each one 3D volume; within-volume block CIs do not quantify between-patient variability',
             'target middle Z marker signal hidden, but source instance geometry derives from full 3D segmentation including target plane',
             'pixel physical calibration for LVILymph unverified in upstream audit',
             'Gd156 antigen inferred from truncated label only',
             'marker results are correlated outcomes within the same target volume; no multiplicity claim',
             'risk predicts within-instance held-out measurement error, not biological assay truth'],
           'sha256_inputs':{k:sha256(v) for k,v in deps.items()}}
    (out/'protocol_and_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf8')
    mainmask=(summary.source_model=='HER2_3B5')&(summary.target_marker=='Gd156')
    pr=summary.loc[mainmask]
    pc=ci.loc[(ci.source_model=='HER2_3B5')&(ci.target_marker=='Gd156')]
    with (out/'RESULTS_README.md').open('w',encoding='utf8') as f:
        f.write('# Portable Reliability Feature Ablation (post-exploration)\n\n')
        f.write('## Primary: SecondHer2 HER2 3B5 -> LVILymph Gd156\n\n```text\n')
        f.write(pr[['method','external_auroc_target_q75','external_auroc_source_fixed_q75','selective_risk_reduction_fraction']].round(5).to_string(index=False))
        f.write('\n```\n\n## Primary paired block CIs\n\n```text\n')
        f.write(pc[['contrast','delta_auroc_target_q75','lower_025','upper_975','delta_auroc_source_fixed_q75']].round(5).to_string(index=False))
        f.write('\n```\n\n## Exploratory five-marker replication\n\n```text\n')
        f.write(summary[(summary.source_model=='HER2_3B5')&(summary.method.isin(['full','signal_only','full_no_absolute_intensity']))][['target_marker','method','external_auroc_target_q75']].round(5).to_string(index=False))
        f.write('\n```\n\nAll measurements are held-out-Z proxy errors from fixed pseudo-instances; this does not establish independent biological truth.\n')
    print('[5/5] Final decision-oriented tables',flush=True)
    print('\n=== PRIMARY FEATURE ABLATIONS: 3B5 -> Gd156 ===')
    print(pr[['method','external_auroc_target_q75','external_auprc_target_q75','external_auroc_source_fixed_q75','selective_risk_reduction_fraction']].round(5).to_string(index=False))
    print('\n=== PRIMARY PAIRED SPATIAL BOOTSTRAP ===')
    print(pc[['contrast','delta_auroc_target_q75','lower_025','upper_975','delta_auroc_source_fixed_q75','source_fixed_lower_025','source_fixed_upper_975']].round(5).to_string(index=False))
    print('\n=== ALL MARKERS: 3B5 SOURCE (AUROC TARGET Q75) ===')
    p=summary.loc[summary.source_model=='HER2_3B5'].pivot(index='target_marker',columns='method',values='external_auroc_target_q75')
    print(p.loc[list(TARGETS),['variation','signal_only','full_no_absolute_intensity','full']].round(5).to_string())
    print('\n=== ALL MARKERS: FULL MINUS SIGNAL_ONLY BLOCK CI ===')
    q=ci[(ci.source_model=='HER2_3B5')&(ci.contrast=='full-minus-signal_only')]
    print(q[['target_marker','delta_auroc_target_q75','lower_025','upper_975']].round(5).to_string(index=False))
    print('\nRESULTS:',out)

if __name__=='__main__':main()
