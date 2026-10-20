#!/usr/bin/env python3
"""Source-only fit / cross-volume evaluation of PORTABLE HER2 reliability features.

The originally reported 12E boundary-sensitive model CANNOT be transferred to
LVILymph: no matched dual segmentation, nor a second HER2 assay, exists there.
This re-fits a reduced, cross-volume-available feature subset on SecondHer2;
it is NOT a transfer of the published exploratory full feature set.

Target intensity errors are NEVER used for model fitting, feature selection,
imputation, scaling, or hyperparameter tuning. Target top-quartile error is a
predeclared evaluation label only. External-volume analysis is post-exploratory.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

SEED=20261019
SOURCE_MARKERS=('HER2_3B5','HER2_29D8')
TARGET_MARKERS=('Gd156','Cd112','Eu151','Yb174','Er168')
SHARED=('n_sections','log_total_other_px','log_near2_px','near2_area_imbalance','non_target_plane_area_cv','n_observed_non_target')
SIGNAL=('variation2','log_nearest2_mean','other_mad_log','near_vs_other_shift')
FULL_COLS=list(SHARED+SIGNAL)
GEO_COLS=list(SHARED)
SRC_REL={
 'per_z':'runs/secondher2_2d_vs_3d_heldout_v0_1/per_object_per_section_her2.csv',
 'heldout':'runs/secondher2_2d_vs_3d_heldout_v0_1/heldout_2d_vs_multiplane_predictions.csv',
 'coords':'runs/secondher2_3d_spatial_discordance_v0_1/3d_her2_object_spatial_coordinates.csv',
 'boundary':'runs/workflow12_secondher2_fixed_boundary_transfer_all92_v0_1/candidate_boundary_metrics.csv',
}
TGT_REL={
 'per_z':'runs/lvilymph_external_sampling_v0_1/per_z_object_signals.csv',
 'heldout':'runs/lvilymph_external_sampling_v0_1/heldout_object_predictions.csv',
 'audit':'runs/lvilymph_external_sampling_v0_1/protocol_and_audit.json',
}

def required(df,keys,tag):
    miss=set(keys)-set(df.columns)
    if miss:raise ValueError(f'{tag}: missing columns {sorted(miss)}')

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def mk_model():
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),
                         LogisticRegression(C=1.0,max_iter=1000,random_state=SEED))

def get_prior_module(root):
    path=root/'scripts/run_secondher2_reliability.py'
    if not path.is_file():
        raise FileNotFoundError(f'Previous reliability source script required: {path}. '
                                'Reinstall secondher2_reliability_model_v0_1.zip first.')
    spec=importlib.util.spec_from_file_location('source_reliability',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def build_source(root, prior):
    inputs={k:root/p for k,p in SRC_REL.items()}
    for f in inputs.values():
        if not f.is_file():raise FileNotFoundError(f'Missing frozen source input: {f}')
    df={k:pd.read_csv(p) for k,p in inputs.items()}
    src=prior.build_features(df['per_z'],df['heldout'],df['coords'],df['boundary'])
    if not src.source_pseudo_id.is_unique:raise ValueError('Source duplicate IDs')
    return src,inputs

def canonicalize_source(src,marker):
    if marker not in SOURCE_MARKERS:raise ValueError('Unknown source marker')
    rename={marker+'_'+name:name for name in SIGNAL}
    rename[marker+'_error_nearest2']='error'
    s=src.rename(columns=rename).copy()
    required(s,[*FULL_COLS,'error','source_pseudo_id','x_um','y_um','z_um'], 'source features')
    if s[FULL_COLS].isna().any().any():raise ValueError('Unexpected missing portable source features')
    if (s['error']<0).any():raise ValueError('Negative source errors')
    return s

def build_target(root, target_marker):
    inputs={k:root/p for k,p in TGT_REL.items()}
    for p in inputs.values():
        if not p.is_file():raise FileNotFoundError(f'Missing frozen LVILymph input: {p}')
    with inputs['audit'].open(encoding='utf8') as f:audit=json.load(f)
    if audit.get('status')!='complete' or audit.get('n_z')!=16:
        raise ValueError('Unexpected LVILymph frozen audit')
    perz=pd.read_csv(inputs['per_z']);held=pd.read_csv(inputs['heldout'])
    required(perz,['pseudo_id','z_section','n_voxels',f'{target_marker}_mean'], 'LVILymph per-Z')
    required(held,['pseudo_id','isotope','target_z_section','n_source_sections','nearest2_prediction','nearest2_error','block'], 'LVILymph heldout')
    perz=perz[['pseudo_id','z_section','n_voxels',f'{target_marker}_mean']]
    sub=held.loc[held.isotope==target_marker].copy()
    if sub.pseudo_id.duplicated().any():raise ValueError('Duplicate target keys')
    if perz.duplicated(['pseudo_id','z_section']).any():raise ValueError('Duplicate target per-Z keys')
    if len(sub)<50:raise ValueError(f'Too few target objects for {target_marker}')
    index=sub.set_index('pseudo_id')
    rows=[]
    for sid,g in perz.loc[perz.pseudo_id.isin(index.index)].groupby('pseudo_id',sort=True):
        g=g.sort_values('z_section')
        z=g.z_section.to_numpy(int)
        target=int(index.loc[sid,'target_z_section'])
        if len(g)<4 or np.any(np.diff(z)!=1) or target!=z[(len(z)-1)//2]:
            raise ValueError(f'Target source pseudo-ID {sid}: ineligible or changed holdout')
        if len(g)!=int(index.loc[sid,'n_source_sections']):raise ValueError('Section-count mismatch')
        weights=g.n_voxels.to_numpy(float)
        vals=g[f'{target_marker}_mean'].to_numpy(float)
        if np.any(weights<=0) or np.any(vals<0) or not np.isfinite(vals).all():
            raise ValueError('Nonfinite/invalid target signals')
        other=g[g.z_section!=target].copy()
        other=other.iloc[np.lexsort((other.z_section.to_numpy(),np.abs(other.z_section.to_numpy()-target)))]
        near=other.iloc[:2]
        n2=float(np.average(near[f'{target_marker}_mean'],weights=near.n_voxels))
        observed=float(index.loc[sid,'nearest2_prediction'])
        if not np.isclose(n2,observed,rtol=1e-7,atol=1e-7):
            raise RuntimeError(f'Frozen nearest2 prediction mismatch for {sid}/{target_marker}: {n2} vs {observed}')
        true=float(g.loc[g.z_section==target,f'{target_marker}_mean'].iloc[0]);err=abs(np.log1p(n2)-np.log1p(true))
        if not np.isclose(err,float(index.loc[sid,'nearest2_error']),rtol=1e-7,atol=1e-7):
            raise RuntimeError(f'Frozen nearest2 error mismatch for {sid}/{target_marker}')
        vv=other[f'{target_marker}_mean'].to_numpy(float)
        row={'pseudo_id':int(sid),'error':err,'block':str(index.loc[sid,'block']),
             'n_sections':len(g),'log_total_other_px':np.log1p(other.n_voxels.sum()),
             'log_near2_px':np.log1p(near.n_voxels.sum()),
             'near2_area_imbalance':abs(np.log(float(near.n_voxels.iloc[0])/float(near.n_voxels.iloc[1]))),
             'non_target_plane_area_cv':float(other.n_voxels.std(ddof=0)/other.n_voxels.mean()),
             'n_observed_non_target':len(other),
             'variation2':float(abs(np.log1p(vv[0])-np.log1p(vv[1]))),
             'log_nearest2_mean':float(np.log1p(n2)),
             'other_mad_log':float(np.median(abs(np.log1p(vv)-np.median(np.log1p(vv))))),
             'near_vs_other_shift':float(abs(np.log1p(n2)-np.log1p(np.average(vv,weights=other.n_voxels))))}
        rows.append(row)
    t=pd.DataFrame(rows)
    if len(t)!=len(sub):raise ValueError(f'Target count mismatch {len(t)} vs {len(sub)}')
    required(t,[*FULL_COLS,'error','block','pseudo_id'], 'target features')
    if not np.isfinite(t[FULL_COLS+['error']].to_numpy(float)).all():raise ValueError('Nonfinite target feature values')
    return t,inputs

def evaluate(d, risk, risk_tag, target_marker, src_marker, source_threshold):
    y=(d.error>float(d.error.quantile(.75))).astype(int).to_numpy()
    ys=(d.error>source_threshold).astype(int).to_numpy()
    vals=risk.astype(float)
    sel=np.lexsort((d.pseudo_id.to_numpy(),vals))[:len(d)//2]
    err=d.error.to_numpy(float)
    auc=float(roc_auc_score(y,vals))
    r,p=spearmanr(vals,err)
    return {'source_model':src_marker,'target_marker':target_marker,'method':risk_tag,
            'n_objects':len(d),'n_positive_target_q75':int(y.sum()),
            'source_q75_threshold_abs_log_error':float(source_threshold),
            'target_fraction_above_source_q75':float(ys.mean()),
            'external_auroc_target_q75':auc,
            'external_auprc_target_q75':float(average_precision_score(y,vals)),
            'risk_error_spearman':float(r),
            'overall_mean_error':float(err.mean()),
            'low_risk_50pct_mean_error':float(err[sel].mean()),
            'selective_risk_reduction_fraction':float(1.-err[sel].mean()/err.mean())}

def block_bootstrap_delta(d, full, baseline, n_boot, seed):
    y=(d.error>d.error.quantile(.75)).astype(int).to_numpy()
    block_ids=pd.factorize(d.block,sort=True)[0]
    unique=np.unique(block_ids)
    groups=[np.flatnonzero(block_ids==i) for i in unique]
    if len(groups)<15:raise ValueError('Too few spatial bootstrap blocks')
    rng=np.random.default_rng(seed)
    diffs=[]
    for _ in range(n_boot):
        idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),size=len(groups))])
        if len(np.unique(y[idx]))!=2:continue
        diffs.append(roc_auc_score(y[idx],full[idx])-roc_auc_score(y[idx],baseline[idx]))
    if len(diffs)<.95*n_boot:raise ValueError('Bootstrap many degenerate draws')
    return {'delta_auroc':float(roc_auc_score(y,full)-roc_auc_score(y,baseline)),
            'lower_025':float(np.quantile(diffs,.025)),'upper_975':float(np.quantile(diffs,.975)),
            'n_blocks':len(groups),'bootstrap_n':len(diffs)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path('.'))
    ap.add_argument('--out',type=Path,default=Path('runs/lvilymph_portable_reliability_transfer_v0_1'))
    ap.add_argument('--bootstrap',type=int,default=499)
    args=ap.parse_args()
    root=args.project_root.resolve();out=(root/args.out).resolve()
    if args.bootstrap<100:raise ValueError('Bootstrap must be at least 100')
    if out.exists():raise FileExistsError(f'Refusing to overwrite frozen output: {out}')
    print('[1/5] Reconstruct source-only portable features; exclude unavailable 12E boundaries',flush=True)
    prior=get_prior_module(root)
    src,src_inputs=build_source(root,prior)
    t_inputs={k:root/p for k,p in TGT_REL.items()}
    print(f'  SecondHer2 objects={len(src)}; portable={len(FULL_COLS)} features; source marker labels={SOURCE_MARKERS}',flush=True)
    print('[2/5] Source-only spatial heldout validation of reduced portable models',flush=True)
    oof=[];models={};thresholds={};sourcedata={}
    for marker in SOURCE_MARKERS:
        ss=canonicalize_source(src,marker)
        sourcedata[marker]=ss
        risk_records=[]
        for fold,tr,te,guard in prior.split_folds(ss,12.0):
            trdf=ss.iloc[tr];tedf=ss.iloc[te]
            thresh=float(trdf.error.quantile(.75));y=(trdf.error>thresh).astype(int)
            p=pd.DataFrame({'marker':marker,'fold':fold,'source_pseudo_id':tedf.source_pseudo_id.to_numpy(),
                            'y':(tedf.error.to_numpy()>thresh).astype(int),
                            'variation_risk':tedf.variation2.to_numpy(float)})
            for tag,features in (('geometry',GEO_COLS),('full',FULL_COLS)):
                model=mk_model();model.fit(trdf[features],y)
                p[tag+'_risk']=model.predict_proba(tedf[features])[:,1]
            risk_records.append(p)
        pred=pd.concat(risk_records,ignore_index=True)
        if len(pred)!=len(ss) or pred.source_pseudo_id.duplicated().any():raise RuntimeError('Bad OOF coverage')
        for tag in ('variation','geometry','full'):
            roc=float(roc_auc_score(pred.y,pred[tag+'_risk']))
            oof.append({'marker':marker,'method':tag,'n_objects':len(pred),'source_spatial_oof_auroc':roc})
        q75=float(ss.error.quantile(.75));thresholds[marker]=q75
        yy=(ss.error>q75).astype(int)
        for tag,features in (('geometry',GEO_COLS),('full',FULL_COLS)):
            mdl=mk_model();mdl.fit(ss[features],yy)
            models[(marker,tag)]=mdl
        print(f'  {marker}: source spatial OOF AUROC full={oof[-1]["source_spatial_oof_auroc"]:.4f}; source fixed q75={q75:.5f}',flush=True)
    print('[3/5] Frozen source models -> LVILymph target volume (no target-based fitting)',flush=True)
    evaluations=[];intervals=[];objectrows=[];shift=[]
    # Only Gd156 source HER2 to candidate target HER2 is intended primary; others are biological-marker controls.
    targets={t:build_target(root,t)[0] for t in TARGET_MARKERS}
    for source_marker in SOURCE_MARKERS:
        ss=sourcedata[source_marker]
        for target_marker,t in targets.items():
            risks={'variation':t.variation2.to_numpy(float)}
            for tag,fields in (('geometry',GEO_COLS),('full',FULL_COLS)):
                risks[tag]=models[(source_marker,tag)].predict_proba(t[fields])[:,1]
            for tag,score in risks.items():
                res=evaluate(t,score,tag,target_marker,source_marker,thresholds[source_marker])
                res['analysis_role']='primary' if (source_marker,target_marker)==('HER2_3B5','Gd156') else 'secondary_exploratory'
                evaluations.append(res)
            for baseline in ('variation','geometry'):
                ci=block_bootstrap_delta(t,risks['full'],risks[baseline],args.bootstrap,SEED+(0 if source_marker=='HER2_3B5' else 31)+TARGET_MARKERS.index(target_marker))
                intervals.append({'source_model':source_marker,'target_marker':target_marker,
                                  'contrast':'full-minus-'+baseline,**ci})
            d=t[['pseudo_id','block','error']].copy()
            d['source_model']=source_marker;d['target_marker']=target_marker
            for tag,arr in risks.items():d[tag+'_risk']=arr
            d['target_top_quartile_error']=(t.error>t.error.quantile(.75)).astype(int)
            objectrows.append(d)
            for feat in FULL_COLS:
                vals=ss[feat].to_numpy(float);tv=t[feat].to_numpy(float)
                a=np.quantile(vals,[.025,.5,.975]);std=float(np.std(vals))
                shift.append({'source_model':source_marker,'target_marker':target_marker,'feature':feat,
                              'source_p025':a[0],'source_median':a[1],'source_p975':a[2],
                              'target_median':float(np.median(tv)),
                              'target_fraction_outside_source_95pct_range':float(np.mean((tv<a[0])|(tv>a[2]))),
                              'median_shift_source_sd':float(abs(np.median(tv)-a[1])/std) if std>1e-12 else np.nan})
    print('[4/5] Create tables and OOD/feature-domain audit',flush=True)
    out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(oof).to_csv(out/'source_portable_spatial_oof.csv',index=False)
    pd.DataFrame(evaluations).to_csv(out/'external_reliability_summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'external_spatial_bootstrap.csv',index=False)
    pd.concat(objectrows,ignore_index=True).to_csv(out/'external_per_object_risks.csv',index=False)
    pd.DataFrame(shift).to_csv(out/'feature_domain_shift.csv',index=False)
    audit={'project':'3D-TissueMap AI','version':'portable_reliability_transfer_v0_1',
           'primary_mapping':'SecondHer2 HER2_3B5 risk -> LVILymph Gd156 c-erbB-/HER2 provisional',
           'secondary_mapping':'SecondHer2 HER2_29D8 -> LVILymph Gd156, plus 4 other markers (exploratory, NOT HER2 replication)',
           'not_the_original_full_model':True,
           'why_not_original':'source uses non-target-Z 12E dual-boundary sensitivity and second HER2 assay; LVILymph has neither',
           'source_model':'Source-only median imputer + source-only standard scaler + logistic regression C=1.0, no target fitting',
           'features':FULL_COLS,'geometry_only_features':GEO_COLS,
           'source_out_of_fold':'3 spatial X stripes with 12um guard, fold train-only error q75',
           'external_primary_outcome':'target top-quartile observed nearest2 absolute log1p error is an EVALUATION label only; never used to train or choose model',
           'fixed_source_thresholds':thresholds,
           'source_marker_names':SOURCE_MARKERS,'target_channels':TARGET_MARKERS,
           'limitations':['post-exploration cross-volume validation, not preregistered blind external test',
               'LVILymph HER2 Gd156 inferred from truncated c-erbB- channel; requires panel confirmation',
               'different IMC panels and antibody clones, source/target intensities may have severe covariate shift',
               'source original dual-boundary metrics unavailable target: this is a REDUCED portable model, not exact reproduction',
               'tissue-object pseudo-IDs are not manual 3D cell ground truth',
               'all test instances nested in one target volume; bootstrap does not create independent biological replicates',
               'LVILymph 3D pixel sizes were assumed in upstream evaluation; verify original study provenance'],
           'sha256':{**{'source_'+k:sha256(v) for k,v in src_inputs.items()},
                     **{'target_'+k:sha256(v) for k,v in t_inputs.items()},
                     'source_model_script':sha256(root/'scripts/run_secondher2_reliability.py')},
           'n_bootstrap':args.bootstrap, 'seed':SEED}
    (out/'protocol_and_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf8')
    with (out/'RESULTS_README.md').open('w',encoding='utf8') as f:
        f.write('# Portable Cross-Volume HER2 Measurement Reliability (post-exploratory)\n\n')
        f.write('**Reduced portable subset** of features, source-only re-fit; not the original full 12E geometry model.\n\n')
        f.write('## Source spatial OOF\n\n```text\n'+pd.DataFrame(oof).round(5).to_string(index=False)+'\n```\n\n')
        primary=pd.DataFrame(evaluations).query('source_model=="HER2_3B5" and target_marker=="Gd156"')
        f.write('## Predeclared primary Gd156 transfer\n\n```text\n'+primary.round(5).to_string(index=False)+'\n```\n\n')
        pri_ci=pd.DataFrame(intervals).query('source_model=="HER2_3B5" and target_marker=="Gd156"')
        f.write('## Primary spatial block uncertainty\n\n```text\n'+pri_ci.round(5).to_string(index=False)+'\n```\n\n')
        f.write('No patient-level inference or independent clinical HER2 truth. The target top quartile is evaluation-only.\n')
    print('[5/5] Final primary summary',flush=True)
    print('\n=== SOURCE PORTABLE SPATIAL OOF ===')
    print(pd.DataFrame(oof).round(5).to_string(index=False))
    print('\n=== PRIMARY SECONDHER2 3B5 -> LVILYMPH Gd156 ===')
    print(primary[['method','n_objects','external_auroc_target_q75','external_auprc_target_q75','risk_error_spearman','selective_risk_reduction_fraction','target_fraction_above_source_q75']].round(5).to_string(index=False))
    print('\n=== PRIMARY PAIRED SPATIAL BOOTSTRAP (DELTA AUROC) ===')
    print(pri_ci.round(5).to_string(index=False))
    print('\n=== TARGET FEATURE DISTRIBUTION SHIFT SUMMARY ===')
    ps=pd.DataFrame(shift).query('source_model=="HER2_3B5" and target_marker=="Gd156"')
    print(ps[['feature','median_shift_source_sd','target_fraction_outside_source_95pct_range']].round(4).to_string(index=False))
    print('\nSUCCESS:',out)

if __name__=='__main__':main()
