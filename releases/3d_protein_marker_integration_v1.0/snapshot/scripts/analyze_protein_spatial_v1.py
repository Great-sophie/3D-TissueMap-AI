#!/usr/bin/env python3
"""Workflow09D reproducible descriptive association and spatial-block sensitivity.

Correlation of mask-restricted raw marker intensity is not biological colocalization.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

PAIRS=[('panCK','HER2'),('CD31','LYVE1'),('Ir193','HER2'),
       ('HER2','n_xy_radius_neighbors'),('LYVE1','n_xy_radius_neighbors'),
       ('Ir193','n_xy_radius_neighbors')]

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def col(name): return name if name=='n_xy_radius_neighbors' else 'mean_raw_'+name

def correlation(df,a,b):
    v=df[[col(a),col(b)]].replace([np.inf,-np.inf],np.nan).dropna()
    if len(v)<3 or v.iloc[:,0].nunique()<2 or v.iloc[:,1].nunique()<2:return None
    rho=float(v.iloc[:,0].corr(v.iloc[:,1],method='spearman'))
    return rho if np.isfinite(rho) else None

def report(df, sizes=(120,200), min_block_n=40):
    required={'instance_id','centroid_y_px','centroid_x_px','qc_flag_review','z_span_planes',
              'n_xy_radius_neighbors'}|{col(m) for m in ('Ir193','panCK','HER2','CD31','LYVE1')}
    miss=required-set(df.columns)
    if miss:raise ValueError(f'Missing Workflow09 output columns: {sorted(miss)}')
    if df['instance_id'].isna().any() or df['instance_id'].duplicated().any():raise ValueError('Duplicate/missing instance_id')
    if len(df)<3:raise ValueError('Insufficient instance count')
    flagged=df['qc_flag_review'].astype(str).str.lower().isin(['true','1'])
    summaries=[];blockrows=[];stratarows=[]
    for a,b in PAIRS:
        r=correlation(df,a,b)
        ru=correlation(df.loc[~flagged],a,b)
        summaries.append({'pair':f'{a}--{b}','n':len(df),'global_rho':r,'unflagged_rho':ru})
        for span in sorted(df['z_span_planes'].dropna().unique()):
            group=df.loc[(~flagged)&(df['z_span_planes']==span)]
            if len(group)>=30:
                stratarows.append({'pair':f'{a}--{b}','z_span_planes':int(span),'n':int(len(group)),
                                   'unflagged_rho':correlation(group,a,b)})
    block_summ=[]
    for sz in sizes:
        if sz<1:raise ValueError('block size must be >=1')
        temp=df.copy()
        temp['_block_y']=np.floor(temp['centroid_y_px']/sz).astype(int)
        temp['_block_x']=np.floor(temp['centroid_x_px']/sz).astype(int)
        blocks=list(temp.groupby(['_block_y','_block_x'],sort=True))
        for a,b in PAIRS:
            vals=[]
            for (by,bx),g in blocks:
                if len(g)<min_block_n:continue
                rho=correlation(g,a,b)
                if rho is None:continue
                vals.append(rho)
                blockrows.append({'block_px':sz,'block_y':int(by),'block_x':int(bx),
                                  'pair':f'{a}--{b}','n_instances':len(g),'spearman_rho':rho})
            arr=np.asarray(vals,dtype=float)
            block_summ.append({'block_px':sz,'pair':f'{a}--{b}','n_blocks':len(vals),
                               'median_rho':float(np.median(arr)) if len(arr) else None,
                               'q25_rho':float(np.quantile(arr,.25)) if len(arr) else None,
                               'q75_rho':float(np.quantile(arr,.75)) if len(arr) else None,
                               'positive_fraction':float(np.mean(arr>0)) if len(arr) else None})
    return summaries,blockrows,block_summ,stratarows

def run(source:Path,out:Path):
    if out.exists():raise FileExistsError('Analysis output already exists; refusing overwrite: '+str(out))
    df=pd.read_csv(source)
    summaries, blocks, block_summ, strata=report(df)
    out.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.workflow09d_tmp_',dir=out.parent))
    try:
        tables={'global_correlations.csv':summaries,'block_correlations.csv':blocks,
                'block_summary.csv':block_summ,'z_span_sensitivity.csv':strata}
        for name,rows in tables.items():
            pd.DataFrame(rows).to_csv(staging/name,index=False)
        summary={'version':'1.0.0-descriptive','source_table':str(source.resolve()),
                 'source_table_sha256':sha(source),'n_instances':int(len(df)),
                 'block_sizes_px':[120,200],'block_min_n':40,
                 'global_results':summaries,'block_results':block_summ,
                 'limitations':['Single specimen; block groups are not biological replicates.',
                     'Mask-restricted intensities do not establish cell-level protein expression.',
                     'Only XY pixel block grouping; Z spacing unknown.',
                     'Spearman association does not imply co-localization or causality.',
                     'Spatial autocorrelation, Z-span, cell size and stain/batch effects remain.']}
        (staging/'association_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        hashes={p.name:sha(p) for p in sorted(staging.iterdir()) if p.is_file()}
        (staging/'output_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
        staging.rename(out)
    finally:
        if staging.exists():shutil.rmtree(staging)
    return summary

def verify(source:Path,out:Path):
    hashes=json.loads((out/'output_sha256.json').read_text())
    present={p.name for p in out.iterdir() if p.is_file()}
    if present!=set(hashes)|{'output_sha256.json'}:raise RuntimeError('Unexpected/missing Workflow09D results')
    for name,expected in hashes.items():
        if sha(out/name)!=expected:raise RuntimeError('Changed: '+name)
    summary=json.loads((out/'association_summary.json').read_text())
    if sha(source)!=summary['source_table_sha256']:raise RuntimeError('Input spatial table changed')
    if len(pd.read_csv(source))!=summary['n_instances']:raise RuntimeError('Instance count changed')
    return {'n_instances':summary['n_instances'],'files_verified':len(hashes)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--table',type=Path,default=Path('runs/3d_protein_marker_integration_v1/instance_marker_spatial_table.csv'))
    ap.add_argument('--out',type=Path,default=Path('runs/3d_protein_spatial_association_v1'))
    ap.add_argument('--verify',action='store_true')
    a=ap.parse_args()
    result=verify(a.table,a.out) if a.verify else run(a.table,a.out)
    print(json.dumps(result if a.verify else {'status':'CREATED','n_instances':result['n_instances'],'block_summary':result['block_results']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
