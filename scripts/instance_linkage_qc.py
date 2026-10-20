#!/usr/bin/env python3
"""3D-TissueMap: audit instance IDs versus measured ROI, area and protein tables.

This is an ID/geometry consistency screen, not a verification that the protein
rows correspond to the label IDs. No image spacing, registration or tumour
phenotype is inferred.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile


def _counts(arr):
    ids, cnt = np.unique(arr, return_counts=True)
    ok = ids > 0
    return dict(zip(ids[ok].astype(int).tolist(), cnt[ok].astype(int).tolist()))


def _comparison(x, y):
    a=np.asarray(x,dtype=float); b=np.asarray(y,dtype=float)
    ok=np.isfinite(a)&np.isfinite(b)
    a,b=a[ok],b[ok]
    if not len(a): return {'n':0, 'fraction_exact':None,'pearson':None,'median_abs_difference':None}
    r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else None
    return {'n':int(len(a)), 'fraction_exact':float(np.mean(np.isclose(a,b,rtol=0,atol=0))),
            'pearson':r, 'median_abs_difference':float(np.median(np.abs(a-b)))}


def run(data_dir:Path, out:Path):
    out.mkdir(parents=True,exist_ok=True)
    full_paths=sorted(data_dir.glob('full_mask*.tif'))
    measured_paths=sorted(data_dir.glob('measured_mask*.tif'))
    if len(full_paths)!=1 or len(measured_paths)!=1:
        raise RuntimeError('Expected exactly one full_mask*.tif and one measured_mask*.tif')
    full=tifffile.imread(full_paths[0]); measured=tifffile.imread(measured_paths[0])
    if full.ndim!=3 or measured.ndim!=3 or full.shape[0]!=measured.shape[0]:
        raise ValueError(f'Incompatible 3D masks: full {full.shape}, measured {measured.shape}')
    full_count=_counts(full); measured_count=_counts(measured)
    del full, measured
    area_file=data_dir/'LVI_lymph_labels_area.csv'
    protein_file=data_dir/'LVI_lymph_mean_intensities.csv'
    area=pd.read_csv(area_file)
    protein=pd.read_csv(protein_file)
    if not {'label','area'}.issubset(area.columns):
        raise ValueError('Area CSV requires columns label and area')
    ids=pd.to_numeric(area.label,errors='raise')
    if ids.isna().any() or (ids<=0).any() or not np.all(ids==ids.astype('int64')):
        raise ValueError('Area IDs must be positive integers')
    if ids.duplicated().any(): raise ValueError('Duplicate area-table label IDs')
    area=area.copy(); area['label']=ids.astype('int64')
    area['area']=pd.to_numeric(area['area'],errors='coerce')
    area['full_mask_voxels']=area['label'].map(full_count)
    area['measured_mask_voxels']=area['label'].map(measured_count)
    area['row_index_0based']=np.arange(len(area))
    area['present_in_full']=area.full_mask_voxels.notna()
    area['present_in_measured']=area.measured_mask_voxels.notna()
    area.to_csv(out/'area_vs_mask_voxels.csv',index=False)
    measured_ids=set(measured_count)
    area_ids=set(area.label.astype(int))
    full_ids=set(full_count)
    result={
        'scope':'single LVILymph specimen; author watershed labels; no independent manual GT',
        'full_mask_unique_ids':len(full_ids),
        'measured_mask_unique_ids':len(measured_ids),
        'area_table_rows':int(len(area)),
        'protein_table_rows':int(len(protein)),
        'area_labels_unique':True,
        'area_label_set_equals_measured_mask':area_ids==measured_ids,
        'measured_IDs_all_in_full_mask':measured_ids.issubset(full_ids),
        'area_table_order_strictly_increasing':bool(np.all(np.diff(area.label.to_numpy())>0)),
        'area_column_vs_measured_voxel_count':_comparison(area['area'],area['measured_mask_voxels']),
        'area_column_vs_full_voxel_count':_comparison(area['area'],area['full_mask_voxels']),
        'protein_has_explicit_label_col':bool('label' in protein.columns),
        'protein_row_mapping_verified':False,
        'critical_blocker':'Matching table lengths or mask IDs is NOT enough to assign protein rows to cell IDs. Seek original analysis script / exported cell-index metadata, or independently verify intensity mapping against aligned source images.',
        'mask_geometry':'Full and measured ROI image coordinates may differ. Do NOT infer crop origin, physical spacing or voxel volume.',
        'future_protein_join_policy':'BLOCKED without explicit unique ID provenance or independently validated marker-intensity correspondence'
    }
    if 'label' in protein.columns:
        prot_ids=pd.to_numeric(protein.label,errors='coerce')
        if prot_ids.notna().all() and not prot_ids.duplicated().any():
            result['protein_ID_set_equals_measured']=set(prot_ids.astype(int))==measured_ids
            result['protein_row_mapping_verified']=bool(result['protein_ID_set_equals_measured'])
            result['future_protein_join_policy']='Join via explicit unique label column only, after checking image provenance'
    (out/'linkage_qc.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print('Saved:',out)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('runs/instance_linkage_qc'))
    a=p.parse_args();run(a.data_dir,a.out)
if __name__=='__main__':main()
