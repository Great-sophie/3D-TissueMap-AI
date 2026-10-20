#!/usr/bin/env python3
"""Audit author-provided 3D watershed instance IDs before training instance models.

Usage (from 3D-TissueMap-AI project root):
  python scripts/instance_reference_audit.py --data-dir data/inspect/LVILymph \
      --out runs/instance_reference_audit

This DOES NOT validate segmentation accuracy, physical registration, or protein-table
row-to-instance alignment. It never modifies source masks or raw data.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi


def audit(data_dir: Path, out: Path, check_top_n: int = 12):
    out.mkdir(parents=True, exist_ok=True)
    masks = sorted(data_dir.glob('full_mask*.tif'))
    if len(masks) != 1:
        raise ValueError(f'Expected exactly one full_mask*.tif, found {len(masks)}')
    mask = tifffile.imread(masks[0])
    if mask.ndim != 3 or mask.dtype.kind not in ('i','u'):
        raise ValueError(f'Expected uint/int 3D instance mask, got {mask.shape}, {mask.dtype}')
    if np.any(mask < 0):
        raise ValueError('Negative instance IDs are unexpected')
    ids, counts = np.unique(mask, return_counts=True)
    keep = ids > 0
    ids = ids[keep].astype(np.int64)
    counts = counts[keep].astype(np.int64)
    if not len(ids):
        raise ValueError('No non-background instance IDs found')
    presence = np.zeros(len(ids), dtype=np.uint16)
    per_slice = []
    for z in range(mask.shape[0]):
        iz = np.unique(mask[z]); iz = iz[iz > 0]
        presence[np.searchsorted(ids,iz)] += 1
        per_slice.append({'z_index_0based':z,'unique_instance_ids':int(len(iz)),
                          'foreground_fraction':float(np.mean(mask[z] > 0))})
    table = pd.DataFrame({'cell_id':ids,'voxel_count':counts,
                          'n_z_planes_present':presence})
    table['present_in_multiple_planes'] = table.n_z_planes_present > 1
    table.sort_values('voxel_count',ascending=False).to_csv(out/'reference_instances.csv',index=False)
    pd.DataFrame(per_slice).to_csv(out/'slice_summary.csv',index=False)

    # Diagnose whether a persistent ID is a single 3D connected component.
    # This is an indicator of potential label fragmentation, not a proof of validity.
    sample = table.nlargest(min(check_top_n,len(table)),'voxel_count')
    checked = []
    structure = ndi.generate_binary_structure(3,1) # six-connectivity
    for rec in sample.itertuples(index=False):
        xyz = np.nonzero(mask == rec.cell_id)
        lo = [int(a.min()) for a in xyz]; hi = [int(a.max())+1 for a in xyz]
        crop = mask[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]] == rec.cell_id
        comp,num = ndi.label(crop, structure=structure)
        if num:
            sizes=np.bincount(comp.ravel())[1:]
            largest=float(sizes.max()/sizes.sum())
        else:
            largest=0.0
        checked.append({'cell_id':int(rec.cell_id),
                        'voxel_count':int(rec.voxel_count),
                        'n_z_planes_present':int(rec.n_z_planes_present),
                        'n_connected_components_6conn':int(num),
                        'largest_component_fraction':largest,
                        'bbox_zyx_start':lo, 'bbox_zyx_stop':hi})
    pd.DataFrame(checked).to_csv(out/'sampled_instance_connectivity.csv',index=False)

    labels_area = data_dir/'LVI_lymph_labels_area.csv'
    protein_table = data_dir/'LVI_lymph_mean_intensities.csv'
    join = {'area_table_present':labels_area.exists(),
            'protein_table_present':protein_table.exists(),
            'row_to_instance_mapping_verified':False}
    if labels_area.exists():
        area = pd.read_csv(labels_area)
        join['area_rows']=int(len(area))
        if 'label' in area:
            a_ids = set(pd.to_numeric(area['label'],errors='coerce').dropna().astype('int64').tolist())-{0}
            m_ids = set(int(x) for x in ids)
            join['area_labels_unique']=len(a_ids)==len(area)
            join['area_IDs_in_full_mask']=int(len(a_ids&m_ids))
            join['area_IDs_not_in_full_mask']=int(len(a_ids-m_ids))
    if protein_table.exists():
        expression=pd.read_csv(protein_table)
        join['protein_rows']=int(len(expression))
        join['protein_has_explicit_label_column']='label' in expression.columns
        if labels_area.exists():
            join['area_protein_row_counts_equal']=join['area_rows']==join['protein_rows']
            join['note']='Row-count equality is NOT evidence that CSV row order matches instance IDs.'
    measured_masks = sorted(data_dir.glob('measured_mask*.tif'))
    if len(measured_masks)==1:
        measured=tifffile.imread(measured_masks[0])
        measured_ids=set(int(x) for x in np.unique(measured) if x>0)
        join['measured_mask_shape_ZYX']=list(measured.shape)
        join['measured_ID_count']=len(measured_ids)
        join['measured_IDs_in_full_mask']=int(len(measured_ids&set(int(x) for x in ids)))

    meta={'provenance':'Author-produced watershed instance masks; NOT manual ground truth',
          'mask_file':str(masks[0].name),'shape_ZYX':list(mask.shape),
          'n_unique_instance_IDs':int(len(ids)),
          'fraction_IDs_in_multiple_Z_planes':float(np.mean(presence>1)),
          'reference_foreground_fraction':float(np.mean(mask>0)),
          'largest_IDs_checked_for_3D_connectivity':len(checked),
          'sampled_IDs_multiple_connected_components':int(sum(row['n_connected_components_6conn']>1 for row in checked)),
          'marker_table_join':join,
          'physical_spacing':'NOT VERIFIED - voxel counts are not physical cell volumes',
          'registration':'NOT PROVEN by these checks; audit does not validate slice-to-mask accuracy',
          'next':'After checking labels, compare reference instance IDs with predicted semantic components; do not equate connected components with verified cells.'}
    (out/'instance_audit.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(meta,indent=2,ensure_ascii=False))
    print('\nSample instance connectivity:')
    for x in checked[:6]:
        print('  ID',x['cell_id'],'planes',x['n_z_planes_present'],
              'components',x['n_connected_components_6conn'],
              'largest_fraction',round(x['largest_component_fraction'],3))
    print('\nSaved:',out)
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data/inspect/LVILymph'))
    p.add_argument('--out',type=Path,default=Path('runs/instance_reference_audit'))
    p.add_argument('--check-top-n',type=int,default=12)
    args=p.parse_args()
    if args.check_top_n<1 or args.check_top_n>100:
        p.error('--check-top-n must be between 1 and 100')
    audit(args.data_dir,args.out,args.check_top_n)

if __name__=='__main__': main()
