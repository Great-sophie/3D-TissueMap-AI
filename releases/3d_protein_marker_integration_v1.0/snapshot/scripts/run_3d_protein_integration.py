#!/usr/bin/env python3
"""Workflow09: audit marker stacks then aggregate raw signal only from explicit validated-shaped manifest."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from protein_marker_integration_core import marker_discovery,extract,verify_run


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--discover',action='store_true',help='Read image headers, create empty marker manifest template')
    group.add_argument('--extract',action='store_true',help='Aggregate marker pixel values within pseudo-ID masks; exploratory only')
    group.add_argument('--verify-run',action='store_true',help='Verify output SHA-256')
    p.add_argument('--data-dir',type=Path,default=Path('data/inspect/LVILymph'))
    p.add_argument('--mask',type=Path,default=None)
    p.add_argument('--manifest',type=Path,default=None)
    p.add_argument('--morphometry-dir',type=Path,default=Path('runs/3d_cell_morphometry_v1'))
    p.add_argument('--spatial-dir',type=Path,default=Path('runs/3d_spatial_organization_v1'))
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--max-files',type=int,default=6000)
    p.add_argument('--allow-unverified-geometry',action='store_true',
                   help='Explicit consent for exploratory index-grid extraction; not registration validation')
    p.add_argument('--no-figures',action='store_true')
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if a.discover:
        out=a.out or Path('runs/protein_marker_discovery_v1')
        result=marker_discovery(a.data_dir,out,a.mask,max_files=a.max_files)
        print('DISCOVERY COMPLETE (NOT a registration pass):',json.dumps(result,indent=2,ensure_ascii=False))
        print('Fill',out/'marker_manifest_template.json','with explicitly verified marker image paths/order/channel indices.')
        return result
    if a.extract:
        if a.manifest is None:raise SystemExit('Provide --manifest after checking marker files and channel order')
        out=a.out or Path('runs/3d_protein_marker_integration_v1')
        result=extract(a.manifest,a.morphometry_dir,a.spatial_dir,out,
                       allow_unverified_geometry=a.allow_unverified_geometry,figures=not a.no_figures)
        print('EXPLORATORY EXTRACTION (not independent registration validation):',
              json.dumps({k:result[k] for k in ('n_instances','n_markers','marker_names','marker_geometry_status')},
                         indent=2,ensure_ascii=False))
        print('SAVED:',out)
        return result
    out=a.out or Path('runs/3d_protein_marker_integration_v1')
    result=verify_run(out,a.manifest)
    print('PASS: Workflow09 output SHA-256:',json.dumps(result,ensure_ascii=False))
    return result

if __name__=='__main__':main()
