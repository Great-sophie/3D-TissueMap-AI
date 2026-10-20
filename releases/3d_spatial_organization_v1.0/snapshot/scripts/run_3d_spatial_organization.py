#!/usr/bin/env python3
"""Workflow 08 v1: spatial graphs derived from Workflow07 and author 3D pseudo-ID TIFF."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tifffile
from spatial_organization_core import save_run, verify_run


def get_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mask',type=Path,default=None,help='Exact author pseudo-ID TIFF; otherwise use data-dir')
    p.add_argument('--data-dir',type=Path,default=Path('data/inspect/LVILymph'))
    p.add_argument('--morphometry-dir',type=Path,default=Path('runs/3d_cell_morphometry_v1'))
    p.add_argument('--out',type=Path,default=Path('runs/3d_spatial_organization_v1'))
    p.add_argument('--xy-radius-px',type=float,default=24.,help='XY centroid neighborhood radius, pixels')
    p.add_argument('--knn-k',type=int,default=5,help='Top-k per node WITHIN radius only')
    p.add_argument('--assume-zyx',action='store_true',help='Required explicit ZYX index-order acknowledgment')
    p.add_argument('--no-figures',action='store_true')
    p.add_argument('--verify-run',action='store_true',help='Read-only SHA-256 verification; does not recompute graphs')
    return p


def main(argv=None):
    args=get_parser().parse_args(argv)
    if args.verify_run:
        result=verify_run(args.out,args.mask)
        print('PASS: Workflow08 output SHA-256:',json.dumps(result,ensure_ascii=False))
        return result
    if not args.assume_zyx:
        raise SystemExit('Add --assume-zyx; physical registration and Z spacing remain unverified')
    if args.mask is not None:
        path=args.mask
    else:
        options=sorted(args.data_dir.glob('full_mask*.tif'))+sorted(args.data_dir.glob('full_mask*.tiff'))
        if len(options)!=1:
            raise ValueError(f'Expected one full_mask TIFF at {args.data_dir}, got {len(options)}; specify --mask')
        path=options[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    if args.out.exists():
        raise FileExistsError(f'Output already exists: {args.out}')
    with tifffile.TiffFile(path) as tf:
        if len(tf.series)!=1:
            raise ValueError('Expected exactly one TIFF series')
        print('TIFF series axes:',tf.series[0].axes,'shape:',tf.series[0].shape)
        mask=tf.asarray()
    result=save_run(mask,path,args.morphometry_dir,args.out,
                    radius_px=args.xy_radius_px,knn_k=args.knn_k,figures=not args.no_figures)
    show={k:result[k] for k in ('n_instances','n_xy_radius_pairs','n_interface_pairs',
                                'n_interface_pairs_xy','n_interface_pairs_z',
                                'sum_interface_faces_xy','sum_interface_faces_z')}
    print(json.dumps(show,indent=2,ensure_ascii=False))
    print('SAVED:',args.out)
    print('CAUTION: voxel-grid adjacency is NOT validated cell contact; XY distance is not 3D physical distance.')
    return result

if __name__=='__main__':
    main()
