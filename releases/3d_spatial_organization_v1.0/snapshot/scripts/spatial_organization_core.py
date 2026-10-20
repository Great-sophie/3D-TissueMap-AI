"""Workflow 08 v1.0 -- voxel-grid pseudo-instance spatial organization.

No physical Z spacing, no 3D Euclidean lengths, no verified biological contacts.
XY proximity is measured between within-plane centroids of distinct IDs that
co-occur in at least one indexed plane. This is not a physical distance.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

VERSION = "1.0.0"
MORPH_COLUMNS = ("instance_id", "voxel_count", "centroid_z_index", "centroid_y_px", "centroid_x_px", "z_span_planes")
PROFILE_COLUMNS = ("instance_id", "z_index", "area_px2", "centroid_y_px", "centroid_x_px")
QC_COLUMNS = ("instance_id", "qc_flag_review", "touches_z_stack_boundary", "components_6connected")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(frame: pd.DataFrame, columns, name: str):
    absent = sorted(set(columns).difference(frame.columns))
    if absent:
        raise ValueError(f"{name}: missing columns {absent}")


def _int_column(frame: pd.DataFrame, column: str, name: str):
    values = frame[column].to_numpy()
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"{name}.{column} must contain finite numbers")
    if not np.all(values == np.floor(values)):
        raise ValueError(f"{name}.{column} must contain integer values")
    return values.astype(np.int64)


def validate_source(mask: np.ndarray, morph: pd.DataFrame, profile: pd.DataFrame,
                    qc: pd.DataFrame, summary: dict, mask_path: Path) -> None:
    """Reject swapped tables, missing IDs, corrupt provenance and profile inconsistencies."""
    if mask.ndim != 3 or any(dim < 1 for dim in mask.shape):
        raise ValueError(f"Expected nonempty ZYX mask; got shape {mask.shape}")
    if not np.issubdtype(mask.dtype, np.integer) or np.issubdtype(mask.dtype, np.bool_):
        raise ValueError("Mask must contain integer IDs")
    if np.issubdtype(mask.dtype, np.signedinteger) and np.any(mask < 0):
        raise ValueError("Mask must not have negative IDs")
    for df, cols, name in ((morph, MORPH_COLUMNS, "morphometry"),
                           (profile, PROFILE_COLUMNS, "profiles"), (qc, QC_COLUMNS, "QC")):
        _require(df, cols, name)
        ids = _int_column(df, "instance_id", name)
        if len(ids) and (ids <= 0).any():
            raise ValueError(f"{name}: instance_id must be positive")
    ids_mask, counts = np.unique(mask, return_counts=True)
    valid = ids_mask > 0
    ids_mask, counts = ids_mask[valid].astype(np.int64), counts[valid].astype(np.int64)
    if not len(ids_mask):
        raise ValueError("No foreground pseudo-instance IDs")
    mids = _int_column(morph, "instance_id", "morphometry")
    qids = _int_column(qc, "instance_id", "QC")
    if morph.instance_id.duplicated().any() or qc.instance_id.duplicated().any():
        raise ValueError("Duplicate instance IDs in morphometry or QC")
    if set(mids) != set(ids_mask) or set(qids) != set(ids_mask):
        raise ValueError("ID mismatch between mask, morphometry, and QC")
    mcounts = pd.Series(_int_column(morph, "voxel_count", "morphometry"), index=mids)
    if not np.array_equal(mcounts.loc[ids_mask].to_numpy(), counts):
        raise ValueError("Morphometry voxel counts disagree with source mask")
    pids = _int_column(profile, "instance_id", "profiles")
    pz = _int_column(profile, "z_index", "profiles")
    areas = _int_column(profile, "area_px2", "profiles")
    if any((pz < 0) | (pz >= mask.shape[0])) or any(areas <= 0):
        raise ValueError("Invalid z_index or area_px2 in profiles")
    if profile.duplicated(["instance_id", "z_index"]).any():
        raise ValueError("Duplicate (instance_id, z_index) in profiles")
    if set(pids) != set(ids_mask):
        raise ValueError("Profile IDs disagree with mask")
    for col in ("centroid_y_px", "centroid_x_px"):
        if not np.isfinite(profile[col].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Invalid profile centroids: {col}")
    # Exact per-plane label area check: affordable for 16 indexed planes.
    for z in range(mask.shape[0]):
        observed_labels, observed_counts = np.unique(mask[z], return_counts=True)
        fg = observed_labels > 0
        observed = dict(zip(observed_labels[fg].astype(int), observed_counts[fg].astype(int)))
        subset = profile.loc[profile.z_index == z]
        expected = dict(zip(subset.instance_id.astype(int), subset.area_px2.astype(int)))
        if observed != expected:
            raise ValueError(f"Plane {z}: morphometry profile differs from mask")
    shape = summary.get("shape_assumed_zyx")
    if shape is not None and list(mask.shape) != list(shape):
        raise ValueError("Mask differs from Workflow07 stored shape")
    if summary.get("n_instance_ids") is not None and summary["n_instance_ids"] != len(ids_mask):
        raise ValueError("Workflow07 instance count mismatch")
    if summary.get("foreground_voxels") is not None and summary["foreground_voxels"] != int(counts.sum()):
        raise ValueError("Workflow07 foreground voxel count mismatch")
    digest = summary.get("source_mask_sha256")
    if not digest:
        raise ValueError("Workflow07 summary lacks source_mask_sha256; cannot verify provenance")
    if sha256(mask_path) != digest:
        raise ValueError("Source mask SHA-256 differs from Workflow07 provenance")


def xy_radius_graph(profile: pd.DataFrame, radius_px: float, k: int) -> pd.DataFrame:
    """Per-plane centroid distances, radius-limited top-k; no cross-Z Euclidean metric."""
    if not math.isfinite(radius_px) or radius_px <= 0 or not isinstance(k, int) or k < 1:
        raise ValueError("radius_px must be finite positive and k must be >=1")
    # [a,b] -> [min 2D distance, set of Z indices, z index at minimum]
    edges = {}
    for z, frame in profile.groupby("z_index", sort=True):
        ids = frame.instance_id.to_numpy(dtype=np.int64)
        xy = frame[["centroid_y_px", "centroid_x_px"]].to_numpy(dtype=np.float64)
        if len(ids) < 2:
            continue
        pairs = cKDTree(xy).query_pairs(r=radius_px, output_type="ndarray")
        for i, j in pairs:
            a, b = sorted((int(ids[i]), int(ids[j])))
            d = float(np.linalg.norm(xy[i] - xy[j]))
            key = (a, b)
            if key not in edges:
                edges[key] = [d, {int(z)}, int(z)]
            else:
                old = edges[key]
                old[1].add(int(z))
                if d < old[0]:
                    old[0], old[2] = d, int(z)
    columns = ["instance_a", "instance_b", "min_xy_plane_centroid_distance_px",
               "min_distance_z_index", "shared_z_planes_within_radius", "rank_a_within_radius",
               "rank_b_within_radius", "topk_by_either", "topk_by_both"]
    if not edges:
        return pd.DataFrame(columns=columns)
    ranked = defaultdict(list)
    for (a,b), (d, _, _) in edges.items():
        ranked[a].append((d,b))
        ranked[b].append((d,a))
    ranks = {}
    for node, ds in ranked.items():
        for rank, (_, other) in enumerate(sorted(ds, key=lambda t: (t[0], t[1])), start=1):
            ranks[(node,other)] = rank
    rows = []
    for (a,b), (d,zset,min_z) in sorted(edges.items()):
        ra, rb = ranks[(a,b)], ranks[(b,a)]
        rows.append((a,b,d,min_z,len(zset),ra,rb,ra <= k or rb <= k,ra <= k and rb <= k))
    return pd.DataFrame(rows, columns=columns)


def _count_interfaces(a: np.ndarray, b: np.ndarray, orient: int, agg: dict):
    keep = (a > 0) & (b > 0) & (a != b)
    if not np.any(keep):
        return
    left = a[keep].astype(np.int64)
    right = b[keep].astype(np.int64)
    # Unique encoding via pair matrix; no possible overflow from max-ID multiplication.
    pairs = np.column_stack((np.minimum(left,right), np.maximum(left,right)))
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    for (la,lb), count in zip(unique, counts):
        key = (int(la), int(lb))
        if key not in agg:
            agg[key] = [0,0,0]  # [z,y,x]
        agg[key][orient] += int(count)


def voxel_interface_graph(mask: np.ndarray) -> pd.DataFrame:
    """Count each face once; Z/Y/X faces remain separate, not calibrated contact area."""
    agg = {}
    for z in range(mask.shape[0]):
        plane = mask[z]
        if z + 1 < mask.shape[0]:
            _count_interfaces(plane, mask[z+1], 0, agg)
        if plane.shape[0] >= 2:
            _count_interfaces(plane[:-1,:], plane[1:,:], 1, agg)
        if plane.shape[1] >= 2:
            _count_interfaces(plane[:,:-1], plane[:,1:], 2, agg)
    columns = ["instance_a", "instance_b", "z_face_count_voxel_grid",
               "y_face_count_voxel_grid", "x_face_count_voxel_grid", "xy_face_count_voxel_grid",
               "all_face_count_voxel_grid", "interface_type"]
    rows = []
    for (a,b), (z,y,x) in sorted(agg.items()):
        label = "mixed_XY_and_Z" if (x+y)>0 and z>0 else "XY_only" if (x+y)>0 else "Z_only"
        rows.append((a,b,z,y,x,x+y,z+y+x,label))
    return pd.DataFrame(rows,columns=columns)


def spatial_features(morph: pd.DataFrame, qc: pd.DataFrame, neighbors: pd.DataFrame,
                     interfaces: pd.DataFrame) -> pd.DataFrame:
    ids = morph.instance_id.astype(np.int64).tolist()
    near = Counter(); topk = Counter(); mutual = Counter(); closest = {}
    for rec in neighbors.itertuples(index=False):
        a,b = int(rec.instance_a),int(rec.instance_b)
        near[a] += 1; near[b] += 1
        d = float(rec.min_xy_plane_centroid_distance_px)
        closest[a] = min(d,closest.get(a, float('inf')))
        closest[b] = min(d,closest.get(b, float('inf')))
        if rec.topk_by_either:
            topk[a] += 1; topk[b] += 1
        if rec.topk_by_both:
            mutual[a] += 1; mutual[b] += 1
    xy_degree = Counter(); z_degree=Counter(); total_faces=Counter()
    for rec in interfaces.itertuples(index=False):
        a,b=int(rec.instance_a),int(rec.instance_b)
        for node in (a,b):
            if rec.xy_face_count_voxel_grid:
                xy_degree[node]+=1
            if rec.z_face_count_voxel_grid:
                z_degree[node]+=1
            total_faces[node] += int(rec.all_face_count_voxel_grid)
    df = morph[["instance_id","centroid_z_index","centroid_y_px","centroid_x_px","z_span_planes","voxel_count"]].copy()
    flags = qc[["instance_id","qc_flag_review","touches_z_stack_boundary","components_6connected"]].copy()
    df = df.merge(flags, on="instance_id", validate="one_to_one", how="left")
    df["n_xy_radius_neighbors"] = [near[int(i)] for i in ids]
    df["n_xy_radius_limited_topk_either"] = [topk[int(i)] for i in ids]
    df["n_xy_radius_limited_topk_mutual"] = [mutual[int(i)] for i in ids]
    df["nearest_xy_within_radius_px"] = [closest.get(int(i), np.nan) for i in ids]
    df["n_xy_face_interface_neighbors"] = [xy_degree[int(i)] for i in ids]
    df["n_z_face_interface_neighbors"] = [z_degree[int(i)] for i in ids]
    df["total_neighbor_interface_faces_voxel_grid"] = [total_faces[int(i)] for i in ids]
    return df.sort_values('instance_id').reset_index(drop=True)


def build_spatial(mask: np.ndarray, morph: pd.DataFrame, profiles: pd.DataFrame, qc: pd.DataFrame,
                  radius_px: float = 24.0, knn_k: int = 5) -> dict[str,pd.DataFrame]:
    neighbors = xy_radius_graph(profiles,radius_px,knn_k)
    interfaces = voxel_interface_graph(mask)
    features = spatial_features(morph,qc,neighbors,interfaces)
    centroids = features[["instance_id","centroid_z_index","centroid_y_px","centroid_x_px",
                          "z_span_planes","touches_z_stack_boundary","qc_flag_review"]].copy()
    return {"instance_centroids.csv":centroids,
            "spatial_neighbors.csv":neighbors,
            "instance_contact_graph.csv":interfaces,
            "instance_spatial_features.csv":features}


def summary_data(mask: np.ndarray, tables: dict[str,pd.DataFrame], radius_px: float, knn_k: int) -> dict:
    n=tables["spatial_neighbors.csv"]; c=tables["instance_contact_graph.csv"]; f=tables["instance_spatial_features.csv"]
    strat={}
    for label, sel in [("unflagged",f.loc[~f.qc_flag_review.astype(bool)]),
                       ("flagged",f.loc[f.qc_flag_review.astype(bool)])]:
        strat[label] = {"n_instances":len(sel),
                        "mean_xy_radius_neighbors":float(sel.n_xy_radius_neighbors.mean()) if len(sel) else None,
                        "median_xy_radius_neighbors":float(sel.n_xy_radius_neighbors.median()) if len(sel) else None,
                        "mean_xy_interface_neighbors":float(sel.n_xy_face_interface_neighbors.mean()) if len(sel) else None}
    return {"version": VERSION,"shape_assumed_zyx":list(mask.shape),
            "n_instances":len(f),"xy_radius_px":radius_px,"radius_limited_knn_k":knn_k,
            "n_xy_radius_pairs":len(n),"n_radius_limited_knn_pairs_either":int(n.topk_by_either.sum()),
            "n_radius_limited_knn_pairs_mutual":int(n.topk_by_both.sum()),
            "n_interface_pairs":len(c),
            "n_interface_pairs_xy":int((c.xy_face_count_voxel_grid>0).sum()),
            "n_interface_pairs_z":int((c.z_face_count_voxel_grid>0).sum()),
            "sum_interface_faces_xy":int(c.xy_face_count_voxel_grid.sum()),
            "sum_interface_faces_z":int(c.z_face_count_voxel_grid.sum()),
            "qc_strata":strat,
            "interpretation_limits":[
                "Single specimen; author watershed pseudo-ID is not independent biological cell truth.",
                "XY neighbor edges require at least one shared indexed Z plane and use per-plane centroid XY pixels only.",
                "Radius-limited top-k is not unrestricted k-nearest-neighbor search.",
                "Voxel-face interfaces are segmentation-grid adjacency, not verified physical cell contacts or surface area.",
                "No calibrated 3D distance, density per mm^3, inferred interaction, or protein-to-instance assignment.",
                "Boundary-touching instances are flagged for stratification, not automatically errors."]}


def draw_figures(tables:dict[str,pd.DataFrame], output:Path, radius_px:float) -> list[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    f=tables['instance_spatial_features.csv']
    names=[]
    fig, ax = plt.subplots(figsize=(9,7))
    sc=ax.scatter(f.centroid_x_px, f.centroid_y_px, c=f.n_xy_radius_neighbors,
                  s=2.5, alpha=.65, rasterized=True)
    ax.invert_yaxis();ax.set_aspect('equal')
    ax.set(xlabel='X pixel',ylabel='Y pixel',title=f'XY centroid neighbor count (radius {radius_px:g} px; shared Z only)')
    fig.colorbar(sc,ax=ax,label='Number of distinct pseudo-ID neighbors within XY radius')
    fig.tight_layout();fn='xy_neighborhood_map.png';fig.savefig(output/fn,dpi=160);plt.close(fig);names.append(fn)
    fig,ax=plt.subplots(figsize=(8,5))
    data=f.n_xy_radius_neighbors.to_numpy()
    ax.hist(data,bins=min(65,max(5,int(np.max(data)+1))))
    ax.set(xlabel='XY radius neighbors (distinct IDs)',ylabel='Pseudo-instance count',title='Same-Z-plane XY neighborhood distribution')
    fig.tight_layout();fn='xy_neighbor_count_distribution.png';fig.savefig(output/fn,dpi=150);plt.close(fig);names.append(fn)
    fig,ax=plt.subplots(figsize=(8,5))
    data=[f.loc[~f.qc_flag_review.astype(bool),'n_xy_radius_neighbors'],
          f.loc[f.qc_flag_review.astype(bool),'n_xy_radius_neighbors']]
    ax.boxplot(data, tick_labels=['Unflagged','Flagged'],showfliers=False)
    ax.set(ylabel='XY radius neighbors',title='QC-stratified XY neighborhood counts (descriptive)')
    fig.tight_layout();fn='qc_stratified_neighborhood.png';fig.savefig(output/fn,dpi=150);plt.close(fig);names.append(fn)
    return names


def save_run(mask:np.ndarray, mask_path:Path, morph_dir:Path, out:Path,
             radius_px:float=24., knn_k:int=5, figures:bool=True) -> dict:
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {out}')
    files = ("cell_morphometry.csv","cell_slice_profiles.csv","cell_morphology_qc.csv","morphometry_summary.json")
    for name in files:
        if not (morph_dir/name).is_file():
            raise FileNotFoundError(f'Missing Workflow07 prerequisite: {morph_dir/name}')
    morph=pd.read_csv(morph_dir/files[0]); profile=pd.read_csv(morph_dir/files[1]); qc=pd.read_csv(morph_dir/files[2])
    summary=json.loads((morph_dir/files[3]).read_text(encoding='utf8'))
    validate_source(mask,morph,profile,qc,summary,mask_path)
    if not math.isfinite(radius_px) or radius_px<=0 or knn_k<1:
        raise ValueError('Invalid radius or top-k setting')
    # Ensure booleans remain boolean after CSV import; avoid string 'False' -> True.
    for c in ('qc_flag_review','touches_z_stack_boundary'):
        if not pd.api.types.is_bool_dtype(qc[c]):
            raise ValueError(f'QC column must parse as true booleans: {c}')
    tables=build_spatial(mask,morph,profile,qc,radius_px,knn_k)
    report=summary_data(mask,tables,radius_px,knn_k)
    # Cross-check against Workflow07 surface-face count: each inter-ID face counted twice there.
    if 'other_instance_interface_faces_voxel' in morph:
        expected=int(morph.other_instance_interface_faces_voxel.sum())
        seen=int(tables['instance_contact_graph.csv'].all_face_count_voxel_grid.sum())*2
        if expected!=seen:
            raise ValueError(f'Interface face-count disagreement with Workflow07: {expected} vs {seen}')
        report['workflow07_interface_face_crosscheck'] = 'PASS'
    report['source_mask_sha256']=sha256(mask_path)
    report['workflow07_summary_sha256']=sha256(morph_dir/files[3])
    report['workflow07_files_sha256']={name:sha256(morph_dir/name) for name in files}
    report['software_versions']={'python':sys.version.split()[0], 'numpy':np.__version__,
                                  'pandas':pd.__version__}
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.workflow08_staging_',dir=out.parent))
    try:
        for name,df in tables.items():
            df.to_csv(stage/name,index=False,float_format='%.10g')
        if figures:
            report['figures']=draw_figures(tables,stage/'figures',radius_px)
        else:
            report['figures']=[]
        (stage/'spatial_organization_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        digest={str(p.relative_to(stage)):sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        (stage/'output_file_sha256.json').write_text(json.dumps(digest,indent=2),encoding='utf8')
        if out.exists():
            raise FileExistsError('Concurrent output exists; refusing to overwrite')
        stage.rename(out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return report


def verify_run(out:Path, mask_path:Path | None=None) -> dict:
    path=out/'output_file_sha256.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing output manifest: {path}')
    hashes=json.loads(path.read_text(encoding='utf8'))
    for name,expected in hashes.items():
        p=out/name
        if not p.is_file() or sha256(p)!=expected:
            raise ValueError(f'Missing or changed Workflow08 output: {name}')
    # Reject untracked output files (except the manifest itself).
    actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
    if actual != set(hashes) | {'output_file_sha256.json'}:
        raise ValueError('Untracked Workflow08 output files found')
    report=json.loads((out/'spatial_organization_summary.json').read_text(encoding='utf8'))
    if mask_path is not None and sha256(mask_path)!=report['source_mask_sha256']:
        raise ValueError('Source mask does not match output provenance')
    return {'n_verified_files':len(hashes),'n_instances':report['n_instances'],
            'n_xy_radius_pairs':report['n_xy_radius_pairs'], 'n_interface_pairs':report['n_interface_pairs']}
