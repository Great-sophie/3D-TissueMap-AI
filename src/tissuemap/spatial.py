"""Exploratory voxel-spaced 3D morphology and marker-expression analysis.

No marker-positive cell in this module is a validated immune-cell phenotype.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import label as ndi_label, center_of_mass, find_objects, distance_transform_edt
from scipy.spatial import cKDTree
from .data import read_manifest, load_sample


def _table_for_sample(image, mask, names, spacing, sample_id, patient_id, mask_source):
    ids = np.unique(mask)
    ids = ids[ids > 0]
    if not len(ids):
        return pd.DataFrame()
    voxvol = float(np.prod(spacing))
    out = []
    # Iterate one ID at a time so instance IDs remain stable and auditable.
    for idx in ids:
        zz,yy,xx = np.nonzero(mask == idx)
        volume = len(zz)*voxvol
        pos=np.column_stack([zz*spacing[0],yy*spacing[1],xx*spacing[2]])
        if len(pos)>3:
            eig=np.linalg.eigvalsh(np.cov(pos,rowvar=False))
            elongation=float(np.sqrt(max(eig[-1],1e-8)/max(eig[0],1e-8)))
        else:
            elongation=1.
        result = dict(sample_id=sample_id,patient_id=patient_id,cell_id=int(idx),
                      mask_source=mask_source, voxel_count=len(zz), volume_um3=volume,
                      elongation_3d=elongation,
                      centroid_z_um=float(np.mean(zz)*spacing[0]),
                      centroid_y_um=float(np.mean(yy)*spacing[1]),
                      centroid_x_um=float(np.mean(xx)*spacing[2]))
        for ci, name in enumerate(names):
            result[f"mean_{name}"] = float(np.mean(image[ci,zz,yy,xx]))
        out.append(result)
    return pd.DataFrame(out)


def _candidate_gate(cells, marker):
    col = f"mean_{marker}"
    if col not in cells.columns:
        return np.zeros(len(cells),dtype=bool), None
    # A within-sample exploratory high-signal gate, NOT a biological cell type label.
    threshold = float(cells[col].quantile(.75))
    return cells[col].to_numpy() > threshold, threshold


def _spatial_edges(cells, max_distance_um=30.):
    coords = cells[["centroid_z_um","centroid_y_um","centroid_x_um"]].to_numpy()
    if len(coords) < 2:
        return pd.DataFrame(columns=["sample_id","cell_a","cell_b","distance_um"])
    pairs = cKDTree(coords).query_pairs(r=max_distance_um, output_type="ndarray")
    return pd.DataFrame([dict(sample_id=cells.iloc[0].sample_id,
                              cell_a=int(cells.iloc[a].cell_id),cell_b=int(cells.iloc[b].cell_id),
                              distance_um=float(np.linalg.norm(coords[a]-coords[b]))) for a,b in pairs])


def _proximity_2d_3d(cells, marker_a="CD8", marker_b="panCK", spacing_z=2.):
    a,_ = _candidate_gate(cells, marker_a)
    b,_ = _candidate_gate(cells, marker_b)
    aa = cells[a]; bb = cells[b]
    results = []
    if len(aa) == 0 or len(bb) == 0:
        return pd.DataFrame(columns=["sample_id","source_cell_id","nearest_3d_um","nearest_same_section_2d_um"])
    positions_b = bb[["centroid_z_um","centroid_y_um","centroid_x_um"]].to_numpy()
    ids_b = bb.cell_id.to_numpy()
    for cell in aa.itertuples():
        v = np.array([cell.centroid_z_um,cell.centroid_y_um,cell.centroid_x_um])
        keep = ids_b != cell.cell_id
        if not keep.any():
            continue
        distance3d = float(np.linalg.norm(positions_b[keep]-v,axis=1).min())
        # Slice-aware 2D comparison: only target cells whose centroid lies in the same Z section.
        same_z = keep & (np.abs(positions_b[:,0]-v[0]) <= spacing_z/2)
        distance2d = float(np.linalg.norm(positions_b[same_z,1:] - v[1:],axis=1).min()) if same_z.any() else np.nan
        results.append(dict(sample_id=cell.sample_id,source_cell_id=int(cell.cell_id),
                            nearest_3d_um=distance3d, nearest_same_section_2d_um=distance2d))
    return pd.DataFrame(results)


def quantify(manifest, output, split="test", predictions=None, use_reference_instances=True, edge_distance=30.):
    df = read_manifest(manifest)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    all_cells, all_edges, all_distances, gates = [],[],[],[]
    for row in df[df.split==split].itertuples():
        image, ref_mask, names = load_sample(row.path)
        if use_reference_instances:
            instances, mask_source = ref_mask, "reference_supplied"
        else:
            if predictions is None:
                raise ValueError("--predictions required to quantify predicted foreground components")
            with np.load(Path(predictions)/f"{row.sample_id}.npz", allow_pickle=False) as p:
                # Connected components are approximate instances; touching cells may merge.
                instances,_ = ndi_label(p["semantic_mask"] > 0)
            mask_source = "predicted_connected_components_not_validated_instances"
        spacing=(row.spacing_z_um,row.spacing_y_um,row.spacing_x_um)
        cells=_table_for_sample(image,instances,names,spacing,row.sample_id,row.patient_id,mask_source)
        if len(cells) == 0:
            continue
        for marker in ("CD8","panCK","CD31"):
            gate,threshold = _candidate_gate(cells,marker)
            cells[f"{marker}_high_exploratory"] = gate
            gates.append(dict(sample_id=row.sample_id,marker=marker,threshold=threshold,
                              method="within-sample 75th-percentile cell intensity; not validated biological gating"))
        all_cells.append(cells)
        all_edges.append(_spatial_edges(cells,max_distance_um=edge_distance))
        all_distances.append(_proximity_2d_3d(cells,spacing_z=spacing[0]))
    for name, parts in (("cells.csv",all_cells),("neighbours_3d.csv",all_edges),("proximity_2d_vs_3d.csv",all_distances)):
        pd.concat(parts,ignore_index=True).to_csv(output/name,index=False) if parts else pd.DataFrame().to_csv(output/name,index=False)
    pd.DataFrame(gates).to_csv(output/"candidate_marker_gates.csv",index=False)
    meta={"samples":len(set(x.sample_id for x in pd.concat(all_cells).itertuples())) if all_cells else 0,
          "analysis":"exploratory spatial protein + physical-space morphology",
          "2d_control":"closest candidate panCK-high cell whose centroid lies within 0.5 Z-spacing of source cell",
          "important":"Expression percentile gates do NOT establish validated immune or tumour-cell identity."
                     " Ground-truth instance masks were not independently audited; reference quality must be checked."}
    (output/"analysis_metadata.json").write_text(json.dumps(meta,indent=2))
    return meta
