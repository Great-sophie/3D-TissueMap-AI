"""Exploratory morphology + spatial protein fusion without invented clinical labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2


def _standardize(x):
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Feature table contains non-finite values")
    sd=x.std(axis=0)
    return (x-x.mean(axis=0))/np.where(sd>1e-8,sd,1.)


def _kmeans(x,k,seed):
    # Deterministic retries to avoid empty clusters in very small demos.
    for attempt in range(5):
        with np.errstate(all="ignore"):
            _,labels=kmeans2(x,k,iter=30,minit="points",seed=seed+attempt)
        if len(np.unique(labels))==k:
            return labels
    raise ValueError("Clustering degenerate; inspect marker variance and reduce k")


def fuse(cell_table, out, k=2, seed=7, morphology_weight=1., protein_weight=1.):
    """Cluster fused cell-level features. *Not* trained multimodal disease prediction."""
    cells=pd.read_csv(cell_table)
    if len(cells)<k or k<2:
        raise ValueError("Need at least k cells and k>=2")
    markers=[c for c in cells.columns if c.startswith("mean_")]
    if not markers:
        raise ValueError("No measured marker columns (mean_*)")
    morph_cols=[c for c in ("volume_um3","elongation_3d") if c in cells]
    if not morph_cols:
        raise ValueError("Missing 3D morphology columns")
    xm=cells[morph_cols].to_numpy(dtype=np.float64)
    for j,c in enumerate(morph_cols):
        if c=="volume_um3": xm[:,j]=np.log1p(np.maximum(xm[:,j],0))
    xp=np.arcsinh(cells[markers].to_numpy(dtype=np.float64)/5.)
    morph=_standardize(xm)
    protein=_standardize(xp)
    joint=np.concatenate([morphology_weight*morph / np.sqrt(morph.shape[1]),
                          protein_weight*protein / np.sqrt(protein.shape[1])],axis=1)
    labels_m=_kmeans(morph,k,seed)
    labels_p=_kmeans(protein,k,seed)
    labels_j=_kmeans(joint,k,seed)
    embedded=joint-joint.mean(axis=0)
    _,_,vh=np.linalg.svd(embedded,full_matrices=False)
    projection=embedded@vh[:2].T
    if projection.shape[1]==1: projection=np.column_stack((projection,np.zeros(len(cells))))
    result=cells[["sample_id","patient_id","cell_id"]].copy()
    result["morphology_cluster"]=labels_m
    result["protein_cluster"]=labels_p
    result["joint_cluster"]=labels_j
    result["joint_pc1"]=projection[:,0]
    result["joint_pc2"]=projection[:,1]
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/"exploratory_clusters.csv",index=False)
    joined=result.merge(cells,on=["sample_id","patient_id","cell_id"],validate="one_to_one")
    stats=joined.groupby("joint_cluster")[morph_cols+markers].median(numeric_only=True)
    stats.to_csv(out/"cluster_feature_medians.csv")
    metadata=dict(n_cells=int(len(cells)),n_clusters=k,
                  morphology_features=morph_cols,protein_features=markers,
                  method="independent z-score scaling; concatenation; unsupervised k-means; PCA visualization",
                  limitations="Exploratory morpho-proteomic integration only. No ground truth cell types or disease prediction; clusters require biological validation."
                  " Sample/batch effects have not been removed; single-sample demos cannot assess generalization.")
    (out/"fusion_metadata.json").write_text(json.dumps(metadata,indent=2))
    return metadata
