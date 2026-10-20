"""Convert a frozen XY region merge classifier into additive RAG edge evidence.

XY scores come from the saved classifier, evaluated on ALL pure XY interfaces.
Z-involved interfaces retain original voxel-affinity evidence because the region
classifier was trained on XY ONLY. These are NOT calibrated probabilities.
Reference IDs do not enter the construction of these edge features or scores.
After regions merge, scores are weighted averages of original *atomic* learned
edge scores, NOT a newly inferred classifier prediction on the merged geometry.
"""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np

from dynamic_region_merging import build_region_evidence
from region_interface_features import FEATURE_COLUMNS, extract_region_interfaces


def load_region_classifier(path: Path):
    """Load only a trusted, locally created joblib model."""
    artifact = joblib.load(Path(path))
    if not isinstance(artifact, dict) or 'model' not in artifact:
        raise ValueError('Expected locally trained region model bundle')
    cols = artifact.get('feature_columns')
    if cols != FEATURE_COLUMNS:
        raise ValueError('Feature mismatch: model must use original leak-free region features')
    if artifact.get('positive_label') != 1:
        raise ValueError('Expected P(same=1) classifier')
    if not hasattr(artifact['model'], 'predict_proba'):
        raise ValueError('Model must implement predict_proba')
    return artifact


def learned_interface_evidence(sv, p_same, image, bundle, *, z_weight=0.25, max_logit=4.0):
    """Return [N,10] additive evidence, metadata; unknown XY edges ARE scored.

    Pure XY original edges receive a learned score. Their raw negative fraction
    remains unchanged for an explicit raw-repulsion-veto ablation. Edges with
    ANY Z contact retain their unaltered original X/Y/Z affinity evidence.
    """
    sv = np.asarray(sv)
    img = np.asarray(image)
    p_same = np.asarray(p_same)
    if sv.ndim != 3 or img.shape != sv.shape or p_same.shape != (3, *sv.shape):
        raise ValueError('Expected sv/image [Z,Y,X], affinity [3,Z,Y,X]')
    if max_logit <= 0 or not np.isfinite(max_logit):
        raise ValueError('max_logit must be finite and positive')
    if bundle.get('feature_columns') != FEATURE_COLUMNS:
        raise ValueError('Model feature schema mismatch')
    raw_evidence = build_region_evidence(sv,p_same,z_weight)
    # All-zero dummy ref prohibits a reference label from entering inference.
    features, _ = extract_region_interfaces(sv,p_same,img,np.zeros_like(sv),
                                           z_weight=z_weight)
    xy_mask = (features['interface_z_faces'].to_numpy()==0) if len(features) else np.zeros(0,bool)
    xy_count=int(xy_mask.sum())
    mixed_count=int(len(features)-xy_count)
    out=raw_evidence.copy()
    stats = {'n_edges':int(len(features)), 'n_learned_pure_xy':xy_count,
             'n_raw_z_involved':mixed_count,
             'aggregation':'weighted mean of frozen per-atomic-interface classifier logits',
             'z_fallback':'original affinity for any Z-involved interface',
             'repulsion_veto':'original voxel-affinity negative-share when enabled',
             'reference_used_for_prediction':False,
             'p_same_q10_q50_q90':None,
             'fraction_learned_xy_p_same_ge_0_5':None}
    if not len(features):
        if len(out):
            raise AssertionError('Feature/evidence interface count mismatch')
        return out, stats
    feature_pairs=features[['sv_a','sv_b']].to_numpy(np.int64)
    evidence_pairs=out[:,:2].astype(np.int64)
    if not np.array_equal(feature_pairs,evidence_pairs):
        raise AssertionError('Feature/evidence interface pair alignment mismatch')
    if xy_count:
        model=bundle['model']
        cols=bundle['feature_columns']
        pred=model.predict_proba(features.loc[xy_mask,cols])
        labels=list(model.classes_)
        if 1 not in labels:
            raise ValueError('Classifier has no positive class 1')
        p=np.asarray(pred[:,labels.index(1)],dtype=np.float64)
        if not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):
            raise ValueError('Model returned invalid probability')
        p=np.clip(p,1e-4,1-1e-4)
        learned_logit=np.clip(np.log(p/(1-p)),-max_logit,max_logit)
        nxy=features.loc[xy_mask,['interface_x_faces','interface_y_faces']].sum(axis=1).to_numpy(dtype=float)
        if not np.all(nxy>=1):
            raise AssertionError('Pure XY interfaces must have XY faces')
        out[xy_mask,2]=learned_logit*nxy    # XY weighted sum log odds
        out[xy_mask,3]=nxy                  # XY interface-face weight
        # out[:,4] retains raw affinity-based negative-share evidence
        # out[:,5] retains true number of contacting XY faces
        q=np.quantile(p,[.10,.50,.90])
        stats['p_same_q10_q50_q90']=[float(z) for z in q]
        stats['fraction_learned_xy_p_same_ge_0_5']=float(np.mean(p>=.5))
    return out,stats
