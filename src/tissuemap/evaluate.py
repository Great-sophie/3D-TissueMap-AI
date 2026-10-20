"""Segmentation evaluation with test-only inference and spacing-aware HD95."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt
from .data import read_manifest, load_sample


def binary_metrics(ref, pred, spacing=(1.,1.,1.)):
    ref, pred = np.asarray(ref, dtype=bool), np.asarray(pred, dtype=bool)
    if ref.shape != pred.shape:
        raise ValueError("Predicted and reference segmentation shapes differ")
    if not ref.any() and not pred.any():
        return dict(dice=1., iou=1., hd95_um=0.)
    if not ref.any() or not pred.any():
        return dict(dice=0., iou=0., hd95_um=None)
    intersection = np.logical_and(ref, pred).sum()
    dice = 2*intersection/(ref.sum()+pred.sum())
    iou = intersection/np.logical_or(ref,pred).sum()
    ref_surface = ref & ~binary_erosion(ref)
    pred_surface = pred & ~binary_erosion(pred)
    d_to_ref = distance_transform_edt(~ref_surface, sampling=spacing)
    d_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([d_to_ref[pred_surface],d_to_pred[ref_surface]])
    return dict(dice=float(dice), iou=float(iou), hd95_um=float(np.percentile(distances,95)))


def evaluate(manifest, predictions, output, split="test"):
    df = read_manifest(manifest)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in df[df.split==split].itertuples():
        _, mask, _ = load_sample(row.path)
        f = Path(predictions)/f"{row.sample_id}.npz"
        if not f.is_file():
            raise FileNotFoundError(f"Missing prediction: {f}")
        with np.load(f, allow_pickle=False) as dat:
            pred = dat["semantic_mask"]
        mm = binary_metrics(mask > 0, pred, spacing=(row.spacing_z_um,row.spacing_y_um,row.spacing_x_um))
        rows.append(dict(sample_id=row.sample_id, patient_id=row.patient_id, split=row.split, **mm))
    if not rows:
        raise ValueError("No held-out samples available")
    pd.DataFrame(rows).to_csv(output/"metrics.csv", index=False)
    summary = {"split": split, "n_samples": len(rows), "n_patients": len(set(r['patient_id'] for r in rows)),
               "mean_dice": float(np.mean([r['dice'] for r in rows])),
               "mean_iou": float(np.mean([r['iou'] for r in rows])),
               "hd95_note": "HD95 in physical micrometers, null if one segmentation is empty",
               "caution": "These are semantic foreground metrics, not validated individual-cell instance detection metrics."}
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary
