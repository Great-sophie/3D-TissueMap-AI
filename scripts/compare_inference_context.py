#!/usr/bin/env python3
"""3D-TissueMap AI: validation-only inference-context diagnostic.

Use previously trained 3D U-Net to compare full-volume inference and
sliding-window inference with the same Z/XY patch size as training.
No model retraining, no hyperparameter tuning on test tiles.

From the project root:
 python scripts/compare_inference_context.py \
   --data-dir data/inspect/LVILymph \
   --run-dir runs/segmentation_baselines \
   --out runs/inference_context_qc
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_segmentation import UNet, predict_tile, read_data  # noqa: E402


def starts(n: int, window: int):
    """Half-overlapping window starts that guarantee full coverage."""
    if window > n or window < 2:
        raise ValueError(f'Invalid patch dimension: {window} for axis {n}')
    stride = max(1, window // 2)
    ans = list(range(0, n - window + 1, stride))
    if ans[-1] != n - window:
        ans.append(n - window)
    return ans


@torch.inference_mode()
def matched_window_predict(model, volume: np.ndarray, device: str,
                           patch_z: int, patch_xy: int) -> np.ndarray:
    """Blend per-window probabilities (uniform average; diagnostic only)."""
    if volume.ndim != 3:
        raise ValueError('Expected ZYX image')
    z, y, x = volume.shape
    acc = np.zeros((z, y, x), dtype=np.float32)
    count = np.zeros((z, y, x), dtype=np.uint16)
    model.eval()
    for z0 in starts(z, patch_z):
        for y0 in starts(y, patch_xy):
            for x0 in starts(x, patch_xy):
                patch = np.ascontiguousarray(volume[
                    z0:z0+patch_z, y0:y0+patch_xy, x0:x0+patch_xy
                ], dtype=np.float32)
                xt = torch.from_numpy(patch)[None, None].to(device)
                pred = torch.sigmoid(model(xt))[0, 0].cpu().numpy()
                acc[z0:z0+patch_z, y0:y0+patch_xy, x0:x0+patch_xy] += pred
                count[z0:z0+patch_z, y0:y0+patch_xy, x0:x0+patch_xy] += 1
    if np.any(count == 0):
        raise RuntimeError('Some voxels were not predicted')
    return acc / count


def confusion_metrics(pred_prob, ref, guard=12):
    if guard:
        pred_prob = pred_prob[:, guard:-guard, guard:-guard]
        ref = ref[:, guard:-guard, guard:-guard]
    positive = ref > 0
    decision = pred_prob >= 0.5
    tp = int((decision & positive).sum())
    fp = int((decision & ~positive).sum())
    fn = int((~decision & positive).sum())
    tn = int((~decision & ~positive).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                reference_fg=float(positive.mean()),
                predicted_fg=float(decision.mean()),
                mean_probability=float(pred_prob.mean()),
                background_median=float(np.median(pred_prob[~positive])) if (~positive).any() else None,
                dice=float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else None,
                precision=float(tp/(tp+fp)) if (tp+fp) else None,
                recall=float(tp/(tp+fn)) if (tp+fn) else None,
                background_fp_rate=float(fp/(fp+tn)) if (fp+tn) else None)


def aggregate(rows):
    tp=sum(int(r['tp']) for r in rows);fp=sum(int(r['fp']) for r in rows)
    fn=sum(int(r['fn']) for r in rows);tn=sum(int(r['tn']) for r in rows)
    return dict(n_validation_tiles=len(rows),
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                pred_fg=(tp+fp)/(tp+fp+tn+fn) if tp+fp+tn+fn else None,
                reference_fg=(tp+fn)/(tp+fp+tn+fn) if tp+fp+tn+fn else None,
                bg_fp=fp/(fp+tn) if fp+tn else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data/inspect/LVILymph'))
    p.add_argument('--run-dir',type=Path,default=Path('runs/segmentation_baselines'))
    p.add_argument('--out',type=Path,default=Path('runs/inference_context_qc'))
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args()
    if a.device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
    else: device=a.device
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    summary=json.loads((a.run_dir/'summary.json').read_text(encoding='utf-8'))
    params=summary['training_parameters']
    pz=int(params['patch_z']);pxy=int(params['patch_xy']);guard=int(params['guard'])
    if pxy%2 or pz%2:
        raise ValueError('Training patch sizes must be even')
    checkpoint=torch.load(a.run_dir/'unet_3d.pt',map_location='cpu',weights_only=True)
    model=UNet(3,base=int(checkpoint['base'])).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    im,inst,indices,_=read_data(a.data_dir)
    df=pd.read_csv(a.run_dir/'tile_splits.csv')
    val=df[df['split']=='val']
    if len(val)==0:raise ValueError('No validation tiles')
    a.out.mkdir(parents=True,exist_ok=True)
    results=[]
    print(f'Device: {device}; validation tiles: {len(val)}; training patch Z={pz}, XY={pxy}')
    for i,(_,t) in enumerate(val.iterrows(),1):
        tile=dict(t)
        z,y,x=im.shape[0],int(t['y']),int(t['x']);n=int(t['w'])
        volume=im[:,y:y+n,x:x+n]
        ref=inst[:,y:y+n,x:x+n]
        full=predict_tile(model,3,im,tile,device)
        matched=matched_window_predict(model,volume,device,pz,pxy)
        for mode,prob in [('full_volume',full),('matched_windows',matched)]:
            result=confusion_metrics(prob,ref,guard)
            results.append(dict(tile_y=y,tile_x=x,mode=mode,**result))
        print(f'Validation tile {i}/{len(val)} ({y},{x}) complete',flush=True)
    pd.DataFrame(results).to_csv(a.out/'per_validation_tile.csv',index=False)
    report={}
    for mode in ('full_volume','matched_windows'):
        report[mode]=aggregate([r for r in results if r['mode']==mode])
    report['scope']='VALIDATION tiles only, within one specimen; watershed pseudo-reference'
    report['purpose']='Diagnose train/inference patch-context mismatch WITHOUT retraining or selecting thresholds on test tiles'
    report['threshold']=0.5
    report['training_patch_zyx']=[pz,pxy,pxy]
    report['full_inference_zyx']=[int(im.shape[0]),int(val.iloc[0]['w']),int(val.iloc[0]['w'])]
    (a.out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Saved:',a.out/'per_validation_tile.csv',a.out/'comparison.json')
    print('WARNING: this isolates a potential inference-context effect; not patient-level validation.')

if __name__=='__main__':main()
