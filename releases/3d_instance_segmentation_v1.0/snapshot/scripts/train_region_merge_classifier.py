#!/usr/bin/env python3
"""Exploratory second-stage XY region-merge baselines, grouped by TRAIN tile.

IMPORTANT: Source affinity encoder was trained on the same specimen/TRAIN tiles.
OOF only removes second-stage tile-level overlap; it is NOT independent validation.
Author-watershed reference labels are pseudo-labels, not cell ground truth.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


def load_training_tables(data_dir: Path):
    info_path = data_dir / 'region_interface_dataset_summary.json'
    summary = json.loads(info_path.read_text(encoding='utf-8'))
    if summary.get('parameters', {}).get('include_z_pseudo_labels'):
        raise ValueError('Z pseudo-labels were enabled; this baseline is strictly XY-only')
    cols = summary.get('classification_feature_whitelist')
    if not isinstance(cols, list) or not cols or len(cols) != len(set(cols)):
        raise ValueError('Missing or repeated feature whitelist in dataset summary')
    for c in cols:
        if (not isinstance(c, str) or c.startswith(('reference_', 'majority_id_'))
            or c in {'pseudo_label','tile_id','eligible_for_training','label_status',
                     'baseline_static_gate','sv_a','sv_b','tile_x','tile_y','tile_w'}):
            raise ValueError(f'Unsafe feature whitelist column: {c!r}')
    path = data_dir / 'region_interface_labeled_train.csv'
    df = pd.read_csv(path)
    required = ['tile_id', *cols, 'pseudo_label']
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if df.empty:
        raise ValueError('Labeled table is empty')
    if not set(AFFINITY_INTERFACE_COLUMNS).issubset(set(cols)):
        raise ValueError('Feature whitelist lacks required affinity-interface baseline columns')
    if not set(df.pseudo_label.dropna().unique()).issubset({0, 1}) or df.pseudo_label.isna().any():
        raise ValueError('Only known binary XY pseudo-labels 0/1 are permitted')
    if df.tile_id.isna().any() or (df.tile_id.astype(str).str.len() == 0).any():
        raise ValueError('tile_id required for grouped evaluation')
    if df.tile_id.nunique() < 3:
        raise ValueError('Need >=3 distinct TRAIN spatial tiles')
    if df.pseudo_label.nunique() < 2:
        raise ValueError('Need both same and different pseudo-labels')
    # Strictly use metadata whitelisted columns. No audit table is read here.
    X = df[cols].apply(pd.to_numeric, errors='raise').replace([np.inf,-np.inf], np.nan)
    if X.isna().all(axis=0).any():
        # Z-only features are legitimately all missing on XY-only dataset.
        # They get a 0 (absent) sentinel consistently in all folds.
        pass
    y = df.pseudo_label.to_numpy(dtype=np.int8)
    groups = df.tile_id.astype(str).to_numpy()
    return df, X, y, groups, cols, summary


AFFINITY_INTERFACE_COLUMNS = [
    'interface_n_faces', 'interface_z_faces', 'interface_y_faces', 'interface_x_faces',
    'interface_z_fraction', 'interface_y_fraction', 'interface_x_fraction',
    'affinity_p_mean', 'affinity_p_std', 'affinity_p10', 'affinity_p50',
    'affinity_p90', 'affinity_logit_mean_weighted', 'affinity_negative_fraction_weighted',
    'affinity_p_mean_z', 'affinity_p_mean_y', 'affinity_p_mean_x',
]


def model_specs(seed: int):
    return {
        'logistic_affinity_interface_only': Pipeline([
            ('select', ColumnTransformer([
                ('affinity_interface', 'passthrough', AFFINITY_INTERFACE_COLUMNS)
            ], remainder='drop')),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0, keep_empty_features=True)),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000,
                                               solver='lbfgs', random_state=seed)),
        ]),
        'logistic_balanced': Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0, keep_empty_features=True)),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000,
                                               solver='lbfgs', random_state=seed)),
        ]),
        'hgb_balanced': Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0, keep_empty_features=True)),
            ('classifier', HistGradientBoostingClassifier(
                loss='log_loss', learning_rate=0.06, max_iter=150,
                max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=10.0,
                early_stopping=False, random_state=seed)),
        ]),
    }


def binary_metrics(y: np.ndarray, p_same: np.ndarray, threshold: float=0.5):
    """Unlike overall accuracy, shows both false merges and missed same edges."""
    y = np.asarray(y, dtype=np.int8)
    p_same = np.asarray(p_same, dtype=float)
    if len(y) != len(p_same) or np.any(~np.isfinite(p_same)):
        raise ValueError('Metric input sizes/non-finite values')
    pred = (p_same >= threshold).astype(np.int8)
    false_merges = int(((pred == 1) & (y == 0)).sum())
    missed_same = int(((pred == 0) & (y == 1)).sum())
    out = {
        'n': int(len(y)), 'n_same': int((y == 1).sum()),
        'n_different': int((y == 0).sum()),
        'threshold_same': float(threshold),
        'same_precision': float(precision_score(y, pred, pos_label=1, zero_division=0)),
        'same_recall': float(recall_score(y, pred, pos_label=1, zero_division=0)),
        'different_precision': float(precision_score(1-y, 1-pred, pos_label=1, zero_division=0)),
        'different_recall': float(recall_score(1-y, 1-pred, pos_label=1, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)) if len(np.unique(y))==2 else None,
        'false_merge_edges': false_merges,
        'missed_same_edges': missed_same,
        'same_average_precision': (float(average_precision_score(y, p_same)) if len(np.unique(y))==2 else None),
        'different_average_precision': (float(average_precision_score(1-y, 1-p_same)) if len(np.unique(y))==2 else None),
        'roc_auc': (float(roc_auc_score(y, p_same)) if len(np.unique(y))==2 else None),
    }
    return out


def compute_baselines(X: pd.DataFrame):
    # Existing fixed-gate baseline: score>=0.2, negfrac<=.35, nfaces>=2.
    # The static gate is computed from whitelisted model features only.
    score = X['affinity_logit_mean_weighted'].to_numpy(dtype=float)
    neg = X['affinity_negative_fraction_weighted'].to_numpy(dtype=float)
    faces = X['interface_n_faces'].to_numpy(dtype=float)
    if np.any(~np.isfinite(score)) or np.any(~np.isfinite(neg)) or np.any(~np.isfinite(faces)):
        raise ValueError('Missing baseline features')
    return {
        'affinity_mean': X['affinity_p_mean'].to_numpy(dtype=float),
        'affinity_pooled_logit': expit(score),
        'original_static_gate': ((score >= 0.2) & (neg <= 0.35) & (faces >= 2)).astype(float),
    }


def grouped_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
               seed=42, folds=4, fitted_models=None):
    """OOF on TRAIN tiles. Returns predictions and per-fold diagnostics."""
    folds = min(int(folds), len(np.unique(groups)))
    if folds < 3:
        raise ValueError('At least 3 spatial folds are required')
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    specs = model_specs(seed) if fitted_models is None else fitted_models
    oof = {name: np.full(len(y), np.nan, dtype=float) for name in specs}
    fold_number = np.full(len(y), -1, dtype=int)
    diagnostics=[]
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups)):
        overlap=set(groups[tr]).intersection(groups[va])
        if overlap:
            raise AssertionError(f'Leaked tile IDs across grouped split: {overlap}')
        if np.unique(y[tr]).size < 2:
            raise ValueError(f'Fold {fold} training has a single class; reduce --folds or revise grouping')
        for name,model in specs.items():
            weights = compute_sample_weight('balanced', y=y[tr])
            if name == 'hgb_balanced':
                model.fit(X.iloc[tr], y[tr], classifier__sample_weight=weights)
            else:
                model.fit(X.iloc[tr], y[tr])
            oof[name][va] = model.predict_proba(X.iloc[va])[:, list(model.classes_).index(1)]
        fold_number[va] = fold
        diagnostics.append({
            'fold': fold, 'n_fit': len(tr), 'n_score':len(va),
            'n_fit_different':int((y[tr]==0).sum()),
            'n_score_different':int((y[va]==0).sum()),
            'fit_tiles':sorted(set(groups[tr])), 'score_tiles':sorted(set(groups[va]))
        })
    if (fold_number<0).any() or any(~np.isfinite(p).all() for p in oof.values()):
        raise AssertionError('Incomplete OOF predictions')
    return oof, fold_number, diagnostics


def train_and_save(data_dir: Path, out: Path, seed=42, folds=4, threshold=.5):
    df, X, y, groups, cols, source = load_training_tables(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    required_out = [out/'region_merge_oof_predictions.csv', out/'region_merge_oof_metrics.csv',
                    out/'region_merge_training_summary.json']
    if any(p.exists() for p in required_out):
        raise FileExistsError('Output files already exist. Choose new --out path.')
    base=compute_baselines(X)
    oof, fold_id, fold_diagnostics = grouped_cv(X,y,groups,seed=seed,folds=folds)
    outputs={**base, **oof}
    prediction = pd.DataFrame({'tile_id':groups,'fold':fold_id,'pseudo_label':y})
    for name,prob in outputs.items():
        prediction['p_same_'+name]=prob
    prediction.to_csv(out/'region_merge_oof_predictions.csv',index=False)
    rows=[]
    for name, prob in outputs.items():
        rows.append(dict(method=name, split='all_train_group_oof' if name in oof else 'fixed_baseline_all_train',
                         **binary_metrics(y,prob,threshold=threshold)))
        for i in range(int(fold_id.max())+1):
            sel=(fold_id==i)
            rows.append(dict(method=name,split=f'fold_{i}',**binary_metrics(y[sel],prob[sel],threshold=threshold)))
    pd.DataFrame(rows).to_csv(out/'region_merge_oof_metrics.csv',index=False)
    pd.DataFrame(fold_diagnostics).to_json(out/'region_merge_fold_details.json', orient='records', indent=2)
    pd.DataFrame([{'tile_id':g, 'n_edges':int((groups==g).sum()), 'n_different':int(((groups==g)&(y==0)).sum()),
                  'n_same':int(((groups==g)&(y==1)).sum())} for g in sorted(set(groups))]).to_csv(
                        out/'region_merge_tile_counts.csv',index=False)
    for name,model in model_specs(seed).items():
        weights = compute_sample_weight('balanced', y=y)
        if name=='hgb_balanced':
            model.fit(X,y,classifier__sample_weight=weights)
        else:
            model.fit(X,y)
        joblib.dump({'model': model, 'feature_columns':cols,'train_scope':'TRAIN only; same specimen',
                     'positive_label':1,'threshold':threshold}, out/f'{name}_trained_on_all_train.joblib')
    result={
        'purpose':'Exploratory region-interface pseudo-label classification, NOT instance reconstruction PQ',
        'source_dir':str(data_dir), 'n_labeled_xy':len(y),
        'n_train_tiles':int(len(np.unique(groups))),
        'n_same':int((y==1).sum()),'n_different':int((y==0).sum()),
        'fraction_different':float((y==0).mean()),
        'feature_whitelist':cols, 'n_folds':int(fold_id.max())+1,
        'decision_rule':'P(same)>=0.5; no threshold optimized on OOF scores',
        'baseline_rule':'Original static score>=0.2 and negative<=0.35 and n_faces>=2',
        'metrics_overall':{r['method']:{k:v for k,v in r.items() if k not in {'method','split'}}
                           for r in rows if r['split'] in ('all_train_group_oof','fixed_baseline_all_train')},
        'critical_limitations':[
            'Only labeled high-purity XY edges; unknown edges excluded and not classified as different.',
            'All reference labels are author watershed pseudo labels.',
            'Affinity encoder upstream was already trained on the same TRAIN tiles; grouped OOF here does NOT confer independent upstream validation.',
            'All groups are from one specimen; spatial proximity may remain correlated.',
            'Z-related interfaces are not included in this classifier.',
            'Class prevalence is strongly imbalanced and selected for reference coverage; it may not represent deployment interfaces.',
            'No predictions have been decoded to 3D cell instances; classification metrics do NOT establish PQ improvement.',
            'Joblib models load executable Python objects; load only your own generated files.',
        ],
    }
    (out/'region_merge_training_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(pd.DataFrame(rows).query("split=='all_train_group_oof' or split=='fixed_baseline_all_train'")[
        ['method','n','n_different','same_precision','same_recall','different_precision','different_recall',
         'different_average_precision','false_merge_edges','missed_same_edges']].to_string(index=False))
    print('Saved:',out)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True,help='Region Interface Dataset output directory')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--folds',type=int,default=4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threshold',type=float,default=.5,
                        help='Fixed P(same) decision threshold (default 0.5)')
    args=parser.parse_args()
    if args.folds<3 or not 0<args.threshold<1:
        parser.error('Require --folds>=3 and 0<--threshold<1')
    train_and_save(args.data_dir,args.out,seed=args.seed,folds=args.folds,threshold=args.threshold)

if __name__=='__main__':
    main()
