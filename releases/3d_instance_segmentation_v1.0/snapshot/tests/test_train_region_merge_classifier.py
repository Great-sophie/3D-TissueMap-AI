from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_region_merge_classifier import (
    binary_metrics, load_training_tables, compute_baselines,
    grouped_cv, train_and_save
)

FEATURES = [
    'interface_n_faces','interface_z_faces','interface_y_faces','interface_x_faces',
    'interface_z_fraction','interface_y_fraction','interface_x_fraction',
    'affinity_p_mean','affinity_p_std','affinity_p10','affinity_p50',
    'affinity_p90','affinity_logit_mean_weighted','affinity_negative_fraction_weighted',
    'affinity_p_mean_z','affinity_p_mean_y','affinity_p_mean_x',
    'ir193_face_absdiff_mean',
    'region_volume_min','region_volume_max','region_volume_ratio_min_max',
    'region_ir193_mean_absdiff','region_ir193_std_mean',
    'region_bbox_z_extent_min','region_bbox_z_extent_max',
    'region_bbox_y_extent_min','region_bbox_y_extent_max',
    'region_bbox_x_extent_min','region_bbox_x_extent_max',
]

def build_fixture(root, groups=6, per_group=80, include_z=False):
    rng=np.random.default_rng(123)
    samples=[]
    for gid in range(groups):
        for i in range(per_group):
            different = i % 14 == 0
            prob = np.clip(.74 + rng.normal(0,.13) - (.38 if different else 0),.01,.99)
            logit = np.log(prob/(1-prob))
            d={c:float(rng.normal()) for c in FEATURES}
            d.update(tile_id=f'train_y{gid*192}_x{gid*192}_w192',
                     pseudo_label=0 if different else 1, interface_n_faces=3,
                     interface_z_faces=0,interface_y_faces=1,interface_x_faces=2,
                     affinity_p_mean=prob,affinity_logit_mean_weighted=logit,
                     affinity_negative_fraction_weighted=float(prob<.5),
                     affinity_p_mean_z=np.nan)
            samples.append(d)
    root.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(samples)
    df.to_csv(root/'region_interface_labeled_train.csv',index=False)
    meta={'classification_feature_whitelist':FEATURES,
          'parameters':{'include_z_pseudo_labels':include_z}}
    (root/'region_interface_dataset_summary.json').write_text(json.dumps(meta))
    return df


def test_binary_metrics_cases():
    metrics=binary_metrics(np.array([1,1,0,0]),np.array([.8,.3,.1,.9]))
    assert metrics['false_merge_edges']==1
    assert metrics['missed_same_edges']==1
    assert abs(metrics['same_recall']-.5)<1e-8
    assert abs(metrics['different_recall']-.5)<1e-8


def test_dataset_uses_feature_whitelist_only(tmp_path):
    df=build_fixture(tmp_path/'input')
    df['majority_id_a']=123
    df.to_csv(tmp_path/'input'/'region_interface_labeled_train.csv',index=False)
    data,X,y,g,columns,_=load_training_tables(tmp_path/'input')
    assert columns==FEATURES
    assert 'majority_id_a' not in X
    assert len(X)==len(df)


def test_rejects_reference_feature_leakage(tmp_path):
    build_fixture(tmp_path/'input')
    p=tmp_path/'input'/'region_interface_dataset_summary.json'
    d=json.loads(p.read_text())
    d['classification_feature_whitelist'].append('reference_purity_a')
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Unsafe'):
        load_training_tables(tmp_path/'input')


def test_z_flag_not_allowed(tmp_path):
    build_fixture(tmp_path/'input',include_z=True)
    with pytest.raises(ValueError,match='Z pseudo'):
        load_training_tables(tmp_path/'input')


def test_static_gate_is_reconstructed(tmp_path):
    build_fixture(tmp_path/'input')
    _,X,_,_,_,_=load_training_tables(tmp_path/'input')
    p=compute_baselines(X)
    score=X['affinity_logit_mean_weighted'].to_numpy()
    neg=X['affinity_negative_fraction_weighted'].to_numpy()
    known=(score>=.2)&(neg<=.35)&(X['interface_n_faces'].to_numpy()>=2)
    assert np.array_equal(p['original_static_gate'],known)


def test_grouped_cv_no_overlap(tmp_path):
    build_fixture(tmp_path/'input')
    _,X,y,g,_,_=load_training_tables(tmp_path/'input')
    oof,fold_ids,details=grouped_cv(X,y,g,folds=3)
    assert np.all(fold_ids>=0)
    assert all(np.all(np.isfinite(p)) for p in oof.values())
    assert all(not (set(d['fit_tiles']) & set(d['score_tiles'])) for d in details)
    assert len(np.unique(fold_ids))==3


def test_full_smoke(tmp_path):
    src=tmp_path/'src'
    build_fixture(src,groups=6,per_group=80)
    dst=tmp_path/'out'
    result=train_and_save(src,dst,folds=3)
    assert result['n_labeled_xy']==480
    assert (dst/'hgb_balanced_trained_on_all_train.joblib').exists()
    assert (dst/'logistic_balanced_trained_on_all_train.joblib').exists()
    oof=pd.read_csv(dst/'region_merge_oof_predictions.csv')
    assert len(oof)==480
    assert {'p_same_logistic_balanced','p_same_hgb_balanced'} <= set(oof.columns)
    assert (dst/'region_merge_tile_counts.csv').exists()
    with pytest.raises(FileExistsError):
        train_and_save(src,dst,folds=3)
