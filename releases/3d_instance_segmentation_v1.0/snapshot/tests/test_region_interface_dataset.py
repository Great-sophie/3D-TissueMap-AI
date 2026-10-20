"""Synthetic tests: no private patient data or extra checkpoints required."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from region_interface_features import (
    FEATURE_COLUMNS, extract_region_interfaces, region_reference_diagnostics,
)


def sample():
    sv=np.zeros((2,4,6),np.int32)
    # Four small regions per plane; XY same, XY different, cross-Z contacts.
    sv[0,:2,:3]=1
    sv[0,:2,3:]=2
    sv[0,2:,:3]=3
    sv[0,2:,3:]=4
    sv[1,:2,:3]=5
    sv[1,:2,3:]=6
    sv[1,2:,:3]=7
    sv[1,2:,3:]=8
    reference=np.zeros_like(sv)
    for i,refid in ((1,11),(2,11),(3,22),(4,33),(5,11),(6,11),(7,22),(8,33)):
        reference[sv==i]=refid
    # Orphan region with no reference overlap.
    reference[sv==8]=0
    img=np.arange(sv.size,dtype=np.float32).reshape(sv.shape)/sv.size
    p=np.full((3,*sv.shape),.8,dtype=np.float32)
    return sv,p,img,reference


def edge(df,a,b):
    r=df[(df.sv_a==a)&(df.sv_b==b)]
    assert len(r)==1
    return r.iloc[0]


def test_reliable_same_and_different_xy():
    sv,p,img,ref=sample()
    df,audit=extract_region_interfaces(sv,p,img,ref)
    ab=edge(df,1,2)
    cd=edge(df,1,3)
    assert ab.pseudo_label==1 and ab.label_status=='reliable_xy' and ab.eligible_for_training==1
    assert cd.pseudo_label==0 and cd.label_status=='reliable_xy' and cd.eligible_for_training==1
    assert len(audit)==len(df)
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    assert not any(x.startswith('reference_') or x.startswith('majority_id') for x in FEATURE_COLUMNS)


def test_provisional_z_is_not_training_by_default():
    sv,p,img,ref=sample()
    df,_=extract_region_interfaces(sv,p,img,ref)
    zz=edge(df,1,5)
    assert zz.pseudo_label==1 and zz.label_status=='provisional_z'
    assert zz.eligible_for_training==0 and zz.interface_z_faces>0
    df2,_=extract_region_interfaces(sv,p,img,ref,include_z_labels=True)
    assert edge(df2,1,5).eligible_for_training==1


def test_unknown_orphan_never_negative():
    sv,p,img,ref=sample()
    df,_=extract_region_interfaces(sv,p,img,ref)
    ab=edge(df,7,8)
    assert ab.pseudo_label==-1 and ab.eligible_for_training==0
    assert ab.label_status=='unknown_low_reference_support'


def test_unknown_when_low_overlap_coverage():
    sv,p,img,ref=sample()
    ref[0,:2,:3]=0
    ref[0,0,0]=11
    df,_=extract_region_interfaces(sv,p,img,ref,min_coverage=.70)
    assert edge(df,1,2).pseudo_label==-1


def test_unknown_when_mixed_low_purity():
    sv,p,img,ref=sample()
    ref[0,:1,:3]=44
    df,_=extract_region_interfaces(sv,p,img,ref,min_purity=.90)
    assert edge(df,1,2).pseudo_label==-1


def test_numeric_interface_stats_and_baseline_gate():
    sv,p,img,ref=sample()
    p[2,0,0,2]=.2
    p[2,0,1,2]=.8
    df,_=extract_region_interfaces(sv,p,img,ref)
    row=edge(df,1,2)
    assert row.interface_x_faces==2
    assert row.interface_n_faces==2
    assert row.interface_y_faces==0
    assert row.affinity_p_mean==pytest.approx(.5,abs=1e-6)
    assert row.affinity_p10==pytest.approx(.26,abs=1e-6)
    assert row.affinity_p50==pytest.approx(.5,abs=1e-6)
    assert row.affinity_p90==pytest.approx(.74,abs=1e-6)
    assert row.affinity_negative_fraction_weighted==pytest.approx(.5)
    assert row.baseline_static_gate==0


def test_pooled_features_match_original_region_graph():
    sv,p,img,ref=sample()
    df,_=extract_region_interfaces(sv,p,img,ref,z_weight=.25)
    # Compatibility test: model input aggregation stays identical to original module.
    from evaluate_affinity_region_graph import interface_edges
    orig=interface_edges(sv,p,z_weight=.25)
    table={(int(r.sv_a),int(r.sv_b)):r for r in df.itertuples(index=False)}
    for a,b,score,neg,nfaces in orig:
        row=table[int(a),int(b)]
        assert row.interface_n_faces==int(nfaces)
        assert row.affinity_logit_mean_weighted==pytest.approx(score,abs=1e-8)
        assert row.affinity_negative_fraction_weighted==pytest.approx(neg,abs=1e-8)


def test_empty_and_invalid():
    sv,p,img,ref=sample()
    df,audit=extract_region_interfaces(np.zeros_like(sv),p,img,ref)
    assert df.empty and audit.empty
    with pytest.raises(ValueError):
        extract_region_interfaces(sv,p[:2],img,ref)
    with pytest.raises(ValueError):
        extract_region_interfaces(sv,p,img,ref,min_coverage=1.1)


def test_deterministic_majority_tie():
    sv,p,img,ref=sample()
    ref[0,:1,:3]=99
    d=region_reference_diagnostics(sv,ref)
    assert d['majority'][1]==11  # tie between 11 and 99: smaller ID wins
    assert d['purity'][1]==.5


def test_cli_end_to_end_synthetic(tmp_path,monkeypatch):
    import json
    import build_region_interface_dataset as cli
    from argparse import Namespace
    splits=tmp_path/'runs';splits.mkdir()
    (splits/'data_audit.json').write_text(json.dumps({'stack_order_check':{'status':'FILENAME_ORDER_MATCH'}}))
    pd.DataFrame([{'y':0,'x':0,'w':32,'split':'train'},
                  {'y':0,'x':32,'w':32,'split':'val'},
                  {'y':32,'x':0,'w':32,'split':'test'}]).to_csv(splits/'tile_splits.csv',index=False)
    im=np.ones((4,64,64),np.float32)
    gt=np.zeros((4,64,64),np.int32)
    gt[:,:32,:16]=1;gt[:,:32,16:32]=2
    monkeypatch.setattr(cli,'read_data',lambda path:(im,gt,[],''))
    monkeypatch.setattr(cli.torch,'load',lambda *a,**kw:{'base':4,'z_weight':.25,'state_dict':{}})
    class Fake:
        def load_state_dict(self,*a,**kw):pass
        def to(self,*a,**kw):return self
        def eval(self):return self
    monkeypatch.setattr(cli,'make_network',lambda *a,**kw:Fake())
    monkeypatch.setattr(cli,'predict_prob',lambda model,volume,device:
        np.broadcast_to(np.array([.9,.9,.9,.9],np.float32)[:,None,None,None],
                        (4,*volume.shape)).copy())
    cfg=Namespace(accept_provisional_registration=True,include_z_pseudo_labels=False,
        max_tiles=0,guard=2,region_z=1,region_xy=4,device='cpu',
        split_run_dir=splits,affinity_run_dir=tmp_path/'affinity',data_dir=tmp_path/'data',
        out=tmp_path/'results',foreground_threshold=.5,z_weight=.25,
        min_reference_coverage=.7,min_reference_purity=.9,min_reference_overlap_voxels=3)
    result=cli.run(cfg)
    assert result['n_train_tiles']==1
    assert result['totals']['n_interfaces']>0
    assert result['totals']['n_train_eligible']>0
    df=pd.read_csv(cfg.out/'region_interfaces_train.csv')
    assert set(df.tile_id)=={'train_y0_x0_w32'}
    assert all(col not in df for col in ('majority_id_a','reference_coverage_a'))
    with pytest.raises(FileExistsError):
        cli.run(cfg)
