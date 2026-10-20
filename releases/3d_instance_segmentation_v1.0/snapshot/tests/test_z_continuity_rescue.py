from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from z_continuity_rescue import propose_z_links,rescue_z_links


def paired():
    mask=np.zeros((2,4,6),np.int32)
    mask[0,:,:3]=1;mask[1,:,:3]=2
    mask[0,:,3:]=3;mask[1,:,3:]=4
    p=np.full(mask.shape,.75)
    return mask,p


def test_two_perfect_z_links():
    m,p=paired()
    links=propose_z_links(m,p)
    assert len(links)==2 and links.geometry_eligible.all() and links.mutual_best.all()
    out,d=rescue_z_links(m,links,max_component_voxels=40)
    assert d['n_accepted_merges']==2 and d['n_after']==2
    assert out[0,0,0]==out[1,0,0]!=out[0,0,-1]==out[1,0,-1]
    assert not np.any((out==0)!=(m==0))


def test_z_affinity_veto_specific_links():
    m,p=paired()
    p[0,:,:3]=.3
    links=propose_z_links(m,p)
    out,d=rescue_z_links(m,links,require_affinity=True,min_mean_pz=.45,max_component_voxels=40)
    assert d['n_accepted_merges']==1
    assert out[0,0,0]!=out[1,0,0] and out[0,0,5]==out[1,0,5]


def test_reciprocal_best_prevents_one_to_many():
    m=np.zeros((2,4,6),np.int32)
    m[0,:,:]=1;m[1,:,:3]=2;m[1,:,3:]=3
    links=propose_z_links(m,np.full(m.shape,.75),min_iou=.10)
    assert links.geometry_eligible.sum()==1
    out,d=rescue_z_links(m,links,max_component_voxels=100)
    assert d['n_accepted_merges']==1 and d['n_after']==2


def test_self_interface_blocks_spurious_link():
    m=np.zeros((2,4,6),np.int32)
    m[0,:,:]=1;m[1,:,:5]=1;m[1,:,5:]=2
    links=propose_z_links(m,np.full(m.shape,.75),min_iou=.01)
    assert links.loc[links.a==1].set_index('b').loc[1,'mutual_best']
    assert not links.loc[links.b==2,'geometry_eligible'].any()
    out,d=rescue_z_links(m,links,max_component_voxels=100)
    assert d['n_accepted_merges']==0


def test_no_overlap_or_single_layer():
    m=np.zeros((2,4,6),np.int32)
    m[0,:,:2]=1;m[1,:,3:]=2
    assert len(propose_z_links(m,np.full(m.shape,.7)))==0
    out,d=rescue_z_links(m,propose_z_links(m,np.full(m.shape,.7)))
    assert d['n_accepted_merges']==0 and (out>0).sum()==(m>0).sum()
    ones=m[:1]
    assert len(propose_z_links(ones,np.full(ones.shape,.5)))==0


def test_size_cap_prevents_large_merge():
    m,p=paired()
    links=propose_z_links(m,p)
    out,d=rescue_z_links(m,links,max_component_voxels=20)
    assert d['n_volume_rejections']==2 and d['n_accepted_merges']==0


def test_invalid_inputs():
    m,p=paired()
    with pytest.raises(ValueError):propose_z_links(m,p[:1])
    with pytest.raises(ValueError):propose_z_links(m,p,min_iou=1.2)
    with pytest.raises(ValueError):propose_z_links(m,p+3)
    with pytest.raises(ValueError):rescue_z_links(m,propose_z_links(m,p),max_component_voxels=0)


def test_immutable_and_output_inference_only():
    m,p=paired();orig=m.copy()
    links=propose_z_links(m,p)
    out,d=rescue_z_links(m,links)
    np.testing.assert_array_equal(m,orig)
    assert d['never_accesses_reference']
    assert np.count_nonzero(out)==np.count_nonzero(m)


def test_driver_synthetic_stack(tmp_path,monkeypatch):
    import json
    import evaluate_z_continuity_rescue as app
    from region_interface_features import FEATURE_COLUMNS
    class FixedClassifier:
        classes_=np.array([0,1])
        def predict_proba(self,X):
            assert list(X.columns)==FEATURE_COLUMNS
            prob=np.full(len(X),.85)
            return np.stack((1-prob,prob),axis=1)
    class FakeModel:
        def load_state_dict(self,*x,**kw):pass
        def to(self,*x,**kw):return self
        def eval(self):return self
    split=tmp_path/'splits';split.mkdir()
    (split/'data_audit.json').write_text(json.dumps({'stack_order_check':{'status':'FILENAME_ORDER_MATCH'}}))
    pd.DataFrame([{'y':0,'x':0,'w':16,'split':'train'}]).to_csv(split/'tile_splits.csv',index=False)
    image=np.full((2,16,16),.6,np.float32)
    ref=np.zeros_like(image,dtype=np.int32)
    ref[:,:,:8]=1;ref[:,:,8:]=2
    def fake_predict(*args):
        p=np.full((4,2,16,16),.72,np.float32)
        p[0]=.95
        return p
    monkeypatch.setattr(app,'read_data',lambda *a:(image,ref,None,None))
    monkeypatch.setattr(app,'predict_prob',fake_predict)
    monkeypatch.setattr(app,'make_network',lambda *a:FakeModel())
    monkeypatch.setattr(app.torch,'load',lambda *a,**k:{'base':4,'z_weight':.25,'state_dict':{}})
    monkeypatch.setattr(app,'load_region_classifier',lambda *a:{'model':FixedClassifier(),'feature_columns':FEATURE_COLUMNS,'positive_label':1})
    from argparse import Namespace
    args=Namespace(split='train',max_tiles=1,data_dir=tmp_path,split_run_dir=split,
        affinity_run_dir=tmp_path,predictor_dir=tmp_path,out=tmp_path/'output',
        region_z=1,region_xy=4,accept_provisional_registration=True,
        device='cpu',guard=1,foreground_threshold=.5,z_weight=.25,score_threshold=.2,
        min_faces=2,max_component_voxels=1200,z_min_faces=2,min_overlap=.35,min_iou=.15,min_mean_pz=.45)
    out=app.run(args)
    assert len(out)==3
    assert (args.out/'z_link_axis_breaks.csv').exists()
    assert out[0]['pq']>=0
    assert out[1]['n_predicted']<=out[0]['n_predicted']
    assert out[2]['n_predicted']<=out[0]['n_predicted']
