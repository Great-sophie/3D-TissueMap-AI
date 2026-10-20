from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from learned_region_graph import learned_interface_evidence,load_region_classifier
from dynamic_region_merging import build_region_evidence,merge_dynamic_regions
from region_interface_features import FEATURE_COLUMNS,extract_region_interfaces


class FixedClassifier:
    classes_=np.array([0,1])
    def __init__(self,p=0.9): self.p=p
    def predict_proba(self,X):
        assert list(X.columns)==FEATURE_COLUMNS
        p=np.full(len(X),self.p,dtype=float)
        return np.column_stack((1-p,p))


def example():
    sv=np.zeros((2,3,4),np.int32)
    sv[0,:,:2]=1;sv[0,:,2:]=2
    sv[1,:,:2]=3;sv[1,:,2:]=4
    prob=np.full((3,2,3,4),0.65,dtype=float)
    image=np.arange(sv.size,dtype=float).reshape(sv.shape)/sv.size
    return sv,prob,image


def bundle(p=0.9):
    return {'model':FixedClassifier(p),'feature_columns':FEATURE_COLUMNS,'positive_label':1}


def test_learned_xy_only_and_z_retains_raw():
    sv,prob,img=example()
    raw=build_region_evidence(sv,prob,0.25)
    learned,stats=learned_interface_evidence(sv,prob,img,bundle())
    assert np.array_equal(raw[:,:2],learned[:,:2])
    assert stats['n_learned_pure_xy']==2
    assert stats['n_raw_z_involved']==2
    xy=learned[:,6]==0
    np.testing.assert_allclose(learned[xy,2]/learned[xy,3],np.log(9.0),atol=1e-7)
    np.testing.assert_allclose(learned[~xy],raw[~xy])
    # Keep original negative fraction for optional raw-repulsion-veto ablation.
    np.testing.assert_allclose(learned[xy,4:6],raw[xy,4:6])


def test_oracle_reference_can_never_be_used_in_feature_construction(monkeypatch):
    from learned_region_graph import extract_region_interfaces as original
    def guarded(sv,prob,img,ref,**kwargs):
        assert np.count_nonzero(ref)==0
        return original(sv,prob,img,ref,**kwargs)
    monkeypatch.setattr('learned_region_graph.extract_region_interfaces',guarded)
    sv,prob,img=example()
    learned_interface_evidence(sv,prob,img,bundle())


def test_feature_alignment_and_recompute_consistency():
    sv,prob,img=example()
    feat,_=extract_region_interfaces(sv,prob,img,np.zeros_like(sv))
    evidence,stat=learned_interface_evidence(sv,prob,img,bundle())
    assert np.array_equal(feat[['sv_a','sv_b']].to_numpy(),evidence[:,:2])
    # Two independent XY interfaces are scored, not individual voxels.
    assert stat['n_learned_pure_xy'] == int((feat.interface_z_faces==0).sum())


def test_invalid_feature_schema_and_probabilities():
    sv,prob,img=example()
    with pytest.raises(ValueError,match='schema'):
        learned_interface_evidence(sv,prob,img,{'model':FixedClassifier(),'feature_columns':FEATURE_COLUMNS[:-1]})
    with pytest.raises(ValueError,match='invalid probability'):
        learned_interface_evidence(sv,prob,img,bundle(1.5))


def test_empty_graph():
    sv=np.zeros((2,3,4),np.int32)
    p=np.full((3,2,3,4),0.5)
    out,stat=learned_interface_evidence(sv,p,np.ones_like(sv,dtype=float),bundle())
    assert out.shape==(0,10) and stat['n_edges']==0


def test_graph_decoder_compatibility():
    sv,prob,img=example()
    evidence,_=learned_interface_evidence(sv,prob,img,bundle(.90))
    seg,diag=merge_dynamic_regions(sv,evidence,phase_order='joint',use_repulsion=False,
                                    score_threshold=.2,min_faces=1,max_component_voxels=100)
    assert seg.shape==sv.shape
    assert diag['initial_interfaces']==4
    assert diag['final_components']>=1


def test_model_load_schema(tmp_path):
    import joblib
    path=tmp_path/'known.joblib'
    joblib.dump(bundle(),path)
    loaded=load_region_classifier(path)
    assert loaded['positive_label']==1
    joblib.dump({'model':FixedClassifier(),'feature_columns':['reference_id'],'positive_label':1},path)
    with pytest.raises(ValueError,match='mismatch'):
        load_region_classifier(path)


def test_mixed_z_interface_fallback():
    sv,prob,img=example()
    # Merge labels so pair 1-2 contacts both in XY and Z.
    sv[1,:,:2]=2
    sv[1,:,2:]=3
    raw=build_region_evidence(sv,prob,0.25)
    learned,stats=learned_interface_evidence(sv,prob,img,bundle())
    mixed=(raw[:,7]>0) # Z faces, not Z weighted neg count
    assert mixed.any()
    np.testing.assert_allclose(learned[mixed],raw[mixed])


def test_evaluation_driver_on_synthetic_stack(tmp_path,monkeypatch):
    import json
    import pandas as pd
    import evaluate_learned_region_graph as app
    root=tmp_path/'split';root.mkdir()
    (root/'data_audit.json').write_text(json.dumps({'stack_order_check':{'status':'FILENAME_ORDER_MATCH'}}))
    pd.DataFrame([{'y':0,'x':0,'w':16,'split':'train'}]).to_csv(root/'tile_splits.csv',index=False)
    pred_dir=tmp_path/'models';pred_dir.mkdir()
    img=np.ones((2,16,16),np.float32)
    ref=np.zeros((2,16,16),np.int32)
    ref[:,:,:8]=1;ref[:,:,8:]=2
    # Predict foreground with some FP; same model prediction for all decoders.
    def fake_prob(model,image,device):
        out=np.full((4,2,16,16),0.7,np.float32)
        out[0]=.9
        return out
    class FakeNet:
        def load_state_dict(self,*a,**k):pass
        def to(self,*a,**k):return self
        def eval(self):return self
    monkeypatch.setattr(app,'read_data',lambda d:(img,ref,None,None))
    monkeypatch.setattr(app,'predict_prob',fake_prob)
    monkeypatch.setattr(app,'make_network',lambda *a,**k:FakeNet())
    monkeypatch.setattr(app.torch,'load',lambda *a,**k:{'base':4,'z_weight':.25,'state_dict':{}})
    monkeypatch.setattr(app,'load_region_classifier',lambda *a,**k:bundle(.9))
    from argparse import Namespace
    args=Namespace(data_dir=tmp_path,split_run_dir=root,affinity_run_dir=tmp_path,
                   predictor_dir=pred_dir,out=tmp_path/'out',split='train',max_tiles=1,
                   region_z=1,region_xy=4,z_weight=.25,score_threshold=.2,
                   max_negative_fraction=.35,max_component_voxels=1200,min_faces=2,
                   foreground_threshold=.5,guard=1,device='cpu',accept_provisional_registration=True)
    result=app.run(args)
    assert len(result['rows'])==7
    assert result['in_sample_warning'] is True
    assert result['reference_used_by_graph_inference'] is False
    assert (args.out/'learned_region_instance_comparison.csv').exists()
