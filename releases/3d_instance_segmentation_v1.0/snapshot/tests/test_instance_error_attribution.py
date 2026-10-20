from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from instance_error_attribution import overlap_objects,directional_faces


def test_fragment_orphan_exact_matching():
    ref=np.zeros((1,3,6),np.int32)
    ref[:,:,:3]=1;ref[:,:,3:5]=2
    pred=np.zeros_like(ref)
    pred[:,:,0]=11;pred[:,:,1:3]=12;pred[:,:,3:5]=13;pred[:,:,5]=14
    stats,objs,refs,ov=overlap_objects(ref,pred,min_overlap_voxels=2,min_reference_fraction=.1)
    assert stats['tp']==2
    assert stats['fp']==2
    obj=objs.set_index('predicted_id').category.to_dict()
    assert obj[11]=='fragment_candidate'
    assert obj[14]=='orphan_no_reference_overlap'
    assert objs.category.value_counts().to_dict()['matched_iou_ge_0_5']==2
    assert stats['n_reference_with_2plus_substantial_prediction_components']==1


def test_mixed_reference_and_background_dominated():
    ref=np.zeros((1,4,8),np.int32)
    ref[:,:,0:3]=1;ref[:,:,3:6]=2
    pred=np.zeros_like(ref)
    pred[:,:,2:4]=10  # 1 column each ref, IoU <.5
    pred[:,:,6:8]=20 # orphan
    stats,objs,_,_=overlap_objects(ref,pred,min_overlap_voxels=2)
    assert objs.set_index('predicted_id').loc[10,'category']=='multi_reference_overlap'
    assert objs.set_index('predicted_id').loc[20,'category']=='orphan_no_reference_overlap'


def test_directional_faces_separates_z_y_x():
    ref=np.ones((2,2,3),np.int32)
    pred=np.ones_like(ref)
    pred[1,:,:]=2
    rows=directional_faces(ref,pred).set_index('axis')
    assert rows.loc['Z','conditional_same_prediction_rate']==0
    assert rows.loc['X','conditional_same_prediction_rate']==1
    assert rows.loc['Y','conditional_same_prediction_rate']==1


def test_match_identity_parity():
    ref=np.array([[[0,100,100,0],[0,200,200,0]]],dtype=np.int32)
    summary,*_=overlap_objects(ref,ref)
    assert summary['pq']==1 and summary['tp']==2 and summary['fp']==0


def test_empty_and_shape_checks():
    ref=np.zeros((2,2,3),np.int32)
    stats,objs,refs,over=overlap_objects(ref,ref)
    assert stats['pq']==1 and len(objs)==0
    assert directional_faces(ref,ref).n_same_reference_voxel_faces.sum()==0
    with pytest.raises(ValueError):overlap_objects(ref,np.zeros((3,2,2),np.int32))


def test_end_to_end_driver_synthetic(tmp_path, monkeypatch):
    import json, pandas as pd
    from argparse import Namespace
    # This test uses only pre-existing model interfaces and deterministic mock predictions.
    import audit_instance_error_attribution as app
    root=tmp_path/'splits';root.mkdir()
    (root/'data_audit.json').write_text(json.dumps({'stack_order_check':{'status':'FILENAME_ORDER_MATCH'}}))
    pd.DataFrame([{'y':0,'x':0,'w':16,'split':'val'}]).to_csv(root/'tile_splits.csv',index=False)
    img=np.ones((2,16,16),np.float32)
    ref=np.zeros((2,16,16),np.int32)
    ref[:,:,:7]=11;ref[:,:,7:14]=22
    def fake_predict(model,image,device):
        p=np.full((4,2,16,16),.65,np.float32)
        p[0]=.9
        return p
    class Net:
        def load_state_dict(self,*a,**k):pass
        def to(self,*a,**k):return self
        def eval(self):return self
    from region_interface_features import FEATURE_COLUMNS
    class MockClassifier:
        classes_=np.array([0,1])
        def predict_proba(self,X):
            assert list(X.columns)==FEATURE_COLUMNS
            return np.tile([[.2,.8]],(len(X),1))
    monkeypatch.setattr(app,'read_data',lambda d:(img,ref,None,None))
    monkeypatch.setattr(app,'predict_prob',fake_predict)
    monkeypatch.setattr(app,'make_network',lambda *a,**k:Net())
    monkeypatch.setattr(app.torch,'load',lambda *a,**k:{'base':4,'z_weight':.25,'state_dict':{}})
    monkeypatch.setattr(app,'load_region_classifier',lambda *a,**k:{'model':MockClassifier(), 'feature_columns':FEATURE_COLUMNS,'positive_label':1})
    args=Namespace(data_dir=tmp_path,split_run_dir=root,affinity_run_dir=tmp_path,
                   predictor_dir=tmp_path,out=tmp_path/'out',split='val',max_tiles=1,
                   region_z=1,region_xy=4,z_weight=.25,score_threshold=.2,
                   max_negative_fraction=.35,max_component_voxels=1200,min_faces=2,
                   foreground_threshold=.5,min_overlap_voxels=2,min_reference_fraction=.05,
                   guard=1,device='cpu',accept_provisional_registration=True)
    out=app.run(args)
    assert len(out['tiles'])==2
    df=pd.read_csv(args.out/'instance_error_summary.csv')
    assert len(df)==2
    assert set(df.method)=={'distance_watershed','logistic_balanced_frozen_edge_score_only'}
    assert len(pd.read_csv(args.out/'xy_z_face_continuity.csv'))==6
    assert (args.out/'unmatched_category_counts.csv').exists()
    assert out['inference_uses_reference'] is False
