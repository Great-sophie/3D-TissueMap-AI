import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_z_link_safety import component_reference_stats,link_reference_categories,descendant_ids,summarize_links,matched_preservation

def test_categories_three_types():
    # IDs 1 & 2 same reference, 3 different reference, 4 reference background
    initial=np.array([[[1,1,2,2],[3,3,4,4]]],dtype=np.int32)
    ref=np.array([[[10,10,10,10],[20,20,0,0]]],dtype=np.int32)
    rid,pur,cov,reliable=component_reference_stats(initial,ref)
    assert list(rid[1:])==[10,10,20,0]
    p=pd.DataFrame({'a':[1,2,1],'b':[2,3,4],'z':[0,0,0],
         'geometry_eligible':[True,True,True], 'mean_pz':[.6,.3,.6],
         'footprint_iou':[.4,.4,.4]})
    ann=link_reference_categories(p,rid,pur,cov,reliable)
    assert ann.reference_category.tolist()==['reliable_same','reliable_different','uncertain']
    sums,_=summarize_links(ann,.45)
    assert sums.iloc[0].n==3 and sums.iloc[1].n==2
    assert sums.iloc[2].reliable_different==1

def test_descendants_union():
    a=np.array([[[1,1,2],[0,3,3]]])
    b=np.array([[[1,1,1],[0,2,2]]])
    assert descendant_ids(a,b)=={1:1,2:1,3:2}

def test_descendants_reject_split():
    a=np.array([[[1,1]]]);b=np.array([[[1,2]]])
    try: descendant_ids(a,b)
    except ValueError:pass
    else:raise AssertionError('split not detected')

def test_match_preserve_single_object():
    a=np.zeros((2,3,3),dtype=np.int32)
    a[:,0,:]=1; a[:,2,:]=2
    b=a.copy();b[b==2]=1
    preserved=matched_preservation(a,a,a)
    assert preserved['baseline_matched_instances']==2
    assert preserved['baseline_matches_lost_or_changed']==0
    merged=matched_preservation(a,a,b)
    assert merged['post_matched_instances']<=2

def test_saved_artifacts_end_to_end(tmp_path,monkeypatch):
    import types,json
    from argparse import Namespace
    from audit_z_link_safety import run
    mask=np.zeros((2,6,6),np.int32)
    mask[:,0:2,0:2]=10
    mask[:,3:5,3:5]=20
    base=np.zeros_like(mask)
    base[0,0:2,0:2]=1
    base[1,0:2,0:2]=2
    base[:,3:5,3:5]=3
    # simulated union of the same-reference fragments 1 and 2
    geom=base.copy();geom[geom==2]=1;geom[geom==3]=2
    gated=geom.copy()
    module=types.ModuleType('train_segmentation')
    module.read_data=lambda _: (np.zeros_like(mask,dtype=float),mask,None,None)
    monkeypatch.setitem(sys.modules,'train_segmentation',module)
    split_dir=tmp_path/'split';split_dir.mkdir()
    pd.DataFrame([{'split':'val','y':0,'x':0,'w':6}]).to_csv(split_dir/'tile_splits.csv',index=False)
    rescue=tmp_path/'rescue';rescue.mkdir()
    np.savez_compressed(rescue/'z_link_masks_0_0.npz',baseline=base,geometry=geom,geometry_pz=gated)
    pd.DataFrame([{'z':0,'a':1,'b':2,'n_faces':4,'area_a':4,'area_b':4,
                   'overlap_min':1.,'footprint_iou':1.,'mean_pz':0.6,
                   'mutual_best':True,'geometry_eligible':True}]).to_csv(rescue/'z_link_candidates_0_0.csv',index=False)
    args=Namespace(data_dir=tmp_path,split_run_dir=split_dir,rescue_run_dir=rescue,
                   out=tmp_path/'result',tile_y=0,tile_x=0,split='val',guard=0,
                   min_purity=.8,min_coverage=.5,min_mean_pz=.45)
    run(args)
    result=json.loads((args.out/'z_link_safety_summary.json').read_text())
    assert result['link_categories'][0]['reliable_same']==1
    assert result['baseline_match_preservation']['z_overlap_mutual']['baseline_matched_instances']==2
