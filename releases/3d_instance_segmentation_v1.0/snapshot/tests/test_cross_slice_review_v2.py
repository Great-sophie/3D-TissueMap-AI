import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cross_slice_review_v2_core import (parse_review_ids,fixed_window,mask_outline,
    common_intensity_limits,plane_indices,center_crop_for_candidate)
from render_cross_slice_identity_v2 import candidate_figure,run,make_index


def test_parse_review_ids():
    assert parse_review_ids('ZPAIR_004, zpair_041')==['ZPAIR_004','ZPAIR_041']
    for value in ('','ZPAIR_004,ZPAIR_004','a42'):
        with pytest.raises(ValueError):parse_review_ids(value)


def test_stable_crop_clamping_and_contains_both():
    shape=(30,40)
    box=fixed_window(shape,(12,17),5,[(1,3),(24,35)])
    y0,y1,x0,x1=box
    assert 0<=y0<y1<=30 and 0<=x0<x1<=40
    assert y0<=1<y1 and y0<=24<y1
    assert x0<=3<x1 and x0<=35<x1


def test_stack_edges_do_not_invent_planes():
    assert plane_indices(0,4)==[-1,0,1,2]
    assert plane_indices(2,4)==[1,2,3,4]
    with pytest.raises(ValueError):plane_indices(3,4)


def test_shared_intensity_scale():
    im=np.zeros((4,25,25),float)
    im[1]=10;im[2]=150;im[3]=220
    lo,hi=common_intensity_limits(im,[0,1,2,3],(0,25,0,25))
    assert lo==0 and hi>=200


def test_outline_nonempty_and_other_pixels_false():
    a=np.zeros((12,13),bool);a[2:8,3:9]=True
    o=mask_outline(a)
    assert o.any() and (o&~a).sum()==0 and o.sum()<a.sum()


def test_crop_include_two_components():
    pred=np.zeros((4,36,36),np.int32)
    pred[1,2:5,3:7]=77
    pred[2,26:29,28:31]=88
    box=center_crop_for_candidate(pred,1,77,88,4)
    y0,y1,x0,x1=box
    assert y0<=2 and y1>=29 and x0<=3 and x1>=31


def make_data():
    z,y,x=6,42,42
    img=np.random.default_rng(10).normal(12,2,(z,y,x)).astype(np.float32)
    reference=np.zeros((z,y,x),dtype=np.int32)
    baseline=np.zeros_like(reference)
    cases=[('ZPAIR_004',0,11,12,'reliable_same',111,111),
           ('ZPAIR_006',1,21,22,'reliable_same',123,123),
           ('ZPAIR_024',2,31,32,'reliable_different',135,136),
           ('ZPAIR_026',3,41,42,'reliable_different',145,146),
           ('ZPAIR_041',4,51,52,'uncertain',0,0)]
    for k,(_,zz,a,b,cat,ra,rb) in enumerate(cases):
        iy=6+k*6;ix=11+k*3
        baseline[zz,iy:iy+3,ix:ix+3]=a
        baseline[zz+1,iy+1:iy+4,ix:ix+3]=b
        if ra:reference[zz,iy:iy+3,ix:ix+3]=ra
        if rb:reference[zz+1,iy+1:iy+4,ix:ix+3]=rb
        img[zz,iy:iy+3,ix:ix+3]+=100
        img[zz+1,iy+1:iy+4,ix:ix+3]+=80
    return img,reference,baseline,cases


def test_figure_meta_and_same_id_color_behavior():
    img,ref,pred,cases=make_data()
    rid,z,a,b,cat,ra,rb=cases[0]
    row=dict(review_id=rid,z=z,a=a,b=b,reference_category=cat,
             a_ref_majority=ra,b_ref_majority=rb,mean_pz=.5,footprint_iou=.4)
    fig,metadata=candidate_figure(img,ref,pred,row,100,200,8,80)
    assert metadata['display_z_1_based']==[None,1,2,3]
    assert metadata['reference_a']==metadata['reference_b']==111
    assert len(fig.axes)==8
    assert metadata['crop_global_yxyx'][0]>=100
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_figure_different_ref_ids():
    img,ref,pred,cases=make_data()
    rid,z,a,b,cat,ra,rb=cases[2]
    fig,meta=candidate_figure(img,ref,pred,dict(review_id=rid,z=z,a=a,b=b,
                 reference_category=cat,a_ref_majority=ra,b_ref_majority=rb,
                 mean_pz=.42,footprint_iou=.6),0,0,8)
    assert meta['reference_a']!=meta['reference_b']
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_e2e_no_overwrite_and_provenance(tmp_path,monkeypatch):
    img,ref,pred,cases=make_data()
    data=tmp_path/'data';data.mkdir()
    split=tmp_path/'splits';split.mkdir()
    pd.DataFrame([{'split':'val','y':0,'x':0,'w':48}]).to_csv(split/'tile_splits.csv',index=False)
    rescue=tmp_path/'rescue';rescue.mkdir()
    np.savez_compressed(rescue/'z_link_masks_0_0.npz',baseline=pred)
    audit=tmp_path/'audit';audit.mkdir()
    rows=[]
    for rid,z,a,b,cat,ra,rb in cases:
        rows.append(dict(review_id=rid,z=z,a=a,b=b,reference_category=cat,
              mean_pz=.5,footprint_iou=.5,a_ref_majority=ra,b_ref_majority=rb,
              manual_identity_assessment='UNREVIEWED',manual_registration_assessment='UNREVIEWED',
              reviewer_notes=''))
    pd.DataFrame(rows).to_csv(audit/'manual_z_link_review_template.csv',index=False)
    pd.DataFrame(rows).to_csv(rescue/'z_link_candidates_0_0.csv',index=False)
    (audit/'z_registration_identity_summary.json').write_text(json.dumps({
        'stack_filename_order_validation':{'status':'FILENAME_ORDER_MATCH'},
        'roi':{'tile_y':0,'tile_x':0,'guard':3,'shape_zyx':[6,42,42]}}))
    from types import ModuleType
    m=ModuleType('train_segmentation');m.read_data=lambda _: (
        np.pad(img,((0,0),(3,3),(3,3))),np.pad(ref,((0,0),(3,3),(3,3))),
        list(range(6)),'full_mask_fixture.tif')
    monkeypatch.setitem(sys.modules,'train_segmentation',m)
    out=tmp_path/'new'
    args=argparse.Namespace(data_dir=data,split_run_dir=split,rescue_run_dir=rescue,
         audit_dir=audit,out=out,review_ids=','.join(q[0] for q in cases),guard=3,
         halfwidth=10,dpi=65)
    result=run(args)
    assert len(result)==5
    assert len(list(out.glob('ZPAIR_*.png')))==5
    assert (out/'review_gallery.html').exists()
    assert len(pd.read_csv(out/'manual_z_link_review_v2.csv'))==5
    assert pd.read_csv(out/'manual_z_link_review_v2.csv').v2_manual_identity_assessment.eq('UNREVIEWED').all()
    assert pd.read_csv(audit/'manual_z_link_review_template.csv').manual_identity_assessment.eq('UNREVIEWED').all()
    assert json.loads((out/'review_manifest.json').read_text())['no_registration_transform_applied']
    with pytest.raises(FileExistsError):run(args)


def test_html_gallery_escapes_source_text():
    page=make_index([dict(review_id='ZPAIR_001',reference_category='<x>',image_file='x.png')])
    assert '<img' in page and '&lt;x&gt;' in page
    assert 'Pseudo-labels' in make_index([])


def test_provenance_candidate_mismatch(tmp_path,monkeypatch):
    img,ref,pred,cases=make_data()
    data=tmp_path/'data';data.mkdir()
    split=tmp_path/'splits';split.mkdir()
    pd.DataFrame([{'split':'val','y':0,'x':0,'w':48}]).to_csv(split/'tile_splits.csv',index=False)
    rescue=tmp_path/'rescue';rescue.mkdir()
    np.savez_compressed(rescue/'z_link_masks_0_0.npz',baseline=pred)
    audit=tmp_path/'audit';audit.mkdir()
    rid,z,a,b,cat,ra,rb=cases[0]
    pd.DataFrame([dict(review_id=rid,z=z,a=a,b=b,reference_category=cat,
              a_ref_majority=ra,b_ref_majority=rb,
              mean_pz=.5,footprint_iou=.5)]).to_csv(audit/'manual_z_link_review_template.csv',index=False)
    # Intentionally mismatched source tuple; reject instead of inventing provenance.
    pd.DataFrame([dict(z=z,a=a,b=b+999)]).to_csv(rescue/'z_link_candidates_0_0.csv',index=False)
    (audit/'z_registration_identity_summary.json').write_text(json.dumps({
       'stack_filename_order_validation':{'status':'FILENAME_ORDER_MATCH'},
       'roi':{'tile_y':0,'tile_x':0,'guard':3,'shape_zyx':[6,42,42]}}))
    from types import ModuleType
    m=ModuleType('train_segmentation')
    m.read_data=lambda _: (np.pad(img,((0,0),(3,3),(3,3))),
                           np.pad(ref,((0,0),(3,3),(3,3))),[], 'mask.tif')
    monkeypatch.setitem(sys.modules,'train_segmentation',m)
    args=argparse.Namespace(data_dir=data,split_run_dir=split,rescue_run_dir=rescue,
          audit_dir=audit,out=tmp_path/'new',review_ids=rid,guard=3,halfwidth=10,dpi=60)
    with pytest.raises(RuntimeError,match='do not match original'):
        run(args)
