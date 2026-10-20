import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest
import tifffile

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from z_registration_identity_core import (overlapping_views,normalized_corr,alignment_screen,
    adjacent_mask_screen,id_slice_statistics,stratified_samples,pair_crop_center,crop_bounds,
    image_mask_enrichment_screen)
from audit_z_registration_identity import run


def test_shift_views_align():
    rng=np.random.default_rng(5)
    x=rng.normal(size=(48,48))
    y=np.zeros_like(x)
    y[3:,5:]=x[:-3,:-5]
    # Shift Y by (-3,-5) to recover original X.
    a,b=overlapping_views(x,y,-3,-5)
    assert np.array_equal(a,b)
    assert normalized_corr(x,y,-3,-5)==pytest.approx(1.)


def test_alignment_screen_identifies_known_shift():
    rng=np.random.default_rng(8)
    x=rng.random((64,64));y=np.zeros_like(x);y[3:,4:]=x[:-3,:-4]
    r=alignment_screen(x,y,max_shift=6)
    assert (r['shift_dy_to_align_next'],r['shift_dx_to_align_next'])==(-3,-4)
    assert r['best_shift_corr']>r['zero_shift_corr']


def test_alignment_screen_no_shift():
    rng=np.random.default_rng(1);x=rng.random((48,48))
    r=alignment_screen(x,x,max_shift=5)
    assert (r['shift_dy_to_align_next'],r['shift_dx_to_align_next'])==(0,0)
    assert r['best_shift_corr']==pytest.approx(1)


def test_connected_id_persistence_and_gaps():
    r=np.zeros((5,4,4),np.int32)
    r[0,1,1]=1;r[1,1,1]=1;r[3,1,1]=1;r[0,0,0]=2
    stats=id_slice_statistics(r).set_index('reference_id')
    assert int(stats.loc[1,'n_gaps_in_occupied_slices'])==1
    assert int(stats.loc[1,'n_occupied_slices'])==3
    assert int(stats.loc[1,'n_3d_6_connected_components'])==2
    assert not bool(stats.loc[2,'is_multi_slice'])


def test_ir193_mask_alignment_enrichment():
    labels=np.zeros((20,20),np.int16);labels[5:15,5:15]=1
    sig=np.ones((20,20),np.float32);sig[6:14,6:14]=100
    stats=image_mask_enrichment_screen(sig,labels)
    assert stats["bright_ir193_capture_by_mask"]==pytest.approx(1.0)
    assert stats["bright_capture_enrichment_vs_mask_area"]>1


def test_adjacent_label_screen():
    a=np.array([[1,2,0],[1,0,0]])
    b=np.array([[1,4,0],[0,0,0]])
    r=adjacent_mask_screen(a,b)
    assert r['both_foreground_faces']==2
    assert r['same_id_face_fraction_both_foreground']==0.5
    assert r['n_shared_ids']==1


def test_sampling_does_not_invent_gold_labels():
    df=pd.DataFrame([dict(z=0,a=i+1,b=i+4,reference_category=c,geometry_eligible=True,
                          mean_pz=p,footprint_iou=.4) for i,(c,p) in enumerate(
        [('reliable_same',.3),('reliable_same',.5),('reliable_different',.3),
         ('reliable_different',.5),('uncertain',.3),('uncertain',.5)])])
    x=stratified_samples(df,n_per_category=2,seed=7)
    assert len(x)==6
    assert x['manual_identity_assessment'].eq('UNREVIEWED').all()
    assert set(x.reference_category)==set(df.reference_category)


def test_crop_center_overlap():
    mask=np.zeros((2,25,25),np.int32)
    mask[0,4:10,3:10]=1;mask[1,6:11,5:12]=2
    row=pd.Series({'z':0,'a':1,'b':2})
    y,x=pair_crop_center(mask,mask,row)
    assert 6<=y<=9 and 5<=x<=9
    assert crop_bounds((y,x),mask.shape[1:],5)[0]>=0


def test_stratified_missing_category_and_empty():
    df=pd.DataFrame([{'z':0,'a':1,'b':2,'reference_category':'uncertain',
                      'geometry_eligible':True,'mean_pz':.2,'footprint_iou':.2}])
    r=stratified_samples(df,8)
    assert len(r)==1


def make_fixture(tmp_path):
    folder=tmp_path/'data';folder.mkdir()
    Z,H,W=16,64,64
    rng=np.random.default_rng(56)
    yy,xx=np.mgrid[:H,:W]
    base=(np.exp(-((xx-22)**2+(yy-22)**2)/35)+np.exp(-((xx-42)**2+(yy-42)**2)/40))
    mask=np.zeros((Z,H,W),np.uint16)
    mask[:,19:27,19:27]=1
    mask[:,40:47,41:48]=2
    mask[4,19:27,19:27]=0  # artificial ID gap
    tifffile.imwrite(folder/'full_mask_fixture.tif',mask,photometric='minisblack',metadata={'axes':'ZYX'})
    from zipfile import ZIP_DEFLATED
    with ZipFile(folder/'fixture_SIMILARITY.zip','w',ZIP_DEFLATED) as zf:
        for z in range(Z):
            arr=np.clip(10+base*100+rng.normal(0,2,(H,W)),0,255).astype(np.uint8)
            if z==3:arr=np.roll(arr,2,axis=1)
            bio=io.BytesIO();tifffile.imwrite(bio,arr)
            zf.writestr(f'reg/Ir193/{z+1:04d}_sample_section_{z+1:02d}__Ir193.tiff',bio.getvalue())
    (folder/'final_3D_stack_order.csv').write_text(''.join(f'sample_section_{z+1:02d}.ome.tiff,{z+1}\n' for z in range(Z)))
    split=tmp_path/'runs'/'split';split.mkdir(parents=True)
    pd.DataFrame([{'split':'val','y':0,'x':0,'w':64}]).to_csv(split/'tile_splits.csv',index=False)
    (split/'data_audit.json').write_text(json.dumps({'stack_order_check':{'status':'FILENAME_ORDER_MATCH'}}))
    rescue=tmp_path/'runs'/'rescue';rescue.mkdir()
    np.savez_compressed(rescue/'z_link_masks_0_0.npz',baseline=mask,geometry=mask,geometry_pz=mask)
    safety=tmp_path/'runs'/'safety';safety.mkdir()
    rows=[{'z':int(z),'a':1,'b':2,'reference_category':cat,'geometry_eligible':True,
          'mean_pz':p,'footprint_iou':.3,'a_ref_majority':1,'b_ref_majority':2}
          for z,(cat,p) in enumerate([('reliable_same',.55),('reliable_different',.2),('uncertain',.6)])]
    # b has no overlap with a. Center fallback uses two components.
    pd.DataFrame(rows).to_csv(safety/'z_link_candidates_reference_audit.csv',index=False)
    pd.DataFrame(rows).to_csv(rescue/'z_link_candidates_0_0.csv',index=False)
    return folder,split,rescue,safety


def test_end_to_end(tmp_path):
    folder,split,rescue,safety=make_fixture(tmp_path)
    import argparse
    out=tmp_path/'runs'/'identity_audit'
    args=argparse.Namespace(data_dir=folder,split_run_dir=split,rescue_run_dir=rescue,
       safety_audit_dir=safety,out=out,tile_y=0,tile_x=0,guard=0,max_shift=4,
       seed=42,n_per_category=2,crop_halfwidth=14)
    summary=run(args)
    assert summary['stack_filename_order_validation']['status']=='FILENAME_ORDER_MATCH'
    assert summary['registration_screen']['n_adjacent_pairs']==15
    assert summary['reference_ID_continuity']['n_ids_with_nonconsecutive_slice_occupancy']>=1
    assert len(pd.read_csv(out/'manual_z_link_review_template.csv'))==3
    assert (out/'z_registration_overview.png').exists()
    assert (out/'review_atlas_reliable_same_page_01.png').exists()
    assert (out/'review_atlas_reliable_different_page_01.png').exists()
    assert (out/'review_atlas_uncertain_page_01.png').exists()
    assert len(pd.read_csv(out/'z_pair_registration_screen.csv'))==15
    assert len(pd.read_csv(out/'ir193_mask_alignment_screen.csv'))==16
    with pytest.raises(FileExistsError):run(args)
