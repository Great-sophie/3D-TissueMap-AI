import importlib.util
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parents[1]/'scripts'/'train_segmentation.py'
spec=importlib.util.spec_from_file_location('train_segmentation',p)
segment=importlib.util.module_from_spec(spec)
spec.loader.exec_module(segment)


def fixture_mask():
    mask=np.zeros((8,256,256),np.uint32)
    label=1
    for y in range(4):
      for x in range(4):
        mask[:,y*64+16:y*64+46,x*64+16:x*64+46]=label
        label+=1
    return mask


def test_splits_are_exclusive_and_reproducible():
    mask=fixture_mask()
    a=segment.spatial_tiles(mask,64,8,42)
    b=segment.spatial_tiles(mask,64,8,42)
    assert a==b
    for i,t in enumerate(a):
      if t['split'] not in ('train','val','test'):continue
      for q in a[i+1:]:
        if q['split'] not in ('train','val','test'):continue
        overlap=(t['x']<q['x1'] and q['x']<t['x1'] and t['y']<q['y1'] and q['y']<t['y1'])
        assert not overlap
    assert {t['split'] for t in a}=={'train','val','test'}


def test_instance_id_persistence():
    report=segment.instance_audit(fixture_mask())
    assert report['fraction_IDs_seen_in_multiple_slices']==1.
    assert len(report['adjacent_plane_ID_continuity'])==7
    assert all(r['same_ID_fraction_where_both_foreground']==1. for r in report['adjacent_plane_ID_continuity'])


def test_metrics_perfect_and_empty():
    true=np.zeros((4,32,32),np.uint8);true[:,4:20,4:20]=1
    pred=true.astype(np.float32)
    m=segment.metrics(pred,true,2)
    assert m['dice']==1.0 and m['iou']==1.0
    null=segment.metrics(np.zeros_like(pred),true,2)
    assert null['dice']<0.01


def test_unet_2d_3d_shapes():
    import torch
    with torch.no_grad():
        m2=segment.UNet(2,4);m3=segment.UNet(3,4)
        assert m2(torch.ones((1,1,32,32))).shape==(1,1,32,32)
        assert m3(torch.ones((1,1,4,32,32))).shape==(1,1,4,32,32)


def test_norm_finite():
    a=np.array([[np.nan,0,1,2],[3,4,np.inf,5]],dtype=np.float32)
    assert np.isfinite(segment.normalize_slice(a)).all()


def test_stack_order_headerless_0based(tmp_path):
    lines=[]; entries=[]
    for i in range(1,17):
        section=f"20190220_lvi03_bc_test02_s1_p9_r{i}_a{i}_ac"
        lines.append(f"{section}.ome.tiff,{i-1}")
        entries.append((i,f"SIMILARITY10_In115/Ir193/{i:04d}_{section}__Ir193.tiff"))
    path=tmp_path/'final_3D_stack_order.csv'
    path.write_text('\n'.join(lines)+'\n')
    result=segment.inspect_stack_order(path, entries, 16)
    assert result['status']=='FILENAME_ORDER_MATCH',result
    assert result['csv_rows']==16 and result['csv_index_base']==0

def test_stack_order_wrong_mapping_is_blocked(tmp_path):
    lines=[]; entries=[]
    for i in range(1,17):
        section=f"20190220_lvi03_bc_test02_s1_p9_r{i}_a{i}_ac"
        lines.append(f"{section}.ome.tiff,{15-i}")
        entries.append((i,f"SIMILARITY10_In115/Ir193/{i:04d}_{section}__Ir193.tiff"))
    path=tmp_path/'final_3D_stack_order.csv'
    path.write_text('\n'.join(lines)+'\n')
    result=segment.inspect_stack_order(path, entries, 16)
    assert result['status']=='BLOCKED'
