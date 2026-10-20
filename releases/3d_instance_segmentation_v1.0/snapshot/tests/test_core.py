import numpy as np
import pandas as pd
import pytest
import torch
from tissuemap.data import generate_synthetic, read_manifest, load_sample, read_stack
from tissuemap.models import MiniUNet, dice_bce_loss
from tissuemap.evaluate import binary_metrics
from tissuemap.spatial import _table_for_sample, _proximity_2d_3d


def test_synthetic_schema(tmp_path):
    mf=generate_synthetic(tmp_path)
    df=read_manifest(mf)
    assert len(df)==6
    image,mask,channels=load_sample(df.iloc[0]["path"])
    assert image.shape==(4,12,48,48)
    assert mask.shape==(12,48,48)
    assert channels==["Nuclear","CD8","panCK","CD31"]


def test_patient_split_leakage_rejected(tmp_path):
    mf=generate_synthetic(tmp_path)
    df=pd.read_csv(mf)
    df.loc[5,"patient_id"]=df.loc[0,"patient_id"]
    df.to_csv(mf,index=False)
    with pytest.raises(ValueError,match="leakage"):
        read_manifest(mf)


def test_axes_reorder(tmp_path):
    src=np.arange(3*4*6*8).reshape(3,4,6,8)
    np.save(tmp_path/"image.npy",np.transpose(src,(1,0,2,3)))
    x=read_stack(tmp_path/"image.npy","ZCYX")
    np.testing.assert_array_equal(x,src)


def test_models_forward_and_loss():
    for dim,size in ((2,(16,16)),(3,(8,16,16))):
        m=MiniUNet(dim=dim,base=4)
        x=torch.randn((1,1,*size));y=(x>.5).float()
        logits=m(x)
        assert logits.shape==x.shape
        assert torch.isfinite(dice_bce_loss(logits,y))


def test_binary_metrics_physical_spacing():
    a=np.zeros((4,8,8),bool);a[1:3,2:5,2:5]=1
    m=binary_metrics(a,a,spacing=(3,1,1))
    assert m["dice"]==pytest.approx(1)
    assert m["hd95_um"]==pytest.approx(0)
    empty=binary_metrics(a,np.zeros_like(a))
    assert empty["hd95_um"] is None


def test_spatial_physical_volume():
    image=np.ones((2,2,4,4),dtype=np.float32)
    mask=np.zeros((2,4,4),int)
    mask[:,1:3,1:3]=1
    cells=_table_for_sample(image,mask,["CD8","panCK"],(2.,1.,1.),"s1","p1","reference")
    assert cells.iloc[0].volume_um3==pytest.approx(16)
    assert cells.iloc[0]["mean_CD8"]==pytest.approx(1)


def test_proximity_does_not_fake_2d_matches():
    cells=pd.DataFrame({"sample_id":["s1","s1","s1","s1"],"cell_id":[1,2,3,4],
                        "centroid_z_um":[0,2,4,8],"centroid_y_um":[1,2,4,8],"centroid_x_um":[1,2,4,8],
                        "mean_CD8":[10,1,1,1],"mean_panCK":[1,1,1,10]})
    d=_proximity_2d_3d(cells,spacing_z=2)
    assert len(d)==1 and np.isnan(d.iloc[0].nearest_same_section_2d_um)


def test_unsupervised_fusion(tmp_path):
    from tissuemap.fusion import fuse
    data=pd.DataFrame({"sample_id":["s"]*8,"patient_id":["p"]*8,"cell_id":list(range(1,9)),
                       "volume_um3":[5,6,7,8,20,21,22,23],
                       "elongation_3d":[1.,1.2,1.1,1.3,2.,2.1,2.3,2.2],
                       "mean_CD8":[1.,1.1,1.2,1.1,8.,9.,7.,8.5],
                       "mean_panCK":[8.,7.,8.,7.5,1.,1.1,1.2,1.]})
    source=tmp_path/"cells.csv"; data.to_csv(source,index=False)
    meta=fuse(source,tmp_path/"fusion",k=2)
    assert meta["n_cells"]==8
    assert (tmp_path/"fusion"/"exploratory_clusters.csv").exists()
