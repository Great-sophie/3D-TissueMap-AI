import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from dynamic_region_merging import build_region_evidence,merge_dynamic_regions,_summary


def row(a,b,xylog=0., xyweight=1., xyneg=0., xyfaces=1,zlog=0.,zweight=0.,zneg=0.,zfaces=0):
    return [a,b,xylog,xyweight,xyneg,xyfaces,zlog,zweight,zneg,zfaces]


def test_dynamic_interface_reaggregation_prevents_false_bridge():
    # XY interfaces 1-2=high, 2-3=positive, 1-3=negative.
    # After merging 1+2, the A-to-3 interface uses BOTH original contacts:
    # (2 + (-2)) / 2 = 0, thus must NOT merge 3.
    sv=np.array([[[1,2,3]]],np.int32)
    e=np.array([row(1,2,xylog=3.),row(1,3,xylog=-2.),row(2,3,xylog=2.)],float)
    out,meta=merge_dynamic_regions(sv,e,score_threshold=.2,min_faces=1,max_component_voxels=50,
                                   phase_order='joint',use_repulsion=False)
    assert meta['n_merges']==1
    assert out[0,0,0]==out[0,0,1]!=out[0,0,2]


def test_repulsive_veto_changes_eligible_merges():
    sv=np.array([[[1,2]]],np.int32)
    e=np.array([row(1,2,xylog=1., xyweight=3., xyneg=1.5, xyfaces=3)],float)
    free, fmeta=merge_dynamic_regions(sv,e,score_threshold=.2,max_negative_fraction=.35,min_faces=1,
                                      phase_order='joint',use_repulsion=False)
    veto, vmeta=merge_dynamic_regions(sv,e,score_threshold=.2,max_negative_fraction=.35,min_faces=1,
                                      phase_order='joint',use_repulsion=True)
    assert fmeta['n_merges']==1 and vmeta['n_merges']==0
    assert free[0,0,0]==free[0,0,1] and veto[0,0,0]!=veto[0,0,1]


def test_xy_then_z_is_two_stages():
    sv=np.array([[[1]],[[2]]],np.int32)
    e=np.array([row(1,2,xyweight=0.,xyfaces=0,zlog=1.,zweight=1.,zfaces=1)],float)
    out,meta=merge_dynamic_regions(sv,e,phase_order='xy_then_z',min_faces=1)
    assert meta['phase_stats'][0]['merges']==0
    assert meta['phase_stats'][1]['merges']==1
    assert out[0,0,0]==out[1,0,0]


def test_build_evidence_supports_xy_and_z_and_is_additive():
    sv=np.array([[[1,2]],[[3,4]]],np.int32)
    a=np.full((3,*sv.shape),.9,np.float64)
    a[0]=.2
    data=build_region_evidence(sv,a,z_weight=.25)
    assert len(data)==4
    e={(int(r[0]),int(r[1])):r[2:] for r in data}
    assert e[1,2][3]==1 and e[1,2][7]==0
    assert e[1,3][3]==0 and e[1,3][7]==1
    assert np.isclose(e[1,3][5],.25)
    assert np.isclose(_summary(e[1,3],'joint')[0],np.log(.2/.8))


def test_empty_and_high_cap_and_background():
    sv=np.zeros((2,5,5),np.int32)
    a=np.ones((3,*sv.shape),np.float32)
    e=build_region_evidence(sv,a)
    assert e.shape==(0,10)
    out,meta=merge_dynamic_regions(sv,e)
    assert np.array_equal(out,sv) and meta['final_components']==0


def test_oracle_recovers_separated_cells():
    # two separate regions, split each into gridlets, with perfect predictions
    ref=np.zeros((2,10,10),np.int32)
    ref[:,1:4,1:4]=1
    ref[:,6:9,6:9]=2
    from scipy.ndimage import label
    # Assign unique SV to every foreground voxel to stress the merge bookkeeping.
    sv=np.zeros_like(ref)
    sv[ref>0]=np.arange(1,np.count_nonzero(ref)+1)
    p=np.full((3,*ref.shape),.01,np.float32)
    for ax in range(3):
        lo=[slice(None)]*3;hi=[slice(None)]*3
        lo[ax]=slice(None,-1);hi[ax]=slice(1,None)
        r1,r2=ref[tuple(lo)],ref[tuple(hi)]
        p[ax][tuple(lo)] = np.where((r1>0)&(r1==r2),.99,.01)
    e=build_region_evidence(sv,p)
    pred,meta=merge_dynamic_regions(sv,e,min_faces=1,score_threshold=.2,max_component_voxels=1000)
    assert meta['final_components']==2
    assert len(np.unique(pred[ref==1]))==1
    assert len(np.unique(pred[ref==2]))==1
    assert pred[0,1,1]!=pred[0,6,6]


def test_size_cap_blocks_large_union():
    sv=np.array([[[1,1,2,2]]],np.int32)
    e=np.array([row(1,2,xylog=3.)],float)
    pred,meta=merge_dynamic_regions(sv,e,min_faces=1,max_component_voxels=3)
    assert meta['n_merges']==0
    assert meta['phase_stats'][-1]['blocked_by_volume_cap']==1
    assert pred[0,0,0]!=pred[0,0,2]
