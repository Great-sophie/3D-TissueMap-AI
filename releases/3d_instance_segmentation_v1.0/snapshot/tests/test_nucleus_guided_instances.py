"""Synthetic-only algorithmic unit checks; not evidence of biomedical performance."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from nucleus_guided_instance_separation import nucleus_guided_instances


def sample():
    z,y,x=np.ogrid[:8,:32,:32]
    a=(z-3)**2/5+(y-10)**2/20+(x-11)**2/20
    b=(z-4)**2/5+(y-21)**2/20+(x-21)**2/20
    image=(np.exp(-a/2)+np.exp(-b/2)).astype(np.float32)
    foreground=((a<=1.5)|(b<=1.5)).astype(np.float32)
    probability=.15+.75*foreground
    return probability,image


def test_nucleus_guided_separates_two_isolated_objects():
    p, im = sample()
    pred,meta=nucleus_guided_instances(p,im,min_distance=3,intensity_threshold=.2)
    assert pred.shape==p.shape
    assert len(np.unique(pred[p>0.5]))==2
    assert meta['n_seeds']>=2
    assert np.all(pred[p<.5]==0)


def test_nucleus_guided_component_fallback_for_zero_intensity():
    p,im=sample()
    pred,meta=nucleus_guided_instances(p,np.zeros_like(im),min_distance=3,
                                      intensity_threshold=.8)
    assert meta['n_fallback_seeds']>=2
    assert pred.max()>=2


def test_nucleus_guided_reject_shape_mismatch():
    p,im=sample()
    with pytest.raises(ValueError):
        nucleus_guided_instances(p,im[:-1])
