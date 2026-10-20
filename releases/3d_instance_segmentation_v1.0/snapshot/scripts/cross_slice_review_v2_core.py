"""Pure visualization helpers for exploratory Z identity review.

No image/label registration transforms or model changes. Display coordinates
remain common for every panel; labels are AUTHOR watershed pseudo IDs.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def parse_review_ids(value: str) -> list[str]:
    ids = [x.strip().upper() for x in value.split(',') if x.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Provide a nonempty, unique comma-separated review-id list')
    if not all(x.startswith('ZPAIR_') and x[6:].isdigit() for x in ids):
        raise ValueError('Expected review IDs like ZPAIR_004')
    return ids


def fixed_window(shape, center, halfwidth, required_points=(), pad=3):
    """Choose ONE fixed XY crop shared across Z planes; include both candidates.

    If both cannot fit in the image, clamp to data bounds without introducing
    fake shift. All local indices are mapped to the same global pixel coords.
    """
    h, w = map(int, shape)
    if min(h,w)<1 or halfwidth<2:raise ValueError('Invalid crop shape or width')
    pts=np.asarray(list(required_points),dtype=float).reshape(-1,2)
    cy,cx=map(float,center)
    if len(pts):
        # Enlarge symmetrically if necessary, rather than clipping a candidate.
        hw_y=max(halfwidth,int(np.ceil(max(abs(pts[:,0]-cy))))+pad)
        hw_x=max(halfwidth,int(np.ceil(max(abs(pts[:,1]-cx))))+pad)
    else:hw_y=hw_x=halfwidth
    size_y=min(h,2*hw_y+1);size_x=min(w,2*hw_x+1)
    y0=max(0,min(h-size_y,int(round(cy))-size_y//2))
    x0=max(0,min(w-size_x,int(round(cx))-size_x//2))
    return y0,y0+size_y,x0,x0+size_x


def mask_outline(mask):
    m=np.asarray(mask,dtype=bool)
    if m.ndim!=2:raise ValueError('Expected 2D mask')
    return m & ~ndi.binary_erosion(m,structure=np.ones((3,3),bool),border_value=0)


def common_intensity_limits(stack, z_indices, bounds):
    """One grayscale display range for all valid panels of a candidate."""
    y0,y1,x0,x1=bounds
    planes=[np.asarray(stack[z,y0:y1,x0:x1],np.float32).ravel()
            for z in z_indices if 0 <= z < stack.shape[0]]
    if not planes:raise ValueError('No valid planes')
    data=np.concatenate(planes)
    data=data[np.isfinite(data)]
    if data.size<1:return 0.,1.
    lo,hi=np.percentile(data,[1,99.5])
    return float(lo),float(max(hi,lo+1e-6))


def get_candidate_geometry(labels,z,a,b):
    if labels.ndim!=3 or z<0 or z+1>=labels.shape[0]:
        raise ValueError('Invalid 3D Z candidate')
    aa=np.argwhere(labels[z]==a);bb=np.argwhere(labels[z+1]==b)
    if not len(aa) or not len(bb):
        raise ValueError('Missing predicted component in specified candidate plane')
    union=np.concatenate([aa,bb])
    center=np.rint(np.mean(union,axis=0)).astype(int)
    # bbox corners include both candidates (overlap not assumed).
    corners=union[[np.argmin(union[:,0]),np.argmax(union[:,0]),
                   np.argmin(union[:,1]),np.argmax(union[:,1])]]
    return tuple(center.tolist()),corners


def center_crop_for_candidate(labels,z,a,b,halfwidth):
    center,points=get_candidate_geometry(labels,z,a,b)
    return fixed_window(labels.shape[1:],center,halfwidth,points)


def plane_indices(z,nz):
    if z<0 or z+1>=nz:raise ValueError('Out of range Z pair')
    return [z-1,z,z+1,z+2]
