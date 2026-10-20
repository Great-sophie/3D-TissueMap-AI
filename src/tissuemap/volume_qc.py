"""Auditable 3D tissue volume onboarding and spatial-axis QC.

No image axis inference for raw TIFF; metadata are inspected separately. Physical
measurements are computed only when spacing is explicitly provided and verified.

Examples:
  python -m tissuemap.volume_qc --demo --output runs/synthetic_volume_qc
  python -m tissuemap.volume_qc --header-only --image sample.tiff
  python -m tissuemap.volume_qc --image sample.tiff --image-axes CZYX \
      --channel-index 0 --mask instance_mask.tiff --mask-axes ZYX \
      --spacing 2.0 1.0 1.0 --output runs/real_volume_qc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import find_objects

from .data import generate_synthetic, load_sample, read_stack, normalize_image


def tiff_header(path):
    path = Path(path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Header-only mode currently supports TIFF images only")
    with tifffile.TiffFile(str(path)) as tf:
        series = [{"index": i, "shape": list(s.shape), "axes_metadata": s.axes,
                   "dtype": str(s.dtype), "page_count": len(s.pages)}
                  for i, s in enumerate(tf.series)]
        ome = bool(tf.ome_metadata)
        imagej = bool(tf.is_imagej)
    return {"file": str(path.resolve()), "size_bytes": path.stat().st_size,
            "imagej": imagej, "ome_xml_present": ome, "series": series,
            "warning": "TIFF axes metadata are clues, not a substitute for validating channel identity, registration or physical Z spacing."}


def _bbox_features(labels, spacing=None):
    """Cell-level instance summaries in native voxel order (Z,Y,X).

    Bounding lengths are axis-aligned extents, NOT fitted nuclear diameters.
    """
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
        raise ValueError("Expected nonnegative integer instance mask (Z,Y,X)")
    nz = np.nonzero(labels)
    label_vals = labels[nz]
    ids, inverse, counts = np.unique(label_vals, return_inverse=True, return_counts=True)
    data = []
    sz, sy, sx = (float(x) for x in spacing) if spacing is not None else (None, None, None)
    if len(ids):
        zc, yc, xc = (v.astype(np.float64, copy=False) for v in nz)
        sums = [np.bincount(inverse, weights=v, minlength=len(ids)) for v in (zc, yc, xc)]
        mins = [np.full(len(ids), np.iinfo(np.int64).max, dtype=np.int64) for _ in range(3)]
        maxs = [np.full(len(ids), -1, dtype=np.int64) for _ in range(3)]
        for i, coord in enumerate(nz):
            np.minimum.at(mins[i], inverse, coord)
            np.maximum.at(maxs[i], inverse, coord)
        for i, (label, n) in enumerate(zip(ids, counts)):
            centroid = tuple(float(s[i]/n) for s in sums)
            extents = [int(maxs[j][i] - mins[j][i] + 1) for j in range(3)]
            row = {"instance_id": int(label), "n_voxels": int(n),
                   "centroid_z_vox": centroid[0], "centroid_y_vox": centroid[1], "centroid_x_vox": centroid[2],
                   "bbox_z_vox": extents[0], "bbox_y_vox": extents[1], "bbox_x_vox": extents[2]}
            if spacing is not None:
                row.update({"volume_um3": n * sz * sy * sx,
                            "centroid_z_um": centroid[0] * sz,
                            "centroid_y_um": centroid[1] * sy,
                            "centroid_x_um": centroid[2] * sx,
                            "bbox_z_um": extents[0]*sz,
                            "bbox_y_um": extents[1]*sy,
                            "bbox_x_um": extents[2]*sx})
            data.append(row)
    return pd.DataFrame(data, columns=["instance_id", "n_voxels", "centroid_z_vox", "centroid_y_vox", "centroid_x_vox",
                                       "bbox_z_vox", "bbox_y_vox", "bbox_x_vox"] +
                       (["volume_um3", "centroid_z_um", "centroid_y_um", "centroid_x_um", "bbox_z_um", "bbox_y_um", "bbox_x_um"]
                        if spacing is not None else []))


def _orthogonal_plot(image, labels, output, spacing=None, channel="unknown"):
    arr = normalize_image(image)
    z,y,x = [i//2 for i in arr.shape]
    panes = [(arr[z,:,:], labels[z,:,:] if labels is not None else None, "XY — Z midplane", "X", "Y", (2,1)),
             (arr[:,y,:], labels[:,y,:] if labels is not None else None, "XZ — Y midplane", "X", "Z", (2,0)),
             (arr[:,:,x], labels[:,:,x] if labels is not None else None, "YZ — X midplane", "Y", "Z", (1,0))]
    fig, axs = plt.subplots(1,3,figsize=(13,4))
    for ax, (im, lab, title, xlabel, ylabel, idx) in zip(axs,panes):
        if spacing is not None:
            ext = (0, im.shape[1]*spacing[idx[0]], im.shape[0]*spacing[idx[1]], 0)
        else:
            ext = (0, im.shape[1], im.shape[0], 0)
        ax.imshow(im,cmap="gray",interpolation="nearest",extent=ext,aspect="equal")
        if lab is not None:
            overlay = np.ma.masked_equal(lab,0)
            ax.imshow(overlay,alpha=.4,cmap="turbo",interpolation="nearest",extent=ext,aspect="equal")
        unit = "μm" if spacing is not None else "pixels"
        ax.set_title(title,fontsize=10); ax.set_xlabel(f"{xlabel} ({unit})"); ax.set_ylabel(f"{ylabel} ({unit})")
    fig.suptitle(f"3D Volume QC | {channel} | raw voxel planes; Z/Y/X spacing: {list(spacing) if spacing else 'UNKNOWN'} μm",fontsize=11)
    fig.tight_layout(); fig.savefig(output,dpi=150,bbox_inches="tight"); plt.close(fig)


def _projection_plot(image, output):
    arr = normalize_image(image)
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,im,title in zip(axs,[arr.max(axis=0),arr.max(axis=1),arr.max(axis=2)],
                           ["Max projection along Z → XY", "Max projection along Y → XZ", "Max projection along X → YZ"]):
        ax.imshow(im,cmap="gray",interpolation="nearest"); ax.set_title(title,fontsize=10)
        ax.set_xlabel("pixels"); ax.set_ylabel("pixels")
    fig.suptitle("Maximum-intensity projections are NOT physical single tissue sections")
    fig.tight_layout(); fig.savefig(output,dpi=150,bbox_inches="tight"); plt.close(fig)


def run_audit(image, mask, spacing, channel, output, provenance, axes="CZYX"):
    image = np.asarray(image)
    if image.ndim != 4:
        raise ValueError("Image must be canonical 4D (C,Z,Y,X)")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("Image must have numeric voxel intensity")
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite values")
    if mask is not None:
        if mask.shape != image.shape[1:]:
            raise ValueError(f"Misregistered shape: mask={mask.shape}, image={image.shape[1:]}")
        if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
            raise ValueError("Instance mask must contain nonnegative integer labels")
    if spacing is not None:
        if len(spacing) != 3 or not all(np.isfinite(v) and v > 0 for v in spacing):
            raise ValueError("Spacing must be strictly positive Z,Y,X in μm")
    if not 0 <= channel < image.shape[0]:
        raise ValueError(f"Channel index {channel} outside 0..{image.shape[0]-1}")
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    vol = image[channel]
    feats = _bbox_features(mask,spacing) if mask is not None else None
    if feats is not None:
        feats.to_csv(output/"cell_morphology_qc.csv",index=False)
    _orthogonal_plot(vol,mask,output/"orthogonal_slices.png",spacing=spacing,channel=str(channel))
    _projection_plot(vol,output/"maximum_projections.png")
    report = {
        "provenance":provenance,"image_shape_CZYX":list(image.shape),
        "dtype":str(image.dtype),"selected_channel_index":int(channel),
        "selected_channel_dynamic_range":{"min":float(vol.min()),"max":float(vol.max())},
        "finite_intensity":True,"axes_interpreted":"CZYX",
        "spacing_ZYX_um":list(map(float,spacing)) if spacing is not None else None,
        "physical_units_valid": spacing is not None,
        "mask_provided": mask is not None,
        "instance_count":len(feats) if feats is not None else None,
        "foreground_fraction":float(np.mean(mask>0)) if mask is not None else None,
        "annotation_quality":"NOT VERIFIED — source/provenance review required",
        "scientific_limitations":[
            "No inference about disease biology without validating tissue and marker identities.",
            "Spacing is provided by user; it has not been automatically verified against acquisition metadata.",
            "Orthogonal sections are displayed in pixel coordinates; anisotropy must be considered when interpreting morphology.",
            "No 3D segmentation model performance or independent test cohort is claimed."
        ]}
    (output/"volume_qc_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    return report


def run_demo(output):
    from .data import generate_synthetic
    output=Path(output)
    manifest=generate_synthetic(output/"synthetic_source",samples=6)
    sample=output/"synthetic_source"/"synthetic_00.npz"
    im,mask,channels=load_sample(sample)
    r=run_audit(im,mask,(2.,.8,.8),0,output,"SYNTHETIC, generated solely for smoke-test and training")
    # False-neighbourhood intuition: two objects share XY centre but are separated in Z.
    extent=(16,64,64)
    zz,yy,xx=np.ogrid[:extent[0],:extent[1],:extent[2]]
    a=((zz-3)/1.7)**2+((yy-32)/6.0)**2+((xx-32)/6.0)**2<=1
    b=((zz-12)/1.7)**2+((yy-32)/6.0)**2+((xx-32)/6.0)**2<=1
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(a.max(axis=0),cmap="Blues",alpha=.8); axs[0].imshow(np.ma.masked_where(~b.max(axis=0),b.max(axis=0)),cmap="Oranges",alpha=.6)
    axs[0].set_title("XY maximum projection\nlooks overlapping")
    axs[1].imshow((a*1+b*2)[:,32,:],cmap="viridis",vmin=0,vmax=2,interpolation="nearest",aspect="auto")
    axs[1].set_title("XZ reveals Z separation")
    centers_um=np.array([[3*2.,32*.8,32*.8],[12*2.,32*.8,32*.8]])
    axs[2].scatter([centers_um[0,2],centers_um[1,2]],[centers_um[0,0],centers_um[1,0]],c=[0,1],s=110)
    axs[2].set_xlim(20,34);axs[2].set_ylim(0,32);axs[2].set_xlabel("X (μm)");axs[2].set_ylabel("Z (μm)")
    axs[2].set_title("3D centroid gap = 18 μm\nXY-centroid gap = 0 μm")
    fig.suptitle("SYNTHETIC teaching example: projection coincidence ≠ physical cell contact")
    fig.tight_layout();fig.savefig(output/"why_3d_matters.png",dpi=150,bbox_inches="tight");plt.close(fig)
    r["teaching_demo"]={"centroid_XY_um":0,"centroid_3D_um":18.,"explanation":"Objects coincide after projection but are separate along Z; not a single physical 2D section."}
    (output/"volume_qc_report.json").write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding="utf-8")
    return r


def build_parser():
    p=argparse.ArgumentParser(description="3D IMC axes and instance morphology audit; raw TIFF axes require explicit verification")
    p.add_argument("--demo",action="store_true",help="run artificial teaching example; NO clinical data")
    p.add_argument("--header-only",action="store_true",help="inspect TIFF metadata without loading the whole image")
    p.add_argument("--image",type=Path)
    p.add_argument("--image-axes",help="explicit raw image axis order, e.g. CZYX, ZCYX or ZYX")
    p.add_argument("--mask",type=Path,help="optional reference instance label TIFF/NPY")
    p.add_argument("--mask-axes",help="explicit raw instance mask order, normally ZYX")
    p.add_argument("--spacing",nargs=3,type=float,metavar=("Z_UM","Y_UM","X_UM"),
                   help="physical voxel Z,Y,X sizes; only use VERIFIED metadata, never defaults")
    p.add_argument("--channel-index",type=int,default=0)
    p.add_argument("--output",type=Path,default=Path("runs/volume_qc"))
    p.add_argument("--max-file-mb",type=float,default=1200.,help="file-size safety guard; header-only bypasses")
    return p


def main(argv=None):
    p=build_parser(); a=p.parse_args(argv)
    if a.demo:
        r=run_demo(a.output)
    elif a.header_only:
        if not a.image: p.error("--header-only requires --image")
        r=tiff_header(a.image)
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/"tiff_header.json").write_text(json.dumps(r,indent=2),encoding="utf-8")
    else:
        if not a.image: p.error("specify --image or --demo")
        if a.image.stat().st_size> a.max_file_mb*1024**2:
            p.error("Image exceeds safe size threshold; use --header-only first and prepare a crop/chunked dataset")
        if a.image.suffix.lower()==".npz":
            if a.mask: p.error("NPZ sample includes its own mask; do not specify --mask")
            image,mask,channels=load_sample(a.image)
            provenance="NPZ source; provenance unverified";axes="CZYX"
        else:
            if not a.image_axes:p.error("--image-axes required for raw TIFF/NPY")
            image=read_stack(a.image,a.image_axes)
            mask=None;channels=[];axes=a.image_axes
            if a.mask:
                if not a.mask_axes:p.error("--mask-axes is required when mask provided")
                m=read_stack(a.mask,a.mask_axes)
                if m.shape[0]!=1:p.error("mask must have exactly one channel")
                mask=m[0]
            provenance="Raw image supplied by user; specimen and annotation provenance not yet verified"
        r=run_audit(image,mask,a.spacing,a.channel_index,a.output,provenance,axes)
    print(json.dumps(r,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
