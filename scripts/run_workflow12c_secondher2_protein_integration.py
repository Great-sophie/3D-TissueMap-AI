from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import tifffile

base = Path(
    "data/inspect/SecondHer2BreastCancerModel/"
    "SecondHer2BreastCancerModel"
)

ome = base / "aligned_SecondHer2BreastCancer_model_3D_image_stack.ome.tiff"
mask = base / "measured_mask_final_segmentation_hwatershed_bg500_90%.tif"

out = Path("runs/workflow12_secondher2_protein_integration_v0_1")

if out.exists():
    raise SystemExit(f"Output exists; refusing to overwrite: {out}")

markers = {
    "Ir193": "Ir193",
    "panCK": "Lu175",
    "Cadherin": "Yb173",
    "HER2_3B5": "Eu151",
    "HER2_29D8": "Gd156",
    "CD44": "Gd160",
}

tables = []

with tifffile.TiffFile(ome) as tf, tifffile.TiffFile(mask) as mf:
    series = tf.series[0]

    assert series.axes == "ZCYX"
    assert series.shape == (92, 31, 686, 608)
    assert len(series.pages) == 92 * 31
    assert len(mf.pages) == 92

    root = ET.fromstring(tf.ome_metadata)
    pixels = next(
        e for e in root.iter()
        if e.tag.endswith("Pixels")
    )

    names = [
        e.attrib["Name"]
        for e in pixels
        if e.tag.endswith("Channel")
    ]

    assert len(names) == 31
    indices = {key: names.index(tag) for key, tag in markers.items()}

    print("CHANNEL MAPPING:", indices, flush=True)

    for z in range(92):
        m = mf.pages[z].asarray()

        assert m.shape == (686, 608)

        ids, inverse, counts = np.unique(
            m.ravel(),
            return_inverse=True,
            return_counts=True
        )

        valid = ids != 0

        data = {
            "z": np.full(valid.sum(), z, dtype=np.int32),
            "instance_id": ids[valid],
            "voxel_count": counts[valid],
        }

        for marker, ci in indices.items():
            image = series.pages[z * 31 + ci].asarray()

            assert image.shape == m.shape

            sums = np.bincount(
                inverse,
                weights=image.ravel().astype(np.float64),
                minlength=len(ids)
            )

            data[f"sum_raw_{marker}"] = sums[valid]

        tables.append(pd.DataFrame(data))

        if z % 10 == 0 or z == 91:
            print(f"Processed Z: {z}", flush=True)

per_z = pd.concat(tables, ignore_index=True)

assert not per_z.duplicated(["z", "instance_id"]).any()

result = (
    per_z.drop(columns="z")
    .groupby("instance_id", as_index=False)
    .sum()
)

n_planes = (
    per_z.groupby("instance_id")
    .size()
    .rename("n_z_planes")
    .reset_index()
)

result = result.merge(
    n_planes, on="instance_id", validate="one_to_one"
)

for marker in markers:
    result[f"mean_raw_{marker}"] = (
        result[f"sum_raw_{marker}"] / result["voxel_count"]
    )

result = result.drop(
    columns=[f"sum_raw_{marker}" for marker in markers]
).sort_values("instance_id")

# Cross-check with Workflow 12B.
ref = pd.read_csv(
    "runs/workflow12_secondher2_3d_morphometry_v0_1/"
    "secondher2_measured_instance_3d_summary.csv"
)

check = result.merge(
    ref[["instance_id", "voxel_count", "n_z_planes"]],
    on="instance_id",
    suffixes=("_new", "_ref"),
    validate="one_to_one"
)

assert len(result) == len(check) == 81226
assert int(result.voxel_count.sum()) == 14687958
assert (check.voxel_count_new == check.voxel_count_ref).all()
assert (check.n_z_planes_new == check.n_z_planes_ref).all()

signal_cols = [f"mean_raw_{k}" for k in markers]

assert np.isfinite(result[signal_cols].to_numpy()).all()

# Only create output after checks pass.
out.mkdir(parents=True)

result.to_csv(out / "instance_marker_mean_raw.csv", index=False)
per_z.to_csv(out / "instance_marker_per_z_sums.csv", index=False)

summary = {
    "dataset": "SecondHer2BreastCancerModel",
    "n_z": 92,
    "n_instances": int(len(result)),
    "foreground_voxels": int(result.voxel_count.sum()),
    "channels": indices,
    "mask": "author-provided measured pseudo-instance mask",
    "measurement": "uncalibrated raw signal mean within pseudo-instance",
    "morphometry_crosscheck": "PASS",
}

(out / "integration_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n===== PROTEIN INTEGRATION =====")
print("Instances:", len(result))
print("Foreground voxels:", int(result.voxel_count.sum()))
print("Channels:", len(markers))
print("Morphometry cross-check: PASS")

print("\n===== MARKER SUMMARY =====")
print(
    result[signal_cols].describe()
    .loc[["mean", "50%", "std", "min", "max"]]
    .to_string()
)

print("\nSAVED:", out)
print("AUDIT COMPLETE")
