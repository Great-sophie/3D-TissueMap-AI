#!/usr/bin/env python3
"""SecondHer2: exploratory spatial-discordance robustness checks.

Reads the frozen 3D object spatial table (no TIFF reread or mask edits).
Tests the *signed within-group percentile-rank difference* between HER2 clones,
not the absolute discordance. Moran's I uses a row-standardized directed kNN graph.
Permutation tests are descriptive within one tissue volume, not biological replication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DEFAULT_INPUT = Path(
    "runs/secondher2_3d_spatial_discordance_v0_1/"
    "3d_her2_object_spatial_coordinates.csv"
)
DEFAULT_OUT = Path("runs/secondher2_3d_spatial_robustness_v0_1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def moran(v: np.ndarray, near: np.ndarray) -> float:
    centered = v - v.mean()
    denom = np.dot(centered, centered)
    if denom <= 0:
        raise ValueError("Rank difference has no variance")
    return float(np.dot(centered, centered[near].mean(axis=1)) / denom)


def permutation_indices(strata: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(strata == value) for value in np.unique(strata)]


def sample_perm(v: np.ndarray, rng: np.random.Generator,
                indices: list[np.ndarray] | None) -> np.ndarray:
    if indices is None:
        return rng.permutation(v)
    shuffled = v.copy()
    for idx in indices:
        if len(idx) > 1:
            shuffled[idx] = v[rng.permutation(idx)]
    return shuffled


def perm_test(v: np.ndarray, near: np.ndarray, rng: np.random.Generator,
              permutations: int, indices: list[np.ndarray] | None):
    obs = moran(v, near)
    null = np.empty(permutations, dtype=float)
    for b in range(permutations):
        null[b] = moran(sample_perm(v, rng, indices), near)
    midpoint = float(null.mean())
    p_centered_two = (1 + np.count_nonzero(
        np.abs(null - midpoint) >= abs(obs - midpoint)
    )) / (permutations + 1)
    p_positive = (1 + np.count_nonzero(null >= obs)) / (permutations + 1)
    return dict(
        observed_I=obs,
        null_mean=midpoint,
        null_p025=float(np.quantile(null, .025)),
        null_p975=float(np.quantile(null, .975)),
        p_two_sided_descriptive=float(p_centered_two),
        p_positive_descriptive=float(p_positive),
    ), null


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--z-band-um", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    if args.permutations < 99:
        parser.error("--permutations must be at least 99")
    if args.z_band_um <= 0:
        parser.error("--z-band-um must be positive")
    if not args.input.is_file():
        parser.error(f"Missing input file: {args.input}")
    if args.out.exists():
        parser.error(f"Refusing to overwrite existing output: {args.out}")

    df = pd.read_csv(args.input)
    required = {
        "source_pseudo_id", "group", "x_um", "y_um", "z_um", "n_voxels",
        "HER2_3B5_mean", "HER2_29D8_mean", "rank_discordance"
    }
    if not required.issubset(df.columns):
        parser.error(f"Missing columns: {sorted(required - set(df.columns))}")
    if not df.source_pseudo_id.is_unique:
        raise ValueError("source_pseudo_id is not unique")
    if set(df.group) != {"retained", "excluded"}:
        raise ValueError(f"Unexpected groups: {set(df.group)}")
    for key in ["x_um", "y_um", "z_um", "n_voxels", "rank_discordance",
                "HER2_3B5_mean", "HER2_29D8_mean"]:
        if not np.isfinite(df[key]).all():
            raise ValueError(f"Missing/nonfinite values: {key}")
    if (df.n_voxels <= 0).any():
        raise ValueError("Nonpositive voxel count")

    # Recompute the original signed within-group ranks to guard against stale tables.
    recomputed = (
        df.groupby("group")["HER2_3B5_mean"].rank(pct=True)
        - df.groupby("group")["HER2_29D8_mean"].rank(pct=True)
    )
    if not np.allclose(recomputed, df.rank_discordance, atol=1e-12):
        raise ValueError("rank_discordance does not match signed within-group percentile ranks")

    # Common size bins across the complete volume, not separate bins by group.
    df["size_q_common"] = pd.qcut(
        df.n_voxels.rank(method="first"), 4, labels=False
    ).astype(int)
    df["z_bin"] = np.floor(df.z_um / args.z_band_um).astype(int)

    rows = []
    null_rows = []
    for j, (group, g) in enumerate(df.groupby("group", sort=True)):
        g = g.reset_index(drop=True)
        pts = g[["x_um", "y_um", "z_um"]].to_numpy(float)
        v = g.rank_discordance.to_numpy(float)
        k_values = (6, 12, 24)
        if len(g) < max(k_values) + 2:
            raise ValueError(f"Too few objects in group {group}")
        _, ind = cKDTree(pts).query(pts, k=max(k_values) + 2)
        neighbors = np.vstack([ids[ids != i][:max(k_values)]
                               for i, ids in enumerate(ind)])
        if neighbors.shape != (len(g), max(k_values)):
            raise ValueError("Insufficient distinct kNN neighbors")

        bin_z = g.z_bin.to_numpy(dtype=int)
        # Integer combined stratum for common size quartile and Z band.
        bin_joint = bin_z * 4 + g.size_q_common.to_numpy(dtype=int)
        strata_specs = {
            "global": None,
            "within_z_band": permutation_indices(bin_z),
            "within_z_band_and_size": permutation_indices(bin_joint),
        }
        print(f"\n=== {group.upper()} (n={len(g)}) ===", flush=True)
        for k in k_values:
            near = neighbors[:, :k]
            obs = moran(v, near)
            print(f"Observed signed-rank Moran I, k={k}: {obs:.6f}", flush=True)
            if k != 12:
                rows.append(dict(group=group, k=k, null_type="not_tested",
                                 observed_I=obs, n_objects=len(g)))
                continue
            for code, (null_type, strata) in enumerate(strata_specs.items()):
                # RNG streams deterministic and independent across scenarios.
                rng = np.random.default_rng(args.seed + 1000*j + code)
                stats, null = perm_test(v, near, rng, args.permutations, strata)
                eligible_fraction = 1.0 if strata is None else float(
                    sum(len(idx) for idx in strata if len(idx) > 1) / len(g)
                )
                row = dict(group=group, k=k, null_type=null_type,
                           n_objects=len(g), n_strata=1 if strata is None else len(strata),
                           permutable_fraction=eligible_fraction, **stats)
                rows.append(row)
                null_rows.extend(dict(group=group, null_type=null_type,
                                      perm_idx=b+1, moran_I=float(value))
                                 for b, value in enumerate(null))
                print(
                    f"  {null_type}: I={stats['observed_I']:.4f}; "
                    f"null mean={stats['null_mean']:.4f}; "
                    f"two-sided p={stats['p_two_sided_descriptive']:.4f}; "
                    f"permutable={eligible_fraction:.3f}",
                    flush=True
                )

    args.out.mkdir(parents=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "spatial_robustness_summary.csv", index=False)
    pd.DataFrame(null_rows).to_csv(args.out / "permutation_nulls_k12.csv", index=False)
    metadata = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "permutations": args.permutations,
        "z_band_um": args.z_band_um,
        "seed": args.seed,
        "k_values": [6,12,24],
        "permutation_test_k": 12,
        "metric": "signed within-group percentile rank(HER2_3B5)-rank(HER2_29D8)",
        "graph": "directed kNN, row-standardized; physical XYZ from OME nominal spacing",
        "interpretation": "Exploratory spatial structuring within one specimen, not biological replication or validated cell identity",
    }
    (args.out / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print("\n=== SPATIAL ROBUSTNESS SUMMARY ===")
    print(summary.round(4).to_string(index=False))
    print("\nSUCCESS:", args.out)


if __name__ == "__main__":
    main()
