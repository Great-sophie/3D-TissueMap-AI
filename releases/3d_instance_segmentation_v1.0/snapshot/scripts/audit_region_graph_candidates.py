#!/usr/bin/env python3
"""Audit STATIC RAG edge filtering on a reused VAL crop (diagnostics only).

No training or test split access. The script uses existing, fixed XYZ affinity
weights and the exact interface construction in evaluate_affinity_region_graph.
It is a structural audit, not threshold selection or patient-level validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_affinity_region_graph import (
    block_supervoxels,
    interface_edges,
    merge_regions,
)
from train_segmentation import read_data
from train_voxel_affinity import make_network, predict_prob


def connectivity_after_edges(n_nodes: int, edge_pairs: np.ndarray) -> tuple[int, int]:
    """Connected component lower bound when every supplied edge may merge.

    All n_nodes have IDs 1..n_nodes. Ignores scores and volume caps; this is
    *not* the number of true cells or a bound on instance PQ.
    """
    parent = np.arange(n_nodes + 1, dtype=np.int32)
    size = np.ones(n_nodes + 1, dtype=np.int32)

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = int(parent[u])
        return u

    unions = 0
    for a, b in edge_pairs:
        x, y = find(int(a)), find(int(b))
        if x == y:
            continue
        if size[x] < size[y]:
            x, y = y, x
        parent[y] = x
        size[x] += size[y]
        unions += 1
    return n_nodes - unions, unions


def greedy_rejection_stats(sv: np.ndarray, candidate: np.ndarray,
                           max_component_voxels: int) -> dict:
    """Replay current greedy merge order and identify why candidates fail."""
    n = int(sv.max())
    counts = np.bincount(sv.ravel(), minlength=n + 1).astype(np.int64)
    parents = np.arange(n + 1, dtype=np.int32)

    def find(u: int) -> int:
        while parents[u] != u:
            parents[u] = parents[parents[u]]
            u = int(parents[u])
        return u

    counts_out = {"accepted": 0, "already_connected": 0,
                  "blocked_by_volume_cap": 0}
    for row in candidate[np.argsort(-candidate[:, 2])]:
        a, b = find(int(row[0])), find(int(row[1]))
        if a == b:
            counts_out["already_connected"] += 1
            continue
        if counts[a] + counts[b] > max_component_voxels:
            counts_out["blocked_by_volume_cap"] += 1
            continue
        if counts[a] < counts[b]:
            a, b = b, a
        parents[b] = a
        counts[a] += counts[b]
        counts_out["accepted"] += 1
    counts_out["final_component_count"] = n - counts_out["accepted"]
    return counts_out


def audit_interface_filters(sv: np.ndarray, interfaces: np.ndarray,
                            score_threshold: float, max_negative_fraction: float,
                            min_faces: int, max_component_voxels: int) -> dict:
    n = int(sv.max())
    score_ok = interfaces[:, 2] >= score_threshold
    neg_ok = interfaces[:, 3] <= max_negative_fraction
    faces_ok = interfaces[:, 4] >= min_faces
    all_ok = score_ok & neg_ok & faces_ok
    masks = {
        "ALL_REGION_INTERFACES": np.ones(len(interfaces), bool),
        "SCORE_ONLY": score_ok,
        "NEGATIVE_SHARE_ONLY": neg_ok,
        "MIN_FACES_ONLY": faces_ok,
        "SCORE_AND_NEGATIVE_SHARE": score_ok & neg_ok,
        "ALL_THREE_FILTERS": all_ok,
    }
    rows = []
    for name, eligible in masks.items():
        count_components, union_count = connectivity_after_edges(
            n, interfaces[eligible, :2]
        )
        rows.append({
            "filter": name,
            "retained_interfaces": int(eligible.sum()),
            "removed_interfaces": int(len(interfaces) - eligible.sum()),
            "retained_fraction": float(eligible.mean()) if len(eligible) else None,
            "components_if_all_retained_edges_merge_without_size_cap": int(count_components),
            "merges_possible_without_size_cap": int(union_count),
        })

    candidates = interfaces[all_ok]
    replay = greedy_rejection_stats(sv, candidates, max_component_voxels)
    _, official_merges = merge_regions(
        sv, interfaces, score_threshold, max_negative_fraction,
        max_component_voxels, min_faces
    )
    if replay["accepted"] != official_merges:
        raise AssertionError("Replay differs from current implementation")
    return {
        "n_supervoxels": n,
        "n_interfaces_total": int(len(interfaces)),
        "n_candidates": int(all_ok.sum()),
        "separate_fail_counts": {
            "score": int((~score_ok).sum()),
            "negative_share": int((~neg_ok).sum()),
            "min_faces": int((~faces_ok).sum()),
            "note": "Reasons overlap: separate fail counts must not be added",
        },
        "filter_table": rows,
        "current_greedy_rejections": replay,
        "params": {
            "score_threshold_logit": score_threshold,
            "max_negative_fraction": max_negative_fraction,
            "min_faces": min_faces,
            "max_component_voxels": max_component_voxels,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-run-dir", type=Path, required=True)
    parser.add_argument("--affinity-run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--region-z", type=int, default=1)
    parser.add_argument("--region-xy", type=int, default=4)
    parser.add_argument("--max-tiles", type=int, default=1)
    parser.add_argument("--guard", type=int, default=12)
    parser.add_argument("--z-weight", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--max-negative-fraction", type=float, default=0.35)
    parser.add_argument("--min-faces", type=int, default=2)
    parser.add_argument("--max-component-voxels", type=int, default=1200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--accept-provisional-registration", action="store_true")
    args = parser.parse_args()

    audit_path = args.split_run_dir / "data_audit.json"
    previous_audit = json.loads(audit_path.read_text())
    if previous_audit.get("stack_order_check", {}).get("status") != "FILENAME_ORDER_MATCH":
        raise RuntimeError("Stack-order filename audit must pass first")
    if not args.accept_provisional_registration:
        raise RuntimeError("Unverified registration: acknowledge with --accept-provisional-registration")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    img, ref, _, _ = read_data(args.data_dir)
    split = pd.read_csv(args.split_run_dir / "tile_splits.csv")
    split = split[split.split == "val"].head(args.max_tiles)
    if split.empty:
        raise ValueError("No validation tiles")
    ck = torch.load(args.affinity_run_dir / "xyz_downweighted.pt", map_location="cpu", weights_only=True)
    if ck.get("z_weight", 0) <= 0:
        raise ValueError("XYZ-supervised weights are required")
    model = make_network(int(ck["base"]), args.split_run_dir / "unet_3d.pt")
    model.load_state_dict(ck["state_dict"], strict=True)
    model = model.to(device).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    out = []
    for t in split.itertuples(index=False):
        y, x, w = int(t.y), int(t.x), int(t.w)
        g = args.guard
        if w <= 2 * g:
            raise ValueError("Guard exceeds ROI width")
        with torch.inference_mode():
            p = predict_prob(model, img[:, y:y+w, x:x+w], device)
        p = p[:, :, g:w-g, g:w-g]
        ref_crop = ref[:, y+g:y+w-g, x+g:x+w-g]
        sv = block_supervoxels(p[0] >= 0.5,
                              (args.region_z, args.region_xy, args.region_xy))
        edges = interface_edges(sv, p[1:], args.z_weight)
        item = audit_interface_filters(sv, edges, args.score_threshold,
                                       args.max_negative_fraction, args.min_faces,
                                       args.max_component_voxels)
        item.update({"tile_y": y, "tile_x": x, "reference_ids_in_crop": int(np.unique(ref_crop[ref_crop > 0]).size)})
        out.append(item)
        print(f"VAL ({y},{x}): {item['n_supervoxels']} supervoxels; "
              f"{item['n_interfaces_total']} total interfaces; "
              f"{item['n_candidates']} candidates")
        print(pd.DataFrame(item["filter_table"]).to_string(index=False))
        print("Actual merge rejection counts:", item["current_greedy_rejections"], flush=True)
    payload = {
        "scope": "Reused VALIDATION only, ONE specimen; no TEST access",
        "purpose": "Filter attrition and graph-connectivity audit, NOT method ranking",
        "important": [
            "Connected component floor assumes all retained edges can merge and ignores volume cap; it is not PQ or true-cell count.",
            "Full-RAG connectivity is diagnostic, not a license to merge all edges; false bridges can merge distinct cells.",
            "Reference is author-generated watershed, not manually verified cell truth.",
            "Image/mask registration and physical voxel spacing remain unverified.",
        ],
        "tiles": out,
    }
    (args.out / "region_candidate_audit.json").write_text(json.dumps(payload, indent=2))
    frame = []
    for item in out:
        frame.extend({"tile_y": item["tile_y"], "tile_x": item["tile_x"], **r}
                     for r in item["filter_table"])
    pd.DataFrame(frame).to_csv(args.out / "region_candidate_filter_table.csv", index=False)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
