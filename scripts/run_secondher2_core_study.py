#!/usr/bin/env python3
"""SecondHer2 core paper experiment: spatially held-out kernel evaluation & figures.

Requires existing *frozen* 3D pseudo-instance tables. Does NOT alter source or
prior outputs. No HER2 signal at held-out target Z is used in predictors.

Scientific scope: within-one-specimen exploratory held-out Z-plane prediction.
Spatial x-stripe CV is NOT external biological validation or true cell GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, patches
from mpl_toolkits.mplot3d import Axes3D  # noqa F401

M = ("HER2_3B5", "HER2_29D8")
TAUS = (1.0, 2.0, 4.0, 8.0, float("inf"))  # fixed grid, no test-informed changes
SEED = 20261019
SOURCES = {
    "sections": "runs/secondher2_2d_vs_3d_heldout_v0_1/per_object_per_section_her2.csv",
    "heldout": "runs/secondher2_2d_vs_3d_heldout_v0_1/heldout_2d_vs_multiplane_predictions.csv",
    "coordinates": "runs/secondher2_3d_spatial_discordance_v0_1/3d_her2_object_spatial_coordinates.csv",
    "zlag": "runs/secondher2_z_lag_profile_v0_1/z_lag_error_summary.csv",
    "size": "runs/qc_secondher2_common_size_strata_v0_1/common_size_stratified_concordance.csv",
    "spatial_robustness": "runs/secondher2_3d_spatial_robustness_v0_1/spatial_robustness_summary.csv",
}


def truth(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def locate_inputs(root: Path) -> dict[str, Path]:
    paths = {name: root / rel for name, rel in SOURCES.items()}
    essential = ("sections", "heldout", "coordinates")
    missing = [str(paths[k]) for k in essential if not paths[k].is_file()]
    if missing:
        raise FileNotFoundError("Required previous outputs missing:\n" + "\n".join(missing))
    return paths


def kernel_predict(z: np.ndarray, n: np.ndarray, val: np.ndarray, target: int, tau: float) -> float:
    dz = np.abs(z - target) * 2.0
    if not np.isfinite(tau):
        w = n.astype(float)
    else:
        w = n * np.exp(-(dz - dz.min()) / tau)
    if not np.isfinite(w.sum()) or w.sum() <= 0:
        raise RuntimeError("Invalid distance weights")
    return float(np.dot(w, val) / w.sum())


def build_feature_table(sections: pd.DataFrame, old: pd.DataFrame, coord: pd.DataFrame) -> pd.DataFrame:
    req = {"source_pseudo_id", "z", "n_px", *[x + "_sum" for x in M]}
    if req - set(sections):
        raise ValueError(f"Per-Z missing: {req - set(sections)}")
    if sections.duplicated(["source_pseudo_id", "z"]).any():
        raise ValueError("Duplicate source-z rows")
    if (sections.n_px <= 0).any():
        raise ValueError("Non-positive n_px")
    req_old = {"source_pseudo_id", "target_z", "primary_eligible", "retained", *[x + "_heldout_mean" for x in M]}
    if req_old - set(old):
        raise ValueError(f"Held-out missing: {req_old - set(old)}")
    if old.source_pseudo_id.duplicated().any() or coord.source_pseudo_id.duplicated().any():
        raise ValueError("Duplicate source ID in previous tables")
    req_coord = {"source_pseudo_id", "x_um", "y_um", "z_um"}
    if req_coord - set(coord):
        raise ValueError("Missing centroid columns")
    prev = old.set_index("source_pseudo_id")
    rows = []
    for sid, g in sections.loc[sections.source_pseudo_id.isin(prev.index)].groupby("source_pseudo_id", sort=True):
        g = g.sort_values("z")
        zz = g.z.to_numpy(dtype=int)
        if len(zz) < 4 or not np.all(np.diff(zz) == 1):
            raise RuntimeError(f"Noncontiguous eligible pseudo-instance: {sid}")
        ref = prev.loc[sid]
        target = int(ref.target_z)
        if target not in zz or target != zz[len(zz) // 2]:
            raise RuntimeError(f"Mismatched frozen target-z at source ID={sid}")
        other = g.loc[g.z != target].copy()
        d = (other.z - target).abs().to_numpy()
        order = np.lexsort((other.z.to_numpy(), d))
        other = other.iloc[order]
        first_two = other.iloc[:2]
        furthest_two = other.iloc[-2:]
        row: dict[str, Any] = {
            "source_pseudo_id": int(sid), "n_sections": int(len(g)), "target_z": target,
            "primary_eligible": bool(truth(pd.Series([ref.primary_eligible])).iloc[0]),
            "retained": bool(truth(pd.Series([ref.retained])).iloc[0]),
            "nearest2_n_px": int(first_two.n_px.sum()),
            "farthest2_n_px": int(furthest_two.n_px.sum()),
            "all_others_n_px": int(other.n_px.sum()),
        }
        for marker in M:
            vc = marker + "_sum"
            obs = other[vc].to_numpy(dtype=float) / other.n_px.to_numpy(dtype=float)
            if not np.isfinite(obs).all() or (obs < 0).any():
                raise ValueError(f"Invalid per-plane intensities: {marker}")
            target_mean = float(ref[marker + "_heldout_mean"])
            target_check = float(g.loc[g.z == target, vc].iloc[0] / g.loc[g.z == target, "n_px"].iloc[0])
            if not np.isclose(target_mean, target_check, rtol=1e-8, atol=1e-8):
                raise RuntimeError("Held-out reference mismatch")
            n = other.n_px.to_numpy(float)
            z = other.z.to_numpy(int)
            pred1 = float(other.iloc[0][vc] / other.iloc[0].n_px)
            pred2 = float(first_two[vc].sum() / first_two.n_px.sum())
            predall = float(other[vc].sum() / other.n_px.sum())
            predfar = float(furthest_two[vc].sum() / furthest_two.n_px.sum())
            if not np.isclose(pred1, float(ref[marker + "_single_plane_mean"]), rtol=1e-8, atol=1e-8):
                raise RuntimeError("Nearest-1 frozen prediction mismatch")
            if not np.isclose(predall, float(ref[marker + "_other_planes_mean"]), rtol=1e-8, atol=1e-8):
                raise RuntimeError("All-others frozen prediction mismatch")
            row[marker + "_target_mean"] = target_mean
            row[marker + "_nearest2_mean"] = pred2
            row[marker + "_near2_variation"] = abs(np.log1p(obs[0]) - np.log1p(obs[1]))
            for name, pred in (("nearest1", pred1), ("nearest2", pred2), ("all_others", predall), ("farthest2", predfar)):
                row[marker + "_" + name + "_error"] = float(abs(np.log1p(pred) - np.log1p(target_mean)))
            for tau in TAUS:
                tag = "inf" if not np.isfinite(tau) else str(int(tau))
                val = kernel_predict(z, n, obs, target, tau)
                row[f"{marker}_tau{tag}_error"] = float(abs(np.log1p(val) - np.log1p(target_mean)))
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table) != len(old) or set(table.source_pseudo_id) != set(old.source_pseudo_id):
        raise RuntimeError("Mismatch in heldout object coverage")
    table = table.merge(coord[list(req_coord)], on="source_pseudo_id", validate="one_to_one", how="left")
    if table[["x_um", "y_um", "z_um"]].isna().any().any():
        raise RuntimeError("Missing centroids")
    return table


def assign_xfold(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cuts = np.quantile(x, [1/3, 2/3])
    if cuts[1] - cuts[0] < 15:
        raise ValueError("Insufficient X range for 3 spatial stripes")
    folds = np.searchsorted(cuts, x, side="left")
    return folds, cuts


def select_tau(train: pd.DataFrame, marker: str, bin_mask: np.ndarray | None = None) -> tuple[float, float]:
    sub = train if bin_mask is None else train.loc[bin_mask]
    if len(sub) < 30:
        raise ValueError("Too few training pseudo-instances for kernel selection")
    losses = []
    for tau in TAUS:
        tag = "inf" if np.isinf(tau) else str(int(tau))
        losses.append(float(sub[f"{marker}_tau{tag}_error"].mean()))
    idx = int(np.argmin(losses))
    return TAUS[idx], losses[idx]


def get_kernel_errors(df: pd.DataFrame, marker: str, tau: float) -> np.ndarray:
    tag = "inf" if np.isinf(tau) else str(int(tau))
    return df[f"{marker}_tau{tag}_error"].to_numpy(float)


def spatial_x_cv(table: pd.DataFrame, guard_um: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = table.loc[table.primary_eligible].copy()
    if len(primary) < 50:
        raise ValueError("Too few primary retained pseudo-instances")
    fold, cuts = assign_xfold(primary.x_um.to_numpy(float))
    primary["spatial_x_fold"] = fold
    results = []
    selections = []
    for fold_id in range(3):
        test = primary.loc[primary.spatial_x_fold == fold_id].copy()
        if fold_id == 0:
            train = primary.loc[primary.x_um > cuts[0] + guard_um]
        elif fold_id == 1:
            train = primary.loc[(primary.x_um < cuts[0] - guard_um) | (primary.x_um > cuts[1] + guard_um)]
        else:
            train = primary.loc[primary.x_um < cuts[1] - guard_um]
        if len(train) < 40 or len(test) < 15:
            raise ValueError(f"Spatial fold {fold_id}: ntrain={len(train)}, ntest={len(test)} insufficient")
        if set(train.source_pseudo_id) & set(test.source_pseudo_id):
            raise RuntimeError("Training/test ID leakage")
        min_dist = np.min(np.abs(train.x_um.to_numpy()[:, None] - test.x_um.to_numpy()[None, :]))
        if min_dist < guard_um - 1e-6:
            raise RuntimeError(f"Spatial guard violated in fold {fold_id}")
        for marker in M:
            fixed_tau, fixed_train_loss = select_tau(train, marker)
            threshold = float(train[marker + "_near2_variation"].median())
            lo = train[marker + "_near2_variation"].to_numpy(float) <= threshold
            high = ~lo
            lo_tau, lo_loss = select_tau(train, marker, lo)
            hi_tau, hi_loss = select_tau(train, marker, high)
            variations = test[marker + "_near2_variation"].to_numpy(float)
            adaptive = np.where(variations <= threshold,
                                get_kernel_errors(test, marker, lo_tau),
                                get_kernel_errors(test, marker, hi_tau))
            fixed = get_kernel_errors(test, marker, fixed_tau)
            for method, vals in {
                "nearest1": test[marker + "_nearest1_error"].to_numpy(float),
                "nearest2": test[marker + "_nearest2_error"].to_numpy(float),
                "farthest2": test[marker + "_farthest2_error"].to_numpy(float),
                "all_others": test[marker + "_all_others_error"].to_numpy(float),
                "distance_kernel": fixed,
                "variation_adaptive_kernel": adaptive,
            }.items():
                for sid, ee, xx, yy, zz in zip(test.source_pseudo_id, vals, test.x_um, test.y_um, test.z_um):
                    results.append({"source_pseudo_id": int(sid), "fold": int(fold_id), "marker": marker,
                                    "method": method, "abs_log_error": float(ee),
                                    "x_um": float(xx), "y_um": float(yy), "z_um": float(zz)})
            selections.append({
                "fold": int(fold_id), "marker": marker, "n_train": len(train), "n_test": len(test),
                "cut1_um": float(cuts[0]), "cut2_um": float(cuts[1]), "guard_um": guard_um,
                "small_variation_threshold_log1p": threshold,
                "fixed_tau_um": str(fixed_tau), "tau_small_variation_um": str(lo_tau),
                "tau_large_variation_um": str(hi_tau), "fixed_training_mae": fixed_train_loss,
                "small_variation_training_mae": lo_loss, "large_variation_training_mae": hi_loss,
                "train_test_min_x_distance_um": float(min_dist),
            })
    p = pd.DataFrame(results)
    n_primary = len(primary)
    expected = n_primary * len(M) * 6
    if len(p) != expected or p.duplicated(["source_pseudo_id", "marker", "method"]).any():
        raise RuntimeError(f"Missing or duplicate spatial holdout predictions: {len(p)} vs {expected}")
    return p, pd.DataFrame(selections)


def spatial_bootstrap(p: pd.DataFrame, n_boot: int, block_xy_um: float = 32., block_z_um: float = 24.) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, bootstraps = [], []
    rng = np.random.default_rng(SEED)
    for marker in M:
        d = p.loc[p.marker == marker].pivot(index="source_pseudo_id", columns="method", values="abs_log_error")
        coords = p.loc[(p.marker == marker) & (p.method == "nearest2"), ["source_pseudo_id", "x_um", "y_um", "z_um"]]
        coords = coords.set_index("source_pseudo_id")
        d = d.join(coords, how="left", validate="one_to_one")
        if d.isna().any().any():
            raise RuntimeError("Missing predictions/coords in bootstrap")
        block = (np.floor(d.x_um.to_numpy()/block_xy_um).astype(int).astype(str) + ":" +
                 np.floor(d.y_um.to_numpy()/block_xy_um).astype(int).astype(str) + ":" +
                 np.floor(d.z_um.to_numpy()/block_z_um).astype(int).astype(str))
        codes, uniq = pd.factorize(block, sort=True)
        group_indices = [np.flatnonzero(codes == i) for i in range(len(uniq))]
        if len(uniq) < 15:
            raise ValueError(f"Insufficient 3D spatial blocks: {len(uniq)}")
        bootstrap_selections = [np.concatenate([group_indices[i] for i in rng.integers(0, len(uniq), size=len(uniq))]) for _ in range(n_boot)]
        base = d["nearest2"].to_numpy(float)
        for method in ("nearest1", "nearest2", "farthest2", "all_others", "distance_kernel", "variation_adaptive_kernel"):
            val = d[method].to_numpy(float)
            diff = base - val  # positive means method improves over nearest2
            b = np.array([np.mean(diff[idx]) for idx in bootstrap_selections])
            summaries.append({"marker": marker, "method": method, "n_objects": len(d), "n_spatial_blocks": len(uniq),
                              "median_abs_log_error": float(np.median(val)),
                              "mean_abs_log_error": float(np.mean(val)),
                              "mean_paired_improvement_vs_nearest2": float(np.mean(diff)),
                              "bootstrap_lower_025": float(np.quantile(b, .025)),
                              "bootstrap_upper_975": float(np.quantile(b, .975)),
                              "fraction_better_than_nearest2": float(np.mean(val < base)),
                              "fraction_ties": float(np.mean(np.isclose(val, base, rtol=0, atol=1e-12)) )})
            bootstraps.extend(({"marker": marker, "method": method, "bootstrap_iteration": i, "mean_improvement": v}
                               for i, v in enumerate(b)))
    return pd.DataFrame(summaries), pd.DataFrame(bootstraps)


def set_fig_style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.spines.top":False,
                         "axes.spines.right":False, "figure.facecolor":"white",
                         "axes.labelsize":10, "axes.titlesize":12, "savefig.dpi":180})


def save(fig, out: Path, base: str):
    fig.savefig(out / f"{base}.png", dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{base}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure1(out: Path):
    fig, ax = plt.subplots(figsize=(11, 3.3))
    ax.set_xlim(0, 11);ax.set_ylim(0, 4);ax.axis("off")
    labels = [("92-section IMC", "Eu151 / Gd156"),
              ("3D pseudo-IDs", "author watershed mask"),
              ("Frozen HER2 by Z", "held-out middle Z"),
              ("X-stripe CV", "3 folds + 12 µm guard"),
              ("Paired benchmark", "Nearest-2 vs kernels")]
    xvals = [0.1, 2.35, 4.6, 6.85, 9.1]
    for i, ((top,bottom), x) in enumerate(zip(labels,xvals)):
        box = patches.FancyBboxPatch((x,1.4), 1.8, 1.35, boxstyle="round,pad=0.08,rounding_size=0.13",
                                    facecolor="#f1f5f9", edgecolor="#7890a3", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x+0.9,2.28,top,ha="center",va="center",fontsize=9.3,weight="bold")
        ax.text(x+0.9,1.77,bottom,ha="center",va="center",fontsize=8, color="#526477")
        if i<4:
            ax.annotate("",xy=(x+2.19,2.07),xytext=(x+1.88,2.07),arrowprops={"arrowstyle":"-|>","lw":1.5,"color":"#526477"})
    ax.text(5.5, 3.45,"Core study workflow (schematic; not source microscopy)",ha="center",fontsize=14,weight="bold")
    ax.text(5.5,.55,"Single specimen · pseudo-instance measurements · spatial holdout ≠ independent patient validation",ha="center",fontsize=10,color="#9a4b23")
    save(fig,out,"Figure1_workflow_schematic")


def figure2(root: Path, out: Path):
    p=root/SOURCES["size"]
    if not p.exists():
        return False
    d=pd.read_csv(p).sort_values("size_bin")
    fig, ax=plt.subplots(figsize=(8.5,4.3))
    x=d.size_bin.astype(int).to_numpy()
    ax.plot(x,d.rho_retained,"o-",label="PanCK-domain retained",color="#1a6f94",lw=2)
    ax.plot(x,d.rho_excluded,"s-",label="Excluded",color="#da8123",lw=2)
    ax.set(ylim=(0.25,.98),xticks=x,xlabel="Common object-size octile",ylabel="Spearman rho: HER2 clone means")
    ax.grid(axis="y",alpha=.18)
    ax.legend(frameon=False,loc="upper left")
    ax.set_title("Population selection shifts clone concordance across shared size bins")
    ax.text(.01,-.30,"Bins are descriptive; one specimen; within-bin correlations are not patient-level replicates.",
            transform=ax.transAxes,fontsize=9,color="#7c3f21")
    save(fig,out,"Figure2_population_size")
    return True


def figure3(root: Path,out: Path):
    p=root/SOURCES["coordinates"]
    df=pd.read_csv(p)
    req={"source_pseudo_id","x_um","y_um","z_um","group","rank_discordance"}
    if req-set(df):
        return False
    rng=np.random.default_rng(SEED)
    fig=plt.figure(figsize=(12,5.7))
    cmap=plt.get_cmap("coolwarm")
    for i, gname in enumerate(("retained", "excluded"),start=1):
        ax=fig.add_subplot(1,2,i,projection="3d")
        d=df.loc[df.group==gname]
        if len(d)>2400:
            d=d.iloc[np.sort(rng.choice(len(d),2400,replace=False))]
        sc=ax.scatter(d.x_um,d.y_um,d.z_um,c=d.rank_discordance,cmap=cmap,
                      vmin=-.65,vmax=.65,s=4,alpha=.7,rasterized=True)
        ax.set(xlabel="X (µm)",ylabel="Y (µm)",zlabel="Z (µm)")
        ax.set_title(f"{gname.capitalize()} pseudo-instances (subsampled)")
        ax.view_init(elev=19,azim=-60)
        ax.set_box_aspect((1,1,.72))
    fig.suptitle("3D HER2 signed within-group rank discordance",fontsize=14,weight="bold")
    fig.subplots_adjust(left=0, right=.90,wspace=.05)
    cax=fig.add_axes([.91,.24,.012,.50])
    cbar=fig.colorbar(sc,cax=cax)
    cbar.set_label("3B5 rank − 29D8 rank")
    save(fig,out,"Figure3_spatial_discordance_3d")
    return True


def figure4(stats: pd.DataFrame,out: Path):
    methods=("nearest1","nearest2","farthest2","all_others","distance_kernel","variation_adaptive_kernel")
    labels=["Nearest-1","Nearest-2","Farthest-2","All planes","Distance kernel","Variation-adaptive"]
    fig,axs=plt.subplots(1,2,figsize=(12.4,5.1),sharey=False)
    for ax,marker in zip(axs,M):
        g=stats.loc[stats.marker==marker].set_index("method").loc[list(methods)]
        yy=np.arange(len(methods))
        val=g.mean_paired_improvement_vs_nearest2.to_numpy()
        neg=val-g.bootstrap_lower_025.to_numpy()
        pos=g.bootstrap_upper_975.to_numpy()-val
        c=["#7a9aaa" if method not in ("distance_kernel","variation_adaptive_kernel") else "#276b93" for method in methods]
        ax.barh(yy,val,xerr=np.vstack([neg,pos]),color=c,alpha=.85,capsize=3)
        ax.axvline(0,c="black",lw=1,alpha=.6)
        ax.set_yticks(yy,labels)
        ax.invert_yaxis()
        ax.set_xlabel("Mean paired log-error improvement vs Nearest-2")
        ax.set_title(marker + f"  (n={int(g.n_objects.iloc[0])})")
        ax.grid(axis="x",alpha=.14)
    fig.suptitle("Spatially held-out test folds — paired block-bootstrap intervals",fontsize=14,weight="bold")
    fig.text(.02,.005,"Positive improvement favors the indicated method. Intervals quantify within-specimen resampling, not patient-generalization.",fontsize=9,color="#88512e")
    fig.tight_layout(rect=(0,.04,1,.92))
    save(fig,out,"Figure4_spatial_holdout_benchmark")


def figure5(root: Path,out:Path):
    p=root/SOURCES["zlag"]
    if not p.exists():
        return False
    d=pd.read_csv(p)
    d=d[(d.group=="primary_retained_clean") & (d.z_lag<=4)].copy()
    if d.empty:
        return False
    fig,ax=plt.subplots(figsize=(8.5,4.5))
    for marker,color,style in (("HER2_3B5","#1a6f94","o-"),("HER2_29D8","#da8123","s-")):
        sub=d[d.marker==marker].sort_values("physical_lag_um")
        ax.plot(sub.physical_lag_um,sub.median_single_plane_error,style,c=color,lw=2,label=marker)
        for row in sub.itertuples():
            ax.annotate(f"n={int(row.n_objects)}",(row.physical_lag_um,row.median_single_plane_error),
                        textcoords="offset points",xytext=(0,9),ha="center",fontsize=7.5,color=color)
    ax.set(xlabel="Axial lag (µm)",ylabel="Median held-out single-plane absolute log-error",
           title="Z-distance effect (cohort shrinks as lag increases)")
    ax.legend(frameon=False)
    ax.grid(alpha=.18)
    fig.tight_layout()
    save(fig,out,"Figure5_z_lag_profile")
    return True


def brief(stats:pd.DataFrame, choices:pd.DataFrame, table:pd.DataFrame, path:Path, source_hashes:dict):
    lines=["# SecondHer2 core-study readout", "", "## Experimental contract",
           "- One author-produced 3D pseudo-instance mask; same preselected held-out middle Z target used for all predictors.",
           "- 3 spatial X-stripe test folds with a guard between training and test objects; only training objects choose the kernel bandwidth and signal-variation split.",
           "- Baselines: nearest-1, nearest-2, farthest-2 (same plane count), all non-target Z planes (different pixel budget).",
           "- Predictors use ONLY non-target Z signal; within-object source-plane variation uses the nearest two *source* planes.",
           "- Outcome: absolute difference of log1p of raw mean intensity. Only within-volume evidence, not independently validated biology.","",
           "## Main results (positive mean paired improvement is better than nearest-2)",""]
    for marker in M:
        lines.append("### "+marker)
        lines.append("")
        for r in stats[stats.marker==marker].itertuples():
            lines.append(f"- **{r.method}**: mean error {r.mean_abs_log_error:.5f}; mean paired improvement {r.mean_paired_improvement_vs_nearest2:+.5f}; block-bootstrap interval [{r.bootstrap_lower_025:+.5f}, {r.bootstrap_upper_975:+.5f}].")
        cand=stats[(stats.marker==marker)&(stats.method=="variation_adaptive_kernel")].iloc[0]
        if cand.bootstrap_lower_025>0:
            verdict="An exploratory held-out spatial improvement above Nearest-2 was observed. Independent specimens remain essential."
        elif cand.bootstrap_upper_975<0:
            verdict="The adaptive candidate performs worse than Nearest-2 on this held-out task. Do not claim 3D prediction gain."
        else:
            verdict="No clear improvement over Nearest-2 is resolved by within-specimen block bootstrap."
        lines += ["", "**Interpretation:** "+verdict,"",]
    lines += ["## Leakage / bias safeguards",
              f"- Total eligible held-out pseudo-instances: {len(table)}; primary test instances: {int(table.primary_eligible.sum())}.",
              "- Kernel hyperparameters are tuned using *training pseudo-instance targets only*, never held-out test-object targets.",
              "- Spatial separation uses x-only partitioning + guard. Spatial effects and specimen identity can still confound results.",
              "- Farthest-2 fixes number of planes, but pixel counts/quality still vary; true budget-matched voxel resampling is NOT implemented.",
              "- No manual boundary ground truth; no independent specimen. Bootstrap confidence bands are descriptive.",
              "- Current ROI, segmentation and target selection were explored repeatedly; this is not pristine pre-registered test data.",
              "- Larger cohorts and raw-mask-generation provenance are needed before strong novelty/biology claims.",
              "", "## Source checksums", ""]
    for p,h in source_hashes.items():
        lines.append(f"- `{p}`: `{h}`")
    path.write_text("\n".join(lines)+"\n",encoding="utf8")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root",type=Path,default=Path("."))
    ap.add_argument("--out",type=Path,default=Path("runs/secondher2_core_study_v0_1"))
    ap.add_argument("--guard-um",type=float,default=12.0)
    ap.add_argument("--bootstrap",type=int,default=1000)
    args=ap.parse_args()
    if args.guard_um<0 or args.bootstrap<100:
        ap.error("--guard-um must be nonnegative and --bootstrap >=100")
    root=args.project_root.expanduser().resolve()
    inputs=locate_inputs(root)
    out=(root/args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    dat={k:pd.read_csv(v) for k,v in inputs.items() if k in ("sections","heldout","coordinates")}
    print("[1/5] Reconcile frozen per-Z HER2, heldout targets, and centroids",flush=True)
    table=build_feature_table(dat["sections"],dat["heldout"],dat["coordinates"])
    print("[2/5] Fit bandwidths in training stripes; evaluate spatially-held-out stripes",flush=True)
    prediction,selections=spatial_x_cv(table,args.guard_um)
    print("[3/5] Compute paired spatial-block uncertainty",flush=True)
    stats,boots=spatial_bootstrap(prediction,args.bootstrap)
    out.mkdir(parents=True,exist_ok=False)
    figs=out/"figures";figs.mkdir()
    table.to_csv(out/"object_source_features.csv",index=False)
    prediction.to_csv(out/"spatial_fold_heldout_predictions.csv",index=False)
    selections.to_csv(out/"training_only_bandwidth_selections.csv",index=False)
    stats.to_csv(out/"core_comparison_summary.csv",index=False)
    boots.to_csv(out/"spatial_bootstrap_samples.csv",index=False)
    set_fig_style()
    print("[4/5] Render study figures",flush=True)
    figure1(figs)
    plotted={"Figure2_population_size":figure2(root,figs),
             "Figure3_spatial_discordance_3d":figure3(root,figs),
             "Figure5_z_lag_profile":figure5(root,figs)}
    figure4(stats,figs)
    paths={str(v.relative_to(root)):sha256(v) for v in inputs.values() if v.is_file()}
    protocol={"scope":"single-volume; spatially held-out stripe prediction only",
              "markers":list(M),"seed":SEED,"n_spatial_folds":3,"guard_um":args.guard_um,
              "kernel_taus_um":[str(x) for x in TAUS],
              "kernel_weights":"n_px * exp(-(abs(z-target_z)*2um - nearest_source_distance_um)/tau_um)",
              "adaptive_feature":"abs(log1p(mean closest non-target section) - log1p(mean second-closest non-target section))",
              "adaptive_cut":"training-population median per marker, per fold",
              "target":"fixed heldout middle plane",
              "outcome":"absolute log1p raw mean intensity error",
              "cv":"3 quantile-based disjoint X stripes; train excludes 12um buffer nearest test stripe",
              "bootstrap":"resample 32x32x24 um centroid blocks, whole pooled spatial-test predictions",
              "citations":"none; measures are descriptive within one specimen",
              "figure_panels_present":plotted,"sources_sha256":paths}
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding="utf8")
    brief(stats,selections,table,out/"RESULTS_README.md",paths)
    print("[5/5] Completed. Overview:",flush=True)
    print("\n=== CORE SPATIAL HOLDOUT BENCHMARK ===")
    cols=["marker","method","n_objects","median_abs_log_error","mean_paired_improvement_vs_nearest2","bootstrap_lower_025","bootstrap_upper_975"]
    print(stats[cols].round(5).to_string(index=False))
    print("\n=== TRAIN-ONLY BANDWIDTH CHOICES ===")
    print(selections[["fold","marker","n_train","n_test","fixed_tau_um","tau_small_variation_um","tau_large_variation_um","train_test_min_x_distance_um"]].to_string(index=False))
    print("\nSUCCESS:",out)

if __name__=="__main__":
    main()
