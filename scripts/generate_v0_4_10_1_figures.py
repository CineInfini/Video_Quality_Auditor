#!/usr/bin/env python3
"""Final master figures for v0.4.10.1 paper.

Figures:
  fig12_ceiling_progression — 4-bar progression on T2VQA-DB n=422
  fig13_feature_importance — RF importance + p-values for 16 features
  fig14_method_5way — Ridge / RF / GB / regime-aware / per-dataset RF, 3 ds
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def fig12_ceiling_progression(t2vqa_data, ext_results, out_dir: Path):
    """Stacked bars: best single → Ridge → Ridge ext → RF ext → ML SOTA."""
    # Pull T2VQA numbers
    t2vqa_corr = t2vqa_data["correlations_vs_mos"]
    best_single_ext = max(abs(c["spearman"]) for c in t2vqa_corr.values()
                           if c.get("spearman") is not None)
    # Per-dataset oracles
    oracles_ridge = {o["dataset"]: o for o in ext_results["per_dataset_oracle_ridge"]}
    oracles_rf = {o["dataset"]: o for o in ext_results["per_dataset_oracle_rf"]}

    # Build progression for T2VQA
    points = [
        ("best single\nmetric (8 feat)", 0.238, 0),  # from v0.4.10.0
        ("Ridge\n(8 features)", 0.278, 0.111),       # from v0.4.10.0
        ("best single\n(16 features)", best_single_ext, 0),
        (f"Ridge\n(16 features)", oracles_ridge["t2vqa"]["spearman_mean"],
         oracles_ridge["t2vqa"]["spearman_std"]),
        (f"Random Forest\n(16 features)", oracles_rf["t2vqa"]["spearman_mean"],
         oracles_rf["t2vqa"]["spearman_std"]),
        ("DOVER\n(ML, published)", 0.72, 0),
        ("FAST-VQA\n(ML, published)", 0.65, 0),
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(points))
    rhos = [p[1] for p in points]
    stds = [p[2] for p in points]
    colors = ["#cccccc"] * 5 + ["#ff6b6b"] * 2
    bars = ax.bar(x, rhos, yerr=stds, capsize=4, color=colors,
                   edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, rhos):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.018,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in points], fontsize=8.5)
    ax.set_ylabel("Spearman ρ vs MOS")
    ax.set_title("Pure-CV ceiling progression on T2VQA-DB (n=422)\n"
                 "From 0.24 (single metric, v0.4.10.0) to 0.46 (16 features + RF, v0.4.10.1) "
                 "— gap to ML SOTA cut from 3× to 1.6×", fontsize=10)
    ax.set_ylim(0, 0.85)
    fig.tight_layout()
    fig.savefig(out_dir / "fig12_ceiling_progression.pdf")
    fig.savefig(out_dir / "fig12_ceiling_progression.png")
    plt.close(fig)
    print(f"  ✓ fig12_ceiling_progression")


def fig13_feature_importance(ext_results, out_dir: Path):
    feats = ext_results["feature_importance_rf"]
    feats_sorted = sorted(feats, key=lambda x: x["importance"])
    names = [f["feature"] for f in feats_sorted]
    imps = [f["importance"] for f in feats_sorted]

    # Mark new (v0.4.10.1) features differently
    NEW = {"spatial_info", "temporal_info", "edge_density", "color_richness",
           "block_artifact", "noise_level", "hog_consistency", "face_count"}
    colors = ["#d8651e" if n in NEW else "#5e8b3a" for n in names]

    fig, ax = plt.subplots(figsize=(7, 6))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, imps, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title("Feature importance — RF on combined 862 videos.\n"
                 "Orange = new in v0.4.10.1; green = baseline.", fontsize=10)
    for b, v in zip(bars, imps):
        ax.text(v + 0.001, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig13_feature_importance.pdf")
    fig.savefig(out_dir / "fig13_feature_importance.png")
    plt.close(fig)
    print(f"  ✓ fig13_feature_importance")


def fig14_method_5way(ext_results, out_dir: Path):
    """5-method comparison on combined + per-dataset oracles."""
    methods_combined = {m["method"]: m for m in ext_results["combined_methods"]}
    if "regime_aware_ridge" in methods_combined:
        # already there
        pass
    oracles_ridge = {o["dataset"]: o for o in ext_results["per_dataset_oracle_ridge"]}
    oracles_rf = {o["dataset"]: o for o in ext_results["per_dataset_oracle_rf"]}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: combined dataset comparison
    ax = axes[0]
    method_names = ["Ridge (linear)", "Gradient Boosting", "Random Forest",
                    "regime_aware_ridge"]
    rhos = [methods_combined[m]["spearman_mean"] for m in method_names]
    stds = [methods_combined[m]["spearman_std"] for m in method_names]
    x = np.arange(len(method_names))
    palette = ["#cccccc", "#9e9e9e", "#3b7eb6", "#5e8b3a"]
    bars = ax.bar(x, rhos, yerr=stds, capsize=4, color=palette,
                   edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, rhos):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in method_names], rotation=0,
                       fontsize=8.5)
    ax.set_ylabel("Spearman ρ vs z-MOS")
    ax.set_title("Combined n=862 — 5-fold stratified CV\n"
                 "Random Forest beats linear Ridge by +0.077", fontsize=10)
    ax.set_ylim(0, 0.55)

    # Right: per-dataset oracle (Ridge vs RF)
    ax = axes[1]
    ds_list = ["konvid", "t2vqa", "videofeedback"]
    ds_n = {d: oracles_ridge[d]["n"] for d in ds_list if d in oracles_ridge}
    rd_rhos = [oracles_ridge[d]["spearman_mean"] for d in ds_list]
    rd_stds = [oracles_ridge[d]["spearman_std"] for d in ds_list]
    rf_rhos = [oracles_rf[d]["spearman_mean"] for d in ds_list]
    rf_stds = [oracles_rf[d]["spearman_std"] for d in ds_list]

    x = np.arange(len(ds_list))
    width = 0.36
    b1 = ax.bar(x - width/2, rd_rhos, width, yerr=rd_stds, capsize=3,
                 color="#cccccc", edgecolor="black", linewidth=0.6,
                 label="Ridge (linear)")
    b2 = ax.bar(x + width/2, rf_rhos, width, yerr=rf_stds, capsize=3,
                 color="#3b7eb6", edgecolor="black", linewidth=0.6,
                 label="Random Forest")
    for b, v in zip(b1, rd_rhos):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015,
                f"{v:.3f}", ha="center", fontsize=8)
    for b, v in zip(b2, rf_rhos):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015,
                f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={ds_n[d]})" for d in ds_list])
    ax.set_ylabel("Spearman ρ vs MOS")
    ax.set_title("Per-dataset oracle (16 features, 5-fold CV)\n"
                 "RF beats Ridge by +0.05 (KoNViD), +0.12 (T2VQA), -0.13 (VF, n too small)",
                 fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig14_method_5way.pdf")
    fig.savefig(out_dir / "fig14_method_5way.png")
    plt.close(fig)
    print(f"  ✓ fig14_method_5way")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t2vqa-extended-json", required=True, type=Path)
    ap.add_argument("--extended-regression-json", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t2vqa_data = json.loads(args.t2vqa_extended_json.read_text())
    ext_results = json.loads(args.extended_regression_json.read_text())

    fig12_ceiling_progression(t2vqa_data, ext_results, args.out_dir)
    fig13_feature_importance(ext_results, args.out_dir)
    fig14_method_5way(ext_results, args.out_dir)
    print(f"\nFigures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
