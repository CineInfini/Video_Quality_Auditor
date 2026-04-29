#!/usr/bin/env python3
"""v0.4.11 hero figure: per-dataset Ridge ρ comparison between
basic photometric features (v0.4.10, 8 features) and the advanced
pure-CV pack (v0.4.11, 48 features).

The figure shows that BRISQUE NSS + LBP + HOG + DCT push KoNViD-1k from
ρ=0.51 to ρ=0.67 (within reach of ML SOTA 0.83) and VideoFeedback
from 0.43 to 0.67. T2VQA-DB plateaus near 0.26 — the irreducible
ML-required regime.
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
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basic-results-json", required=True, type=Path,
                    help="signed_rank_results.json from v0.4.10")
    ap.add_argument("--advanced-results-json", required=True, type=Path,
                    help="advanced_features_analysis.json from v0.4.11")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    basic = json.loads(args.basic_results_json.read_text())
    adv = json.loads(args.advanced_results_json.read_text())

    # Map dataset names
    map_basic = {"konvid_1k": "konvid", "videofeedback": "videofeedback",
                 "t2vqa_db": "t2vqa"}
    datasets = ["konvid", "videofeedback", "t2vqa"]
    labels = ["KoNViD-1k\n(natural, n=392)",
              "VideoFeedback\n(AIGC small, n=48)",
              "T2VQA-DB\n(AIGC large, n=422)"]
    sota_published = {"konvid": 0.83, "videofeedback": 0.60, "t2vqa": 0.72}

    basic_rho = []
    adv_rho = []
    adv_std = []
    for d_adv in datasets:
        # basic from signed_rank ridge_cv_mean
        b_key = [k for k, v in map_basic.items() if v == d_adv][0]
        basic_rho.append(basic[b_key]["ridge_cv_mean"])
        adv_rho.append(adv["per_dataset_ridge"][d_adv]["rho_mean"])
        adv_std.append(adv["per_dataset_ridge"][d_adv]["rho_std"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(datasets))
    width = 0.30
    palette_basic = "#cccccc"
    palette_adv = "#3b7eb6"

    bars_basic = ax.bar(x - width / 2, basic_rho, width,
                        color=palette_basic, edgecolor="black", linewidth=0.6,
                        label="Basic 8 photometric features (v0.4.10)")
    bars_adv = ax.bar(x + width / 2, adv_rho, width,
                      yerr=adv_std, capsize=3,
                      color=palette_adv, edgecolor="black", linewidth=0.6,
                      label="Advanced 48 pure-CV features (v0.4.11)\n"
                            "BRISQUE NSS + LBP + HOG + DCT + multi-scale Sobel")

    # SOTA reference
    for i, d in enumerate(datasets):
        sota = sota_published[d]
        ax.hlines(sota, x[i] - width, x[i] + width,
                  colors="red", linestyles="--", linewidth=1.3, alpha=0.85)
        ax.text(x[i], sota + 0.025, f"ML SOTA ≈ {sota:.2f}",
                ha="center", fontsize=8, color="red", fontweight="bold")

    # Annotate values
    for b, v in zip(bars_basic, basic_rho):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for b, v in zip(bars_adv, adv_rho):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    # Delta annotations
    for i, (b_v, a_v) in enumerate(zip(basic_rho, adv_rho)):
        delta = a_v - b_v
        if abs(delta) > 0.05:
            color = "darkgreen" if delta > 0 else "darkred"
            ax.annotate(f"Δ {delta:+.2f}",
                        xy=(x[i], (a_v + b_v) / 2),
                        ha="center", color=color, fontsize=10,
                        fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman ρ vs human MOS (Ridge, 5-fold CV)")
    ax.set_title(
        "Pushing the pure-CV ceiling — v0.4.11 vs v0.4.10\n"
        "BRISQUE NSS pushes KoNViD-1k toward ML SOTA. T2VQA-DB stays bounded — "
        "ML features (CLIP/DINOv2) needed for prompt-aligned AIGC quality.",
        fontsize=10
    )
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig12_v0411_advanced_features.pdf")
    fig.savefig(args.out_dir / "fig12_v0411_advanced_features.png")
    plt.close(fig)
    print(f"  ✓ fig12_v0411_advanced_features.{{pdf,png}}")


if __name__ == "__main__":
    main()
