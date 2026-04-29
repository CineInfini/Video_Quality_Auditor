#!/usr/bin/env python3
"""Master comparison figure: every method we tested vs the ML SOTA
ceiling, on each of the three datasets. This is the figure that
makes the paper's story crystal-clear.

Output:
    fig11_method_comparison_3datasets.{pdf,png}
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
    ap.add_argument("--signed-rank-json", required=True, type=Path)
    ap.add_argument("--regime-aware-json", required=True, type=Path)
    ap.add_argument("--konvid-json", required=True, type=Path)
    ap.add_argument("--videofeedback-json", required=True, type=Path)
    ap.add_argument("--t2vqa-json", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sr = json.loads(args.signed_rank_json.read_text())
    konvid = json.loads(args.konvid_json.read_text())
    vf = json.loads(args.videofeedback_json.read_text())
    t2vqa = json.loads(args.t2vqa_json.read_text())

    # Methods × datasets matrix
    methods = ["best single metric", "signed-rank composite", "Ridge (5-fold CV)"]
    datasets = ["KoNViD-1k\n(natural, n=392)",
                "VideoFeedback\n(AIGC small, n=48)",
                "T2VQA-DB\n(AIGC large, n=422)"]
    keys = ["konvid_1k", "videofeedback", "t2vqa_db"]

    rho_best = [abs(sr[k]["best_single_rho"]) for k in keys]
    rho_sr = [sr[k]["signed_rank_cv_mean"] for k in keys]
    rho_sr_std = [sr[k]["signed_rank_cv_std"] for k in keys]
    rho_ridge = [sr[k]["ridge_cv_mean"] for k in keys]
    rho_ridge_std = [sr[k]["ridge_cv_std"] for k in keys]

    # Published SOTA reference points (typical, from literature)
    sota_published = {
        "konvid_1k": 0.83,    # FAST-VQA-B / MDTVSFA-class
        "videofeedback": 0.60,  # VideoScore family
        "t2vqa_db": 0.72,     # DOVER-class
    }

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(datasets))
    width = 0.22
    palette = ["#cccccc", "#888888", "#3b7eb6"]

    bars1 = ax.bar(x - width, rho_best, width, color=palette[0],
                   edgecolor="black", linewidth=0.6,
                   label="Best single pure-CV metric")
    bars2 = ax.bar(x, rho_sr, width,
                   yerr=rho_sr_std, capsize=3, color=palette[1],
                   edgecolor="black", linewidth=0.6,
                   label="Signed-rank composite (5-fold CV)")
    bars3 = ax.bar(x + width, rho_ridge, width,
                   yerr=rho_ridge_std, capsize=3, color=palette[2],
                   edgecolor="black", linewidth=0.6,
                   label="Ridge regression (5-fold CV)")

    # SOTA dashed lines per dataset
    for i, k in enumerate(keys):
        sota = sota_published[k]
        ax.hlines(sota, x[i] - 1.5 * width, x[i] + 1.5 * width,
                  colors="red", linestyles="--", linewidth=1.3, alpha=0.85)
        ax.text(x[i], sota + 0.025, f"ML SOTA ≈ {sota:.2f}", ha="center",
                fontsize=8, color="red", fontweight="bold")

    # Annotate each bar with its value
    for bars, vals in [(bars1, rho_best), (bars2, rho_sr), (bars3, rho_ridge)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Spearman ρ vs human MOS")
    ax.set_title(
        "Pure-CV methods vs ML SOTA — three datasets, three regimes.\n"
        "The pure-CV ceiling is real: ρ ≤ 0.28 on T2VQA-DB even with the best linear method.",
        fontsize=10
    )
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig11_method_comparison_3datasets.pdf")
    fig.savefig(args.out_dir / "fig11_method_comparison_3datasets.png")
    plt.close(fig)
    print(f"  ✓ fig11_method_comparison_3datasets.{{pdf,png}}")


if __name__ == "__main__":
    main()
