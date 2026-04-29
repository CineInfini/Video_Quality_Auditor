#!/usr/bin/env python3
"""Generate publication-quality figures from REAL CineInfini results.

Usage:
    python scripts/generate_figures.py \\
        --results-json out/videofeedback_real_results.json \\
        --out-dir docs/figures/

Outputs (PDF + PNG, 300 dpi, vector-friendly):
    fig1_mos_distribution.{pdf,png}
    fig2_metric_vs_mos.{pdf,png}     (4-panel scatter)
    fig3_spearman_bars.{pdf,png}     (per-metric ρ with CI)
    fig4_pipeline_dag.{pdf,png}      (DAG diagram)

NO placeholder values. Every number in every figure comes from the JSON.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Publication-grade rcParams
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
    "savefig.transparent": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.6,
})


def load(results_path: Path):
    return json.loads(results_path.read_text())


def fig1_mos_distribution(data, out: Path, fname):
    """Figure 1: distribution of MOS labels in the subset."""
    mos = [v["mos"] for v in data["per_video"]]
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    bins = np.arange(0.5, 5.0, 0.5)
    ax.hist(mos, bins=bins, edgecolor="black", color="#3b7eb6", alpha=0.85)
    ax.set_xlabel("VideoFeedback Visual Quality MOS")
    ax.set_ylabel("Count")
    ax.set_title(f"VideoFeedback subset (n={data['n_videos']}) — MOS distribution")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1\nBad", "2\nAvg", "3\nGood", "4\nReal"])
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png")
    plt.close(fig)
    print(f"  ✓ {fname}.{{pdf,png}}")


def fig2_metric_vs_mos(data, out: Path, fname):
    """Figure 2: 4-panel scatter — top-4 metrics by |ρ|."""
    corrs = data["correlations_vs_mos"]
    # Rank metrics by abs(spearman)
    ordered = sorted(
        [(name, c["spearman"]) for name, c in corrs.items()
         if c["spearman"] is not None],
        key=lambda kv: -abs(kv[1])
    )[:4]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    mos = np.array([v["mos"] for v in data["per_video"]])

    for ax, (name, _) in zip(axes.flat, ordered):
        vals = np.array([v.get(name) for v in data["per_video"]], dtype=float)
        mask = np.isfinite(vals) & np.isfinite(mos)
        v, m = vals[mask], mos[mask]
        ax.scatter(v, m, s=16, alpha=0.7, edgecolor="black", linewidth=0.4,
                   color="#d8651e")
        # OLS line
        if len(v) > 2:
            slope, intercept = np.polyfit(v, m, 1)
            xs = np.linspace(v.min(), v.max(), 50)
            ax.plot(xs, slope * xs + intercept, "k--", linewidth=0.9, alpha=0.7)
        c = corrs[name]
        rho = c["spearman"]
        p = c["spearman_p"]
        ci = (c.get("spearman_ci95_lo"), c.get("spearman_ci95_hi"))
        ax.set_xlabel(name.replace("_", " "))
        ax.set_ylabel("MOS")
        ax.set_yticks([1, 2, 3, 4])
        ax.set_title(
            f"{name.replace('_',' ')}\n"
            f"ρ={rho:+.3f}  p={p:.3f}  CI95=[{ci[0]:+.3f}, {ci[1]:+.3f}]",
            fontsize=9,
        )

    fig.suptitle(
        f"Pure-CV metrics vs human MOS — VideoFeedback subset (n={data['n_videos']})",
        fontsize=11, y=1.02
    )
    fig.tight_layout()
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png")
    plt.close(fig)
    print(f"  ✓ {fname}.{{pdf,png}}")


def fig3_spearman_bars(data, out: Path, fname):
    """Figure 3: bar chart of Spearman ρ with CI95 error bars per metric."""
    corrs = data["correlations_vs_mos"]
    items = [(name, c) for name, c in corrs.items() if c["spearman"] is not None]
    items.sort(key=lambda kv: kv[1]["spearman"])

    names = [k.replace("_", "\n") for k, _ in items]
    rhos = np.array([c["spearman"] for _, c in items])
    los = np.array([c["spearman_ci95_lo"] for _, c in items])
    his = np.array([c["spearman_ci95_hi"] for _, c in items])
    err_lo = rhos - los
    err_hi = his - rhos
    pvals = [c["spearman_p"] for _, c in items]

    colors = ["#1f77b4" if p < 0.05 else "#bbbbbb" for p in pvals]

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    x = np.arange(len(items))
    ax.bar(x, rhos, yerr=[err_lo, err_hi], capsize=3, color=colors,
           edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Spearman ρ vs MOS")
    ax.set_title(
        f"Per-metric Spearman correlation (n={data['n_videos']}, "
        f"bootstrap CI95). Colored = p<0.05."
    )
    ax.set_ylim(-0.7, 0.7)
    fig.tight_layout()
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png")
    plt.close(fig)
    print(f"  ✓ {fname}.{{pdf,png}}")


def fig4_pipeline_dag(out: Path, fname):
    """Figure 4: Pipeline DAG (illustrative — not from data)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, label, color="#bdd7ee", edgecolor="black"):
        from matplotlib.patches import FancyBboxPatch
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.05,rounding_size=0.15",
                           linewidth=1.0, facecolor=color, edgecolor=edgecolor)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=9, fontweight="bold")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.9))

    # Layer 1: Input
    box(5, 7, 2, 0.6, "Video V", color="#fce4d6")
    # Layer 2: Decode + Shot
    box(2.5, 5.8, 3, 0.6, "Decode + Shot detection", color="#ffe699")
    arrow(6, 7, 4, 6.4)
    # Layer 3: per-shot processing fan-out
    box(0.2, 4.4, 3, 0.6, "Intra-shot metrics\n(motion, ssim, flicker)", color="#bdd7ee")
    box(3.5, 4.4, 2.8, 0.6, "Identity (DTW)\nArcFace", color="#c5e0b3")
    box(6.6, 4.4, 2.8, 0.6, "Semantic (CLIP/DINO)", color="#e2cfff")
    box(9.7, 4.4, 2.0, 0.6, "VFI artefacts", color="#f4cccc")
    arrow(4, 5.8, 1.7, 5.0)
    arrow(4, 5.8, 4.9, 5.0)
    arrow(4, 5.8, 8.0, 5.0)
    arrow(4, 5.8, 10.7, 5.0)
    # Layer 4: aggregation
    box(0.5, 2.8, 5, 0.6, "Intra aggregation", color="#fff2cc")
    box(6, 2.8, 5, 0.6, "Inter-shot coherence\n(SSIM/HSV/CLIP/DTW)", color="#fff2cc")
    arrow(1.7, 4.4, 3, 3.4)
    arrow(4.9, 4.4, 3, 3.4)
    arrow(8.0, 4.4, 8.5, 3.4)
    arrow(10.7, 4.4, 8.5, 3.4)
    # Layer 5: phase4
    box(3.5, 1.4, 5, 0.6, "Phase-4 verdict gate\n(ACCEPT/REVIEW/REJECT)",
        color="#9dc3e6")
    arrow(3, 2.8, 6, 2.0)
    arrow(8.5, 2.8, 6, 2.0)
    # Layer 6: renderers
    box(0.2, 0.2, 11.5, 0.6,
        "Renderers: Markdown · HTML · PDF · JSON · SVG · Jupyter · Benchmark · VBench",
        color="#f8cbad")
    arrow(6, 1.4, 6, 0.8)

    ax.set_title("CineInfini pipeline — 22 modules, 8 renderers",
                 fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png")
    plt.close(fig)
    print(f"  ✓ {fname}.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-json", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load(args.results_json)
    print(f"Loaded n={data['n_videos']} from {args.results_json.name}")
    print(f"Output dir: {args.out_dir}\n")
    fig1_mos_distribution(data, args.out_dir, "fig1_mos_distribution")
    fig2_metric_vs_mos(data, args.out_dir, "fig2_metric_vs_mos")
    fig3_spearman_bars(data, args.out_dir, "fig3_spearman_bars")
    fig4_pipeline_dag(args.out_dir, "fig4_pipeline_dag")
    print(f"\nAll figures saved to {args.out_dir}")

if __name__ == "__main__":
    main()
