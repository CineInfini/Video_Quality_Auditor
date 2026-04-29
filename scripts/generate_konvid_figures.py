#!/usr/bin/env python3
"""Generate publication-quality figures from the REAL calibration results
on KoNViD-1k (natural videos, n=392) and VideoFeedback (AIGC, n=48).

Outputs (PDF + PNG, 300 dpi):
    fig5_konvid_mos_distribution.{pdf,png}
    fig6_konvid_metric_vs_mos.{pdf,png}      (4-panel scatter)
    fig7_konvid_spearman_bars.{pdf,png}
    fig8_natural_vs_aigc_comparison.{pdf,png}  (side-by-side per-metric ρ)
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
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
})


def fig5_konvid_distribution(data, out: Path, fname: str):
    mos = [v["mos"] for v in data["per_video"]]
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.hist(mos, bins=20, edgecolor="black", color="#5e8b3a", alpha=0.85)
    ax.set_xlabel("KoNViD-1k MOS [1-5]")
    ax.set_ylabel("Count")
    ax.set_title(f"KoNViD-1k subset (n={data['n_videos']}) — MOS distribution")
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png")
    plt.close(fig)
    print(f"  ✓ {fname}.{{pdf,png}}")


def fig6_konvid_scatter(data, out: Path, fname: str):
    corrs = data["correlations_vs_mos"]
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
        ax.scatter(v, m, s=8, alpha=0.4, color="#5e8b3a", edgecolor="none")
        if len(v) > 2:
            slope, intercept = np.polyfit(v, m, 1)
            xs = np.linspace(v.min(), v.max(), 50)
            ax.plot(xs, slope * xs + intercept, "k--", linewidth=1.0, alpha=0.8)
        c = corrs[name]
        rho = c["spearman"]
        p = c["spearman_p"]
        ci = (c["spearman_ci95_lo"], c["spearman_ci95_hi"])
        ax.set_xlabel(name.replace("_", " "))
        ax.set_ylabel("MOS")
        ax.set_title(
            f"{name.replace('_',' ')}\nρ={rho:+.3f}  p={p:.4f}  "
            f"CI95=[{ci[0]:+.3f}, {ci[1]:+.3f}]",
            fontsize=9,
        )

    fig.suptitle(f"Pure-CV metrics vs human MOS — KoNViD-1k (n={data['n_videos']})",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png")
    plt.close(fig)
    print(f"  ✓ {fname}.{{pdf,png}}")


def fig7_konvid_bars(data, out: Path, fname: str):
    corrs = data["correlations_vs_mos"]
    items = [(k, c) for k, c in corrs.items() if c["spearman"] is not None]
    items.sort(key=lambda kv: kv[1]["spearman"])
    names = [k.replace("_", "\n") for k, _ in items]
    rhos = np.array([c["spearman"] for _, c in items])
    los = np.array([c["spearman_ci95_lo"] for _, c in items])
    his = np.array([c["spearman_ci95_hi"] for _, c in items])
    err_lo = rhos - los
    err_hi = his - rhos
    pvals = [c["spearman_p"] for _, c in items]
    colors = ["#5e8b3a" if p < 0.001 else ("#a3c171" if p < 0.05 else "#cccccc")
              for p in pvals]

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    x = np.arange(len(items))
    ax.bar(x, rhos, yerr=[err_lo, err_hi], capsize=3, color=colors,
           edgecolor="black", linewidth=0.6, alpha=0.95)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Spearman ρ vs MOS")
    ax.set_title(f"KoNViD-1k per-metric ρ (n={data['n_videos']}, "
                 f"bootstrap CI95). Dark = p<0.001, light = p<0.05.")
    ax.set_ylim(-0.5, 0.6)
    fig.tight_layout()
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png")
    plt.close(fig)
    print(f"  ✓ {fname}.{{pdf,png}}")


def fig8_comparison(konvid_data, vf_data, out: Path, fname: str):
    """Side-by-side comparison: same metric on natural vs AIGC."""
    common_metrics = ["sharpness", "brightness", "saturation", "contrast",
                      "flicker"]
    konvid_corrs = konvid_data["correlations_vs_mos"]
    vf_corrs = vf_data["correlations_vs_mos"]

    konvid_rhos = [konvid_corrs[k]["spearman"] for k in common_metrics]
    vf_rhos = [vf_corrs[k]["spearman"] for k in common_metrics]

    konvid_lo = [konvid_corrs[k]["spearman_ci95_lo"] for k in common_metrics]
    konvid_hi = [konvid_corrs[k]["spearman_ci95_hi"] for k in common_metrics]
    vf_lo = [vf_corrs[k]["spearman_ci95_lo"] for k in common_metrics]
    vf_hi = [vf_corrs[k]["spearman_ci95_hi"] for k in common_metrics]

    konvid_err_lo = np.array(konvid_rhos) - np.array(konvid_lo)
    konvid_err_hi = np.array(konvid_hi) - np.array(konvid_rhos)
    vf_err_lo = np.array(vf_rhos) - np.array(vf_lo)
    vf_err_hi = np.array(vf_hi) - np.array(vf_rhos)

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    x = np.arange(len(common_metrics))
    width = 0.35
    ax.bar(x - width / 2, konvid_rhos, width,
           yerr=[konvid_err_lo, konvid_err_hi], capsize=3,
           color="#5e8b3a", edgecolor="black", linewidth=0.6,
           label=f"KoNViD-1k (natural, n={konvid_data['n_videos']})")
    ax.bar(x + width / 2, vf_rhos, width,
           yerr=[vf_err_lo, vf_err_hi], capsize=3,
           color="#d8651e", edgecolor="black", linewidth=0.6,
           label=f"VideoFeedback (AIGC, n={vf_data['n_videos']})")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(common_metrics, rotation=15)
    ax.set_ylabel("Spearman ρ vs MOS")
    ax.set_title("Natural-video vs AIGC: per-metric correlation with human MOS\n"
                 "Bootstrap CI95. Sign-flips on brightness/saturation suggest "
                 "regime-dependent calibration.",
                 fontsize=10)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png")
    plt.close(fig)
    print(f"  ✓ {fname}.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--konvid-json", required=True, type=Path)
    ap.add_argument("--videofeedback-json", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    konvid = json.loads(args.konvid_json.read_text())
    vf = json.loads(args.videofeedback_json.read_text())

    print(f"KoNViD-1k: n={konvid['n_videos']}")
    print(f"VideoFeedback: n={vf['n_videos']}")
    print(f"Output: {args.out_dir}\n")

    fig5_konvid_distribution(konvid, args.out_dir, "fig5_konvid_mos_distribution")
    fig6_konvid_scatter(konvid, args.out_dir, "fig6_konvid_metric_vs_mos")
    fig7_konvid_bars(konvid, args.out_dir, "fig7_konvid_spearman_bars")
    fig8_comparison(konvid, vf, args.out_dir, "fig8_natural_vs_aigc_comparison")


if __name__ == "__main__":
    main()
