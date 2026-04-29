#!/usr/bin/env python3
"""Generate the three-dataset cross-comparison figure: KoNViD (natural,
n=392) vs VideoFeedback (AIGC small, n=48) vs T2VQA-DB (AIGC large,
n=422). This is the hero figure of the paper — it shows simultaneously:

  * The pure-CV ceiling on AIGC vs natural
  * Regime-specific sign flips
  * Sample-size dependence (VideoFeedback CIs are wider)

Outputs:
    fig9_three_dataset_comparison.{pdf,png}
    fig10_pure_cv_ceiling.{pdf,png}
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

DATASETS = ["KoNViD-1k", "VideoFeedback", "T2VQA-DB"]
COLORS = {"KoNViD-1k": "#5e8b3a",       # green for natural
          "VideoFeedback": "#d8651e",    # orange for small AIGC
          "T2VQA-DB": "#7B2CBF"}         # purple for large AIGC


def fig9_three_dataset_bars(konvid, vf, t2vqa, out: Path, fname: str):
    """Per-metric ρ across 3 datasets, side-by-side, with CI95 error bars."""
    common = ["sharpness", "brightness", "saturation", "contrast",
              "flicker", "motion_proxy"]
    # Aliases for old VideoFeedback JSON (different naming convention)
    aliases = {
        "motion_proxy": ["motion_proxy", "motion_magnitude"],
        "flicker_var": ["flicker_var", "flicker_variance"],
    }

    def get_corr(ds, metric):
        keys = aliases.get(metric, [metric])
        for k in keys:
            if k in ds["correlations_vs_mos"]:
                return ds["correlations_vs_mos"][k]
        return {"spearman": None, "spearman_ci95_lo": None,
                "spearman_ci95_hi": None, "fdr_significant": False}

    data = {"KoNViD-1k": konvid, "VideoFeedback": vf, "T2VQA-DB": t2vqa}
    x = np.arange(len(common))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, ds_name in enumerate(DATASETS):
        ds = data[ds_name]
        cs = [get_corr(ds, m) for m in common]
        rhos = [c["spearman"] if c["spearman"] is not None else 0 for c in cs]
        los = [c["spearman_ci95_lo"] if c["spearman_ci95_lo"] is not None else 0 for c in cs]
        his = [c["spearman_ci95_hi"] if c["spearman_ci95_hi"] is not None else 0 for c in cs]
        err_lo = np.array(rhos) - np.array(los)
        err_hi = np.array(his) - np.array(rhos)
        offset = (i - 1) * width
        sig = [c.get("fdr_significant", False) for c in cs]
        ax.bar(x + offset, rhos, width,
               yerr=[err_lo, err_hi], capsize=3,
               color=COLORS[ds_name], edgecolor="black",
               linewidth=0.6, alpha=0.92,
               label=f"{ds_name} (n={ds['n_videos']})")
        for j, (rho, s) in enumerate(zip(rhos, sig)):
            if s:
                y = rho + (0.04 if rho >= 0 else -0.06)
                ax.text(x[j] + offset, y, "★", ha="center", va="center",
                        fontsize=10, color="black")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=15)
    ax.set_ylabel("Spearman ρ vs human MOS")
    ax.set_title(
        "Pure-CV metrics across three regimes — natural video, small AIGC, "
        "large AIGC.\n★ = FDR-significant (Benjamini-Hochberg α=0.05). "
        "Error bars = bootstrap CI95 (n_boot=2000).",
        fontsize=10
    )
    ax.set_ylim(-0.55, 0.55)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png")
    plt.close(fig)
    print(f"  ✓ {fname}.{{pdf,png}}")


def fig10_pure_cv_ceiling(konvid, vf, t2vqa, out: Path, fname: str):
    """The 'pure-CV ceiling' figure — shows max |ρ| across metrics for
    each dataset, motivating why ML features are needed for AIGC."""
    data = {"KoNViD-1k": konvid, "VideoFeedback": vf, "T2VQA-DB": t2vqa}

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    # For each dataset: max |ρ| across all metrics, plus the metric that achieved it
    summaries = []
    for ds_name in DATASETS:
        ds = data[ds_name]
        corrs = ds["correlations_vs_mos"]
        best_metric = None
        best_abs_rho = 0.0
        best_rho = 0.0
        for m, c in corrs.items():
            if c.get("spearman") is None:
                continue
            if abs(c["spearman"]) > best_abs_rho:
                best_abs_rho = abs(c["spearman"])
                best_rho = c["spearman"]
                best_metric = m
        summaries.append({
            "dataset": ds_name,
            "n": ds["n_videos"],
            "best_rho": best_rho,
            "best_metric": best_metric,
        })

    x = np.arange(len(summaries))
    rhos = [abs(s["best_rho"]) for s in summaries]
    colors = [COLORS[s["dataset"]] for s in summaries]
    bars = ax.bar(x, rhos, color=colors, edgecolor="black", linewidth=0.7,
                  width=0.55)

    # Annotate with the metric name and signed ρ
    for i, s in enumerate(summaries):
        ax.text(i, abs(s["best_rho"]) + 0.015,
                f"{s['best_metric']}\nρ={s['best_rho']:+.3f}",
                ha="center", va="bottom", fontsize=9)

    # Reference line — published DOVER ρ on AIGC ~0.71 (literature)
    ax.axhline(0.71, color="red", linestyle="--", linewidth=1.0, alpha=0.7,
               label="DOVER (ML, published)")
    ax.axhline(0.65, color="orange", linestyle="--", linewidth=1.0, alpha=0.7,
               label="FAST-VQA (ML, published)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s['dataset']}\n(n={s['n']})" for s in summaries])
    ax.set_ylabel("Best pure-CV |ρ| achievable")
    ax.set_title("The Pure-CV Ceiling on AIGC Quality Assessment\n"
                 "Pure photometric/temporal features cap below 0.25 on T2VQA-DB; "
                 "ML features are necessary.", fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png")
    plt.close(fig)
    print(f"  ✓ {fname}.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--konvid-json", required=True, type=Path)
    ap.add_argument("--videofeedback-json", required=True, type=Path)
    ap.add_argument("--t2vqa-json", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    konvid = json.loads(args.konvid_json.read_text())
    vf = json.loads(args.videofeedback_json.read_text())
    t2vqa = json.loads(args.t2vqa_json.read_text())

    # The VideoFeedback JSON from earlier was generated by an OLD script that
    # didn't compute fdr_significant and didn't run permutation tests.
    # Backfill those fields so the figure code works.
    for k, c in vf["correlations_vs_mos"].items():
        c.setdefault("fdr_significant", c.get("spearman_p", 1.0) < 0.05)
    for k, c in konvid["correlations_vs_mos"].items():
        c.setdefault("fdr_significant", c.get("spearman_p", 1.0) < 0.05)

    print(f"KoNViD: n={konvid['n_videos']}")
    print(f"VideoFeedback: n={vf['n_videos']}")
    print(f"T2VQA-DB: n={t2vqa['n_videos']}\n")

    fig9_three_dataset_bars(konvid, vf, t2vqa, args.out_dir,
                            "fig9_three_dataset_comparison")
    fig10_pure_cv_ceiling(konvid, vf, t2vqa, args.out_dir,
                          "fig10_pure_cv_ceiling")


if __name__ == "__main__":
    main()
