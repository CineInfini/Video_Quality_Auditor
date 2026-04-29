#!/usr/bin/env python3
"""Aggregate the per-video results produced by run_konvid_chunked.py
and compute Spearman/Pearson + bootstrap CI95 against MOS.

Usage:
    python scripts/aggregate_konvid.py \
        --results-jsonl out/konvid_chunks/results.jsonl \
        --out-json docs/figures/konvid_real_results.json
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

METRIC_KEYS = ["sharpness", "brightness", "saturation", "contrast",
               "flicker", "flicker_var", "motion_proxy"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-jsonl", required=True, type=Path,
                    help="Path to results.jsonl produced by run_konvid_chunked.py")
    ap.add_argument("--out-json", required=True, type=Path,
                    help="Output JSON path")
    args = ap.parse_args()

    rows = []
    for line in args.results_jsonl.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    print(f"Loaded {len(rows)} per-video records")

    mos = np.array([r["mos"] for r in rows], dtype=float)
    print(f"MOS range: [{mos.min():.3f}, {mos.max():.3f}]  mean={mos.mean():.3f}")

    correlations = {}
    rng = np.random.default_rng(42)
    n_boot = 2000

    for k in METRIC_KEYS:
        vals = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        mask = np.isfinite(vals) & np.isfinite(mos)
        v, m = vals[mask], mos[mask]
        if len(v) < 5 or v.std() == 0:
            correlations[k] = {"n": int(len(v)), "spearman": None}
            continue
        rho, p = spearmanr(v, m)
        pr, pp = pearsonr(v, m)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(v), len(v))
            if v[idx].std() == 0:
                continue
            r, _ = spearmanr(v[idx], m[idx])
            if not math.isnan(r):
                boots.append(r)
        boots = np.array(boots)
        correlations[k] = {
            "n": int(mask.sum()),
            "spearman": float(rho),
            "spearman_p": float(p),
            "pearson": float(pr),
            "pearson_p": float(pp),
            "spearman_ci95_lo": float(np.quantile(boots, 0.025)),
            "spearman_ci95_hi": float(np.quantile(boots, 0.975)),
        }

    out = {
        "dataset": "KoNViD-1k subset (user upload, n=392 of 1200)",
        "labels_file": "KoNViD_mos_fr.csv (cnn-tlvqm mirror)",
        "n_videos": len(rows),
        "labels_range": [float(mos.min()), float(mos.max())],
        "metrics_run": METRIC_KEYS,
        "correlations_vs_mos": correlations,
        "n_bootstrap": n_boot,
        "per_video": rows,
        "note": (
            "Pure-CV metrics only (no torch/CLIP/DINOv2). Same metric set "
            "as VideoFeedback calibration. Run on Intel sandbox CPU, "
            "in chunks of 50 videos with checkpointing."
        ),
    }
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {args.out_json}")

    # Console report
    print("\n" + "=" * 72)
    print(f"REAL Spearman ρ vs MOS — KoNViD-1k subset (n={len(rows)})")
    print("=" * 72)
    print(f"{'metric':<14} {'n':>4} {'rho':>8} {'p':>10} {'CI95':>22}")
    for k in sorted(correlations, key=lambda x: -abs(correlations[x].get("spearman") or 0)):
        c = correlations[k]
        if c.get("spearman") is None:
            print(f"  {k:<14}  insufficient data")
            continue
        rho, p = c["spearman"], c["spearman_p"]
        lo, hi = c["spearman_ci95_lo"], c["spearman_ci95_hi"]
        print(f"  {k:<14} {c['n']:>4} {rho:>+8.3f} {p:>10.4f}  [{lo:+.3f}, {hi:+.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
