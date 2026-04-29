#!/usr/bin/env python3
"""Aggregate per-video results from run_chunked_calibration.py and compute
publication-grade statistics:

  * Spearman ρ + Pearson r + Kendall τ (three correlation measures)
  * Bootstrap CI95 (n_boot=2000)
  * Two-sided permutation test for ρ (n_perm=2000)
  * Benjamini-Hochberg FDR correction across all metrics
  * Effect-size buckets (small <0.10, medium <0.30, large ≥0.30)

Usage:
    python scripts/aggregate_chunked.py \\
        --results-jsonl out/<dataset>_chunks/results.jsonl \\
        --out-json docs/figures/<dataset>_real_results.json \\
        --dataset-name <dataset>
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau

DEFAULT_KEYS = [
    "sharpness", "brightness", "saturation", "contrast",
    "flicker", "flicker_var", "motion_proxy", "motion_var",
]


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Return list of bool: True if rejected at FDR alpha."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresh
    # Largest k where p_k ≤ k/n * alpha → reject all up to k
    if not np.any(passed):
        return [False] * n
    k_max = np.max(np.where(passed)[0])
    rejected = np.zeros(n, dtype=bool)
    rejected[order[: k_max + 1]] = True
    return rejected.tolist()


def bootstrap_ci(values: np.ndarray, mos: np.ndarray, n_boot: int = 2000,
                 seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        if values[idx].std() == 0 or mos[idx].std() == 0:
            continue
        r, _ = spearmanr(values[idx], mos[idx])
        if not math.isnan(r):
            boots.append(r)
    if not boots:
        return float("nan"), float("nan")
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def permutation_test(values: np.ndarray, mos: np.ndarray, observed_rho: float,
                     n_perm: int = 2000, seed: int = 43) -> float:
    """Two-sided permutation test for Spearman correlation."""
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        permuted = rng.permutation(mos)
        r, _ = spearmanr(values, permuted)
        if not math.isnan(r) and abs(r) >= abs(observed_rho):
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def effect_size_bucket(rho: float) -> str:
    a = abs(rho)
    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "small"
    if a < 0.50:
        return "medium"
    return "large"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-jsonl", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--dataset-name", default="dataset")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--n-permutation", type=int, default=2000)
    ap.add_argument("--metrics", nargs="*", default=DEFAULT_KEYS,
                    help="Metric keys to evaluate (default: standard 8)")
    ap.add_argument("--fdr-alpha", type=float, default=0.05)
    args = ap.parse_args()

    rows = []
    for line in args.results_jsonl.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    print(f"Loaded {len(rows)} per-video records from {args.results_jsonl}")

    mos = np.array([r["mos"] for r in rows], dtype=float)
    print(f"MOS range: [{mos.min():.3f}, {mos.max():.3f}]  mean={mos.mean():.3f}  "
          f"std={mos.std():.3f}")

    # Filter metrics that actually exist
    available = [k for k in args.metrics if any(k in r for r in rows)]
    print(f"Metrics: {available}")

    correlations = {}
    pvals_for_fdr = []
    for k in available:
        vals = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        mask = np.isfinite(vals) & np.isfinite(mos)
        v, m = vals[mask], mos[mask]
        if len(v) < 5 or v.std() == 0:
            correlations[k] = {"n": int(len(v)), "spearman": None}
            continue
        s_rho, s_p = spearmanr(v, m)
        p_rho, p_p = pearsonr(v, m)
        k_tau, k_p = kendalltau(v, m)
        ci_lo, ci_hi = bootstrap_ci(v, m, n_boot=args.n_bootstrap)
        perm_p = permutation_test(v, m, s_rho, n_perm=args.n_permutation)
        pvals_for_fdr.append(s_p)

        correlations[k] = {
            "n": int(mask.sum()),
            "spearman": float(s_rho),
            "spearman_p": float(s_p),
            "spearman_perm_p": float(perm_p),
            "pearson": float(p_rho),
            "pearson_p": float(p_p),
            "kendall": float(k_tau),
            "kendall_p": float(k_p),
            "spearman_ci95_lo": float(ci_lo),
            "spearman_ci95_hi": float(ci_hi),
            "effect_size": effect_size_bucket(s_rho),
        }

    # Apply FDR correction
    rejected = benjamini_hochberg(pvals_for_fdr, alpha=args.fdr_alpha)
    significant_metrics = [k for k, rej in zip(available, rejected) if rej]
    for k, rej in zip(available, rejected):
        correlations[k]["fdr_significant"] = bool(rej)

    out = {
        "dataset": args.dataset_name,
        "labels_file": str(args.results_jsonl),
        "n_videos": len(rows),
        "labels_range": [float(mos.min()), float(mos.max())],
        "labels_mean": float(mos.mean()),
        "labels_std": float(mos.std()),
        "n_bootstrap": args.n_bootstrap,
        "n_permutation": args.n_permutation,
        "fdr_alpha": args.fdr_alpha,
        "fdr_significant_metrics": significant_metrics,
        "metrics_run": available,
        "correlations_vs_mos": correlations,
        "per_video": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {args.out_json}")

    # Console report
    print("\n" + "=" * 88)
    print(f"REAL Spearman ρ vs MOS — {args.dataset_name} (n={len(rows)})")
    print("=" * 88)
    print(f"{'metric':<14} {'rho':>8} {'p':>10} {'perm-p':>9} "
          f"{'CI95':>22} {'effect':>10} {'FDR':>4}")
    for k in sorted(correlations,
                    key=lambda x: -abs(correlations[x].get("spearman") or 0)):
        c = correlations[k]
        if c.get("spearman") is None:
            continue
        rho, p = c["spearman"], c["spearman_p"]
        perm_p = c["spearman_perm_p"]
        lo, hi = c["spearman_ci95_lo"], c["spearman_ci95_hi"]
        eff = c["effect_size"]
        fdr = "*" if c["fdr_significant"] else " "
        print(f"  {k:<14} {rho:>+8.3f} {p:>10.4f} {perm_p:>9.4f}  "
              f"[{lo:+.3f}, {hi:+.3f}] {eff:>10}  {fdr}")
    print(f"\nFDR-significant metrics (α={args.fdr_alpha}): {significant_metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
