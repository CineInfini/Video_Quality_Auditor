#!/usr/bin/env python3
"""Signed-rank aggregation of pure-CV metrics.

Standard linear/ridge regression assumes additive contributions with
fixed signs across datasets. Our cross-regime analysis shows this
assumption fails: sharpness flips sign between VideoFeedback and
T2VQA-DB. We propose a non-parametric alternative:

    1. For each (metric m, dataset d), compute the sign of correlation
       sign_md = sign(rho_md).
    2. For each video v, convert each metric value to its within-
       dataset rank, normalized to [0, 1].
    3. Composite score := sum_m sign_md * rank_md(v).

This composite is:
    * parameter-free (no learned weights),
    * sign-corrected per regime,
    * monotone-transformation invariant (depends on ranks only).

We compare vs. (a) best single metric on each dataset, and (b) ridge
regression baseline.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

METRIC_KEYS = ["sharpness", "brightness", "saturation", "contrast",
               "flicker", "flicker_var", "motion_proxy", "motion_var"]
ALIASES = {
    "motion_proxy": "motion_magnitude",
    "flicker_var": "flicker_variance",
}


def features_and_mos(data: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, used_keys) from a dataset JSON."""
    rows = data["per_video"]
    keys = []
    cols = []
    for k in METRIC_KEYS:
        col = []
        for r in rows:
            if k in r:
                col.append(r[k])
            elif ALIASES.get(k) and ALIASES[k] in r:
                col.append(r[ALIASES[k]])
            else:
                col = None
                break
        if col is not None:
            keys.append(k)
            cols.append(col)
    X = np.array(cols, dtype=float).T  # n × m
    y = np.array([r["mos"] for r in rows], dtype=float)
    return X, y, keys


def sign_per_metric(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sign of Spearman ρ per column."""
    signs = []
    for j in range(X.shape[1]):
        rho, _ = spearmanr(X[:, j], y)
        signs.append(1 if rho >= 0 else -1)
    return np.array(signs)


def signed_rank_score(X: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Per-row signed rank composite. X shape (n, m), signs shape (m,)."""
    from scipy.stats import rankdata
    ranks = np.zeros_like(X)
    for j in range(X.shape[1]):
        ranks[:, j] = rankdata(X[:, j]) / len(X)
    return (ranks * signs).sum(axis=1)


def best_single_metric(X: np.ndarray, y: np.ndarray, keys: list[str]) \
        -> tuple[str, float]:
    best_k, best_abs_rho, best_rho = None, 0.0, 0.0
    for j, k in enumerate(keys):
        rho, _ = spearmanr(X[:, j], y)
        if abs(rho) > best_abs_rho:
            best_abs_rho = abs(rho)
            best_rho = rho
            best_k = k
    return best_k, best_rho


def evaluate_dataset(data: dict, name: str) -> dict:
    X, y, keys = features_and_mos(data)
    print(f"\n=== {name} ===  n={len(y)}, m={len(keys)}")
    n = len(y)

    # Single best metric (oracle baseline)
    best_metric, best_rho = best_single_metric(X, y, keys)
    print(f"  best single metric: {best_metric}  ρ={best_rho:+.3f}")

    # Signed-rank aggregation — full-data (in-sample, oracle for sign)
    signs = sign_per_metric(X, y)
    composite = signed_rank_score(X, signs)
    rho_in, _ = spearmanr(composite, y)
    print(f"  signed-rank composite (in-sample): ρ={rho_in:+.3f}  "
          f"signs={dict(zip(keys, signs.tolist()))}")

    # 5-fold CV to avoid in-sample bias
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rhos = []
    for tr, te in kf.split(X):
        signs_tr = sign_per_metric(X[tr], y[tr])
        # Use train-fold ranks normalized to [0, 1]
        composite_te = signed_rank_score(X[te], signs_tr)
        if y[te].std() == 0 or composite_te.std() == 0:
            continue
        rho_te, _ = spearmanr(composite_te, y[te])
        cv_rhos.append(rho_te)
    cv_mean = float(np.mean(cv_rhos))
    cv_std = float(np.std(cv_rhos))
    print(f"  signed-rank composite (5-fold CV): "
          f"ρ={cv_mean:+.3f} ± {cv_std:.3f}")

    # Ridge regression baseline (5-fold CV)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    ridge_rhos = []
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        if y[te].std() == 0 or pred.std() == 0:
            continue
        rho_te, _ = spearmanr(pred, y[te])
        ridge_rhos.append(rho_te)
    ridge_mean = float(np.mean(ridge_rhos))
    ridge_std = float(np.std(ridge_rhos))
    print(f"  Ridge regression (5-fold CV): "
          f"ρ={ridge_mean:+.3f} ± {ridge_std:.3f}")

    return {
        "n": n,
        "metrics_used": keys,
        "best_single_metric": best_metric,
        "best_single_rho": float(best_rho),
        "signed_rank_in_sample": float(rho_in),
        "signed_rank_cv_mean": cv_mean,
        "signed_rank_cv_std": cv_std,
        "ridge_cv_mean": ridge_mean,
        "ridge_cv_std": ridge_std,
        "signs": dict(zip(keys, signs.tolist())),
        "improvement_over_best_single": float(cv_mean) - abs(float(best_rho)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--konvid-json", required=True, type=Path)
    ap.add_argument("--videofeedback-json", required=True, type=Path)
    ap.add_argument("--t2vqa-json", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    konvid = json.loads(args.konvid_json.read_text())
    vf = json.loads(args.videofeedback_json.read_text())
    t2vqa = json.loads(args.t2vqa_json.read_text())

    out = {
        "konvid_1k": evaluate_dataset(konvid, "KoNViD-1k (natural)"),
        "videofeedback": evaluate_dataset(vf, "VideoFeedback (AIGC small)"),
        "t2vqa_db": evaluate_dataset(t2vqa, "T2VQA-DB (AIGC large)"),
    }

    print("\n" + "=" * 78)
    print("SIGNED-RANK AGGREGATION — improvement over best single metric")
    print("=" * 78)
    print(f"{'dataset':<28} {'best single':>16} {'signed-rank CV':>18} {'Δ':>8}")
    for ds, r in out.items():
        delta = r["signed_rank_cv_mean"] - abs(r["best_single_rho"])
        print(f"  {ds:<26}  {r['best_single_metric']:>10} "
              f"({abs(r['best_single_rho']):.3f})  "
              f"{r['signed_rank_cv_mean']:+.3f} ± {r['signed_rank_cv_std']:.3f}     "
              f"{delta:+.3f}")

    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
