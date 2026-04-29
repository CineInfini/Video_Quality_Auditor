#!/usr/bin/env python3
"""Analyze the advanced pure-CV features (BRISQUE NSS + LBP + HOG + DCT
+ Canny + multi-scale Sobel) and compare against the basic photometric
metrics from v0.4.10.

Key questions:
  1. Which advanced features have the strongest per-dataset Spearman ρ?
  2. Does Ridge with all 48+8 features beat the v0.4.10 ceiling?
  3. Does the regime-aware approach gain?
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler


def load_results(jsonl: Path) -> list[dict]:
    return [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]


def feature_keys_advanced(rows: list[dict]) -> list[str]:
    """All 48 advanced feature keys."""
    fixed = ["brisque_alpha_native", "brisque_sigma2_native",
             "brisque_alpha_halfscale", "brisque_sigma2_halfscale",
             "lbp_entropy", "hog_cell_var", "canny_edge_density",
             "dct_block_energy", "shannon_entropy_y",
             "sobel_scale_1", "sobel_scale_half", "sobel_scale_quarter"]
    brisque_indexed = [f"brisque_f{i:02d}" for i in range(36)]
    return fixed + brisque_indexed


def to_X_y_regime(rows: list[dict], feat_keys: list[str]) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X = np.array([[r.get(k, np.nan) for k in feat_keys] for r in rows],
                 dtype=float)
    y = np.array([r["mos"] for r in rows], dtype=float)
    regimes = np.array([0 if r["regime"] == "natural" else 1 for r in rows])
    # Replace NaNs with column means
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if len(inds[0]):
        X[inds] = np.take(col_means, inds[1])
    datasets = [r["dataset"] for r in rows]
    return X, y, regimes, datasets


def per_dataset_per_feature(rows: list[dict], feat_keys: list[str]) -> dict:
    """Spearman ρ per (dataset, feature). Sorted by absolute ρ."""
    out = {}
    for ds in sorted({r["dataset"] for r in rows}):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        y = np.array([r["mos"] for r in ds_rows])
        feats_ds = []
        for k in feat_keys:
            vals = np.array([r.get(k, np.nan) for r in ds_rows], dtype=float)
            mask = np.isfinite(vals) & np.isfinite(y)
            if mask.sum() < 5 or vals[mask].std() == 0:
                continue
            rho, p = spearmanr(vals[mask], y[mask])
            feats_ds.append((k, float(rho), float(p)))
        feats_ds.sort(key=lambda kv: -abs(kv[1]))
        out[ds] = feats_ds
    return out


def cv_compare(rows: list[dict], feat_keys: list[str], n_splits: int = 5) -> dict:
    """5-fold stratified CV: Ridge / RF on this feature set."""
    X, y, regimes, datasets = to_X_y_regime(rows, feat_keys)
    # z-normalize MOS within dataset (same as before for cross-comparability)
    y_norm = y.copy()
    for ds in set(datasets):
        mask = np.array([d == ds for d in datasets])
        m, s = y_norm[mask].mean(), y_norm[mask].std()
        if s > 0:
            y_norm[mask] = (y_norm[mask] - m) / s

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    res = {"single_ridge": [], "rf": [], "regime_aware": [], "regime_clf_acc": []}
    for tr, te in kf.split(X, regimes):
        Xtr, Xte, ytr, yte = X[tr], X[te], y_norm[tr], y_norm[te]
        rtr, rte = regimes[tr], regimes[te]
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

        # Single Ridge
        m = Ridge(alpha=1.0).fit(Xtr_s, ytr)
        pred = m.predict(Xte_s)
        if yte.std() > 0 and pred.std() > 0:
            rho, _ = spearmanr(pred, yte)
            res["single_ridge"].append(float(rho))

        # Random Forest
        m = RandomForestRegressor(n_estimators=100, max_depth=10,
                                  random_state=42, n_jobs=1).fit(Xtr, ytr)
        pred = m.predict(Xte)
        if yte.std() > 0 and pred.std() > 0:
            rho, _ = spearmanr(pred, yte)
            res["rf"].append(float(rho))

        # Regime-aware Ridge
        clf = LogisticRegression(max_iter=2000, random_state=42).fit(Xtr_s, rtr)
        rpred = clf.predict(Xte_s)
        res["regime_clf_acc"].append(float((rpred == rte).mean()))
        pred_te = np.zeros(len(Xte))
        for r_label in [0, 1]:
            mtr = rtr == r_label
            if mtr.sum() < 5:
                continue
            head = Ridge(alpha=1.0).fit(Xtr_s[mtr], ytr[mtr])
            mte = rpred == r_label
            if mte.sum():
                pred_te[mte] = head.predict(Xte_s[mte])
        if yte.std() > 0 and pred_te.std() > 0:
            rho, _ = spearmanr(pred_te, yte)
            res["regime_aware"].append(float(rho))

    out = {}
    for k, vals in res.items():
        if not vals:
            out[k] = None
            continue
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def per_dataset_ridge(rows: list[dict], feat_keys: list[str], n_splits: int = 5) -> dict:
    """Ridge 5-fold per dataset — to see if advanced features push the
    individual-dataset ceilings."""
    out = {}
    for ds in sorted({r["dataset"] for r in rows}):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if len(ds_rows) < 30:
            continue
        X = np.array([[r.get(k, np.nan) for k in feat_keys] for r in ds_rows],
                     dtype=float)
        y = np.array([r["mos"] for r in ds_rows])
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        if len(inds[0]):
            X[inds] = np.take(col_means, inds[1])

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        rhos = []
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
            pred = m.predict(sc.transform(X[te]))
            if y[te].std() > 0 and pred.std() > 0:
                rho, _ = spearmanr(pred, y[te])
                rhos.append(float(rho))
        out[ds] = {"n": len(ds_rows), "rho_mean": float(np.mean(rhos)),
                   "rho_std": float(np.std(rhos))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-jsonl", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    rows = load_results(args.results_jsonl)
    print(f"Loaded {len(rows)} rows")
    feat_keys = feature_keys_advanced(rows)
    print(f"Using {len(feat_keys)} advanced features\n")

    # Per-dataset per-feature ρ — show top 5 per dataset
    pdpf = per_dataset_per_feature(rows, feat_keys)
    print("=" * 72)
    print("TOP-5 advanced features by |ρ| per dataset:")
    print("=" * 72)
    for ds, feats in pdpf.items():
        print(f"\n  {ds}")
        for k, rho, p in feats[:5]:
            print(f"    {k:<28} ρ={rho:+.3f}  p={p:.4f}")

    # Per-dataset Ridge with full feature set
    print("\n" + "=" * 72)
    print("PER-DATASET RIDGE (5-fold CV, advanced features only):")
    print("=" * 72)
    pdr = per_dataset_ridge(rows, feat_keys)
    for ds, r in pdr.items():
        print(f"  {ds:<10} n={r['n']:>4}  ρ = {r['rho_mean']:+.3f} ± {r['rho_std']:.3f}")

    # Cross-dataset CV (the global comparison)
    print("\n" + "=" * 72)
    print("CROSS-DATASET 5-fold CV (n=862, MOS z-normalized within dataset):")
    print("=" * 72)
    cv = cv_compare(rows, feat_keys)
    for k, v in cv.items():
        if v is None:
            continue
        print(f"  {k:<22}  ρ = {v['mean']:+.3f} ± {v['std']:.3f}")

    out = {
        "n_rows": len(rows),
        "n_features": len(feat_keys),
        "feature_keys": feat_keys,
        "per_dataset_per_feature_top": {ds: feats[:10] for ds, feats in pdpf.items()},
        "per_dataset_ridge": pdr,
        "cross_dataset_cv": cv,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
