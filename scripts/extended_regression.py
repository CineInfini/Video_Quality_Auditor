#!/usr/bin/env python3
"""Extended-feature regression evaluation. Uses ALL 16 pure-CV features
from v0.4.10.1 (the 8 baseline + 8 new BT.500/HOG/face/DCT metrics).

Compares 5 methods via stratified 5-fold CV on n=862 (z-MOS within
dataset):
  * Single Ridge — regime-blind linear baseline
  * Random Forest (200 trees, depth 12)
  * Gradient Boosting Regressor (200 trees)
  * Regime-aware Ridge (proposed in §4.4)
  * Per-dataset Ridge oracle

Also reports per-dataset best:
  * KoNViD-1k oracle
  * VideoFeedback oracle
  * T2VQA-DB oracle
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

EXT_KEYS = [
    "sharpness", "brightness", "saturation", "contrast",
    "flicker", "flicker_var", "motion_proxy", "motion_var",
    "spatial_info", "temporal_info", "edge_density", "color_richness",
    "block_artifact", "noise_level", "hog_consistency", "face_count",
]


def load_combined(konvid_json, vf_json, t2vqa_json):
    rows = []
    for path, ds, regime in [(konvid_json, "konvid", 0),
                              (vf_json, "videofeedback", 1),
                              (t2vqa_json, "t2vqa", 1)]:
        d = json.loads(Path(path).read_text())
        ys = np.array([r["mos"] for r in d["per_video"]])
        ys_z = (ys - ys.mean()) / (ys.std() if ys.std() > 0 else 1)
        for r, y_z in zip(d["per_video"], ys_z):
            feat = []
            for k in EXT_KEYS:
                feat.append(r.get(k, np.nan))
            rows.append({"feat": feat, "y": float(y_z),
                         "regime": regime, "dataset": ds,
                         "video_id": r.get("video_id") or r.get("flickr_id")})
    X = np.array([r["feat"] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows])
    regime = np.array([r["regime"] for r in rows])
    ds_arr = np.array([r["dataset"] for r in rows])
    # Impute NaNs with column means
    means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(means, inds[1])
    return X, y, regime, ds_arr


def evaluate_method(method_name, model_factory, X, y, regime, n_splits=5):
    """5-fold stratified CV."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rhos, prs = [], []
    for tr, te in kf.split(X, regime):
        model, needs_scaling = model_factory()
        if needs_scaling:
            sc = StandardScaler().fit(X[tr])
            model.fit(sc.transform(X[tr]), y[tr])
            pred = model.predict(sc.transform(X[te]))
        else:
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
        if y[te].std() == 0 or pred.std() == 0:
            continue
        r, _ = spearmanr(pred, y[te])
        p, _ = pearsonr(pred, y[te])
        if not np.isnan(r): rhos.append(r); prs.append(p)
    return {
        "method": method_name,
        "n_folds": len(rhos),
        "spearman_mean": float(np.mean(rhos)),
        "spearman_std": float(np.std(rhos)),
        "pearson_mean": float(np.mean(prs)),
        "pearson_std": float(np.std(prs)),
    }


def regime_aware_ridge_cv(X, y, regime, n_splits=5):
    """Regime classifier + 2 ridge heads."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rhos = []; clf_accs = []
    for tr, te in kf.split(X, regime):
        sc = StandardScaler().fit(X[tr])
        Xs_tr, Xs_te = sc.transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, random_state=42).fit(Xs_tr, regime[tr])
        r_pred = clf.predict(Xs_te)
        clf_accs.append(float((r_pred == regime[te]).mean()))
        pred = np.zeros(len(X[te]))
        for r_lbl in [0, 1]:
            mask_tr = regime[tr] == r_lbl
            if mask_tr.sum() < 5: continue
            head = Ridge(alpha=1.0).fit(Xs_tr[mask_tr], y[tr][mask_tr])
            mask_te = r_pred == r_lbl
            if mask_te.sum() > 0:
                pred[mask_te] = head.predict(Xs_te[mask_te])
        if y[te].std() == 0 or pred.std() == 0: continue
        r, _ = spearmanr(pred, y[te])
        if not np.isnan(r): rhos.append(r)
    return {"method": "regime_aware_ridge",
            "spearman_mean": float(np.mean(rhos)),
            "spearman_std": float(np.std(rhos)),
            "clf_acc_mean": float(np.mean(clf_accs))}


def per_dataset_oracle(X, y, ds_arr, ds_name, X_full=None, y_full=None,
                       ds_arr_full=None, n_splits=5):
    """Train Ridge on a single dataset, 5-fold CV within it."""
    mask = ds_arr == ds_name
    if mask.sum() < 25: return None
    Xm, ym = X[mask], y[mask]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rhos = []
    for tr, te in kf.split(Xm):
        sc = StandardScaler().fit(Xm[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(Xm[tr]), ym[tr])
        pred = m.predict(sc.transform(Xm[te]))
        r, _ = spearmanr(pred, ym[te])
        if not np.isnan(r): rhos.append(r)
    return {"dataset": ds_name, "n": int(mask.sum()),
            "spearman_mean": float(np.mean(rhos)),
            "spearman_std": float(np.std(rhos))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--konvid-json", required=True, type=Path)
    ap.add_argument("--videofeedback-json", required=True, type=Path)
    ap.add_argument("--t2vqa-json", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    X, y, regime, ds_arr = load_combined(args.konvid_json, args.videofeedback_json,
                                          args.t2vqa_json)
    print(f"Combined dataset: {len(y)} videos, {X.shape[1]} features")
    print(f"  natural: {(regime==0).sum()}  aigc: {(regime==1).sum()}")
    print(f"  KoNViD: {(ds_arr=='konvid').sum()}  VF: {(ds_arr=='videofeedback').sum()}  T2VQA: {(ds_arr=='t2vqa').sum()}")

    methods = [
        ("Ridge (linear)", lambda: (Ridge(alpha=1.0), True)),
        ("Random Forest", lambda: (RandomForestRegressor(n_estimators=200, max_depth=12,
                                                          random_state=42, n_jobs=1), False)),
        ("Gradient Boosting", lambda: (GradientBoostingRegressor(n_estimators=200,
                                                                   max_depth=4,
                                                                   learning_rate=0.05,
                                                                   random_state=42), False)),
    ]
    print("\n=== Combined dataset (n=862, z-MOS within dataset) ===")
    results = []
    for name, factory in methods:
        r = evaluate_method(name, factory, X, y, regime, n_splits=5)
        results.append(r)
        print(f"  {r['method']:<22}  ρ = {r['spearman_mean']:+.3f} ± {r['spearman_std']:.3f}")

    r = regime_aware_ridge_cv(X, y, regime, n_splits=5)
    results.append(r)
    print(f"  {r['method']:<22}  ρ = {r['spearman_mean']:+.3f} ± {r['spearman_std']:.3f}  (clf_acc={r['clf_acc_mean']:.3f})")

    # Per-dataset oracles
    print("\n=== Per-dataset oracle (Ridge, 5-fold within each dataset) ===")
    oracles = []
    for ds_name in ["konvid", "videofeedback", "t2vqa"]:
        o = per_dataset_oracle(X, y, ds_arr, ds_name)
        if o:
            oracles.append(o)
            print(f"  {ds_name:<14}  n={o['n']:>3}  ρ = {o['spearman_mean']:+.3f} ± {o['spearman_std']:.3f}")

    # Per-dataset RF
    print("\n=== Per-dataset Random Forest oracle (5-fold within each dataset) ===")
    rf_oracles = []
    for ds_name in ["konvid", "videofeedback", "t2vqa"]:
        mask = ds_arr == ds_name
        if mask.sum() < 25: continue
        Xm, ym = X[mask], y[mask]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rhos = []
        for tr, te in kf.split(Xm):
            m = RandomForestRegressor(n_estimators=200, max_depth=12,
                                       random_state=42, n_jobs=1)
            m.fit(Xm[tr], ym[tr])
            pred = m.predict(Xm[te])
            r, _ = spearmanr(pred, ym[te])
            if not np.isnan(r): rhos.append(r)
        rf_oracles.append({"dataset": ds_name, "n": int(mask.sum()),
                            "spearman_mean": float(np.mean(rhos)),
                            "spearman_std": float(np.std(rhos))})
        print(f"  {ds_name:<14}  n={int(mask.sum()):>3}  ρ = {np.mean(rhos):+.3f} ± {np.std(rhos):.3f}")

    # Feature importance from RF on full data
    rf_full = RandomForestRegressor(n_estimators=200, max_depth=12,
                                      random_state=42, n_jobs=1).fit(X, y)
    imp = sorted(zip(EXT_KEYS, rf_full.feature_importances_), key=lambda kv: -kv[1])
    print("\n=== Feature importance (Random Forest, all 862) ===")
    for k, v in imp:
        print(f"  {k:<18}  {v:.4f}")

    out = {
        "n_videos": len(y),
        "n_features": X.shape[1],
        "feature_names": EXT_KEYS,
        "combined_methods": results,
        "per_dataset_oracle_ridge": oracles,
        "per_dataset_oracle_rf": rf_oracles,
        "feature_importance_rf": [{"feature": k, "importance": float(v)} for k, v in imp],
    }
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
