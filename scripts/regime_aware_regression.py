#!/usr/bin/env python3
"""Regime-aware dual-head quality regression.

The motivation. Pure-CV photometric metrics correlate with MOS
*differently* across regimes:

    metric        KoNViD (natural)   VideoFeedback (AIGC)   T2VQA-DB (AIGC)
    sharpness     +0.394             +0.369                  -0.218
    brightness    +0.407             -0.402                  -0.009
    saturation    -0.269             +0.336                  -0.082

A single linear regressor cannot accommodate these sign flips. We
propose a **two-stage estimator**:

    1. A regime classifier  c: x -> {natural, aigc}  predicts the
       regime from the same pure-CV features.
    2. Two specialized regressors r_natural, r_aigc : x -> mos
       are trained, and dispatched by c.

We compare against three baselines:

    a. **Single Ridge** : one Ridge regression on all data (regime-
       blind).
    b. **Per-dataset Ridge** : oracle, trains separately on each
       dataset.
    c. **Random Forest** : non-linear baseline that should auto-
       discover regime splits.

Output:
    out/regime_aware_results.json — full numerical results
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

METRIC_KEYS = ["sharpness", "brightness", "saturation", "contrast",
               "flicker", "flicker_var", "motion_proxy", "motion_var"]

# Backfill aliases for the older VideoFeedback JSON
ALIASES = {
    "motion_proxy": "motion_magnitude",
    "flicker_var": "flicker_variance",
    "motion_var": None,  # not available in old VF
}


def load_videos_with_regime(json_path: Path, regime_label: str) -> list[dict]:
    """Return list of {features: dict, mos: float, regime: str, dataset: str}."""
    data = json.loads(json_path.read_text())
    rows = []
    for v in data["per_video"]:
        feats = {}
        for k in METRIC_KEYS:
            if k in v:
                feats[k] = v[k]
            elif ALIASES.get(k) and ALIASES[k] in v:
                feats[k] = v[ALIASES[k]]
            else:
                feats[k] = None
        rows.append({
            "features": feats,
            "mos": v["mos"],
            "regime": regime_label,
            "dataset": data.get("dataset", json_path.stem),
        })
    return rows


def to_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, regime) where regime is 0=natural, 1=aigc."""
    X = np.array([[r["features"].get(k, np.nan) for k in METRIC_KEYS]
                  for r in rows], dtype=float)
    y = np.array([r["mos"] for r in rows], dtype=float)
    regime = np.array([0 if r["regime"] == "natural" else 1 for r in rows])
    # Replace NaNs with column means (only an issue for VF's missing motion_var)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    return X, y, regime


def ridge_per_split(X_tr, y_tr, X_te, y_te) -> tuple[float, float]:
    sc = StandardScaler().fit(X_tr)
    model = Ridge(alpha=1.0).fit(sc.transform(X_tr), y_tr)
    pred = model.predict(sc.transform(X_te))
    if y_te.std() == 0 or pred.std() == 0:
        return float("nan"), float("nan")
    rho, _ = spearmanr(pred, y_te)
    pr, _ = pearsonr(pred, y_te)
    return float(rho), float(pr)


def rf_per_split(X_tr, y_tr, X_te, y_te) -> tuple[float, float]:
    model = RandomForestRegressor(n_estimators=100, max_depth=8,
                                  random_state=42, n_jobs=1).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    if y_te.std() == 0 or pred.std() == 0:
        return float("nan"), float("nan")
    rho, _ = spearmanr(pred, y_te)
    pr, _ = pearsonr(pred, y_te)
    return float(rho), float(pr)


def regime_aware_per_split(X_tr, y_tr, r_tr, X_te, y_te, r_te) \
        -> tuple[float, float, dict]:
    """Train regime classifier + 2 regression heads, predict, score."""
    # Standardize features for both models
    sc = StandardScaler().fit(X_tr)
    Xs_tr = sc.transform(X_tr)
    Xs_te = sc.transform(X_te)

    # Stage 1: regime classifier on train
    clf = LogisticRegression(max_iter=2000, random_state=42).fit(Xs_tr, r_tr)
    r_pred_te = clf.predict(Xs_te)
    clf_acc = float((r_pred_te == r_te).mean())

    # Stage 2: per-regime regressors trained on train data of that regime
    pred_te = np.zeros(len(X_te))
    for r_label in [0, 1]:
        mask_tr = r_tr == r_label
        if mask_tr.sum() < 5:
            # Fallback: use global ridge for this regime
            global_ridge = Ridge(alpha=1.0).fit(Xs_tr, y_tr)
            mask_te = r_pred_te == r_label
            if mask_te.sum() > 0:
                pred_te[mask_te] = global_ridge.predict(Xs_te[mask_te])
            continue
        head = Ridge(alpha=1.0).fit(Xs_tr[mask_tr], y_tr[mask_tr])
        # Apply head to test rows whose predicted regime matches
        mask_te = r_pred_te == r_label
        if mask_te.sum() > 0:
            pred_te[mask_te] = head.predict(Xs_te[mask_te])

    if y_te.std() == 0 or pred_te.std() == 0:
        return float("nan"), float("nan"), {"clf_acc": clf_acc}
    rho, _ = spearmanr(pred_te, y_te)
    pr, _ = pearsonr(pred_te, y_te)
    return float(rho), float(pr), {"clf_acc": clf_acc}


def cross_validate_all(rows: list[dict], n_splits: int = 5) -> dict:
    X, y, regime = to_matrix(rows)
    n = len(rows)
    print(f"\nDataset combined: n={n}  natural={int((regime==0).sum())}  "
          f"aigc={int((regime==1).sum())}")
    print(f"MOS range per regime:")
    for r_lbl, name in [(0, "natural"), (1, "aigc")]:
        mask = regime == r_lbl
        if mask.sum():
            print(f"  {name}: [{y[mask].min():.2f}, {y[mask].max():.2f}]")

    # Stratified by regime so each fold has both regimes
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {
        "single_ridge": {"spearman": [], "pearson": []},
        "rf_blind": {"spearman": [], "pearson": []},
        "regime_aware": {"spearman": [], "pearson": [], "clf_acc": []},
        "ridge_natural_only": {"spearman": [], "pearson": []},
        "ridge_aigc_only": {"spearman": [], "pearson": []},
    }

    for fold_id, (tr, te) in enumerate(kf.split(X, regime)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        r_tr, r_te = regime[tr], regime[te]

        rho, pr = ridge_per_split(X_tr, y_tr, X_te, y_te)
        results["single_ridge"]["spearman"].append(rho)
        results["single_ridge"]["pearson"].append(pr)

        rho, pr = rf_per_split(X_tr, y_tr, X_te, y_te)
        results["rf_blind"]["spearman"].append(rho)
        results["rf_blind"]["pearson"].append(pr)

        rho, pr, info = regime_aware_per_split(X_tr, y_tr, r_tr, X_te, y_te, r_te)
        results["regime_aware"]["spearman"].append(rho)
        results["regime_aware"]["pearson"].append(pr)
        results["regime_aware"]["clf_acc"].append(info["clf_acc"])

        # Per-regime ridge — only score on test rows of matching regime
        nat_tr = r_tr == 0
        if nat_tr.sum() >= 5:
            sc = StandardScaler().fit(X_tr[nat_tr])
            head = Ridge(alpha=1.0).fit(sc.transform(X_tr[nat_tr]), y_tr[nat_tr])
            nat_te = r_te == 0
            if nat_te.sum() >= 3:
                pred = head.predict(sc.transform(X_te[nat_te]))
                if y_te[nat_te].std() > 0 and pred.std() > 0:
                    rho, _ = spearmanr(pred, y_te[nat_te])
                    pr, _ = pearsonr(pred, y_te[nat_te])
                    results["ridge_natural_only"]["spearman"].append(float(rho))
                    results["ridge_natural_only"]["pearson"].append(float(pr))

        aigc_tr = r_tr == 1
        if aigc_tr.sum() >= 5:
            sc = StandardScaler().fit(X_tr[aigc_tr])
            head = Ridge(alpha=1.0).fit(sc.transform(X_tr[aigc_tr]), y_tr[aigc_tr])
            aigc_te = r_te == 1
            if aigc_te.sum() >= 3:
                pred = head.predict(sc.transform(X_te[aigc_te]))
                if y_te[aigc_te].std() > 0 and pred.std() > 0:
                    rho, _ = spearmanr(pred, y_te[aigc_te])
                    pr, _ = pearsonr(pred, y_te[aigc_te])
                    results["ridge_aigc_only"]["spearman"].append(float(rho))
                    results["ridge_aigc_only"]["pearson"].append(float(pr))

    # Aggregate
    summary = {}
    for method, vals in results.items():
        rhos = [r for r in vals["spearman"] if r is not None and not np.isnan(r)]
        prs = [r for r in vals["pearson"] if r is not None and not np.isnan(r)]
        s = {
            "n_folds": len(rhos),
            "spearman_mean": float(np.mean(rhos)) if rhos else None,
            "spearman_std": float(np.std(rhos)) if rhos else None,
            "spearman_per_fold": rhos,
            "pearson_mean": float(np.mean(prs)) if prs else None,
            "pearson_std": float(np.std(prs)) if prs else None,
        }
        if method == "regime_aware" and "clf_acc" in vals:
            accs = vals["clf_acc"]
            s["clf_acc_mean"] = float(np.mean(accs))
            s["clf_acc_std"] = float(np.std(accs))
        summary[method] = s
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--konvid-json", required=True, type=Path)
    ap.add_argument("--videofeedback-json", required=True, type=Path)
    ap.add_argument("--t2vqa-json", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    rows = []
    rows += load_videos_with_regime(args.konvid_json, "natural")
    rows += load_videos_with_regime(args.videofeedback_json, "aigc")
    rows += load_videos_with_regime(args.t2vqa_json, "aigc")
    print(f"Loaded {len(rows)} videos total")

    # Note: MOS scales differ across datasets (KoNViD [1-5], VF [1-4], T2VQA [0-100]).
    # We z-normalize MOS within each dataset to make them commensurable.
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds, rs in by_ds.items():
        ys = np.array([x["mos"] for x in rs])
        m, s = ys.mean(), ys.std()
        for x in rs:
            x["mos"] = (x["mos"] - m) / (s if s > 0 else 1.0)
        print(f"  {ds}: n={len(rs)}, MOS z-normalized")

    summary = cross_validate_all(rows, n_splits=args.n_folds)

    # Console report
    print("\n" + "=" * 78)
    print("CROSS-VALIDATED SPEARMAN ρ (5-fold, stratified by regime)")
    print("=" * 78)
    print(f"{'method':<22} {'n_folds':>8} {'ρ (mean ± std)':>22} {'P (mean)':>12}")
    for method, s in summary.items():
        if s["spearman_mean"] is None:
            continue
        print(f"  {method:<20}  {s['n_folds']:>5}     "
              f"{s['spearman_mean']:+.3f} ± {s['spearman_std']:.3f}     "
              f"{s.get('pearson_mean', 0):+.3f}")
        if "clf_acc_mean" in s:
            print(f"      regime classifier accuracy: "
                  f"{s['clf_acc_mean']:.3f} ± {s['clf_acc_std']:.3f}")

    out = {"summary": summary, "n_total": len(rows), "n_folds": args.n_folds}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
