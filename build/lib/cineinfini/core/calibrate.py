"""Threshold calibration utilities (CSV-driven grid + logistic regression)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class CalibrationResult:
    thresholds: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    method: str = "grid"
    n_samples: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thresholds": dict(self.thresholds),
            "weights": dict(self.weights),
            "score": float(self.score),
            "method": self.method,
            "n_samples": int(self.n_samples),
            "extra": dict(self.extra),
        }


def _read_csv(path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    rows: List[List[str]] = []
    with p.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)
    if "label" not in header:
        raise ValueError("CSV must contain a 'label' column (0/1)")
    label_idx = header.index("label")
    feature_names = [h for i, h in enumerate(header) if i != label_idx]
    X = np.array(
        [[float(row[i]) for i in range(len(header)) if i != label_idx] for row in rows],
        dtype=np.float64,
    )
    y = np.array([int(float(row[label_idx])) for row in rows], dtype=np.int64)
    return feature_names, X, y


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def grid_search_thresholds(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
    n_grid: int = 25, higher_is_better: Optional[Dict[str, bool]] = None,
) -> CalibrationResult:
    higher_is_better = higher_is_better or {}
    thresholds: Dict[str, float] = {}
    f1s: Dict[str, float] = {}
    for j, name in enumerate(feature_names):
        col = X[:, j]
        col_finite = col[np.isfinite(col)]
        if col_finite.size == 0:
            continue
        grid = np.linspace(col_finite.min(), col_finite.max(), n_grid)
        higher = higher_is_better.get(name, False)
        best_t, best_f = float(np.nanmedian(col_finite)), 0.0
        for t in grid:
            pred = (col >= t).astype(np.int64) if higher else (col <= t).astype(np.int64)
            f = _f1(y, pred)
            if f > best_f:
                best_f, best_t = f, float(t)
        thresholds[name] = best_t
        f1s[name] = best_f
    return CalibrationResult(
        thresholds=thresholds,
        score=float(np.mean(list(f1s.values()))) if f1s else 0.0,
        method="grid", n_samples=int(len(y)),
        extra={"per_feature_f1": f1s},
    )


def logistic_regression_weights(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
) -> CalibrationResult:
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        clf.fit(X, y)
        weights = {n: float(c) for n, c in zip(feature_names, clf.coef_[0])}
        return CalibrationResult(
            weights=weights, score=float(clf.score(X, y)),
            method="logreg-sklearn", n_samples=int(len(y)),
            extra={"intercept": float(clf.intercept_[0])},
        )
    except Exception:
        pass
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Xa = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
    w = np.zeros(Xa.shape[1])
    for _ in range(500):
        z = Xa @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xa.T @ (p - y) / max(1, len(y))
        w -= 0.1 * grad
    weights = {n: float(w[i]) for i, n in enumerate(feature_names)}
    pred = (1.0 / (1.0 + np.exp(-(Xa @ w))) >= 0.5).astype(np.int64)
    return CalibrationResult(
        weights=weights, score=float((pred == y).mean()),
        method="logreg-gd", n_samples=int(len(y)),
        extra={"intercept": float(w[-1])},
    )


def bayesian_optimize_thresholds(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
    n_calls: int = 30, higher_is_better: Optional[Dict[str, bool]] = None,
) -> CalibrationResult:
    try:
        from skopt import gp_minimize
        from skopt.space import Real
        higher_is_better = higher_is_better or {}
        bounds = []
        for j in range(X.shape[1]):
            col = X[:, j][np.isfinite(X[:, j])]
            bounds.append(Real(float(col.min()), float(col.max())) if col.size else Real(0.0, 1.0))

        def neg_f1(ts):
            preds = []
            for j, t in enumerate(ts):
                higher = higher_is_better.get(feature_names[j], False)
                preds.append((X[:, j] >= t) if higher else (X[:, j] <= t))
            return -_f1(y, np.all(preds, axis=0).astype(np.int64))

        res = gp_minimize(neg_f1, bounds, n_calls=n_calls, random_state=42)
        return CalibrationResult(
            thresholds={n: float(v) for n, v in zip(feature_names, res.x)},
            score=float(-res.fun), method="bayes-skopt", n_samples=int(len(y)),
        )
    except Exception:
        return grid_search_thresholds(
            X, y, feature_names, n_grid=max(50, n_calls),
            higher_is_better=higher_is_better,
        )


def calibrate_from_csv(
    path, method: str = "grid",
    higher_is_better: Optional[Dict[str, bool]] = None,
) -> CalibrationResult:
    feature_names, X, y = _read_csv(path)
    if method == "logreg":
        return logistic_regression_weights(X, y, feature_names)
    if method == "bayes":
        return bayesian_optimize_thresholds(X, y, feature_names, higher_is_better=higher_is_better)
    return grid_search_thresholds(X, y, feature_names, higher_is_better=higher_is_better)


__all__ = [
    "CalibrationResult", "calibrate_from_csv", "grid_search_thresholds",
    "logistic_regression_weights", "bayesian_optimize_thresholds",
]
