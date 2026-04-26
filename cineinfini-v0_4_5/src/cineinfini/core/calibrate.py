"""
Threshold calibration for CineInfini metrics (added in v0.4.0).

Why this matters
----------------
The default thresholds (motion=25, ssim3d=0.45, flicker=0.1, …) were set
empirically on three reference videos (BBB, Sintel, Tears of Steel) with no
perceptual ground truth. For a publishable result, thresholds must be
validated against human judgement.

Three calibration methods are provided, in increasing scientific rigour:

1. ``grid_search_thresholds``  — fast, no deps beyond sklearn.
   Sweeps each threshold independently to maximise accuracy on a labelled
   dataset. Suitable for a quick first pass.

2. ``logistic_regression_weights``  — moderate, requires sklearn.
   Learns a single probability model from all metrics jointly. More powerful
   than independent grid search because it captures interactions between
   metrics. Recommended for workshop papers.

3. ``bayesian_optimize_thresholds``  — requires optuna.
   Joint optimisation of all thresholds simultaneously. Most rigorous.
   Use when you have ≥ 200 labelled examples.

Input format
------------
The calibration functions expect a ``pandas.DataFrame`` (or CSV file) with:

    video_name  shot_id  motion  ssim3d  flicker  identity  label
    bbb         1        12.3    0.52    0.05     0.32      ACCEPT
    bbb         2        28.7    0.38    0.12     0.71      REJECT
    …

``label`` must be "ACCEPT" or "REJECT" (case-insensitive).

Usage
-----
    # From CSV
    from cineinfini.core.calibrate import calibrate_from_csv
    result = calibrate_from_csv("annotations.csv", method="logistic")
    print(result.thresholds)
    result.save("~/.cineinfini/thresholds_calibrated.yaml")

    # CLI
    cineinfini calibrate --annotations annotations.csv \
                         --method logistic \
                         --output thresholds.yaml
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# Metric columns, direction ("below" = low values are good), and default threshold
METRIC_CONFIG: list[tuple[str, str, float]] = [
    ("motion",           "below", 25.0),
    ("ssim3d",           "above", 0.45),
    ("flicker",          "below", 0.10),
    ("identity_drift",   "below", 0.60),
    ("ssim_long_range",  "above", 0.45),
    ("clip_temp",        "above", 0.25),
    ("flicker_hf",       "below", 0.01),
]

METRIC_COLUMN_ALIASES: dict[str, str] = {
    "motion_peak_div":       "motion",
    "ssim3d_self":           "ssim3d",
    "identity_intra":        "identity_drift",
    "ssim_long_range":       "ssim_long_range",
    "clip_temp_consistency": "clip_temp",
    "flicker_hf_var":        "flicker_hf",
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    method: str
    thresholds: dict[str, float]
    weights: Optional[dict[str, float]] = None   # for logistic
    metrics: dict[str, Any] = field(default_factory=dict)  # accuracy, AUC, etc.

    def summary(self) -> str:
        lines = [f"CalibrationResult (method={self.method})"]
        lines.append("  Thresholds:")
        for k, v in self.thresholds.items():
            lines.append(f"    {k:<22}: {v:.4f}")
        if self.metrics:
            lines.append("  Metrics:")
            for k, v in self.metrics.items():
                lines.append(f"    {k:<22}: {v}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save thresholds to YAML."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pip install pyyaml")
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": self.method,
            "thresholds": {k: float(v) for k, v in self.thresholds.items()},
            "metrics": self.metrics,
        }
        if self.weights:
            payload["weights"] = {k: float(v) for k, v in self.weights.items()}
        with open(p, "w") as f:
            yaml.dump(payload, f, default_flow_style=False, sort_keys=True)
        print(f"Saved calibration to {p}")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_annotations(path: str | Path) -> "pd.DataFrame":
    """Load an annotations CSV and normalise column names."""
    import pandas as pd
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=METRIC_COLUMN_ALIASES)
    if "label" not in df.columns:
        raise ValueError("CSV must have a 'label' column (ACCEPT/REJECT)")
    df["label_bin"] = (df["label"].str.upper() == "ACCEPT").astype(int)
    return df


def _feature_matrix(df, metric_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays, dropping rows with any NaN in the metric columns."""
    available = [m for m in metric_names if m in df.columns]
    if not available:
        raise ValueError(f"None of {metric_names} found in dataframe")
    sub = df[available + ["label_bin"]].dropna()
    X = sub[available].values
    y = sub["label_bin"].values
    return X, y, available


# ---------------------------------------------------------------------------
# Method 1: independent grid search
# ---------------------------------------------------------------------------

def grid_search_thresholds(
    df,
    metric_configs: list[tuple[str, str, float]] | None = None,
    n_steps: int = 50,
    metric: str = "youden",
) -> CalibrationResult:
    """Find optimal threshold for each metric independently.

    For each metric, sweeps ``n_steps`` values between the 5th and 95th
    percentile of observed values and finds the threshold that maximises
    the Youden index (sensitivity + specificity - 1) or accuracy.

    Parameters
    ----------
    df : pd.DataFrame
        Labelled dataset from ``load_annotations``.
    metric_configs : list of (col, direction, default), optional
        Defaults to METRIC_CONFIG.
    n_steps : int
        Number of threshold candidates per metric.
    metric : "youden" | "accuracy" | "f1"

    Returns
    -------
    CalibrationResult
    """
    from sklearn.metrics import roc_curve, f1_score, accuracy_score
    if metric_configs is None:
        metric_configs = METRIC_CONFIG
    y = df["label_bin"].values
    thresholds: dict[str, float] = {}
    aucs: dict[str, float] = {}

    for col, direction, default in metric_configs:
        if col not in df.columns:
            thresholds[col] = default
            continue
        vals = df[col].dropna().values
        y_clean = df.loc[df[col].notna(), "label_bin"].values

        # For ROC, "good" class (label=1) should correspond to:
        # - low values if direction=="below" → negate
        # - high values if direction=="above" → use as-is
        scores = -vals if direction == "below" else vals

        try:
            from sklearn.metrics import roc_auc_score
            fpr, tpr, ths = roc_curve(y_clean, scores)
            youden = tpr - fpr
            best_idx = int(np.argmax(youden))
            raw_threshold = ths[best_idx]
            # Convert back to metric space
            opt_thresh = -raw_threshold if direction == "below" else raw_threshold
            aucs[col] = float(roc_auc_score(y_clean, scores))
        except Exception:
            opt_thresh = default
        thresholds[col] = round(float(opt_thresh), 4)

    return CalibrationResult(
        method="grid_search",
        thresholds=thresholds,
        metrics={"per_metric_auc": aucs},
    )


# ---------------------------------------------------------------------------
# Method 2: logistic regression (joint)
# ---------------------------------------------------------------------------

def logistic_regression_weights(
    df,
    metric_configs: list[tuple[str, str, float]] | None = None,
    prob_threshold: float = 0.5,
    cv_folds: int = 5,
) -> CalibrationResult:
    """Learn a joint probability model from all metrics.

    Fits a logistic regression on all available metric columns. The
    ``prob_threshold`` (default 0.5) converts predicted probability to
    ACCEPT/REJECT. Cross-validation accuracy is reported.

    Returns a CalibrationResult with ``weights`` (logistic coefficients)
    and a representative per-metric threshold derived from the model.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score

    if metric_configs is None:
        metric_configs = METRIC_CONFIG
    cols = [c for c, _, _ in metric_configs if c in df.columns]
    X, y, used = _feature_matrix(df, cols)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
    ])
    pipe.fit(X, y)
    cv_acc = cross_val_score(pipe, X, y, cv=cv_folds, scoring="accuracy").mean()
    proba = pipe.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, proba))

    lr = pipe.named_steps["lr"]
    scaler = pipe.named_steps["scaler"]
    coefs = lr.coef_[0]
    weights = {col: float(coef) for col, coef in zip(used, coefs)}

    # Derive representative thresholds (value where model gives prob=0.5
    # when other metrics are at their median)
    medians = np.median(X, axis=0)
    derived_thresholds: dict[str, float] = {}
    for i, col in enumerate(used):
        # Vary col while fixing others at median; find where prob=0.5
        try:
            from scipy.optimize import brentq
            col_scaled_offset = (medians[i] - scaler.mean_[i]) / scaler.scale_[i]

            def prob_at(raw_val):
                x_test = medians.copy()
                x_test[i] = raw_val
                x_scaled = (x_test - scaler.mean_) / scaler.scale_
                log_odds = lr.intercept_[0] + np.dot(lr.coef_[0], x_scaled)
                return 1 / (1 + np.exp(-log_odds)) - prob_threshold

            lo, hi = float(X[:, i].min()), float(X[:, i].max())
            if prob_at(lo) * prob_at(hi) < 0:
                opt = brentq(prob_at, lo, hi)
                derived_thresholds[col] = round(opt, 4)
            else:
                derived_thresholds[col] = round(float(medians[i]), 4)
        except Exception:
            derived_thresholds[col] = round(float(medians[i]), 4)

    return CalibrationResult(
        method="logistic",
        thresholds=derived_thresholds,
        weights=weights,
        metrics={"cv_accuracy": round(cv_acc, 4), "auc": round(auc, 4),
                 "n_samples": int(len(y)), "features": used},
    )


# ---------------------------------------------------------------------------
# Method 3: Bayesian optimisation (optuna)
# ---------------------------------------------------------------------------

def bayesian_optimize_thresholds(
    df,
    metric_configs: list[tuple[str, str, float]] | None = None,
    n_trials: int = 100,
    objective: str = "accuracy",
) -> CalibrationResult:
    """Joint Bayesian optimisation of all thresholds via Optuna.

    Parameters
    ----------
    df : pd.DataFrame
    metric_configs : list, optional
    n_trials : int
        Number of Optuna trials (100 is usually enough for ≤10 metrics).
    objective : "accuracy" | "f1" | "youden"

    Returns
    -------
    CalibrationResult
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("pip install optuna")
    from sklearn.metrics import accuracy_score, f1_score

    if metric_configs is None:
        metric_configs = METRIC_CONFIG
    available = [(c, d, t) for c, d, t in metric_configs if c in df.columns]
    y = df["label_bin"].values

    def _verdict(row, trial_thresholds: dict) -> int:
        violations = 0
        for col, direction, _ in available:
            v = row.get(col)
            if v is None or np.isnan(float(v)):
                continue
            th = trial_thresholds[col]
            if direction == "below" and float(v) > th:
                violations += 1
            elif direction == "above" and float(v) < th:
                violations += 1
        return 1 if violations == 0 else 0

    def optuna_objective(trial):
        ths: dict[str, float] = {}
        for col, direction, default in available:
            vals = df[col].dropna().values
            lo, hi = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
            ths[col] = trial.suggest_float(col, lo, hi)
        preds = [_verdict(row, ths) for _, row in df[
            [c for c, _, _ in available] + ["label_bin"]
        ].iterrows()]
        if objective == "f1":
            return f1_score(y[:len(preds)], preds, zero_division=0)
        return accuracy_score(y[:len(preds)], preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    acc = study.best_value

    return CalibrationResult(
        method="bayesian",
        thresholds={col: round(best[col], 4) for col, _, _ in available},
        metrics={"best_score": round(acc, 4), "objective": objective,
                 "n_trials": n_trials},
    )


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def calibrate_from_csv(
    annotations_csv: str | Path,
    method: str = "logistic",
    **kwargs,
) -> CalibrationResult:
    """One-stop calibration from a CSV file.

    Parameters
    ----------
    annotations_csv : str or Path
    method : "grid" | "logistic" | "bayesian"
    **kwargs : forwarded to the underlying calibration function.
    """
    df = load_annotations(annotations_csv)
    print(f"Loaded {len(df)} rows from {annotations_csv}")
    print(f"  ACCEPT: {df['label_bin'].sum()}  REJECT: {(df['label_bin']==0).sum()}")

    if method in ("grid", "grid_search"):
        result = grid_search_thresholds(df, **kwargs)
    elif method in ("logistic", "lr"):
        result = logistic_regression_weights(df, **kwargs)
    elif method in ("bayesian", "optuna"):
        result = bayesian_optimize_thresholds(df, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose: grid | logistic | bayesian")

    print(result.summary())
    return result
