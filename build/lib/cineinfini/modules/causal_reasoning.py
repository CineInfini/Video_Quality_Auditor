"""Causal reasoning module: gravity / vertical-motion plausibility.

The single most distinctive failure mode of AI-generated videos is the
violation of basic physics — objects floating, falling sideways,
trajectories that curve the wrong way. This module catches the most
robust subset: **persistent unjustified upward motion**.

Three complementary sub-metrics (all in [0, 1], higher = more suspicious):

* ``upward_flow_ratio``: among pixels with significant motion, what
  fraction is moving up? In a typical real shot with normal subjects,
  gravity drags moving things down, so this ratio sits around 0.4–0.5.
  When it's persistently 0.7+ across a shot, something is rising without
  a visible cause.
* ``vertical_flow_imbalance``: ratio of total upward flow magnitude to
  total downward flow magnitude. Mirrors the above in a continuous form.
* ``trajectory_curvature_violation``: tracks the centroid of the moving
  mask across consecutive frames; a free-falling object should accelerate
  *downward* (positive second derivative on y in image coords). When the
  centroid persistently accelerates upward, that's a gravity violation.

The composite ``causal_violation`` is the weighted average. It crosses
the configured threshold (``cfg.thresholds.causal_violation``, default
0.35) when motion is consistently unphysical.

This module is **disabled by default**. Enable it via::

    modules:
      causal_reasoning:
        enabled: true

All inputs come from ``get_config()`` — no hardcoded values.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.causal_reasoning")

MOD_ID = "causal_reasoning"
VERSION = "0.4.8.1.3"


# ---------------------------------------------------------------------------
# Optical-flow helpers
# ---------------------------------------------------------------------------
def _to_gray(frame: np.ndarray) -> np.ndarray:
    import cv2
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _compute_flow(prev_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
    """Dense Farnebäck optical flow. Returns HxWx2 array (dx, dy)."""
    import cv2
    return cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


# ---------------------------------------------------------------------------
# Sub-metrics
# ---------------------------------------------------------------------------
def upward_flow_ratio(flow: np.ndarray, motion_threshold: float = 0.5) -> Optional[float]:
    """Fraction of significantly-moving pixels with upward (negative-y) flow."""
    if flow is None or flow.size == 0:
        return None
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)
    moving = mag > motion_threshold
    if moving.sum() < 50:
        return None
    # Image coords: y grows downward, so dy < 0 means upward motion
    upward = (dy < 0) & moving
    return float(upward.sum() / moving.sum())


def vertical_flow_imbalance(flow: np.ndarray, motion_threshold: float = 0.5) -> Optional[float]:
    """Ratio (in [0,1]) of upward flow magnitude to total vertical flow."""
    if flow is None or flow.size == 0:
        return None
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)
    moving = mag > motion_threshold
    if moving.sum() < 50:
        return None
    up_mag = float(np.abs(dy[moving & (dy < 0)]).sum())
    down_mag = float(np.abs(dy[moving & (dy > 0)]).sum())
    total = up_mag + down_mag
    if total < 1e-6:
        return None
    return up_mag / total


def _moving_centroid(flow: np.ndarray, motion_threshold: float = 0.5) -> Optional[Tuple[float, float]]:
    """Return (cx, cy) of the moving region in pixel coordinates."""
    if flow is None or flow.size == 0:
        return None
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    moving = mag > motion_threshold
    if moving.sum() < 50:
        return None
    ys, xs = np.indices(moving.shape)
    cx = float(xs[moving].mean())
    cy = float(ys[moving].mean())
    return cx, cy


def trajectory_curvature_violation(
    centroid_history: List[Tuple[float, float]],
) -> Optional[float]:
    """Score in [0,1]: if the centroid persistently accelerates upward
    (second derivative of y is negative more often than positive), this
    metric returns a violation score equal to that fraction.

    Free-falling objects in image coords have positive y-acceleration
    (downward). Anti-gravity = negative y-acceleration over time.
    """
    if len(centroid_history) < 3:
        return None
    ys = np.array([c[1] for c in centroid_history], dtype=np.float64)
    # Second derivative
    accel = np.diff(ys, n=2)
    if accel.size == 0:
        return None
    upward_accel = (accel < -1.0).sum()  # > 1px/frame² consistent upward accel
    total = accel.size
    return float(upward_accel / total) if total > 0 else None


# ---------------------------------------------------------------------------
# Per-shot aggregator
# ---------------------------------------------------------------------------
def score_shot(
    frames: List[np.ndarray],
    *,
    motion_threshold: float = 0.5,
    upward_weight: float = 0.4,
    imbalance_weight: float = 0.4,
    curvature_weight: float = 0.2,
    max_pairs: int = 30,
) -> Dict[str, Any]:
    """Compute causal-violation metrics over a shot."""
    if len(frames) < 3:
        return {
            "causal_violation": None,
            "upward_flow_ratio": None,
            "vertical_flow_imbalance": None,
            "trajectory_curvature_violation": None,
            "n_pairs": 0,
        }

    grays = [_to_gray(f) for f in frames]
    n_pairs = min(len(grays) - 1, max_pairs)
    step = max(1, (len(grays) - 1) // n_pairs)

    upward_ratios: List[float] = []
    imbalances: List[float] = []
    centroids: List[Tuple[float, float]] = []

    for i in range(0, len(grays) - 1, step):
        try:
            flow = _compute_flow(grays[i], grays[i + 1])
        except Exception as e:  # noqa: BLE001
            logger.debug("flow failed at pair %d: %s", i, e)
            continue
        u = upward_flow_ratio(flow, motion_threshold)
        v = vertical_flow_imbalance(flow, motion_threshold)
        c = _moving_centroid(flow, motion_threshold)
        if u is not None:
            upward_ratios.append(u)
        if v is not None:
            imbalances.append(v)
        if c is not None:
            centroids.append(c)

    upward_mean = float(np.mean(upward_ratios)) if upward_ratios else None
    imbalance_mean = float(np.mean(imbalances)) if imbalances else None
    curv = trajectory_curvature_violation(centroids)

    components: List[Tuple[float, float]] = []
    if upward_mean is not None:
        # Map: 0.5 (balanced) -> 0; 1.0 (all upward) -> 1
        components.append((float(upward_weight), max(0.0, 2 * (upward_mean - 0.5))))
    if imbalance_mean is not None:
        components.append((float(imbalance_weight), max(0.0, 2 * (imbalance_mean - 0.5))))
    if curv is not None:
        components.append((float(curvature_weight), curv))

    if components:
        wsum = sum(w for w, _ in components)
        composite = (
            float(sum(w * v for w, v in components) / wsum)
            if wsum > 1e-9 else 0.0
        )
    else:
        composite = None

    return {
        "causal_violation": composite,
        "upward_flow_ratio": upward_mean,
        "vertical_flow_imbalance": imbalance_mean,
        "trajectory_curvature_violation": curv,
        "n_pairs": len(upward_ratios),
    }


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------
@register_module(
    MOD_ID,
    requires=[],
    description="Causal-physics violation: anti-gravity / unphysical vertical motion.",
    version=VERSION,
)
def run(context: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    motion_threshold = float(mod_cfg.get("motion_threshold", 0.5))
    upward_weight = float(mod_cfg.get("upward_weight", 0.4))
    imbalance_weight = float(mod_cfg.get("imbalance_weight", 0.4))
    curvature_weight = float(mod_cfg.get("curvature_weight", 0.2))
    max_pairs = int(mod_cfg.get("max_pairs", 30))
    threshold = float(cfg.thresholds.get("causal_violation", 0.35))

    per_shot: Dict[int, Dict[str, Any]] = {}
    for sid, frames in context.shot_frames.items():
        per_shot[sid] = score_shot(
            frames,
            motion_threshold=motion_threshold,
            upward_weight=upward_weight,
            imbalance_weight=imbalance_weight,
            curvature_weight=curvature_weight,
            max_pairs=max_pairs,
        )

    violations = [s["causal_violation"]
                  for s in per_shot.values()
                  if s["causal_violation"] is not None]
    summary = {
        "n_shots": len(per_shot),
        "mean_violation": float(np.mean(violations)) if violations else None,
        "max_violation": float(np.max(violations)) if violations else None,
        "n_above_threshold": sum(1 for v in violations if v >= threshold),
    }
    return {
        "module": MOD_ID,
        "version": VERSION,
        "threshold": threshold,
        "weights": {
            "upward": upward_weight,
            "imbalance": imbalance_weight,
            "curvature": curvature_weight,
        },
        "per_shot": per_shot,
        "summary": summary,
    }


__all__ = [
    "upward_flow_ratio", "vertical_flow_imbalance",
    "trajectory_curvature_violation", "score_shot",
    "run", "MOD_ID", "VERSION",
]
