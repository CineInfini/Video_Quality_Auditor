"""Aesthetic / cinematic scoring (optional, disabled by default).

Three visually-meaningful sub-scores combined into one composite per shot:

* **Rule-of-thirds**: detect salient regions via Sobel-edge density, measure
  how close the dominant region's centroid is to one of the four
  rule-of-thirds intersections (the closer, the better).
* **Color harmony**: HSV hue histogram → score how well the dominant hues
  match a known harmony scheme (monochrome / analogous / complementary /
  triadic). Closer to a scheme = higher score.
* **Contrast**: standard deviation of the luminance channel, normalised
  by the theoretical max (≈64 for 8-bit luma). Punishes flat shots.

Each sub-score is in [0, 1]; the composite is a weighted average using
weights from ``cfg.modules.aesthetic_cinematic``. All inputs are read
through ``get_config()`` — no hardcoded values.

This module is **disabled by default**. Enable it by setting::

    modules:
      aesthetic_cinematic:
        enabled: true

in your ``cfg/config.yaml``, or programmatically via
``get_config().modules['aesthetic_cinematic']['enabled'] = True``.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.aesthetic_cinematic")

MOD_ID = "aesthetic_cinematic"
VERSION = "0.4.8.1.2"

# Reference hue offsets (in degrees) that define each known harmony scheme.
# Compared against the gaps between the top-K dominant hues in the frame.
_HARMONY_TARGETS: Dict[str, Tuple[float, ...]] = {
    "monochrome":     (0.0,),
    "analogous":      (30.0,),
    "complementary":  (180.0,),
    "split_complementary": (150.0, 210.0),
    "triadic":        (120.0, 240.0),
    "tetradic":       (90.0, 180.0, 270.0),
}


# ---------------------------------------------------------------------------
# Sub-scorers
# ---------------------------------------------------------------------------
def _to_gray(frame: np.ndarray) -> np.ndarray:
    import cv2
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def rule_of_thirds_score(frame: np.ndarray) -> float:
    """Score in [0, 1]: distance of the salient centroid to the nearest
    rule-of-thirds intersection, normalised by the frame diagonal/4."""
    import cv2
    gray = _to_gray(frame).astype(np.float32)
    h, w = gray.shape[:2]
    if h < 4 or w < 4:
        return 0.0
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    saliency = np.abs(sx) + np.abs(sy)
    total = float(saliency.sum())
    if total < 1e-6:
        return 0.0
    ys, xs = np.indices(saliency.shape)
    cx = float((xs * saliency).sum() / total)
    cy = float((ys * saliency).sum() / total)
    intersections = [
        (w / 3, h / 3), (2 * w / 3, h / 3),
        (w / 3, 2 * h / 3), (2 * w / 3, 2 * h / 3),
    ]
    dist = min(math.hypot(cx - ix, cy - iy) for ix, iy in intersections)
    # Normalise: a centroid >= w/4 from any intersection -> score 0
    norm = math.hypot(w, h) / 4.0
    return float(max(0.0, 1.0 - dist / norm))


def color_harmony_score(frame: np.ndarray, top_k: int = 3,
                        bins: int = 36) -> Tuple[float, str]:
    """Score in [0, 1]: how well the top-K dominant hues match a known
    harmony scheme. Returns (score, best_scheme_name)."""
    import cv2
    if frame.ndim != 3 or frame.shape[2] != 3:
        return 0.0, "none"
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (s_chan > 30) & (v_chan > 30)  # ignore pure greys / blacks
    if mask.sum() < 100:
        return 0.0, "none"
    # OpenCV hue is 0-179 -> rescale to 0-360
    hues = (h_chan[mask].astype(np.float32) * 2.0)
    hist, edges = np.histogram(hues, bins=bins, range=(0, 360))
    if hist.sum() < 1:
        return 0.0, "none"
    centers = (edges[:-1] + edges[1:]) / 2.0
    top_idx = np.argsort(hist)[::-1][:top_k]
    top_hues = sorted(centers[top_idx])
    # Pairwise hue gaps (circular distance in degrees)
    if len(top_hues) < 2:
        return 0.5, "monochrome"
    gaps = sorted(
        min(abs(b - a), 360.0 - abs(b - a))
        for a, b in zip(top_hues[:-1], top_hues[1:])
    )
    best_score, best_scheme = 0.0, "none"
    for scheme, targets in _HARMONY_TARGETS.items():
        for target in targets:
            # Score = 1 - (mean_gap_error / 60.0), clamped to [0,1]
            err = float(np.mean([abs(g - target) for g in gaps]))
            score = max(0.0, 1.0 - err / 60.0)
            if score > best_score:
                best_score, best_scheme = score, scheme
    return float(best_score), best_scheme


def contrast_score(frame: np.ndarray) -> float:
    """Score in [0, 1]: luma std-dev / 64 (cinematic shots typically ≥ 0.8)."""
    gray = _to_gray(frame).astype(np.float32)
    if gray.size < 4:
        return 0.0
    return float(min(1.0, gray.std() / 64.0))


# ---------------------------------------------------------------------------
# Per-shot aggregation
# ---------------------------------------------------------------------------
def score_frames(
    frames: List[np.ndarray],
    *,
    use_rule_of_thirds: bool = True,
    color_harmony_weight: float = 0.4,
    contrast_weight: float = 0.3,
    composition_weight: float = 0.3,
    sample_n: int = 6,
) -> Dict[str, Any]:
    """Score a list of frames and return per-shot aesthetic metrics."""
    if not frames:
        return {"composite": None, "n_sampled": 0}
    idxs = np.linspace(0, len(frames) - 1, min(sample_n, len(frames)), dtype=int)
    rot_scores: List[float] = []
    color_scores: List[float] = []
    contrast_scores: List[float] = []
    schemes: List[str] = []
    for i in idxs:
        f = frames[int(i)]
        if use_rule_of_thirds:
            rot_scores.append(rule_of_thirds_score(f))
        cs, scheme = color_harmony_score(f)
        color_scores.append(cs)
        schemes.append(scheme)
        contrast_scores.append(contrast_score(f))

    rot_mean = float(np.mean(rot_scores)) if rot_scores else None
    color_mean = float(np.mean(color_scores)) if color_scores else 0.0
    contrast_mean = float(np.mean(contrast_scores)) if contrast_scores else 0.0

    components: List[Tuple[float, float]] = []  # (weight, value)
    if rot_mean is not None and use_rule_of_thirds:
        components.append((float(composition_weight), rot_mean))
    components.append((float(color_harmony_weight), color_mean))
    components.append((float(contrast_weight), contrast_mean))
    wsum = sum(w for w, _ in components)
    composite = (
        float(sum(w * v for w, v in components) / wsum)
        if wsum > 1e-9 else 0.0
    )

    # Pick the most-frequent scheme as the shot's harmony label
    if schemes:
        scheme_label = max(set(schemes), key=schemes.count)
    else:
        scheme_label = "none"

    return {
        "composite": composite,
        "rule_of_thirds": rot_mean,
        "color_harmony": color_mean,
        "color_scheme": scheme_label,
        "contrast": contrast_mean,
        "n_sampled": len(idxs),
    }


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------
@register_module(
    MOD_ID,
    requires=[],
    description="Aesthetic & cinematic scoring (rule of thirds + color harmony + contrast).",
    version=VERSION,
)
def run(context: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    use_rot = bool(mod_cfg.get("use_rule_of_thirds", True))
    w_color = float(mod_cfg.get("color_harmony_weight", 0.4))
    w_contrast = float(mod_cfg.get("contrast_weight", 0.3))
    w_comp = float(mod_cfg.get("composition_weight", 0.3))
    sample_n = int(mod_cfg.get("sample_frames_per_shot",
                               cfg.processing.get("n_frames_per_shot", 6)))
    threshold = float(cfg.thresholds.get("aesthetic_score", 0.60))

    per_shot: Dict[int, Dict[str, Any]] = {}
    for sid, frames in context.shot_frames.items():
        per_shot[sid] = score_frames(
            frames,
            use_rule_of_thirds=use_rot,
            color_harmony_weight=w_color,
            contrast_weight=w_contrast,
            composition_weight=w_comp,
            sample_n=sample_n,
        )
    composites = [s["composite"] for s in per_shot.values() if s["composite"] is not None]
    summary = {
        "n_shots": len(per_shot),
        "mean_composite": float(np.mean(composites)) if composites else None,
        "min_composite": float(np.min(composites)) if composites else None,
        "n_below_threshold": sum(1 for c in composites if c < threshold),
    }
    return {
        "module": MOD_ID,
        "version": VERSION,
        "threshold": threshold,
        "weights": {"composition": w_comp, "color": w_color, "contrast": w_contrast},
        "per_shot": per_shot,
        "summary": summary,
    }


__all__ = [
    "rule_of_thirds_score", "color_harmony_score", "contrast_score",
    "score_frames", "run", "MOD_ID", "VERSION",
]
