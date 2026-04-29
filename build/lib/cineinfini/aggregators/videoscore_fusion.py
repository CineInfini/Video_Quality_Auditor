"""VideoScore-style 5-axis aggregation + global composite score.

VideoScore (HuggingFace TIGER, 2024) is the strongest published predictor
of human preference for AIGC video as of April 2026, with Spearman 77.1
on VideoFeedback-test. It outputs five regression scores per video:

  1. Visual Quality
  2. Temporal Consistency
  3. Dynamic Degree
  4. Text-to-Video Alignment
  5. Factual Consistency

We can't *replicate* VideoScore (their model is an 8B-parameter MLLM),
but we can produce **the same output shape** by fusing CineInfini's
native metrics into the same five axes. This makes our reports drop-in
comparable to VideoScore output and lets users rank videos with the
same dimension names every other AIGC paper uses.

We also produce a single composite quality score (a weighted sum of
the five axes) for one-number ranking — the equivalent of VMAF's
"100" or VideoScore's average. Weights default to equal but are
configurable via ``cfg.thresholds["videoscore_weights"]``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..core.config import get_config

logger = logging.getLogger("cineinfini.videoscore_fusion")


VIDEOSCORE_AXES = [
    "visual_quality",
    "temporal_consistency",
    "dynamic_degree",
    "text_to_video_alignment",
    "factual_consistency",
]


# ---------------------------------------------------------------------------
# Per-axis aggregation rules.
#
# Each axis is the mean of a small set of CineInfini gates, after
# normalisation to [0, 1] with "higher = better quality". Shots with no
# signal are dropped from the mean.
# ---------------------------------------------------------------------------
def _norm_clip(x: Optional[float], lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
    if x is None:
        return None
    if hi == lo:
        return None
    return max(0.0, min(1.0, (float(x) - lo) / (hi - lo)))


def _visual_quality(gate: Dict[str, Any]) -> Optional[float]:
    """Sharpness, noise, aesthetic — pixel-level quality."""
    parts: List[float] = []
    if gate.get("sharpness_blur") is not None:
        parts.append(_norm_clip(gate["sharpness_blur"], 0, 500) or 0.0)
    if gate.get("noise_estimation") is not None:
        # invert: more noise = worse
        n = _norm_clip(gate["noise_estimation"], 0, 50)
        if n is not None:
            parts.append(1.0 - n)
    if gate.get("aesthetic_composite") is not None:
        parts.append(float(gate["aesthetic_composite"]))
    elif gate.get("ssim3d_self") is not None:
        parts.append(float(gate["ssim3d_self"]))
    return sum(parts) / len(parts) if parts else None


def _temporal_consistency(gate: Dict[str, Any]) -> Optional[float]:
    """SSIM + flicker + motion smoothness."""
    parts: List[float] = []
    if gate.get("ssim3d_self") is not None:
        parts.append(float(gate["ssim3d_self"]))
    if gate.get("ssim_long_range") is not None:
        parts.append(float(gate["ssim_long_range"]))
    if gate.get("flicker_score") is not None:
        f = _norm_clip(gate["flicker_score"], 0, 50)
        if f is not None:
            parts.append(1.0 - f)
    if gate.get("clip_temp_consistency") is not None:
        parts.append(float(gate["clip_temp_consistency"]))
    return sum(parts) / len(parts) if parts else None


def _dynamic_degree(gate: Dict[str, Any]) -> Optional[float]:
    """Reward genuine motion (not flickering — that's penalised separately)."""
    if gate.get("motion_peak_div") is None:
        return None
    return _norm_clip(gate["motion_peak_div"], 0, 5)


def _text_alignment(gate: Dict[str, Any]) -> Optional[float]:
    """Only available when prompt_alignment_fine module is enabled."""
    return gate.get("prompt_alignment_fine") or gate.get("clip_alignment_score")


def _factual_consistency(gate: Dict[str, Any]) -> Optional[float]:
    """Causal reasoning + physics + identity = the closest analogue."""
    parts: List[float] = []
    if gate.get("identity_within_shot_dtw") is not None:
        d = float(gate["identity_within_shot_dtw"])
        parts.append(max(0.0, 1.0 - 2.0 * d))
    if gate.get("causal_score") is not None:
        parts.append(float(gate["causal_score"]))
    if gate.get("physics_plausibility") is not None:
        parts.append(float(gate["physics_plausibility"]))
    return sum(parts) / len(parts) if parts else None


_AXIS_FUNCS = {
    "visual_quality": _visual_quality,
    "temporal_consistency": _temporal_consistency,
    "dynamic_degree": _dynamic_degree,
    "text_to_video_alignment": _text_alignment,
    "factual_consistency": _factual_consistency,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_videoscore_axes(
    audit_data: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """Aggregate per-shot gates into 5 VideoScore-axis scores in [0, 1].

    Each axis is the mean of its components, computed shot-wise then averaged
    across shots. Returns ``None`` for an axis when no shot has any of the
    required signals (e.g. text alignment when no prompt was given).
    """
    gates = audit_data.get("gates") or {}
    if not gates:
        return {axis: None for axis in VIDEOSCORE_AXES}

    out: Dict[str, Optional[float]] = {}
    for axis, func in _AXIS_FUNCS.items():
        vals = [func(g) for g in gates.values()]
        vals = [v for v in vals if v is not None]
        out[axis] = sum(vals) / len(vals) if vals else None
    return out


def compute_global_score(
    axes: Dict[str, Optional[float]],
    weights: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Single composite quality score in [0, 1].

    Default weighting is uniform across measured axes (text alignment is
    typically null for audits without a prompt, so it self-excludes). Pass
    ``weights`` to override; missing axes default to 1.0.
    """
    if weights is None:
        try:
            cfg = get_config()
            weights = cfg.thresholds.get("videoscore_weights") or {}
        except Exception:
            weights = {}
    measured = {k: v for k, v in axes.items() if v is not None}
    if not measured:
        return None
    w = {k: float(weights.get(k, 1.0)) for k in measured}
    total_w = sum(w.values())
    if total_w == 0:
        return None
    return sum(measured[k] * w[k] for k in measured) / total_w


def attach_videoscore_to_audit(audit_data: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate (and return) audit_data: adds 'videoscore_axes' and
    'composite_score' top-level keys."""
    axes = compute_videoscore_axes(audit_data)
    composite = compute_global_score(axes)
    audit_data["videoscore_axes"] = axes
    audit_data["composite_score"] = composite
    return audit_data


__all__ = [
    "VIDEOSCORE_AXES",
    "compute_videoscore_axes",
    "compute_global_score",
    "attach_videoscore_to_audit",
]
