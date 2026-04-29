"""VBench-compatible 16-dimension mapping.

VBench (NeurIPS 2024) defines 16 disentangled dimensions for AIGC video
quality, in two groups:

  Video Quality
    1.  subject_consistency
    2.  background_consistency
    3.  temporal_flickering
    4.  motion_smoothness
    5.  dynamic_degree
    6.  aesthetic_quality
    7.  imaging_quality

  Video–Condition Consistency
    8.  object_class
    9.  multiple_objects
    10. human_action
    11. color
    12. spatial_relationship
    13. scene
    14. appearance_style
    15. temporal_style
    16. overall_consistency

This module maps CineInfini's native metrics to the seven Video Quality
dimensions (the only ones we can compute without the prompt-suite that
VBench ships with). The Video–Condition Consistency dimensions
require a text prompt + a reference policy, which is *out of scope* for
CineInfini's no-reference auditor and we report ``null`` for them in
the export, with the convention used by VBench's own evaluation kit
when a dimension is not measured.

The exporter writes JSON in the exact schema VBench expects so the
output can be uploaded to the VBench leaderboard or fed into VBench's
analysis scripts unmodified.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...core.config import get_config

logger = logging.getLogger("cineinfini.vbench")


# ---------------------------------------------------------------------------
# Authoritative dimension list (do not reorder; VBench leaderboard uses this)
# ---------------------------------------------------------------------------
VBENCH_DIMENSIONS_QUALITY = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]

VBENCH_DIMENSIONS_CONDITION = [
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
]

VBENCH_ALL = VBENCH_DIMENSIONS_QUALITY + VBENCH_DIMENSIONS_CONDITION


# ---------------------------------------------------------------------------
# CineInfini metric → VBench dimension mapping.
#
# Each VBench dimension is computed as a *normalised* value in [0, 1] from
# one or more native CineInfini gates. Mappings are deliberately simple and
# documented so a reviewer can trace any number back to its source.
#
# Convention: higher = better quality (VBench's convention).
# ---------------------------------------------------------------------------
def _identity_to_subject_consistency(gate: Dict[str, Any]) -> Optional[float]:
    """ArcFace identity DTW → subject consistency.

    Lower DTW distance = higher subject consistency. We invert and clip
    to [0, 1] using a soft-saturation: 0 distance → 1.0, distance ≥ 0.5 → 0.0.
    """
    d = gate.get("identity_within_shot_dtw")
    if d is None:
        d = gate.get("identity_within_shot")  # mean fallback
    if d is None:
        return None
    return max(0.0, min(1.0, 1.0 - 2.0 * float(d)))


def _bg_to_background_consistency(gate: Dict[str, Any]) -> Optional[float]:
    """SSIM long-range → background consistency."""
    s = gate.get("ssim_long_range")
    return float(s) if s is not None else None


def _flicker_to_temporal_flickering(gate: Dict[str, Any]) -> Optional[float]:
    """Inverse of flicker score."""
    f = gate.get("flicker_score")
    if f is None:
        return None
    # Typical flicker_score range is [0, 50]; saturate above 50.
    return max(0.0, min(1.0, 1.0 - float(f) / 50.0))


def _ssim3d_to_motion_smoothness(gate: Dict[str, Any]) -> Optional[float]:
    """3D-SSIM self-referenced → motion smoothness."""
    s = gate.get("ssim3d_self")
    return float(s) if s is not None else None


def _motion_to_dynamic_degree(gate: Dict[str, Any]) -> Optional[float]:
    """Optical-flow peak divergence → dynamic degree.

    VBench's "dynamic degree" rewards motion (not penalises it).
    """
    m = gate.get("motion_peak_div")
    if m is None:
        return None
    # Saturate at 5.0 (very high motion); 0 stays 0.
    return max(0.0, min(1.0, float(m) / 5.0))


def _aesthetic_to_aesthetic_quality(gate: Dict[str, Any]) -> Optional[float]:
    """Aesthetic cinematic composite, when the module is enabled."""
    return gate.get("aesthetic_composite") or gate.get("aesthetic_score")


def _imaging_quality(gate: Dict[str, Any]) -> Optional[float]:
    """Sharpness × inverse-noise composite."""
    sharp = gate.get("sharpness_blur")
    noise = gate.get("noise_estimation")
    if sharp is None and noise is None:
        return None
    sharp_n = max(0.0, min(1.0, float(sharp) / 500.0)) if sharp is not None else 0.5
    noise_n = max(0.0, 1.0 - float(noise) / 50.0) if noise is not None else 0.5
    return 0.5 * (sharp_n + noise_n)


_QUALITY_MAPPERS = {
    "subject_consistency": _identity_to_subject_consistency,
    "background_consistency": _bg_to_background_consistency,
    "temporal_flickering": _flicker_to_temporal_flickering,
    "motion_smoothness": _ssim3d_to_motion_smoothness,
    "dynamic_degree": _motion_to_dynamic_degree,
    "aesthetic_quality": _aesthetic_to_aesthetic_quality,
    "imaging_quality": _imaging_quality,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def map_to_vbench(audit_data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Return a {dimension: value | None} dict over all 16 VBench dimensions.

    Per-shot gates are aggregated by mean; shots with insufficient signal
    contribute ``None`` and are excluded from the mean.
    """
    gates = audit_data.get("gates") or {}
    if not gates:
        return {d: None for d in VBENCH_ALL}

    out: Dict[str, Optional[float]] = {}
    for dim, mapper in _QUALITY_MAPPERS.items():
        vals: List[float] = []
        for gate in gates.values():
            v = mapper(gate)
            if v is not None:
                vals.append(float(v))
        out[dim] = sum(vals) / len(vals) if vals else None
    # Condition dimensions stay None — they need a prompt and reference set.
    for dim in VBENCH_DIMENSIONS_CONDITION:
        out[dim] = None
    return out


def export_vbench_json(
    audit_data: Dict[str, Any],
    output_path: Path,
    *,
    model_name: str = "CineInfini",
) -> Path:
    """Write a VBench-compatible JSON file to ``output_path``.

    The schema follows VBench's evaluation kit output:

    ```json
    {
      "model": "CineInfini",
      "version": "0.4.8.2",
      "video": "<basename>",
      "scores": {
        "subject_consistency": 0.87,
        "background_consistency": 0.92,
        ...
      },
      "measured_dimensions": ["subject_consistency", ...],
      "unmeasured_dimensions": ["object_class", ...]
    }
    ```
    """
    cfg = get_config()
    scores = map_to_vbench(audit_data)
    measured = [d for d, v in scores.items() if v is not None]
    unmeasured = [d for d, v in scores.items() if v is None]
    payload = {
        "model": model_name,
        "version": getattr(cfg, "version", "unknown"),
        "video": audit_data.get("video", {}).get("name") or "unknown",
        "scores": scores,
        "measured_dimensions": measured,
        "unmeasured_dimensions": unmeasured,
        "note": (
            "VBench condition-consistency dimensions require a prompt suite "
            "and reference policy; not computed by CineInfini's no-reference "
            "auditor. See docs/COMPARISON.md."
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("VBench export written to %s (%d/%d dimensions measured)",
                output_path, len(measured), len(VBENCH_ALL))
    return output_path


__all__ = [
    "VBENCH_DIMENSIONS_QUALITY",
    "VBENCH_DIMENSIONS_CONDITION",
    "VBENCH_ALL",
    "map_to_vbench",
    "export_vbench_json",
]
