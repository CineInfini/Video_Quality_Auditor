"""Creative composition: rhythm variety from shot durations.

A boring video uses uniform shot lengths; a well-composed one varies
its rhythm. We measure two things at the video level (cross-shot):

* ``rhythm_variance``: coefficient of variation of shot durations.
  Higher = more variety in shot length.
* ``cut_density``: cuts per second over the audited segment.

These are video-level (not per-shot) metrics. Pure Python. Disabled
by default.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.creative_composition")
MOD_ID = "creative_composition"
VERSION = "0.4.8.1.4"


def rhythm_variance(durations_s):
    arr = np.array([d for d in durations_s if d > 0], dtype=np.float64)
    if arr.size < 2:
        return None
    mean = float(arr.mean())
    if mean < 1e-6:
        return None
    return float(arr.std() / mean)  # coefficient of variation


def cut_density(n_shots: int, total_duration_s: float):
    if total_duration_s <= 0 or n_shots <= 0:
        return None
    # n_shots-1 cuts within the segment
    return float(max(0, n_shots - 1) / total_duration_s)


@register_module(MOD_ID, requires=[], description="Shot-rhythm variety + cut density.", version=VERSION)
def run(context: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    weight = float(mod_cfg.get("rhythm_variance_weight", 0.5))
    fps = float(getattr(context.video, "fps", 24.0)) or 24.0
    durations: list = []
    for s in context.shots or []:
        # shots is a list of (start, end[, duration]) tuples
        if len(s) >= 3:
            durations.append(float(s[2]))
        elif len(s) == 2:
            durations.append((int(s[1]) - int(s[0]) + 1) / fps)
    rv = rhythm_variance(durations)
    total_dur = float(sum(durations)) if durations else 0.0
    cd = cut_density(len(durations), total_dur)
    composite = None
    if rv is not None:
        # Map CV: 0 (uniform) -> 0; ≥1 (very varied) -> 1
        composite = float(weight * min(1.0, rv) + (1 - weight) * (1.0 if cd else 0.0))
    return {
        "module": MOD_ID, "version": VERSION,
        "rhythm_variance": rv, "cut_density": cd,
        "n_shots": len(durations), "total_duration_s": total_dur,
        "composite": composite,
    }
