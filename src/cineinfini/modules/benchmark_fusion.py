"""Benchmark fusion: emit our metrics in VBench / EvalCrafter format.

Maps CineInfini's per-shot metrics to the canonical VBench dimension
names so users can compare scores across benchmarks. This is a pure
data-shaping module — no ML — so it always reports ``available: true``.

The mapping is defined per-key and is intentionally conservative:
metrics that don't have a clean VBench analogue are omitted rather
than fudged.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.benchmark_fusion")
MOD_ID = "benchmark_fusion"
VERSION = "0.4.8.1.4"

# CineInfini metric -> (VBench dimension, transform)
_VBENCH_MAP = {
    "ssim3d_self":            ("subject_consistency", lambda v: float(v)),
    "ssim_long_range":        ("background_consistency", lambda v: float(v)),
    "flicker":                ("temporal_flickering", lambda v: 1.0 - float(v)),
    "motion_peak_div":        ("motion_smoothness",
                                lambda v: max(0.0, 1.0 - float(v) / 50.0)),
    "clip_temp_consistency":  ("aesthetic_quality", lambda v: float(v)),
    "identity_intra":         ("subject_consistency",
                                lambda v: max(0.0, 1.0 - float(v))),
    "causal_violation":       ("dynamic_degree",
                                lambda v: max(0.0, 1.0 - float(v))),
    "aesthetic_score":        ("aesthetic_quality", lambda v: float(v)),
    "trust_score":            ("imaging_quality", lambda v: float(v)),
    "background_ssim":        ("background_consistency", lambda v: float(v)),
    "surprise_p95":           ("temporal_flickering",
                                lambda v: max(0.0, 1.0 - float(v))),
}


def _aggregate_to_vbench(per_shot: Mapping[int, Mapping[str, Optional[float]]]) -> Dict[str, float]:
    accum: Dict[str, list] = {}
    for shot in per_shot.values():
        for k, v in shot.items():
            if v is None or k not in _VBENCH_MAP:
                continue
            dim, fn = _VBENCH_MAP[k]
            try:
                accum.setdefault(dim, []).append(fn(v))
            except Exception:
                continue
    return {dim: float(np.mean(vals)) for dim, vals in accum.items() if vals}


@register_module(MOD_ID, requires=[],
                 description="Aggregate metrics into VBench-style dimensions.",
                 version=VERSION)
def run(context: VideoContext) -> Dict[str, Any]:
    gates = context.cache.get("gates")
    if not isinstance(gates, dict) or not gates:
        return {"module": MOD_ID, "version": VERSION, "available": False,
                "error": "no per-shot gate data in context.cache['gates']"}
    vbench = _aggregate_to_vbench(gates)
    return {"module": MOD_ID, "version": VERSION, "available": True,
            "vbench_scores": vbench, "n_dimensions": len(vbench)}
