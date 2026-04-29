"""Explainability: per-module Shapley-style attribution.

Approximates each module's marginal contribution to the overall verdict
by computing the average difference in the composite gate-pass rate
when that module's metrics are included vs excluded across all shots.

This runs *after* the main audit, so it expects the per-shot gate
results to be passed in via ``context.cache['gates']`` (set by the
orchestrator). When that cache is absent, returns ``available: false``.

Pure NumPy. Disabled by default.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.explainability")
MOD_ID = "explainability"
VERSION = "0.4.8.1.4"

# Map of metric name -> module name; used to attribute Shapley to the source.
_METRIC_TO_MODULE = {
    "motion_peak_div": "motion_coherence",
    "ssim3d_self": "motion_coherence",
    "flicker": "motion_coherence",
    "ssim_long_range": "motion_coherence",
    "flicker_hf": "motion_coherence",
    "identity_intra": "identity_consistency",
    "clip_temp_consistency": "semantic_consistency",
    "background_ssim": "background_consistency",
    "causal_violation": "causal_reasoning",
    "aesthetic_score": "aesthetic_cinematic",
    "trust_score": "trustworthiness",
    "surprise_p95": "world_model_surprise",
}


def _gate_pass_rate(metrics: Mapping[str, Optional[float]],
                    thresholds: Mapping[str, float],
                    excluded: Sequence[str] = ()) -> float:
    passed = 0
    total = 0
    for k, v in metrics.items():
        if k in excluded or v is None:
            continue
        thr = thresholds.get(k)
        if thr is None:
            continue
        total += 1
        # "lower is better" defaults; for ssim* we expect higher
        if k.startswith("ssim") or "consistency" in k or k.endswith("_score"):
            if float(v) >= thr:
                passed += 1
        else:
            if float(v) <= thr:
                passed += 1
    return passed / total if total > 0 else 0.0


def shapley_per_metric(
    per_shot: Mapping[int, Mapping[str, Optional[float]]],
    thresholds: Mapping[str, float],
    n_permutations: int = 50,
) -> Dict[str, float]:
    rng = np.random.default_rng(42)
    shots = list(per_shot.values())
    if not shots:
        return {}
    metric_keys = list({k for s in shots for k in s.keys()})
    if not metric_keys:
        return {}
    n_perm = min(n_permutations, max(1, np.math.factorial(min(len(metric_keys), 6))))
    contributions: Dict[str, List[float]] = {k: [] for k in metric_keys}
    for _ in range(n_perm):
        perm = list(metric_keys)
        rng.shuffle(perm)
        excluded: List[str] = list(metric_keys)
        prev = float(np.mean([
            _gate_pass_rate(s, thresholds, excluded) for s in shots
        ]))
        for k in perm:
            excluded.remove(k)
            now = float(np.mean([
                _gate_pass_rate(s, thresholds, excluded) for s in shots
            ]))
            contributions[k].append(now - prev)
            prev = now
    return {k: float(np.mean(v)) if v else 0.0 for k, v in contributions.items()}


@register_module(MOD_ID, requires=[],
                 description="Approximate Shapley attribution over per-shot gate metrics.",
                 version=VERSION)
def run(context: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    n_perm = int(mod_cfg.get("n_permutations", 50))

    gates = context.cache.get("gates")
    if not isinstance(gates, dict) or not gates:
        return {"module": MOD_ID, "version": VERSION, "available": False,
                "error": "no per-shot gate data in context.cache['gates']"}
    per_metric = shapley_per_metric(gates, cfg.thresholds, n_permutations=n_perm)
    by_module: Dict[str, float] = {}
    for k, v in per_metric.items():
        mod = _METRIC_TO_MODULE.get(k, "unknown")
        by_module[mod] = by_module.get(mod, 0.0) + v
    return {
        "module": MOD_ID, "version": VERSION, "available": True,
        "per_metric_shapley": per_metric,
        "per_module_shapley": by_module,
    }
