"""Smoke tests for every module.

Validates the registration pattern uniformly across all 19 modules:

1. Each declared module is in the registry.
2. Each module is disabled by default.
3. Each module's ``run(context)`` returns a dict with keys ``module``
   and ``version`` and does not crash on empty/synthetic input.

For modules that require ML weights, ``run`` may return
``available: false`` — that's still a valid response and the test
accepts it.

Per-module substantive tests live in their own files (e.g.
``test_aesthetic_cinematic.py``, ``test_causal_reasoning.py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from cineinfini.core.config import default_config, set_config
from cineinfini.core.context import VideoContext, VideoInfoLite
from cineinfini.core.registry import all_modules


# Force-import the modules package so registrations happen
import cineinfini.modules  # noqa: F401


ALL_MODULES = [
    # Pure CV
    "motion_coherence", "identity_consistency", "semantic_consistency",
    "background_consistency", "aesthetic_cinematic", "causal_reasoning",
    "temporal_signature", "physics_plausibility", "trustworthiness",
    "world_model_surprise", "creative_composition", "long_term_narrative",
    "subject_consistency_long", "benchmark_forensic", "explainability",
    "benchmark_fusion",
    # ML-required
    "origin_detection", "multi_modal_safety", "prompt_alignment_fine",
]


@pytest.fixture
def two_shot_context():
    rng = np.random.default_rng(0)
    shot_frames: Dict[int, List[np.ndarray]] = {
        1: [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(8)],
        2: [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(8)],
    }
    cfg = default_config()
    set_config(cfg)
    ctx = VideoContext(
        video=VideoInfoLite(path=Path("/tmp/dummy.mp4"), fps=24.0),
        shots=[(0, 7, 0.33), (8, 15, 0.33)],
        shot_frames=shot_frames,
        cfg=cfg,
    )
    return ctx


def test_every_declared_module_is_registered():
    registered = set(all_modules().keys())
    declared = set(default_config().modules.keys())
    missing = declared - registered
    assert not missing, f"Modules in cfg but not registered: {missing}"


def test_every_module_is_disabled_by_default():
    cfg = default_config()
    enabled_optional = [
        m for m in ALL_MODULES
        if cfg.is_module_enabled(m) and m not in
        {"motion_coherence", "identity_consistency", "semantic_consistency"}
    ]
    assert not enabled_optional, f"Optional modules enabled by default: {enabled_optional}"


@pytest.mark.parametrize("mod_id", ALL_MODULES)
def test_module_returns_required_keys(mod_id, two_shot_context):
    """Every module's run() returns {module, version, ...} or {available: false}."""
    entry = all_modules().get(mod_id)
    assert entry is not None, f"{mod_id} not in registry"
    cfg = two_shot_context.cfg
    cfg.modules[mod_id]["enabled"] = True
    set_config(cfg)
    out = entry.func(two_shot_context)
    assert isinstance(out, dict), f"{mod_id} did not return a dict"
    assert "module" in out and out["module"] == mod_id
    assert "version" in out
    # Either it produced data or it gracefully degraded
    has_payload = any(k in out for k in (
        "per_shot", "summary", "vbench_scores", "long_term_narrative",
        "long_term_coherence", "rhythm_variance", "per_metric_shapley",
    ))
    is_unavailable = out.get("available") is False
    assert has_payload or is_unavailable, (
        f"{mod_id}: neither produced data nor reported unavailable: {out}"
    )


@pytest.mark.parametrize("mod_id", ALL_MODULES)
def test_module_can_be_enabled_and_disabled(mod_id):
    cfg = default_config()
    cfg.modules[mod_id]["enabled"] = True
    assert cfg.is_module_enabled(mod_id) is True
    cfg.modules[mod_id]["enabled"] = False
    assert cfg.is_module_enabled(mod_id) is False


def test_get_active_modules_only_returns_enabled():
    from cineinfini.core.registry import get_active_modules
    cfg = default_config()
    # default: only motion/identity/semantic
    set_config(cfg)
    active = {e.mod_id for e in get_active_modules()}
    assert active == {"motion_coherence", "identity_consistency", "semantic_consistency"}
    # enable one more
    cfg.modules["aesthetic_cinematic"]["enabled"] = True
    set_config(cfg)
    active2 = {e.mod_id for e in get_active_modules()}
    assert "aesthetic_cinematic" in active2


def test_required_models_aggregation():
    """Some modules declare they need ML weights via `requires=[...]`."""
    cfg = default_config()
    cfg.modules["multi_modal_safety"]["enabled"] = True
    cfg.modules["origin_detection"]["enabled"] = True
    set_config(cfg)
    from cineinfini.core.registry import get_registry
    needed = set(get_registry().required_models())
    assert "clip_vit_b32" in needed
    assert "dinov2_vitb14" in needed
