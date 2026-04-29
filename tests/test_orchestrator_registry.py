"""End-to-end test: activating a module via config changes audit output.

This is the contract the registry-driven orchestrator must honour:

  cfg.modules['causal_reasoning']['enabled'] = True   →   the audit JSON
                                                          contains a
                                                          'causal_reasoning'
                                                          entry under
                                                          audit_data['modules'].

Without this guarantee, the YAML knob is cosmetic and the ``@register_module``
pattern is wasted.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from cineinfini.core.config import default_config, set_config
from cineinfini.core.context import VideoContext, VideoInfoLite
from cineinfini.core.registry import all_modules, get_active_modules

import cineinfini.modules  # noqa: F401  (force registrations)


@pytest.fixture
def synth_two_shot_ctx():
    """Two synthetic shots with different motion profiles."""
    rng = np.random.default_rng(0)
    static_shot = [np.full((180, 320, 3), 128, dtype=np.uint8) for _ in range(8)]
    moving_shot: List[np.ndarray] = []
    for t in range(8):
        f = np.full((180, 320, 3), 30, dtype=np.uint8)
        x = 20 + t * 12
        f[60:120, x:x + 40] = 220
        moving_shot.append(f)
    cfg = default_config()
    set_config(cfg)
    return VideoContext(
        video=VideoInfoLite(path=Path("/tmp/dummy.mp4"), fps=24.0,
                            total_frames=16, duration_s=0.66),
        shots=[(0, 7, 0.33), (8, 15, 0.33)],
        shot_frames={1: static_shot, 2: moving_shot},
        cfg=cfg,
    )


def _run_active_modules(ctx) -> Dict[str, dict]:
    """Mimic the orchestrator's module-loop and return per-module results."""
    out: Dict[str, dict] = {}
    for entry in get_active_modules():
        out[entry.mod_id] = entry.func(ctx)
    return out


def test_default_config_runs_three_modules(synth_two_shot_ctx):
    cfg = synth_two_shot_ctx.cfg
    set_config(cfg)
    results = _run_active_modules(synth_two_shot_ctx)
    assert set(results.keys()) == {
        "motion_coherence", "identity_consistency", "semantic_consistency"
    }


def test_enabling_causal_reasoning_adds_it_to_output(synth_two_shot_ctx):
    cfg = synth_two_shot_ctx.cfg
    cfg.modules["causal_reasoning"]["enabled"] = True
    set_config(cfg)
    results = _run_active_modules(synth_two_shot_ctx)
    assert "causal_reasoning" in results
    assert results["causal_reasoning"]["module"] == "causal_reasoning"


def test_enabling_aesthetic_adds_per_shot_aesthetic_score(synth_two_shot_ctx):
    cfg = synth_two_shot_ctx.cfg
    cfg.modules["aesthetic_cinematic"]["enabled"] = True
    set_config(cfg)
    results = _run_active_modules(synth_two_shot_ctx)
    assert "aesthetic_cinematic" in results
    per_shot = results["aesthetic_cinematic"]["per_shot"]
    assert set(per_shot.keys()) == {1, 2}
    for sid, metrics in per_shot.items():
        assert "composite" in metrics


def test_enabling_all_pure_cv_modules(synth_two_shot_ctx):
    """Toggle on the 16 pure-CV modules at once and verify each runs."""
    pure_cv = [
        "motion_coherence", "identity_consistency", "semantic_consistency",
        "background_consistency", "aesthetic_cinematic", "causal_reasoning",
        "temporal_signature", "physics_plausibility", "trustworthiness",
        "world_model_surprise", "creative_composition", "long_term_narrative",
        "subject_consistency_long",
        # benchmark_forensic skipped: needs ffmpeg writer codec
        "explainability", "benchmark_fusion",
    ]
    cfg = synth_two_shot_ctx.cfg
    for m in pure_cv:
        cfg.modules[m]["enabled"] = True
    set_config(cfg)
    # explainability + benchmark_fusion need context.cache['gates']
    synth_two_shot_ctx.cache["gates"] = {1: {}, 2: {}}
    results = _run_active_modules(synth_two_shot_ctx)
    for m in pure_cv:
        assert m in results, f"{m} did not run"
        assert results[m]["module"] == m


def test_disabling_a_module_removes_it(synth_two_shot_ctx):
    cfg = synth_two_shot_ctx.cfg
    cfg.modules["motion_coherence"]["enabled"] = False
    set_config(cfg)
    results = _run_active_modules(synth_two_shot_ctx)
    assert "motion_coherence" not in results
    assert "identity_consistency" in results


def test_required_models_match_enabled_modules(synth_two_shot_ctx):
    """Enabling a module that needs CLIP should surface clip_vit_b32 in
    the registry's required_models() output."""
    cfg = synth_two_shot_ctx.cfg
    cfg.modules["multi_modal_safety"]["enabled"] = True
    cfg.modules["origin_detection"]["enabled"] = True
    set_config(cfg)
    from cineinfini.core.registry import get_registry
    needed = set(get_registry().required_models())
    assert "clip_vit_b32" in needed
    assert "dinov2_vitb14" in needed


def test_module_results_can_be_merged_into_audit_data(synth_two_shot_ctx):
    """Mimic _merge_module_result from the orchestrator."""
    cfg = synth_two_shot_ctx.cfg
    cfg.modules["causal_reasoning"]["enabled"] = True
    cfg.modules["aesthetic_cinematic"]["enabled"] = True
    set_config(cfg)
    audit_data: Dict[str, dict] = {"modules": {}, "gates": {}}
    for entry in get_active_modules():
        result = entry.func(synth_two_shot_ctx)
        audit_data["modules"][entry.mod_id] = {
            k: v for k, v in result.items() if k != "per_shot"
        }
        for sid, fields in (result.get("per_shot") or {}).items():
            audit_data["gates"].setdefault(int(sid), {}).update(fields)
    # Each shot has metrics from multiple modules merged
    assert 1 in audit_data["gates"]
    assert 2 in audit_data["gates"]
    # And both of our enabled modules left their summary at the top level
    assert "causal_reasoning" in audit_data["modules"]
    assert "aesthetic_cinematic" in audit_data["modules"]
