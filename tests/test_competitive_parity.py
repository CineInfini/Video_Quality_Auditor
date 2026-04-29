"""End-to-end tests for v0.4.8.2 competitive-parity features.

Validates:
  - VBench 16-dim export produces valid JSON with the right schema
  - VideoScore 5-axis fusion produces values in [0, 1]
  - Composite global score is computed and is in [0, 1]
  - DOVER/FAST-VQA wrappers report ``available: false`` honestly
    (since we don't ship the model code)
  - The full pipeline composes: audit gates → axes → composite → vbench
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cineinfini.aggregators import (
    VIDEOSCORE_AXES,
    attach_videoscore_to_audit,
    compute_global_score,
    compute_videoscore_axes,
)
from cineinfini.core.config import default_config, set_config
from cineinfini.io.exporters import (
    VBENCH_ALL,
    export_vbench_json,
    map_to_vbench,
)
from cineinfini.io.exporters.vbench_export import (
    VBENCH_DIMENSIONS_CONDITION,
    VBENCH_DIMENSIONS_QUALITY,
)


# ---------------------------------------------------------------------------
# Sample audit data fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_audit():
    """Synthetic audit_data with both shots populated."""
    return {
        "video": {"name": "test_video.mp4", "duration_s": 12.5, "fps": 24.0},
        "gates": {
            1: {
                "identity_within_shot": 0.08,
                "identity_within_shot_dtw": 0.12,
                "ssim3d_self": 0.91,
                "ssim_long_range": 0.84,
                "flicker_score": 4.5,
                "motion_peak_div": 1.8,
                "clip_temp_consistency": 0.88,
                "sharpness_blur": 320.0,
                "noise_estimation": 8.0,
                "aesthetic_composite": 0.72,
            },
            2: {
                "identity_within_shot": 0.15,
                "identity_within_shot_dtw": 0.21,
                "ssim3d_self": 0.78,
                "ssim_long_range": 0.69,
                "flicker_score": 12.0,
                "motion_peak_div": 3.2,
                "clip_temp_consistency": 0.74,
                "sharpness_blur": 180.0,
                "noise_estimation": 18.0,
                "aesthetic_composite": 0.55,
            },
        },
        "modules": {},
    }


# ---------------------------------------------------------------------------
# VBench mapping
# ---------------------------------------------------------------------------
def test_vbench_has_16_dimensions():
    assert len(VBENCH_ALL) == 16
    assert len(VBENCH_DIMENSIONS_QUALITY) == 7
    assert len(VBENCH_DIMENSIONS_CONDITION) == 9


def test_vbench_dimension_names_are_canonical():
    """Names must match VBench's spec exactly so the leaderboard accepts them."""
    expected_quality = {
        "subject_consistency", "background_consistency", "temporal_flickering",
        "motion_smoothness", "dynamic_degree", "aesthetic_quality",
        "imaging_quality",
    }
    assert set(VBENCH_DIMENSIONS_QUALITY) == expected_quality
    expected_condition = {
        "object_class", "multiple_objects", "human_action", "color",
        "spatial_relationship", "scene", "appearance_style", "temporal_style",
        "overall_consistency",
    }
    assert set(VBENCH_DIMENSIONS_CONDITION) == expected_condition


def test_map_to_vbench_returns_all_16_keys(sample_audit):
    result = map_to_vbench(sample_audit)
    assert set(result.keys()) == set(VBENCH_ALL)


def test_map_to_vbench_quality_dimensions_are_in_unit_interval(sample_audit):
    result = map_to_vbench(sample_audit)
    for dim in VBENCH_DIMENSIONS_QUALITY:
        v = result[dim]
        if v is not None:
            assert 0.0 <= v <= 1.0, f"{dim} out of [0,1]: {v}"


def test_map_to_vbench_condition_dimensions_are_null_no_prompt(sample_audit):
    """Without a prompt suite, condition-consistency must be null."""
    result = map_to_vbench(sample_audit)
    for dim in VBENCH_DIMENSIONS_CONDITION:
        assert result[dim] is None


def test_map_to_vbench_higher_dtw_means_lower_subject_consistency():
    """Sanity: identity drift ↑ ⇒ subject_consistency ↓."""
    bad = {"gates": {1: {"identity_within_shot_dtw": 0.45}}}
    good = {"gates": {1: {"identity_within_shot_dtw": 0.05}}}
    assert map_to_vbench(bad)["subject_consistency"] < (
        map_to_vbench(good)["subject_consistency"] or 0.0
    )


def test_export_vbench_json_writes_valid_json(sample_audit, tmp_path):
    out = tmp_path / "vbench.json"
    export_vbench_json(sample_audit, out)
    payload = json.loads(out.read_text())
    assert payload["model"] == "CineInfini"
    assert payload["video"] == "test_video.mp4"
    assert "scores" in payload
    assert set(payload["scores"].keys()) == set(VBENCH_ALL)
    assert "measured_dimensions" in payload
    assert "unmeasured_dimensions" in payload
    # Quality dims should be measured given our gates
    for d in VBENCH_DIMENSIONS_QUALITY:
        assert d in payload["measured_dimensions"], f"{d} should be measured"


def test_export_vbench_handles_empty_gates(tmp_path):
    out = tmp_path / "empty.json"
    export_vbench_json({"gates": {}, "video": {"name": "x"}}, out)
    payload = json.loads(out.read_text())
    assert all(v is None for v in payload["scores"].values())


# ---------------------------------------------------------------------------
# VideoScore fusion
# ---------------------------------------------------------------------------
def test_videoscore_has_5_axes():
    assert len(VIDEOSCORE_AXES) == 5


def test_videoscore_axis_names_match_paper():
    expected = {
        "visual_quality", "temporal_consistency", "dynamic_degree",
        "text_to_video_alignment", "factual_consistency",
    }
    assert set(VIDEOSCORE_AXES) == expected


def test_compute_axes_returns_all_5(sample_audit):
    axes = compute_videoscore_axes(sample_audit)
    assert set(axes.keys()) == set(VIDEOSCORE_AXES)


def test_compute_axes_values_in_unit_interval(sample_audit):
    axes = compute_videoscore_axes(sample_audit)
    for name, v in axes.items():
        if v is not None:
            assert 0.0 <= v <= 1.0, f"{name} = {v}"


def test_text_alignment_is_none_without_prompt(sample_audit):
    axes = compute_videoscore_axes(sample_audit)
    assert axes["text_to_video_alignment"] is None


def test_compute_global_score_excludes_null_axes():
    axes = {
        "visual_quality": 0.8,
        "temporal_consistency": 0.7,
        "dynamic_degree": 0.5,
        "text_to_video_alignment": None,  # missing
        "factual_consistency": 0.9,
    }
    g = compute_global_score(axes)
    # Mean of the 4 measured axes = (0.8 + 0.7 + 0.5 + 0.9) / 4 = 0.725
    assert abs(g - 0.725) < 1e-6


def test_compute_global_score_returns_none_when_all_null():
    assert compute_global_score({k: None for k in VIDEOSCORE_AXES}) is None


def test_attach_videoscore_mutates_audit_data(sample_audit):
    out = attach_videoscore_to_audit(sample_audit)
    assert "videoscore_axes" in out
    assert "composite_score" in out
    assert out["composite_score"] is not None


def test_compute_global_score_respects_custom_weights():
    axes = {
        "visual_quality": 1.0,
        "temporal_consistency": 0.0,
        "dynamic_degree": 0.0,
        "text_to_video_alignment": 0.0,
        "factual_consistency": 0.0,
    }
    weights = {"visual_quality": 5.0}
    g = compute_global_score(axes, weights=weights)
    # All other axes get default weight 1.0
    expected = (1.0 * 5.0) / (5.0 + 1.0 + 1.0 + 1.0 + 1.0)
    assert abs(g - expected) < 1e-6


# ---------------------------------------------------------------------------
# DOVER + FAST-VQA wrappers — graceful degradation
# ---------------------------------------------------------------------------
def test_dover_module_registered():
    import cineinfini.modules  # noqa: F401
    from cineinfini.core.registry import all_modules
    assert "dover_score" in all_modules()


def test_fastvqa_module_registered():
    import cineinfini.modules  # noqa: F401
    from cineinfini.core.registry import all_modules
    assert "fastvqa_score" in all_modules()


def test_dover_reports_unavailable_without_weights(tmp_path):
    """DOVER.pth not on disk → module reports available=false honestly."""
    cfg = default_config()
    cfg.paths["models_dir"] = str(tmp_path / "empty_models")
    cfg.modules["dover_score"]["enabled"] = True
    set_config(cfg)
    from cineinfini.core.context import VideoContext, VideoInfoLite
    ctx = VideoContext(
        video=VideoInfoLite(path=Path("/tmp/x.mp4"), fps=24.0,
                            total_frames=10, duration_s=0.4),
        shots=[(0, 9, 0.4)],
        shot_frames={1: []},
        cfg=cfg,
    )
    from cineinfini.modules.dover_score import dover_score
    result = dover_score(ctx)
    assert result["available"] is False
    assert "weights missing" in result["reason"] or "torch" in result["reason"] \
        or "DOVER" in result["reason"]


def test_fastvqa_reports_unavailable_without_weights(tmp_path):
    cfg = default_config()
    cfg.paths["models_dir"] = str(tmp_path / "empty_models2")
    cfg.modules["fastvqa_score"]["enabled"] = True
    set_config(cfg)
    from cineinfini.core.context import VideoContext, VideoInfoLite
    ctx = VideoContext(
        video=VideoInfoLite(path=Path("/tmp/x.mp4"), fps=24.0,
                            total_frames=10, duration_s=0.4),
        shots=[(0, 9, 0.4)],
        shot_frames={1: []},
        cfg=cfg,
    )
    from cineinfini.modules.fastvqa_score import fastvqa_score
    result = fastvqa_score(ctx)
    assert result["available"] is False


def test_modules_default_to_disabled():
    """Don't auto-enable wrappers — user must opt in."""
    cfg = default_config()
    assert cfg.modules["dover_score"]["enabled"] is False
    assert cfg.modules["fastvqa_score"]["enabled"] is False


# ---------------------------------------------------------------------------
# End-to-end pipeline composition
# ---------------------------------------------------------------------------
def test_full_competitive_pipeline_composes(sample_audit, tmp_path):
    """The whole point of v0.4.8.2: one audit → all four output formats."""
    # 1. CineInfini native gates (already in sample_audit)
    assert sample_audit["gates"]
    # 2. VideoScore fusion
    sample_audit = attach_videoscore_to_audit(sample_audit)
    assert sample_audit["composite_score"] is not None
    # 3. VBench export
    vbench_path = tmp_path / "audit.vbench.json"
    export_vbench_json(sample_audit, vbench_path)
    payload = json.loads(vbench_path.read_text())
    # 4. Sanity: composite is in [0, 1] and at least 5 quality dims measured
    assert 0.0 <= sample_audit["composite_score"] <= 1.0
    assert len(payload["measured_dimensions"]) >= 5
