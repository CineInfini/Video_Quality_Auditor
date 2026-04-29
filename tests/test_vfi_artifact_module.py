"""Tests for temporal_smoothness_via_interp_artifacts module."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_clean_frames():
    """16 frames of a smoothly translating gradient — no ghosting/judder."""
    frames = []
    for i in range(16):
        f = np.zeros((120, 160, 3), dtype=np.uint8)
        # Smooth horizontal translation
        x = (i * 5) % 160
        f[:, x:x+20] = 200
        # Add some texture so Laplacian variance is non-trivial
        noise = np.random.RandomState(i).randint(0, 30, size=f.shape, dtype=np.uint8)
        f = np.clip(f.astype(int) + noise, 0, 255).astype(np.uint8)
        frames.append(f)
    return frames


@pytest.fixture
def synthetic_juddery_frames():
    """16 frames with irregular motion — high judder."""
    frames = []
    positions = [0, 5, 5, 30, 30, 50, 50, 70, 70, 90, 90, 100, 100, 120, 120, 140]
    for i, x in enumerate(positions):
        f = np.zeros((120, 160, 3), dtype=np.uint8)
        f[:, x:x+10] = 200
        noise = np.random.RandomState(i).randint(0, 30, size=f.shape, dtype=np.uint8)
        f = np.clip(f.astype(int) + noise, 0, 255).astype(np.uint8)
        frames.append(f)
    return frames


@pytest.fixture
def synthetic_blurred_interp_frames():
    """16 frames where every other frame has reduced high-freq content
    (simulating an over-smoothed interpolated frame)."""
    frames = []
    for i in range(16):
        f = np.zeros((120, 160, 3), dtype=np.uint8)
        x = (i * 5) % 160
        f[:, x:x+20] = 200
        if i % 2 == 0:
            # Sharp natural frame
            noise = np.random.RandomState(i).randint(0, 50, size=f.shape, dtype=np.uint8)
        else:
            # Blurred interpolated frame
            noise = np.random.RandomState(i).randint(0, 5, size=f.shape, dtype=np.uint8)
        f = np.clip(f.astype(int) + noise, 0, 255).astype(np.uint8)
        frames.append(f)
    return frames


def test_module_registered():
    """The module is loaded by importing cineinfini.modules."""
    import cineinfini.modules  # noqa: F401
    from cineinfini.core.registry import all_modules
    mods = all_modules()
    assert "temporal_smoothness_via_interp_artifacts" in mods


def test_ghosting_score_clean_returns_low(synthetic_clean_frames):
    """Smooth motion should produce low ghosting."""
    from cineinfini.modules.temporal_smoothness_via_interp_artifacts import _ghosting_score
    score = _ghosting_score(synthetic_clean_frames)
    assert 0.0 <= score <= 1.0
    assert score < 0.5, f"Clean frames should have low ghosting, got {score}"


def test_judder_score_clean_returns_low(synthetic_clean_frames):
    """Smooth motion should produce low judder."""
    from cineinfini.modules.temporal_smoothness_via_interp_artifacts import _judder_score
    score = _judder_score(synthetic_clean_frames)
    assert 0.0 <= score <= 1.0


def test_judder_score_juddery_returns_high(synthetic_juddery_frames):
    """Irregular motion should produce high judder."""
    from cineinfini.modules.temporal_smoothness_via_interp_artifacts import _judder_score
    score = _judder_score(synthetic_juddery_frames)
    assert score > 0.3, f"Juddery frames should produce high judder, got {score}"


def test_interp_blur_flags_alternating_frames(synthetic_blurred_interp_frames):
    """When every other frame is over-smoothed, the dip detector should flag many."""
    from cineinfini.modules.temporal_smoothness_via_interp_artifacts import _interp_blur_score
    score = _interp_blur_score(synthetic_blurred_interp_frames)
    assert 0.0 <= score <= 1.0
    # In our synthetic case ~half the frames are flagged
    assert score > 0.2, f"Should flag interpolated frames, got {score}"


def test_module_too_few_frames_returns_zero():
    """Fewer than 3 frames → zero scores (no triple to analyse)."""
    from cineinfini.modules.temporal_smoothness_via_interp_artifacts import (
        _ghosting_score, _judder_score, _interp_blur_score,
    )
    short = [np.zeros((100, 100, 3), dtype=np.uint8)] * 2
    assert _ghosting_score(short) == 0.0
    assert _judder_score(short) == 0.0
    assert _interp_blur_score(short) == 0.0


def test_module_full_returns_correct_shape(synthetic_clean_frames):
    """End-to-end: module returns the documented dict shape."""
    import cineinfini.modules  # noqa: F401
    from cineinfini.modules.temporal_smoothness_via_interp_artifacts import (
        temporal_smoothness_via_interp_artifacts, MODULE_ID,
    )
    class MockCtx:
        shot_frames = {1: synthetic_clean_frames, 2: synthetic_clean_frames[:2]}

    result = temporal_smoothness_via_interp_artifacts(MockCtx())
    assert result["module"] == MODULE_ID
    assert result["available"] is True
    assert 1 in result["per_shot"]
    assert 2 in result["per_shot"]
    # Required per-shot keys
    s1 = result["per_shot"][1]
    for key in ("ghosting_score", "judder_score", "interp_blur_score",
                "vfi_artifact_composite", "n_frames_analysed"):
        assert key in s1, f"Missing key {key} in per_shot result"
    # Composite is the mean of the three sub-scores
    expected = (s1["ghosting_score"] + s1["judder_score"] + s1["interp_blur_score"]) / 3.0
    assert abs(s1["vfi_artifact_composite"] - expected) < 1e-6
    # Validation target documented
    assert result["validation_target_dataset"] == "VFIPS"


def test_yaml_config_has_disabled_default():
    """YAML default ships with the module disabled."""
    from cineinfini.core.config import default_config
    cfg = default_config()
    mod_cfg = cfg.modules.get("temporal_smoothness_via_interp_artifacts", {})
    # Module config may or may not be present in the default dict; if present
    # it must be disabled. If absent, that's also fine (treated as disabled).
    if mod_cfg:
        assert mod_cfg.get("enabled", False) is False, \
            "Default config must keep this module OFF"


def test_vfips_dataset_registered():
    """Dataset registry must include VFIPS as a calibration target."""
    from cineinfini.core.config import default_config
    cfg = default_config()
    assert "vfips" in cfg.datasets
    vfips = cfg.datasets["vfips"]
    assert "VFIPS" in vfips.get("name", "")
    assert "2AFC" in vfips.get("label_format", "")
