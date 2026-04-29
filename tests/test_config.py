"""Test the centralized Config singleton.

Sections:
1. Existing v0.4.8.7 tests — preserved verbatim, must keep passing.
2. v0.4.8.8 additions — validate, diff, summary, merge_configs.
"""
from __future__ import annotations

import yaml
import tempfile
from pathlib import Path

import pytest

from cineinfini.core.config import (
    Config, default_config, test_config as make_test_config, get_config,
    set_config, reset_config, load_config, save_config,
    ConfigValidationError, merge_configs,
)


# ---------------------------------------------------------------------------
# 1. Pre-existing tests (kept verbatim from v0.4.8.7)
# ---------------------------------------------------------------------------
def test_default_config_has_required_sections():
    cfg = default_config()
    for section in ("paths", "device", "processing", "thresholds",
                    "model_urls", "test_videos", "modules", "reporting", "logging"):
        assert hasattr(cfg, section), f"missing section {section}"


def test_test_config_isolates_paths():
    cfg = make_test_config()
    for key in ("reports_dir", "benchmark_dir", "test_videos_dir",
                "cache_dir", "logs_dir", "temp_dir", "output_root"):
        path = cfg.paths[key]
        assert path.startswith("/tmp"), f"{key} not under /tmp: {path}"


def test_test_config_disables_optional_modules():
    cfg = make_test_config()
    enabled = cfg.enabled_modules()
    assert set(enabled) == {"motion_coherence", "identity_consistency", "semantic_consistency"}


def test_singleton_memoization():
    """Regression: get_config used to call _default_config every time."""
    reset_config()
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2


def test_set_and_reset_config():
    custom = make_test_config()
    set_config(custom)
    assert get_config() is custom
    reset_config()
    assert get_config() is not custom


def test_is_module_enabled():
    cfg = default_config()
    assert cfg.is_module_enabled("motion_coherence") is True
    assert cfg.is_module_enabled("aesthetic_cinematic") is False
    assert cfg.is_module_enabled("nonexistent") is False


def test_resolve_path_handles_user_expansion():
    cfg = default_config()
    p = cfg.resolve_path("models_dir")
    assert "~" not in str(p)


def test_save_and_load_yaml(tmp_path):
    cfg = default_config()
    cfg.modules["motion_coherence"]["threshold"] = 99.0
    out = tmp_path / "saved.yaml"
    save_config(cfg, out)
    loaded = load_config(out)
    assert loaded.modules["motion_coherence"]["threshold"] == 99.0


def test_from_dict_roundtrip():
    cfg1 = default_config()
    data = cfg1.to_dict()
    cfg2 = Config.from_dict(data)
    assert cfg2.modules.keys() == cfg1.modules.keys()
    assert cfg2.thresholds == cfg1.thresholds


def test_test_videos_section_present():
    cfg = default_config()
    assert "BBB" in cfg.test_videos
    assert cfg.test_videos["BBB"]["filename"] == "BBB.mp4"
    assert cfg.test_video_path("BBB") is not None


def test_active_renderers_explicit_overrides_legacy_flags():
    cfg = default_config()
    cfg.reporting["active_renderers"] = ["html"]
    assert cfg.active_renderers() == ["html"]


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# 2. v0.4.8.8 additions — validation framework
# ---------------------------------------------------------------------------
def test_validate_default_config_passes():
    """default_config() must always satisfy the structural schema."""
    default_config().validate()


def test_validate_test_config_passes():
    """test_config() must also satisfy the schema (parity with prod)."""
    make_test_config().validate()


def test_validate_missing_required_key_raises():
    cfg = default_config()
    cfg.thresholds.pop("motion")
    with pytest.raises(ConfigValidationError) as exc:
        cfg.validate()
    assert "thresholds.motion" in str(exc.value)


def test_validate_wrong_type_raises():
    cfg = default_config()
    cfg.processing["n_frames_per_shot"] = "sixteen"
    with pytest.raises(ConfigValidationError) as exc:
        cfg.validate()
    msg = str(exc.value)
    assert "n_frames_per_shot" in msg
    assert "expected int" in msg


def test_validate_collects_all_errors():
    """Validator must report every problem in one shot, not bail on the first."""
    cfg = default_config()
    cfg.thresholds.pop("motion")
    cfg.thresholds.pop("flicker")
    cfg.processing["num_workers"] = "many"
    with pytest.raises(ConfigValidationError) as exc:
        cfg.validate()
    msg = str(exc.value)
    assert "motion" in msg
    assert "flicker" in msg
    assert "num_workers" in msg


def test_validate_dict_section_must_be_dict():
    cfg = default_config()
    cfg.modules = "not a dict"  # type: ignore[assignment]
    with pytest.raises(ConfigValidationError) as exc:
        cfg.validate()
    assert "modules" in str(exc.value)


# ---------------------------------------------------------------------------
# 3. v0.4.8.8 additions — diff / summary / merge_configs
# ---------------------------------------------------------------------------
def test_diff_identical_configs_is_empty():
    assert default_config().diff(default_config()) == {}


def test_diff_threshold_change():
    a = default_config()
    b = default_config()
    b.thresholds["motion"] = 99.0
    d = a.diff(b)
    assert d == {"thresholds": {"motion": (25.0, 99.0)}}


def test_diff_added_key_in_other():
    a = default_config()
    b = default_config()
    b.thresholds["new_metric"] = 0.5
    d = a.diff(b)
    assert d == {"thresholds": {"new_metric": (None, 0.5)}}


def test_summary_returns_string_with_key_facts():
    cfg = default_config()
    s = cfg.summary()
    assert "CineInfini" in s
    assert "effective=" in s
    assert "max_duration_s" in s
    assert isinstance(s, str)


def test_merge_configs_override_wins():
    base = default_config()
    over = default_config()
    over.thresholds["motion"] = 99.0
    merged = merge_configs(base, over)
    assert merged.thresholds["motion"] == 99.0
    # Originals untouched
    assert base.thresholds["motion"] == 25.0
    assert over.thresholds["motion"] == 99.0


def test_merge_configs_preserves_unchanged_keys():
    base = default_config()
    over = default_config()
    over.thresholds["motion"] = 1.0
    merged = merge_configs(base, over)
    # Other thresholds carried through unchanged
    assert merged.thresholds["flicker"] == base.thresholds["flicker"]
    assert merged.thresholds["ssim3d"] == base.thresholds["ssim3d"]


def test_merge_configs_deep_merges_modules():
    base = default_config()
    over = default_config()
    over.modules["aesthetic_cinematic"]["enabled"] = True
    merged = merge_configs(base, over)
    assert merged.is_module_enabled("aesthetic_cinematic") is True
    # Untouched module config preserved
    assert merged.modules["motion_coherence"]["enabled"] is True
    # And inner keys preserved
    assert "color_harmony_weight" in merged.modules["aesthetic_cinematic"]


def test_merge_configs_returns_validatable():
    """A merged config must still satisfy the schema."""
    base = default_config()
    over = default_config()
    over.thresholds["motion"] = 30.0
    merge_configs(base, over).validate()


def test_yaml_roundtrip_then_validate(tmp_path):
    """Save → load → validate cycle must succeed."""
    out = tmp_path / "cfg.yaml"
    save_config(default_config(), out)
    loaded = load_config(out)
    loaded.validate()
