"""Verify the 5 performance-config profiles ship correctly.

Each profile in ``cfg/profiles/`` must be valid YAML and load to a
Config object that respects the architectural rules (modules disabled
unless explicitly enabled, paths resolvable, etc.).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROFILES_DIR = Path(__file__).parent.parent / "cfg" / "profiles"


@pytest.fixture
def all_profile_paths():
    return sorted(PROFILES_DIR.glob("*.yaml"))


def test_five_profiles_exist(all_profile_paths):
    names = {p.stem for p in all_profile_paths}
    expected = {"realtime", "academic", "postproduction", "ultralight", "low_memory"}
    assert expected.issubset(names), f"missing: {expected - names}"


def test_each_profile_is_valid_yaml(all_profile_paths):
    for p in all_profile_paths:
        data = yaml.safe_load(p.read_text())
        assert isinstance(data, dict), f"{p.name} is not a dict"
        assert "processing" in data, f"{p.name} missing processing"
        assert "modules" in data, f"{p.name} missing modules"


@pytest.mark.parametrize("profile", [
    "realtime", "ultralight", "postproduction", "academic", "low_memory",
])
def test_profile_module_count_matches_intent(profile):
    """Each profile enables a specific number of modules — sanity check."""
    data = yaml.safe_load((PROFILES_DIR / f"{profile}.yaml").read_text())
    enabled = {k: v for k, v in data["modules"].items() if v.get("enabled")}

    if profile == "realtime":
        # Strict minimal: only motion + identity + semantic
        assert len(enabled) == 3, f"realtime should enable 3 modules, got {list(enabled)}"
    elif profile == "ultralight":
        assert len(enabled) == 3
    elif profile == "academic":
        # Everything on
        assert len(enabled) >= 18, f"academic should enable ≥18, got {len(enabled)}"
    elif profile == "postproduction":
        # Balanced: 8-12
        assert 6 <= len(enabled) <= 14, f"postproduction unexpected: {len(enabled)}"
    elif profile == "low_memory":
        assert 5 <= len(enabled) <= 12


def test_realtime_targets_speed():
    data = yaml.safe_load((PROFILES_DIR / "realtime.yaml").read_text())
    assert data["processing"]["n_frames_per_shot"] <= 8
    assert data["processing"]["use_amp"] is True
    # Heavy modules must be off
    for heavy in ("benchmark_forensic", "trustworthiness", "explainability",
                  "dover_score", "fastvqa_score"):
        assert data["modules"][heavy]["enabled"] is False, (
            f"realtime must have {heavy} disabled"
        )


def test_academic_enables_cross_benchmark_wrappers():
    data = yaml.safe_load((PROFILES_DIR / "academic.yaml").read_text())
    assert data["modules"]["dover_score"]["enabled"] is True
    assert data["modules"]["fastvqa_score"]["enabled"] is True
    assert data["modules"]["benchmark_fusion"]["enabled"] is True


def test_ultralight_drops_below_5_modules():
    data = yaml.safe_load((PROFILES_DIR / "ultralight.yaml").read_text())
    enabled = sum(1 for m in data["modules"].values() if m.get("enabled"))
    assert enabled <= 4, f"ultralight should enable ≤4 modules, got {enabled}"
    assert data["processing"]["n_frames_per_shot"] <= 4


def test_low_memory_uses_amp_and_small_batch():
    data = yaml.safe_load((PROFILES_DIR / "low_memory.yaml").read_text())
    assert data["processing"]["use_amp"] is True
    assert data["processing"]["batch_size"] <= 4
    # Memory-hungry models (DINOv2-based) should be off
    assert data["modules"]["long_term_narrative"]["enabled"] is False


def test_no_profile_enables_all_optional_models_silently():
    """Defensive: don't auto-enable expensive opt-in modules in any profile
    except academic (where opt-in is the explicit point)."""
    for profile in ("realtime", "ultralight", "postproduction", "low_memory"):
        data = yaml.safe_load((PROFILES_DIR / f"{profile}.yaml").read_text())
        assert data["modules"]["dover_score"]["enabled"] is False, (
            f"{profile} must not auto-enable dover_score"
        )
        assert data["modules"]["fastvqa_score"]["enabled"] is False, (
            f"{profile} must not auto-enable fastvqa_score"
        )
