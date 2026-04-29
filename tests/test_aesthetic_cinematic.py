"""Tests for the aesthetic_cinematic optional module."""
from __future__ import annotations

import numpy as np
import pytest

from cineinfini.modules.aesthetic_cinematic import (
    rule_of_thirds_score, color_harmony_score, contrast_score,
    score_frames, MOD_ID, VERSION,
)
from cineinfini.core.config import get_config, set_config, default_config
from cineinfini.core.registry import all_modules, get_active_modules


# ---------------------------------------------------------------------------
# Sub-scorers
# ---------------------------------------------------------------------------
class TestRuleOfThirds:
    def test_centered_subject_low_score(self):
        # Bright square in the dead center -> low score (centered, not on thirds)
        f = np.full((180, 320, 3), 30, dtype=np.uint8)
        f[80:100, 150:170] = 220
        s = rule_of_thirds_score(f)
        assert 0.0 <= s < 0.6

    def test_thirds_subject_high_score(self):
        # Bright square at the (1/3, 1/3) intersection -> high score
        f = np.full((180, 320, 3), 30, dtype=np.uint8)
        # Frame 320x180, intersection at (320/3=106, 180/3=60)
        f[55:65, 100:115] = 220
        s = rule_of_thirds_score(f)
        assert s > 0.7

    def test_blank_frame_zero(self):
        f = np.full((180, 320, 3), 128, dtype=np.uint8)
        s = rule_of_thirds_score(f)
        # No edges -> sum=0 -> score 0
        assert s == 0.0


class TestColorHarmony:
    def test_complementary_colors(self):
        # Half blue, half orange (~180° apart in hue)
        f = np.zeros((180, 320, 3), dtype=np.uint8)
        f[:, :160] = (200, 80, 0)    # blue (BGR)
        f[:, 160:] = (0, 100, 220)   # orange-ish
        s, scheme = color_harmony_score(f)
        assert s > 0.4
        # Scheme should plausibly match either complementary or analogous
        assert scheme in {"complementary", "analogous", "split_complementary"}

    def test_monochrome(self):
        # Single hue -> monochrome scheme expected
        f = np.zeros((180, 320, 3), dtype=np.uint8)
        f[:, :] = (50, 100, 200)
        s, scheme = color_harmony_score(f)
        # All hue concentrated -> high mono score
        assert s > 0.4

    def test_grey_frame_returns_zero(self):
        f = np.full((180, 320, 3), 128, dtype=np.uint8)
        s, scheme = color_harmony_score(f)
        assert s == 0.0
        assert scheme == "none"


class TestContrast:
    def test_flat_frame_low(self):
        f = np.full((180, 320, 3), 128, dtype=np.uint8)
        assert contrast_score(f) == 0.0

    def test_high_contrast(self):
        f = np.zeros((180, 320, 3), dtype=np.uint8)
        f[:, :160] = 0
        f[:, 160:] = 255
        s = contrast_score(f)
        assert s > 0.9


# ---------------------------------------------------------------------------
# Integration with VideoContext + registry
# ---------------------------------------------------------------------------
class TestAestheticPipeline:
    def test_module_registered(self):
        registry = all_modules()
        assert MOD_ID in registry
        assert registry[MOD_ID].version == VERSION

    def test_module_disabled_by_default(self):
        cfg = default_config()
        assert cfg.is_module_enabled(MOD_ID) is False

    def test_module_can_be_enabled(self):
        cfg = default_config()
        cfg.modules[MOD_ID]["enabled"] = True
        set_config(cfg)
        assert cfg.is_module_enabled(MOD_ID) is True
        active = [e.mod_id for e in get_active_modules()]
        assert MOD_ID in active

    def test_score_frames_returns_required_keys(self):
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
                  for _ in range(8)]
        result = score_frames(frames)
        for key in ("composite", "rule_of_thirds", "color_harmony",
                    "color_scheme", "contrast", "n_sampled"):
            assert key in result
        assert 0.0 <= result["composite"] <= 1.0

    def test_score_frames_empty_input(self):
        result = score_frames([])
        assert result["composite"] is None
        assert result["n_sampled"] == 0

    def test_run_via_registry(self, two_shot_frames):
        """End-to-end: enable the module and call its registry func."""
        from cineinfini.core.context import VideoContext, VideoInfoLite
        from cineinfini.core.registry import all_modules

        cfg = default_config()
        cfg.modules[MOD_ID]["enabled"] = True
        set_config(cfg)
        from pathlib import Path
        ctx = VideoContext(
            video=VideoInfoLite(path=Path("/tmp/dummy.mp4")),
            shot_frames=two_shot_frames,
            cfg=cfg,
        )
        entry = all_modules()[MOD_ID]
        out = entry.func(ctx)
        assert out["module"] == MOD_ID
        assert "per_shot" in out
        assert set(out["per_shot"].keys()) == set(two_shot_frames.keys())
        assert out["summary"]["n_shots"] == 2

    def test_weights_read_from_config(self):
        cfg = default_config()
        cfg.modules[MOD_ID]["color_harmony_weight"] = 0.7
        cfg.modules[MOD_ID]["contrast_weight"] = 0.2
        cfg.modules[MOD_ID]["composition_weight"] = 0.1
        cfg.modules[MOD_ID]["enabled"] = True
        set_config(cfg)
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
                  for _ in range(4)]
        from cineinfini.core.context import VideoContext, VideoInfoLite
        from cineinfini.core.registry import all_modules
        from pathlib import Path
        ctx = VideoContext(
            video=VideoInfoLite(path=Path("/tmp/dummy.mp4")),
            shot_frames={1: frames}, cfg=cfg,
        )
        out = all_modules()[MOD_ID].func(ctx)
        assert out["weights"]["color"] == 0.7
        assert out["weights"]["contrast"] == 0.2
        assert out["weights"]["composition"] == 0.1
