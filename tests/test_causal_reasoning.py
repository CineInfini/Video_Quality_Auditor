"""Tests for the causal_reasoning optional module.

We craft three deterministic synthetic scenarios:

1. **Falling square**: small bright square moves *down* over 16 frames -
   physically plausible, low violation score.
2. **Rising square**: same square but moves *up* - should trigger a high
   anti-gravity violation score.
3. **Static frames**: no motion - all sub-metrics return None.

Optical flow is computed via OpenCV's Farneback so the asserted scores
are not contrived; they reflect real flow behaviour on synthetic input.
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from cineinfini.modules.causal_reasoning import (
    upward_flow_ratio, vertical_flow_imbalance,
    trajectory_curvature_violation, score_shot,
    MOD_ID, VERSION,
)
from cineinfini.core.config import default_config, set_config
from cineinfini.core.registry import all_modules, get_active_modules


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------
def _falling_frames(n: int = 16, h: int = 240, w: int = 320) -> list:
    """Bright square falling from top to bottom — physically plausible."""
    frames = []
    sq = 30
    for t in range(n):
        f = np.full((h, w, 3), 30, dtype=np.uint8)
        y = 20 + t * 12               # +12 px per frame, downward
        x = w // 2 - sq // 2
        if y + sq < h:
            f[y:y + sq, x:x + sq] = 220
        frames.append(f)
    return frames


def _rising_frames(n: int = 16, h: int = 240, w: int = 320) -> list:
    """Bright square rising from bottom to top — anti-gravity."""
    frames = []
    sq = 30
    for t in range(n):
        f = np.full((h, w, 3), 30, dtype=np.uint8)
        y = h - 50 - t * 12           # moves UP
        x = w // 2 - sq // 2
        if y > 0:
            f[y:y + sq, x:x + sq] = 220
        frames.append(f)
    return frames


def _static_frames(n: int = 16, h: int = 240, w: int = 320) -> list:
    f = np.full((h, w, 3), 128, dtype=np.uint8)
    return [f.copy() for _ in range(n)]


# ---------------------------------------------------------------------------
# Sub-metric tests
# ---------------------------------------------------------------------------
class TestFlowSubMetrics:
    def test_upward_ratio_falling(self):
        """Falling square -> few upward pixels."""
        import cv2
        frames = _falling_frames()
        prev = cv2.cvtColor(frames[5], cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(frames[6], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        ratio = upward_flow_ratio(flow)
        assert ratio is not None
        assert ratio < 0.4, f"falling square: expected low upward ratio, got {ratio}"

    def test_upward_ratio_rising(self):
        """Rising square -> mostly upward pixels."""
        import cv2
        frames = _rising_frames()
        prev = cv2.cvtColor(frames[5], cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(frames[6], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        ratio = upward_flow_ratio(flow)
        assert ratio is not None
        assert ratio > 0.6, f"rising square: expected high upward ratio, got {ratio}"

    def test_imbalance_falling_low(self):
        import cv2
        frames = _falling_frames()
        prev = cv2.cvtColor(frames[5], cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(frames[6], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        v = vertical_flow_imbalance(flow)
        assert v is not None
        assert v < 0.4

    def test_imbalance_rising_high(self):
        import cv2
        frames = _rising_frames()
        prev = cv2.cvtColor(frames[5], cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(frames[6], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        v = vertical_flow_imbalance(flow)
        assert v is not None
        assert v > 0.6


class TestTrajectoryCurvature:
    def test_falling_no_violation(self):
        # y increases roughly linearly -> 2nd derivative ≈ 0, not negative
        ys = [(160.0, 20.0 + t * 12.0) for t in range(16)]
        score = trajectory_curvature_violation(ys)
        assert score is not None
        # Linear motion: 2nd derivative ≈ 0; not "consistently upward accel"
        assert score < 0.3

    def test_anti_gravity_high(self):
        # Strong upward acceleration -> high score
        ys = [(160.0, 200.0 - t * t * 2.0) for t in range(16)]
        score = trajectory_curvature_violation(ys)
        assert score is not None
        assert score > 0.5

    def test_too_few_points(self):
        assert trajectory_curvature_violation([(0.0, 0.0)]) is None
        assert trajectory_curvature_violation([(0.0, 0.0), (1.0, 1.0)]) is None


# ---------------------------------------------------------------------------
# score_shot integration
# ---------------------------------------------------------------------------
class TestScoreShot:
    def test_falling_low_violation(self):
        result = score_shot(_falling_frames())
        assert result["causal_violation"] is not None
        assert result["causal_violation"] < 0.35
        assert result["n_pairs"] > 0

    def test_rising_high_violation(self):
        result = score_shot(_rising_frames())
        assert result["causal_violation"] is not None
        assert result["causal_violation"] > 0.5

    def test_rising_higher_than_falling(self):
        falling = score_shot(_falling_frames())["causal_violation"]
        rising = score_shot(_rising_frames())["causal_violation"]
        assert rising > falling, f"expected rising > falling, got {rising} vs {falling}"

    def test_static_no_signal(self):
        result = score_shot(_static_frames())
        assert result["causal_violation"] is None
        assert result["upward_flow_ratio"] is None

    def test_too_few_frames(self):
        result = score_shot([np.zeros((100, 100, 3), dtype=np.uint8)] * 2)
        assert result["causal_violation"] is None
        assert result["n_pairs"] == 0


# ---------------------------------------------------------------------------
# Pipeline + registry integration
# ---------------------------------------------------------------------------
class TestPipelineIntegration:
    def test_module_registered(self):
        registry = all_modules()
        assert MOD_ID in registry
        assert registry[MOD_ID].version == VERSION

    def test_module_disabled_by_default(self):
        cfg = default_config()
        assert cfg.is_module_enabled(MOD_ID) is False

    def test_module_enables_correctly(self):
        cfg = default_config()
        cfg.modules[MOD_ID]["enabled"] = True
        set_config(cfg)
        active = [e.mod_id for e in get_active_modules()]
        assert MOD_ID in active

    def test_run_via_registry(self):
        from cineinfini.core.context import VideoContext, VideoInfoLite
        cfg = default_config()
        cfg.modules[MOD_ID]["enabled"] = True
        set_config(cfg)
        ctx = VideoContext(
            video=VideoInfoLite(path=Path("/tmp/dummy.mp4")),
            shot_frames={1: _falling_frames(), 2: _rising_frames()},
            cfg=cfg,
        )
        out = all_modules()[MOD_ID].func(ctx)
        assert out["module"] == MOD_ID
        assert out["summary"]["n_shots"] == 2
        # Shot 2 (rising) should violate, shot 1 (falling) should not
        s1 = out["per_shot"][1]["causal_violation"]
        s2 = out["per_shot"][2]["causal_violation"]
        assert s2 > s1

    def test_weights_read_from_config(self):
        from cineinfini.core.context import VideoContext, VideoInfoLite
        cfg = default_config()
        cfg.modules[MOD_ID]["enabled"] = True
        cfg.modules[MOD_ID]["upward_weight"] = 0.7
        cfg.modules[MOD_ID]["imbalance_weight"] = 0.2
        cfg.modules[MOD_ID]["curvature_weight"] = 0.1
        set_config(cfg)
        ctx = VideoContext(
            video=VideoInfoLite(path=Path("/tmp/dummy.mp4")),
            shot_frames={1: _falling_frames()},
            cfg=cfg,
        )
        out = all_modules()[MOD_ID].func(ctx)
        assert out["weights"]["upward"] == 0.7
        assert out["weights"]["imbalance"] == 0.2
        assert out["weights"]["curvature"] == 0.1
