"""Test the core metric functions (motion, SSIM, flicker)."""
from __future__ import annotations

import numpy as np
import pytest

from cineinfini.core.metrics import (
    motion_peak_div, ssim3d_self,
    flicker_score, flicker_highfreq_variance,
    ssim_long_range,
    compute_composite_score, recompute_composite_scores,
    DEFAULT_COMPOSITE_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Static frames -> low motion, high SSIM, low flicker
# ---------------------------------------------------------------------------
def test_motion_peak_div_static(static_frames):
    assert motion_peak_div(static_frames) < 1.0


def test_ssim3d_self_static(static_frames):
    assert ssim3d_self(static_frames) > 0.99


def test_flicker_score_static(static_frames):
    assert flicker_score(static_frames) < 0.01


def test_flicker_highfreq_variance_static(static_frames):
    assert flicker_highfreq_variance(static_frames) < 0.1


def test_ssim_long_range_identical(static_frames):
    assert ssim_long_range(static_frames) > 0.99


# ---------------------------------------------------------------------------
# Noisy frames -> high flicker, low SSIM
# ---------------------------------------------------------------------------
def test_flicker_score_noisy_higher(noisy_frames, static_frames):
    assert flicker_score(noisy_frames) > flicker_score(static_frames)


def test_ssim_lr_noisy_lower(noisy_frames):
    assert ssim_long_range(noisy_frames) < 0.5


# ---------------------------------------------------------------------------
# Moving square -> non-trivial motion
# ---------------------------------------------------------------------------
def test_motion_peak_div_moving(shifted_frames):
    val = motion_peak_div(shifted_frames)
    assert val is not None
    assert val > 0.0


# ---------------------------------------------------------------------------
# Edge cases: too few frames -> None
# ---------------------------------------------------------------------------
def test_motion_peak_div_too_few():
    f = np.zeros((180, 320, 3), dtype=np.uint8)
    assert motion_peak_div([f, f]) is None


def test_ssim3d_self_too_few():
    f = np.zeros((180, 320, 3), dtype=np.uint8)
    assert ssim3d_self([f, f, f]) is None  # needs >=16


def test_flicker_score_too_few():
    f = np.zeros((180, 320, 3), dtype=np.uint8)
    assert flicker_score([f, f]) is None


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------
def test_compute_composite_score_with_partial_metrics():
    metrics = {"motion_mean": 1.0, "ssim_mean": 0.9, "flicker_mean": 0.05}
    score = compute_composite_score(metrics)
    assert isinstance(score, float)


def test_compute_composite_score_ignores_none():
    metrics = {"motion_mean": None, "ssim_mean": 0.9}
    score = compute_composite_score(metrics)
    assert score == DEFAULT_COMPOSITE_WEIGHTS["ssim_mean"] * 0.9


def test_recompute_composite_scores_updates_dict():
    gates = {1: {"motion_peak_div": 1.0, "ssim3d_self": 0.9, "flicker": 0.05}}
    out = recompute_composite_scores(gates)
    assert "composite" in out[1]
    assert isinstance(out[1]["composite"], float)
