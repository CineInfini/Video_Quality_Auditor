import pytest
import numpy as np
from cineinfini.core.metrics import motion_peak_div, ssim3d_self, flicker_score

def test_motion_peak_div_static(static_frames):
    result = motion_peak_div(static_frames)
    assert result is not None
    assert result < 1.0  # static frames → near-zero motion

def test_motion_peak_div_noisy(noisy_frames):
    result = motion_peak_div(noisy_frames)
    assert result is not None
    assert result > 5.0  # noise → high motion

def test_ssim3d_self_static(static_frames):
    result = ssim3d_self(static_frames)
    assert result is not None
    assert result > 0.99

def test_ssim3d_self_noisy(noisy_frames):
    result = ssim3d_self(noisy_frames)
    assert result is not None
    assert result < 0.3

def test_flicker_score_static(static_frames):
    result = flicker_score(static_frames)
    assert result is not None
    assert result < 0.01

def test_flicker_score_noisy(noisy_frames):
    result = flicker_score(noisy_frames)
    assert result is not None
    assert 0.1 < result < 0.5
