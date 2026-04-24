import pytest
import numpy as np
from cineinfini.core.metrics import (
    motion_peak_div,
    ssim3d_self,
    flicker_score,
    ssim_long_range,
    flicker_highfreq_variance,
    compute_composite_score,
    recompute_composite_scores,
)


class TestMotionPeakDiv:
    def test_static_frames(self, static_frames):
        result = motion_peak_div(static_frames)
        assert result is not None
        assert result < 1.0

    def test_noisy_frames(self, noisy_frames):
        result = motion_peak_div(noisy_frames)
        assert result is not None
        assert result > 5.0

    def test_too_few_frames(self):
        f = np.zeros((180, 320, 3), dtype=np.uint8)
        assert motion_peak_div([f, f]) is None


class TestSSIM3D:
    def test_static(self, static_frames):
        result = ssim3d_self(static_frames)
        assert result is not None
        assert result > 0.99

    def test_noisy(self, noisy_frames):
        result = ssim3d_self(noisy_frames)
        assert result is not None
        assert result < 0.3

    def test_too_few_frames(self):
        f = np.zeros((180, 320, 3), dtype=np.uint8)
        assert ssim3d_self([f] * 15) is None


class TestFlicker:
    def test_static(self, static_frames):
        result = flicker_score(static_frames)
        assert result is not None
        assert result < 0.01

    def test_noisy(self, noisy_frames):
        result = flicker_score(noisy_frames)
        assert result is not None
        assert 0.1 < result < 0.5


class TestSSIMLongRange:
    def test_identical(self, static_frames):
        result = ssim_long_range(static_frames)
        assert result is not None
        assert result > 0.99

    def test_single_frame(self):
        f = np.zeros((180, 320, 3), dtype=np.uint8)
        assert ssim_long_range([f]) is None


class TestFlickerHF:
    def test_static(self, static_frames):
        result = flicker_highfreq_variance(static_frames)
        assert result is not None
        assert result >= 0.0
        assert result < 0.1


class TestCompositeScores:
    def test_compute_composite(self):
        metrics = {"motion_mean": 10.0, "ssim_mean": 0.8, "flicker_mean": 0.05,
                   "identity_mean": 0.2, "ssim_lr_mean": 0.7, "clip_temp_mean": 0.9}
        score = compute_composite_score(metrics)
        assert abs(score - (-7.85)) < 0.001

    def test_composite_skips_none(self):
        metrics = {"motion_mean": 10.0, "ssim_mean": 0.8, "identity_mean": None, "flicker_mean": None}
        score = compute_composite_score(metrics)
        assert abs(score - (-9.2)) < 0.001

    def test_recompute_composite_scores(self):
        gates = {
            1: {"motion_peak_div": 10, "ssim3d_self": 0.8, "flicker": 0.05,
                "identity_intra": 0.2, "ssim_long_range": 0.7, "clip_temp_consistency": 0.9},
            2: {"motion_peak_div": 20, "ssim3d_self": 0.6, "flicker": 0.1,
                "identity_intra": 0.5, "ssim_long_range": 0.4, "clip_temp_consistency": 0.8},
        }
        weights = {"motion_mean": -2.0, "ssim_mean": 1.0}
        recompute_composite_scores(gates, weights)
        assert abs(gates[1]["composite"] - (-19.2)) < 0.001
        assert abs(gates[2]["composite"] - (-39.4)) < 0.001
