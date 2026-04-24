"""
Core metrics tests: motion, SSIM, flicker.
"""
import numpy as np
from cineinfini.core.metrics import motion_peak_div, ssim3d_self, flicker_score, ssim_long_range, flicker_highfreq_variance

def test_motion_peak_div_static():
    frame = np.full((180,320,3),128,dtype=np.uint8)
    frames = [frame.copy() for _ in range(16)]
    assert motion_peak_div(frames) < 1.0

def test_ssim3d_self_static():
    frame = np.full((180,320,3),128,dtype=np.uint8)
    frames = [frame.copy() for _ in range(16)]
    assert ssim3d_self(frames) > 0.99

def test_flicker_score_static():
    frame = np.full((180,320,3),128,dtype=np.uint8)
    frames = [frame.copy() for _ in range(16)]
    assert flicker_score(frames) < 0.01

def test_ssim_long_range_identical():
    frame = np.full((180,320,3),128,dtype=np.uint8)
    frames = [frame.copy() for _ in range(10)]
    assert ssim_long_range(frames) > 0.99

def test_flicker_highfreq_variance_static():
    frame = np.full((180,320,3),128,dtype=np.uint8)
    frames = [frame.copy() for _ in range(10)]
    assert flicker_highfreq_variance(frames) < 0.1
