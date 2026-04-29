"""Tests for the advanced pure-CV feature module (v0.4.11)."""
import numpy as np
import pytest

from cineinfini.metrics.advanced_pure_cv import (
    brisque_features, lbp_entropy, canny_edge_density, dct_block_energy,
    shannon_entropy, multi_scale_sobel, all_advanced_features,
)


@pytest.fixture
def random_frame():
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def gray_random(random_frame):
    import cv2
    return cv2.cvtColor(random_frame, cv2.COLOR_BGR2GRAY)


def test_brisque_features_shape(gray_random):
    f = brisque_features(gray_random)
    assert f.shape == (36,)
    assert np.isfinite(f).all()


def test_lbp_entropy_positive(gray_random):
    e = lbp_entropy(gray_random)
    assert e > 0
    assert e < 10


def test_canny_edge_density_in_range(gray_random):
    d = canny_edge_density(gray_random)
    assert 0 <= d <= 1


def test_dct_block_energy_positive(gray_random):
    e = dct_block_energy(gray_random)
    assert e >= 0


def test_shannon_entropy_in_range(gray_random):
    e = shannon_entropy(gray_random)
    assert 0 <= e <= 8


def test_multi_scale_sobel_returns_three(gray_random):
    s1, s2, s3 = multi_scale_sobel(gray_random)
    for s in (s1, s2, s3):
        assert s >= 0
        assert np.isfinite(s)


def test_all_advanced_features_keys(random_frame):
    feats = all_advanced_features([random_frame, random_frame])
    expected_named = {
        "brisque_alpha_native", "brisque_sigma2_native",
        "brisque_alpha_halfscale", "brisque_sigma2_halfscale",
        "lbp_entropy", "hog_cell_var", "canny_edge_density",
        "dct_block_energy", "shannon_entropy_y",
        "sobel_scale_1", "sobel_scale_half", "sobel_scale_quarter",
    }
    assert expected_named.issubset(set(feats.keys()))
    # All 36 brisque_fNN keys
    for i in range(36):
        assert f"brisque_f{i:02d}" in feats


def test_all_advanced_features_handles_empty():
    feats = all_advanced_features([])
    assert feats == {}
