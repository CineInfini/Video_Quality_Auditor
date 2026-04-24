"""Shared pytest fixtures for CineInfini tests."""
import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile


@pytest.fixture
def sample_video():
    """Create a temporary synthetic video for testing.

    Yields a Path to a 5-second, 24fps, 320x240 MP4 with random content.
    The file is cleaned up after the test.
    """
    # Use delete=False so we can close the file before writing with OpenCV
    # (Windows-compatible; Linux works either way).
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    path = Path(tmp.name)

    out = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 240)
    )
    rng = np.random.default_rng(42)
    for _ in range(120):  # 5 seconds at 24 fps
        frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    yield path

    # cleanup
    if path.exists():
        path.unlink()


@pytest.fixture
def sample_shot_frames():
    """Return two dummy shots of 10 frames each for coherence testing."""
    rng = np.random.default_rng(42)
    shot1 = [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(10)]
    shot2 = [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(10)]
    return {1: shot1, 2: shot2}


@pytest.fixture
def static_frames():
    """16 identical gray frames (no motion, no flicker, SSIM=1)."""
    frame = np.full((180, 320, 3), 128, dtype=np.uint8)
    return [frame.copy() for _ in range(16)]


@pytest.fixture
def noisy_frames():
    """16 independent random frames (high motion, low SSIM)."""
    rng = np.random.default_rng(42)
    return [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(16)]
