"""Shared pytest fixtures for CineInfini tests."""
import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile


@pytest.fixture
def sample_video():
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    path = Path(tmp.name)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 240))
    rng = np.random.default_rng(42)
    for _ in range(120):
        frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def sample_shot_frames():
    rng = np.random.default_rng(42)
    shot1 = [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(10)]
    shot2 = [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(10)]
    return {1: shot1, 2: shot2}


@pytest.fixture
def static_frames():
    frame = np.full((180, 320, 3), 128, dtype=np.uint8)
    return [frame.copy() for _ in range(16)]


@pytest.fixture
def noisy_frames():
    rng = np.random.default_rng(42)
    return [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(16)]
