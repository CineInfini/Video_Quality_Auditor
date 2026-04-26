"""Shared pytest fixtures with environment isolation."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_global_paths(tmp_path_factory) -> Generator:
    """Redirect all output paths to /tmp for the entire test session.

    Without this fixture, tests that call `audit_video` write to
    ~/.cineinfini/ and pollute the user's production environment.
    The isolation is transparent to test code.
    """
    from cineinfini import set_global_paths
    from cineinfini.core.face_detection import set_models_dir

    tmp = tmp_path_factory.mktemp("cineinfini_session")
    (tmp / "reports").mkdir()
    (tmp / "bench").mkdir()
    set_global_paths(tmp / "reports", tmp / "bench")
    # Share real models dir (downloaded once, never regenerated)
    real_models = Path.home() / ".cineinfini" / "models"
    set_models_dir(real_models)
    yield
    # No teardown needed — tmp_path_factory cleans up automatically


@pytest.fixture
def sample_video(tmp_path: Path) -> Generator[Path, None, None]:
    """Synthetic 5-second 320×240 video."""
    path = tmp_path / "sample.mp4"
    out = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 240)
    )
    rng = np.random.default_rng(42)
    for _ in range(120):
        frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def sample_shot_frames() -> dict[int, list]:
    """Two dummy shots of 10 random frames each."""
    rng = np.random.default_rng(42)
    return {
        1: [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(10)],
        2: [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(10)],
    }


@pytest.fixture
def static_frames() -> list[np.ndarray]:
    """16 identical gray frames (SSIM=1, flicker=0)."""
    frame = np.full((180, 320, 3), 128, dtype=np.uint8)
    return [frame.copy() for _ in range(16)]


@pytest.fixture
def noisy_frames() -> list[np.ndarray]:
    """16 independent random frames."""
    rng = np.random.default_rng(42)
    return [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(16)]
