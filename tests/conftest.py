"""Shared pytest fixtures + heavy-dep stubs.

The audit pipeline imports ``torch`` / ``open_clip`` / ``transformers`` /
``onnxruntime`` at module load time. When those aren't installed we
register lightweight stubs **before** the first ``cineinfini`` import so
the package's ``__init__.py`` doesn't blow up. On real machines (where
the libraries are installed) this is a no-op.
"""
from __future__ import annotations

import sys
import types


def _ensure_stub(name: str) -> None:
    if name in sys.modules:
        return

    class _Stub(types.ModuleType):
        def __getattr__(self, attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            sub = _Stub(f"{self.__name__}.{attr}")
            setattr(self, attr, sub)
            return sub

        def __call__(self, *a, **k):
            return _Stub(self.__name__)

        def __iter__(self):
            return iter([])

    sys.modules[name] = _Stub(name)


for _m in (
    "torch", "torch.nn", "torch.nn.functional", "torch.cuda",
    "torchvision", "open_clip", "transformers",
    "onnxruntime",
    "plotly", "plotly.express", "plotly.graph_objects", "plotly.subplots",
    "weasyprint",
    "reportlab", "reportlab.lib", "reportlab.lib.pagesizes",
    "reportlab.lib.styles", "reportlab.platypus",
    "fpdf", "fpdf2",
):
    _ensure_stub(_m)


# ---------------------------------------------------------------------------
# Real fixtures
# ---------------------------------------------------------------------------
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def static_frames():
    frame = np.full((180, 320, 3), 128, dtype=np.uint8)
    return [frame.copy() for _ in range(16)]


@pytest.fixture
def noisy_frames():
    rng = np.random.default_rng(42)
    return [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(16)]


@pytest.fixture
def shifted_frames():
    frames = []
    for t in range(16):
        f = np.full((180, 320, 3), 30, dtype=np.uint8)
        x = 20 + t * 8
        f[60:120, x:x + 40] = 220
        frames.append(f)
    return frames


@pytest.fixture
def two_shot_frames():
    rng = np.random.default_rng(0)
    return {
        1: [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(10)],
        2: [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(10)],
    }


@pytest.fixture
def tmp_models_dir():
    with tempfile.TemporaryDirectory() as t:
        yield Path(t)


@pytest.fixture
def real_models_dir():
    """Resolve the real models directory from the production config.

    Skips the test if no model weights have been downloaded yet (i.e. the
    user hasn't run ``cineinfini bootstrap``). This lets a test opt-in to
    real ArcFace / CLIP / DINOv2 weights without forcing the suite to
    download ~270 MB on every CI run.
    """
    from cineinfini.core.config import default_config
    cfg = default_config()
    models_dir = cfg.models_dir()
    if not models_dir.exists() or not any(models_dir.iterdir()):
        pytest.skip(
            f"No models in {models_dir} — run `cineinfini bootstrap` first"
        )
    yield models_dir


@pytest.fixture
def real_test_video():
    """Resolve a real downloaded test video (BBB by default).

    Skips the test if the user has not run ``cineinfini bootstrap``.
    Uses the same Config that production uses, so the test auto-picks
    up whatever the user configured in ``cfg/config.yaml -> paths``.
    """
    from cineinfini.core.config import default_config
    cfg = default_config()
    bbb = cfg.test_video_path("BBB")
    if bbb is None or not bbb.exists():
        pytest.skip(
            f"No test video at {bbb} — run `cineinfini bootstrap --videos-only`"
        )
    yield bbb


@pytest.fixture(autouse=True)
def isolated_config(tmp_path):
    from cineinfini.core.config import set_config, reset_config, test_config
    cfg = test_config()
    cfg.paths["models_dir"] = str(tmp_path / "models")
    cfg.paths["test_videos_dir"] = str(tmp_path / "videos")
    set_config(cfg)
    yield cfg
    reset_config()
