"""
Centralized configuration for CineInfini (added in v0.4.0).

Design goals
------------
1. Single source of truth: all paths, thresholds, model URLs in one place.
2. Zero breaking changes: existing code that reads CONFIG dict or calls
   set_global_paths() / set_models_dir() keeps working unchanged.
3. Gradual migration: new modules use get_config(); old modules are
   progressively migrated in v0.5.0.
4. Test isolation: set_config(test_config()) redirects ALL output to /tmp,
   so tests never pollute ~/.cineinfini/.
5. No circular imports: this module imports nothing from cineinfini.

Usage
-----
Anywhere in the codebase (new code):

    from cineinfini.core.config import get_config
    cfg = get_config()
    max_dur = cfg.processing["max_duration_s"]
    models  = cfg.resolve_path("models_dir")

In tests (conftest.py):

    from cineinfini.core.config import set_config, test_config
    set_config(test_config())          # isolate to /tmp, fast settings

From a YAML file (user):

    from cineinfini.core.config import load_config, set_config
    set_config(load_config("~/.cineinfini/config.yaml"))

Backward compatibility layer
----------------------------
The module exposes ``compat_models_dir()`` and ``compat_reports_dir()``
for use in face_detection.py / audit.py during migration.  Once all
callers are migrated, these helpers can be removed.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Built-in defaults (no external file required)
# ---------------------------------------------------------------------------

_D_PATHS: dict[str, str] = {
    "models_dir":     "~/.cineinfini/models",
    "reports_dir":    "~/.cineinfini/reports",
    "benchmark_dir":  "~/.cineinfini/benchmark",
    "test_videos_dir":"~/.cineinfini/test_videos",
    "cache_dir":      "~/.cineinfini/cache",
    "logs_dir":       "~/.cineinfini/logs",
    "temp_dir":       "/tmp/cineinfini",
}

_D_PROCESSING: dict[str, Any] = {
    "max_duration_s":       60,
    "shot_threshold":       0.2,
    "min_shot_duration_s":  0.5,
    "downsample_to":        [320, 180],
    "n_frames_per_shot":    16,
    "frame_resize":         [320, 180],
    "step":                 2,
    "adaptive_threshold":   True,
    "threshold_percentile": 85,
    "num_workers":          4,
    "parallel_shots":       True,
    "inter_shot_subsample": 5,
    "narrative_coherence":  True,
    "compute_dtw_self":     True,
    "compute_dtw_inter":    True,
    "dtw_max_samples":      16,
    "benchmark_mode":       True,
    "embedder":             "arcface_onnx",
    "semantic_scorer":      "clip",
}

_D_THRESHOLDS: dict[str, float] = {
    "motion":              25.0,
    "ssim3d":              0.45,
    "flicker":             0.1,
    "identity_drift":      0.6,
    "ssim_long_range":     0.45,
    "clip_temp":           0.25,
    "flicker_hf":          0.01,
    "narrative_coherence": 0.7,
}

# URL registry — single source of truth for all downloadable assets.
# Update URLs here only; nothing else in the codebase has them.
_D_URLS: dict[str, dict] = {
    "arcface": {
        "url":      "https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/w600k_r50.onnx",
        "filename": "arcface.onnx",
        "sha256":   None,
    },
    "yunet": {
        "url":      "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "filename": "yunet.onnx",
        "sha256":   None,
    },
    "clip_vit_b32": {
        "url":      "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "filename": "ViT-B-32.pt",
        "sha256":   "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    },
    "bbb_10s": {
        "url":      "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_1MB.mp4",
        "filename": "BigBuckBunny_10s.mp4",
        "sha256":   None,
    },
    "bbb_full": {
        "url":      "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        "filename": "BigBuckBunny.mp4",
        "sha256":   None,
    },
}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Immutable-ish configuration for one CineInfini session.

    All path values are stored as raw strings (~ unexpanded).
    Use ``resolve_path(key)`` to get an absolute Path.
    """
    paths:      dict[str, str]    = field(default_factory=lambda: dict(_D_PATHS))
    processing: dict[str, Any]    = field(default_factory=lambda: dict(_D_PROCESSING))
    thresholds: dict[str, float]  = field(default_factory=lambda: dict(_D_THRESHOLDS))
    model_urls: dict[str, dict]   = field(default_factory=lambda: dict(_D_URLS))
    device:     str               = "cpu"

    # -----------------------------------------------------------------------
    # Path helpers
    # -----------------------------------------------------------------------

    def resolve_path(self, key: str) -> Path:
        """Return an absolute, ~ -expanded Path for a path key."""
        raw = self.paths.get(key, f"~/.cineinfini/{key}")
        return Path(raw).expanduser().resolve()

    def models_dir(self) -> Path:
        return self.resolve_path("models_dir")

    def reports_dir(self) -> Path:
        return self.resolve_path("reports_dir")

    def benchmark_dir(self) -> Path:
        return self.resolve_path("benchmark_dir")

    def test_videos_dir(self) -> Path:
        return self.resolve_path("test_videos_dir")

    # -----------------------------------------------------------------------
    # Model URL helpers
    # -----------------------------------------------------------------------

    def model_url(self, key: str) -> Optional[dict]:
        """Return the URL dict for a model, or None if not registered."""
        return self.model_urls.get(key)

    def model_path(self, key: str) -> Optional[Path]:
        """Resolve the local path for a model."""
        entry = self.model_urls.get(key)
        if entry is None:
            return None
        return self.models_dir() / entry["filename"]

    # -----------------------------------------------------------------------
    # Backward-compat bridge for audit.py / face_detection.py
    # -----------------------------------------------------------------------

    def to_audit_config(self) -> dict:
        """Convert to the legacy CONFIG dict expected by audit.py."""
        cfg = dict(self.processing)
        cfg["thresholds"] = dict(self.thresholds)
        cfg["gpu_device"] = self.device
        return cfg

    # -----------------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Build a Config from a nested dict (e.g. parsed from YAML)."""
        device = data.get("device", {})
        if isinstance(device, dict):
            device = device.get("gpu_device", "cpu")
        return cls(
            paths=      {**_D_PATHS,      **data.get("paths", {})},
            processing= {**_D_PROCESSING, **data.get("processing", {})},
            thresholds= {**_D_THRESHOLDS, **data.get("thresholds", {})},
            model_urls= {**_D_URLS,       **data.get("model_urls", {})},
            device=     device,
        )

    def replace(self, **overrides) -> "Config":
        """Return a new Config with top-level fields replaced."""
        import dataclasses
        return dataclasses.replace(self, **overrides)


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_config: Optional[Config] = None


def get_config() -> Config:
    """Return the active Config (creates a default one on first call)."""
    global _config
    if _config is None:
        _config = _auto_load()
    return _config


def set_config(cfg: Config) -> None:
    """Replace the active Config.  Use in tests before any other import."""
    global _config
    _config = cfg


def reset_config() -> None:
    """Reset to None so the next get_config() re-reads from disk/defaults."""
    global _config
    _config = None


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def default_config() -> Config:
    """Return a fresh Config with built-in defaults."""
    return Config()


def test_config(tmp_dir: Optional[str] = None) -> Config:
    """Return a Config for pytest: fast, isolated, all output to /tmp.

    Parameters
    ----------
    tmp_dir : str, optional
        Base temp directory.  A new ``tempfile.mkdtemp()`` is used if None,
        guaranteeing isolation between parallel test sessions.

    Example (conftest.py)::

        @pytest.fixture(scope="session", autouse=True)
        def isolate_config():
            from cineinfini.core.config import set_config, test_config, reset_config
            set_config(test_config())
            yield
            reset_config()
    """
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="cineinfini_test_")
    tmp = Path(tmp_dir)
    return Config(
        paths={
            "models_dir":     str(Path.home() / ".cineinfini" / "models"),  # shared
            "reports_dir":    str(tmp / "reports"),
            "benchmark_dir":  str(tmp / "benchmark"),
            "test_videos_dir":str(tmp / "videos"),
            "cache_dir":      str(tmp / "cache"),
            "logs_dir":       str(tmp / "logs"),
            "temp_dir":       str(tmp / "temp"),
        },
        processing={
            **_D_PROCESSING,
            "max_duration_s":    10,
            "n_frames_per_shot": 8,
            "num_workers":       2,
            "narrative_coherence": False,   # skip DINOv2 (slow)
            "parallel_shots":    False,
            "benchmark_mode":    False,
        },
        thresholds=dict(_D_THRESHOLDS),
        model_urls=dict(_D_URLS),
        device="cpu",
    )


def load_config(path: str | Path) -> Config:
    """Load a Config from a YAML file.

    Requires ``pyyaml``: pip install pyyaml

    Parameters
    ----------
    path : str or Path
        Path to the YAML file. ``~`` is expanded.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml required: pip install pyyaml")
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config.from_dict(data)


def save_config(cfg: Config, path: str | Path) -> None:
    """Serialize a Config to YAML."""
    try:
        import yaml, dataclasses
    except ImportError:
        raise ImportError("pyyaml required: pip install pyyaml")
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(dataclasses.asdict(cfg), f, default_flow_style=False,
                  allow_unicode=True, sort_keys=True)


# ---------------------------------------------------------------------------
# Auto-loader: YAML → env var → defaults (no error if absent)
# ---------------------------------------------------------------------------

def _auto_load() -> Config:
    """Try ~/.cineinfini/config.yaml, then CINEINFINI_CONFIG env var, then defaults."""
    for candidate in [
        os.environ.get("CINEINFINI_CONFIG"),
        str(Path.home() / ".cineinfini" / "config.yaml"),
    ]:
        if candidate and Path(candidate).expanduser().exists():
            try:
                return load_config(candidate)
            except Exception as exc:
                print(f"[cineinfini.config] Warning: could not load {candidate}: {exc}")
    return default_config()


# ---------------------------------------------------------------------------
# Backward-compatibility helpers (used during migration)
# ---------------------------------------------------------------------------

def compat_models_dir() -> Path:
    """Return models_dir from the active config. Use in face_detection.py."""
    return get_config().models_dir()


def compat_reports_dir() -> Path:
    return get_config().reports_dir()


def compat_benchmark_dir() -> Path:
    return get_config().benchmark_dir()
