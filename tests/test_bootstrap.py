"""Test the bootstrap module (system deps, models, test videos)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from cineinfini.core.bootstrap import (
    AssetStatus, BootstrapReport,
    check_system_dependency, ensure_system_deps,
    ensure_models, ensure_test_videos, bootstrap_all,
    _ensure_one, _verify_asset, _sha256_file,
)
from cineinfini.core.config import default_config, set_config, get_config


# ---------------------------------------------------------------------------
# System deps
# ---------------------------------------------------------------------------
def test_check_system_dependency_known_binary():
    # python should always be on PATH where pytest runs
    present, version = check_system_dependency("python3")
    if present:
        assert version is not None


def test_check_system_dependency_missing_binary():
    present, version = check_system_dependency("definitely_not_installed_xyz_42")
    assert present is False
    assert version is None


def test_ensure_system_deps_returns_dict():
    result = ensure_system_deps(("python3",))
    assert "python3" in result
    assert "present" in result["python3"]


def test_ensure_system_deps_raises_when_required_missing():
    with pytest.raises(RuntimeError, match="Missing required"):
        ensure_system_deps(("definitely_not_installed_xyz_42",), raise_on_missing=True)


def test_ensure_system_deps_includes_install_hint():
    result = ensure_system_deps(("definitely_not_installed_xyz_42",))
    entry = result["definitely_not_installed_xyz_42"]
    assert entry["present"] is False
    assert "hint" in entry


# ---------------------------------------------------------------------------
# Asset verification
# ---------------------------------------------------------------------------
def test_verify_asset_missing(tmp_path):
    present, sha_ok, size = _verify_asset(tmp_path / "nope", None)
    assert present is False
    assert size == 0


def test_verify_asset_existing_no_sha(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    present, sha_ok, size = _verify_asset(p, None)
    assert present is True
    assert sha_ok is None
    assert size == 5


def test_verify_asset_sha_mismatch(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    present, sha_ok, size = _verify_asset(p, "0" * 64)
    assert present is True
    assert sha_ok is False


def test_verify_asset_sha_match(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    real_sha = _sha256_file(p)
    present, sha_ok, size = _verify_asset(p, real_sha)
    assert sha_ok is True


# ---------------------------------------------------------------------------
# ensure_one (with skip_download)
# ---------------------------------------------------------------------------
def test_ensure_one_skip_download(tmp_path):
    target = tmp_path / "missing.bin"
    s = _ensure_one("test", "http://example.com", target, skip_download=True)
    assert isinstance(s, AssetStatus)
    assert s.present is False
    assert s.error == "asset missing and skip_download=True"


def test_ensure_one_already_present(tmp_path):
    target = tmp_path / "exists.bin"
    target.write_bytes(b"already here")
    s = _ensure_one("test", "http://example.com", target)
    assert s.present is True
    assert s.downloaded is False
    assert s.size_bytes == len(b"already here")


def test_ensure_one_no_url_no_file(tmp_path):
    target = tmp_path / "missing.bin"
    s = _ensure_one("test", "", target)
    assert s.present is False
    assert s.error == "no URL configured"


# ---------------------------------------------------------------------------
# ensure_models / ensure_test_videos
# ---------------------------------------------------------------------------
def test_ensure_models_skip_download_returns_one_status_per_key(tmp_path):
    cfg = default_config()
    cfg.paths["models_dir"] = str(tmp_path / "models")
    set_config(cfg)
    statuses = ensure_models(cfg, skip_download=True)
    assert len(statuses) == len(cfg.model_urls)
    assert all(s.present is False for s in statuses)


def test_ensure_models_only_filter(tmp_path):
    cfg = default_config()
    cfg.paths["models_dir"] = str(tmp_path / "models")
    set_config(cfg)
    statuses = ensure_models(cfg, only=["yunet"], skip_download=True)
    assert len(statuses) == 1
    assert statuses[0].name == "model:yunet"


def test_ensure_test_videos_uses_config_paths(tmp_path):
    cfg = default_config()
    cfg.paths["test_videos_dir"] = str(tmp_path / "v")
    set_config(cfg)
    statuses = ensure_test_videos(cfg, skip_download=True)
    assert len(statuses) == len(cfg.test_videos)
    for s in statuses:
        assert str(tmp_path / "v") in str(s.path)


# ---------------------------------------------------------------------------
# bootstrap_all
# ---------------------------------------------------------------------------
def test_bootstrap_all_returns_report(tmp_path):
    cfg = default_config()
    cfg.paths["models_dir"] = str(tmp_path / "m")
    cfg.paths["test_videos_dir"] = str(tmp_path / "v")
    set_config(cfg)
    report = bootstrap_all(cfg, require_ffmpeg=False, skip_download=True)
    assert isinstance(report, BootstrapReport)
    assert isinstance(report.system, dict)
    assert len(report.models) == len(cfg.model_urls)
    assert len(report.videos) == len(cfg.test_videos)


def test_bootstrap_report_serialisable(tmp_path):
    cfg = default_config()
    cfg.paths["models_dir"] = str(tmp_path / "m")
    cfg.paths["test_videos_dir"] = str(tmp_path / "v")
    report = bootstrap_all(cfg, require_ffmpeg=False, skip_download=True)
    data = report.to_dict()
    assert "ok" in data
    assert "models" in data
    # Should round-trip through JSON
    encoded = json.dumps(data)
    assert "models" in json.loads(encoded)


def test_bootstrap_uses_config_paths_no_hardcoding(tmp_path):
    """No model/video should ever land outside the configured paths."""
    cfg = default_config()
    cfg.paths["models_dir"] = str(tmp_path / "isolated_models")
    cfg.paths["test_videos_dir"] = str(tmp_path / "isolated_videos")
    set_config(cfg)
    report = bootstrap_all(cfg, require_ffmpeg=False, skip_download=True)
    for m in report.models:
        assert str(tmp_path / "isolated_models") in str(m.path)
    for v in report.videos:
        assert str(tmp_path / "isolated_videos") in str(v.path)
