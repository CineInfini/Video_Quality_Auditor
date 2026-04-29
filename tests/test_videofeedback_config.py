"""Tests for v0.4.8.7 — corrected VideoFeedback config + HF fetcher dispatch."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def _run_cli(*args, timeout=60):
    return subprocess.run(
        [sys.executable, "-m", "cineinfini.cli.main", *args],
        cwd=REPO,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=timeout,
    )


def test_videofeedback_size_corrected():
    """The bogus 50 GB / .tar / MIT entries are gone."""
    from cineinfini.core.config import default_config
    cfg = default_config()
    vfb = cfg.datasets["videofeedback"]
    assert vfb.get("size_mb") == 12.6, "Real size is 12.6 MB, not 50 GB"
    assert "size_gb" not in vfb, "size_gb should not be present"
    assert vfb.get("license") == "Apache-2.0", "License is Apache-2.0, not MIT"
    assert vfb.get("n_samples") == 37661
    assert vfb.get("label_range") == [1, 4]


def test_videofeedback_has_label_fields():
    """All 5 MOS dimensions are documented."""
    from cineinfini.core.config import default_config
    vfb = default_config().datasets["videofeedback"]
    expected = {
        "visual quality", "temporal consistency", "dynamic degree",
        "text-to-video alignment", "factual consistency",
    }
    assert set(vfb["label_fields"]) == expected


def test_videofeedback_uses_hf_fetcher():
    """Fetch method should route through `huggingface_datasets`, not a tar URL."""
    from cineinfini.core.config import default_config
    vfb = default_config().datasets["videofeedback"]
    assert vfb.get("fetch_method") == "huggingface_datasets"
    assert "load_dataset" in vfb.get("fetch_command", "")
    assert "TIGER-Lab/VideoFeedback" in vfb.get("fetch_command", "")


def test_videofeedback_has_videos_mirror():
    """The separate MP4 mirror is documented."""
    from cineinfini.core.config import default_config
    vfb = default_config().datasets["videofeedback"]
    assert "hexuan21/VideoFeedback-videos-mp4" in vfb.get("videos_mirror", "")


def test_videofeedback_has_no_legacy_url():
    """The old `data.tar` URL was a guess and is removed (HF doesn't expose data.tar)."""
    from cineinfini.core.config import default_config
    vfb = default_config().datasets["videofeedback"]
    # A tar URL is wrong; HF datasets are loaded via the datasets library.
    assert "data.tar" not in (vfb.get("url") or "")


def test_yaml_and_python_defaults_in_sync():
    """The on-disk YAML default and the in-code Python defaults agree on size_mb."""
    import yaml
    from cineinfini.core.config import default_config
    yaml_doc = yaml.safe_load((REPO / "cfg/config.yaml").read_text())
    yaml_vfb = yaml_doc["datasets"]["videofeedback"]
    py_vfb = default_config().datasets["videofeedback"]
    assert yaml_vfb["size_mb"] == py_vfb["size_mb"] == 12.6
    assert yaml_vfb["license"] == py_vfb["license"] == "Apache-2.0"
    assert yaml_vfb["n_samples"] == py_vfb["n_samples"] == 37661


def test_datasets_info_command_runs():
    """`cineinfini datasets --info videofeedback` should still work."""
    r = _run_cli("datasets", "--info", "videofeedback")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "videofeedback" in out.lower() or "VideoFeedback" in out


def test_build_calibration_csv_script_exists():
    """The new helper script is shipped."""
    s = REPO / "scripts" / "build_calibration_csv.py"
    assert s.exists()
    assert "load_dataset" in s.read_text()
