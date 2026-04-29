"""Smoke tests for v0.4.8.6 CLI commands: watch, serve, calibrate."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(*args, timeout: int = 10) -> subprocess.CompletedProcess:
    repo = Path(__file__).parent.parent
    return subprocess.run(
        [sys.executable, "-m", "cineinfini.cli.main", *args],
        cwd=repo,
        env={"PYTHONPATH": str(repo / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.parametrize("cmd", ["watch", "serve", "calibrate"])
def test_command_help_succeeds(cmd):
    """`cineinfini <cmd> --help` should exit 0 and print description."""
    r = _run_cli(cmd, "--help")
    assert r.returncode == 0, f"{cmd} --help failed: {r.stderr}"
    assert cmd in r.stdout.lower() or "usage" in r.stdout.lower()


def test_watch_help_describes_directory_argument():
    r = _run_cli("watch", "--help")
    assert r.returncode == 0
    assert "DIRECTORY" in r.stdout or "directory" in r.stdout.lower()


def test_serve_help_lists_endpoints():
    r = _run_cli("serve", "--help")
    assert r.returncode == 0
    # endpoints documented in the docstring
    assert "/audit" in r.stdout or "health" in r.stdout.lower()


def test_calibrate_help_documents_csv_format():
    r = _run_cli("calibrate", "--help")
    assert r.returncode == 0
    assert "labels-csv" in r.stdout or "CSV" in r.stdout


def test_calibrate_rejects_missing_csv_column(tmp_path):
    """A CSV without video_path/mos columns should fail with a clear error."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("foo,bar\n1,2\n3,4\n")
    r = _run_cli("calibrate", "--labels-csv", str(bad_csv),
                 "--output", str(tmp_path / "out.json"))
    # Should exit non-zero
    assert r.returncode != 0
    # Error message should explain expected columns
    err = r.stderr.lower() + r.stdout.lower()
    assert "column" in err or "video" in err or "mos" in err


def test_top_level_help_lists_new_commands():
    r = _run_cli("--help")
    assert r.returncode == 0
    for cmd in ("watch", "serve", "calibrate"):
        assert cmd in r.stdout, f"{cmd} missing from top-level --help"
