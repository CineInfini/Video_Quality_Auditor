"""Test ``optional_models`` registry + GitHub release-asset resolver.

Validates two things:
  1. The hard-coded URLs in the registry compose correctly from
     (github_repo, github_tag, asset_name) — i.e. the pinning is internally
     consistent and won't drift if someone edits the URL but not the tag.
  2. ``resolve_github_release_asset()`` and ``resolve_optional_model_url()``
     behave as expected (with mocked HTTP).
"""
from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from cineinfini.core.bootstrap import (
    github_release_url,
    resolve_github_release_asset,
    resolve_optional_model_url,
)
from cineinfini.core.config import default_config


# ---------------------------------------------------------------------------
# Registry consistency tests — no network
# ---------------------------------------------------------------------------
def test_optional_models_section_has_fastvqa():
    cfg = default_config()
    assert "fastvqa" in cfg.optional_models
    assert "fastvqa_m" in cfg.optional_models


def test_fastvqa_url_matches_v0_3_release():
    cfg = default_config()
    fast = cfg.optional_models["fastvqa"]
    assert fast["url"] == (
        "https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/"
        "releases/download/v1.0.0-open-release-weights/fast-vqa_v0_3.pth"
    )
    assert fast["filename"] == "fast-vqa_v0_3.pth"
    assert fast["github_tag"] == "v1.0.0-open-release-weights"


def test_fastvqa_m_url_matches_v0_3_release():
    cfg = default_config()
    fast_m = cfg.optional_models["fastvqa_m"]
    assert fast_m["url"] == (
        "https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/"
        "releases/download/v1.0.0-open-release-weights/fast-vqa_m-v0_3.pth"
    )
    assert fast_m["github_tag"] == "v1.0.0-open-release-weights"


def test_every_optional_model_url_composes_from_repo_tag_asset():
    """Each entry's hard-coded URL MUST equal github_release_url(repo,tag,asset).

    This catches the failure mode where someone edits the tag but not the
    URL, or vice-versa, leaving the entry internally inconsistent.
    """
    cfg = default_config()
    for key, entry in cfg.optional_models.items():
        composed = github_release_url(
            entry["github_repo"], entry["github_tag"], entry["asset_name"],
        )
        assert composed == entry["url"], (
            f"{key}: composed URL {composed!r} != hard-coded {entry['url']!r}"
        )


def test_pinning_uses_a_named_tag_not_latest():
    """`latest` would defeat the whole point of pinning."""
    cfg = default_config()
    for key, entry in cfg.optional_models.items():
        tag = entry["github_tag"]
        assert tag.lower() != "latest", f"{key} uses tag 'latest' — not pinned!"
        assert tag, f"{key} has empty tag"


# ---------------------------------------------------------------------------
# github_release_url — pure function
# ---------------------------------------------------------------------------
def test_github_release_url_compose():
    url = github_release_url(
        "VQAssessment/FAST-VQA-and-FasterVQA",
        "v1.0.0-open-release-weights",
        "fast-vqa_v0_3.pth",
    )
    assert url == (
        "https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/"
        "releases/download/v1.0.0-open-release-weights/fast-vqa_v0_3.pth"
    )


def test_github_release_url_rejects_bad_repo():
    with pytest.raises(ValueError):
        github_release_url("no_slash", "v1", "asset.pth")


# ---------------------------------------------------------------------------
# resolve_github_release_asset — mocked HTTP
# ---------------------------------------------------------------------------
def _mock_response(payload: dict):
    fake = MagicMock()
    fake.read.return_value = json.dumps(payload).encode("utf-8")
    fake.__enter__ = MagicMock(return_value=fake)
    fake.__exit__ = MagicMock(return_value=False)
    return fake


def test_resolver_finds_asset():
    payload = {
        "assets": [
            {"name": "other.pth", "browser_download_url": "https://x/other.pth"},
            {"name": "fast-vqa_v0_3.pth",
             "browser_download_url": "https://x/fast-vqa_v0_3.pth"},
        ]
    }
    with patch("cineinfini.core.bootstrap.urlopen",
               return_value=_mock_response(payload)) as m:
        url = resolve_github_release_asset(
            "VQAssessment/FAST-VQA-and-FasterVQA",
            "v1.0.0-open-release-weights",
            "fast-vqa_v0_3.pth",
        )
    assert url == "https://x/fast-vqa_v0_3.pth"
    m.assert_called_once()


def test_resolver_returns_none_when_asset_missing():
    payload = {"assets": [{"name": "different.pth",
                           "browser_download_url": "https://x/different.pth"}]}
    with patch("cineinfini.core.bootstrap.urlopen",
               return_value=_mock_response(payload)):
        url = resolve_github_release_asset(
            "owner/repo", "v1", "missing.pth",
        )
    assert url is None


def test_resolver_returns_none_on_404():
    from urllib.error import HTTPError
    err = HTTPError("https://api.github.com/...", 404, "Not Found", {}, None)
    with patch("cineinfini.core.bootstrap.urlopen", side_effect=err):
        url = resolve_github_release_asset("owner/repo", "v999", "x.pth")
    assert url is None


def test_resolver_propagates_non_404_http_errors():
    from urllib.error import HTTPError
    err = HTTPError("https://api.github.com/...", 500, "Boom", {}, None)
    with patch("cineinfini.core.bootstrap.urlopen", side_effect=err):
        with pytest.raises(HTTPError):
            resolve_github_release_asset("owner/repo", "v1", "x.pth")


def test_resolver_uses_token_from_env(monkeypatch):
    payload = {"assets": []}
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_TESTTOKEN")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["auth"] = req.headers.get("Authorization")
        return _mock_response(payload)

    with patch("cineinfini.core.bootstrap.urlopen", side_effect=fake_urlopen):
        resolve_github_release_asset("owner/repo", "v1", "x.pth")
    assert captured["auth"] == "Bearer ghp_TESTTOKEN"


def test_resolver_rejects_bad_repo():
    with pytest.raises(ValueError):
        resolve_github_release_asset("no_slash_here", "v1", "x.pth")


# ---------------------------------------------------------------------------
# resolve_optional_model_url — high-level API
# ---------------------------------------------------------------------------
def test_resolve_url_prefers_hardcoded():
    entry = {
        "url": "https://hardcoded.example/x.pth",
        "github_repo": "owner/repo",
        "github_tag": "v1",
        "asset_name": "x.pth",
    }
    assert resolve_optional_model_url(entry) == "https://hardcoded.example/x.pth"


def test_resolve_url_composes_when_no_hardcoded():
    entry = {
        "github_repo": "owner/repo",
        "github_tag": "v1.2.3",
        "asset_name": "x.pth",
    }
    assert resolve_optional_model_url(entry) == (
        "https://github.com/owner/repo/releases/download/v1.2.3/x.pth"
    )


def test_resolve_url_falls_back_to_filename_field():
    entry = {
        "github_repo": "owner/repo",
        "github_tag": "v1",
        "filename": "fallback.pth",  # asset_name absent, filename used
    }
    assert "fallback.pth" in resolve_optional_model_url(entry)


def test_resolve_url_raises_when_nothing_resolvable():
    with pytest.raises(ValueError):
        resolve_optional_model_url({"description": "nothing to go on"})


def test_resolve_actual_fastvqa_entry():
    cfg = default_config()
    url = resolve_optional_model_url(cfg.optional_models["fastvqa"])
    assert url.endswith("/fast-vqa_v0_3.pth")
    assert "v1.0.0-open-release-weights" in url
