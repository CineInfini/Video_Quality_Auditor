"""One-shot environment bootstrapper.

Ensures every external asset CineInfini needs is present:

* **System binaries** (ffmpeg, ffprobe) — checked but never installed (we
  refuse to silently apt/brew/choco — the user gets a clear instruction).
* **ML model weights** — downloaded once into ``cfg.models_dir()`` from the
  URLs declared in ``cfg.model_urls``.
* **Test videos** (BBB, Tears of Steel, Sintel, Elephants Dream) —
  downloaded once into ``cfg.test_videos_dir()`` from ``cfg.test_videos``.

Idempotent: if a file already exists with a non-zero size and a matching
SHA-256 (when declared), it is left untouched.

All paths come from ``Config``; nothing is hardcoded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Config, get_config

logger = logging.getLogger("cineinfini.bootstrap")

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AssetStatus:
    """Status of one asset (model or test video)."""
    name: str
    path: Path
    present: bool = False
    size_bytes: int = 0
    sha256_ok: Optional[bool] = None
    downloaded: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "present": self.present,
            "size_bytes": int(self.size_bytes),
            "sha256_ok": self.sha256_ok,
            "downloaded": self.downloaded,
            "error": self.error,
        }


@dataclass
class BootstrapReport:
    """Aggregate report returned by ``bootstrap_all`` / ``ensure_*``."""
    system: Dict[str, Any] = field(default_factory=dict)
    models: List[AssetStatus] = field(default_factory=list)
    videos: List[AssetStatus] = field(default_factory=list)
    ok: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "system": dict(self.system),
            "models": [m.to_dict() for m in self.models],
            "videos": [v.to_dict() for v in self.videos],
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# System-level checks
# ---------------------------------------------------------------------------
_INSTALL_HINTS = {
    "ffmpeg": (
        "Install ffmpeg:\n"
        "  Linux/Debian: sudo apt-get install -y ffmpeg\n"
        "  macOS:        brew install ffmpeg\n"
        "  Windows:      choco install ffmpeg  (or download a static build)"
    ),
    "ffprobe": "ffprobe ships with ffmpeg; install ffmpeg.",
}


def check_system_dependency(binary: str) -> Tuple[bool, Optional[str]]:
    """Return (present, version_string_or_None) for an executable on PATH."""
    found = shutil.which(binary)
    if not found:
        return False, None
    try:
        out = subprocess.run(
            [found, "-version"], capture_output=True, text=True, timeout=10
        )
        line = (out.stdout or out.stderr or "").splitlines()[0] if (out.stdout or out.stderr) else ""
        return True, line.strip() or found
    except Exception as e:  # noqa: BLE001
        return True, f"{found} (version check failed: {e})"


def ensure_system_deps(
    required: Iterable[str] = ("ffmpeg", "ffprobe"),
    raise_on_missing: bool = False,
) -> Dict[str, Any]:
    """Verify required system binaries are available on PATH.

    Returns a dict like::

        {"ffmpeg": {"present": True, "version": "ffmpeg version 6.0 ..."},
         "ffprobe": {"present": False, "version": None,
                     "hint": "Install ffmpeg: ..."}}

    Set ``raise_on_missing=True`` to raise :class:`RuntimeError` if any
    required binary is missing.
    """
    report: Dict[str, Any] = {}
    missing: List[str] = []
    for binary in required:
        present, version = check_system_dependency(binary)
        entry: Dict[str, Any] = {"present": present, "version": version}
        if not present:
            entry["hint"] = _INSTALL_HINTS.get(binary, f"Install '{binary}' and ensure it is on PATH.")
            missing.append(binary)
        report[binary] = entry
    if missing and raise_on_missing:
        raise RuntimeError(
            "Missing required system dependencies: "
            + ", ".join(missing)
            + "\n"
            + "\n".join(report[m]["hint"] for m in missing)
        )
    return report


# ---------------------------------------------------------------------------
# GitHub release-asset resolver
# ---------------------------------------------------------------------------
# Strategy: when a hard-coded asset URL goes 404 (because the upstream repo
# renamed/moved a release), we hit the GitHub API to dynamically resolve the
# current `browser_download_url` for the asset matching `asset_name` under
# the given tag. This lets `optional_models` entries declare both the
# stable URL *and* the resolver coordinates as a fallback.

_GITHUB_API_BASE = "https://api.github.com"


def resolve_github_release_asset(
    repo: str,
    tag: str,
    asset_name: str,
    *,
    timeout: int = 15,
    token: Optional[str] = None,
) -> Optional[str]:
    """Return the ``browser_download_url`` for an asset, or None.

    Parameters
    ----------
    repo
        ``"owner/repo"`` (e.g. ``"VQAssessment/FAST-VQA-and-FasterVQA"``).
    tag
        The release tag (e.g. ``"v1.0.0-open-release-weights"``).
    asset_name
        Exact asset filename (e.g. ``"fast-vqa_v0_3.pth"``).
    timeout
        Per-HTTP-call timeout in seconds.
    token
        Optional GitHub token (read from ``GITHUB_TOKEN`` env var if None).
        Helps avoid rate-limiting when running inside CI.

    Returns
    -------
    The ``browser_download_url`` string, or None if the release/asset
    cannot be found. Raises only on transport errors that aren't 404.
    """
    if "/" not in repo:
        raise ValueError(f"repo must be 'owner/repo', got: {repo!r}")
    url = f"{_GITHUB_API_BASE}/repos/{repo}/releases/tags/{tag}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "cineinfini/release-resolver",
    }
    tok = token or os.environ.get("GITHUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8"))
    except HTTPError as e:
        if e.code == 404:
            logger.warning("GitHub release not found: %s @ %s", repo, tag)
            return None
        raise
    except URLError as e:
        logger.warning("GitHub API unreachable: %s", e)
        return None
    for asset in payload.get("assets") or []:
        if asset.get("name") == asset_name:
            return asset.get("browser_download_url")
    logger.warning(
        "Asset %r not found in %s @ %s (%d assets present)",
        asset_name, repo, tag, len(payload.get("assets") or []),
    )
    return None


def github_release_url(repo: str, tag: str, asset_name: str) -> str:
    """Return the *stable* GitHub release-asset URL by string composition.

    Once a release with assets is published the path is immutable:
    ``/{repo}/releases/download/{tag}/{asset}``. Use this when you don't
    want to spend an API call.
    """
    if "/" not in repo:
        raise ValueError(f"repo must be 'owner/repo', got: {repo!r}")
    return f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"


def resolve_optional_model_url(
    entry: Dict[str, Any],
    *,
    api_fallback: bool = False,
) -> str:
    """Return the best-known download URL for an ``optional_models`` entry.

    Order of preference:
      1. ``entry["url"]`` (hard-coded)
      2. Composed from (github_repo, github_tag, asset_name)
      3. Live GitHub API lookup (only when ``api_fallback=True``)
    """
    if entry.get("url"):
        return entry["url"]
    repo = entry.get("github_repo")
    tag = entry.get("github_tag")
    asset = entry.get("asset_name") or entry.get("filename")
    if repo and tag and asset:
        return github_release_url(repo, tag, asset)
    if api_fallback and repo and tag and asset:
        resolved = resolve_github_release_asset(repo, tag, asset)
        if resolved:
            return resolved
    raise ValueError(
        f"Cannot determine download URL for entry: keys={list(entry.keys())}"
    )


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def _download_url(
    url: str,
    dest: Path,
    *,
    progress: bool = True,
    timeout: int = 60,
    user_agent: str = "Mozilla/5.0 cineinfini-bootstrap",
) -> None:
    """Stream ``url`` to ``dest`` atomically (download to .part then rename).

    Uses ``requests`` if available (with progress), otherwise falls back to
    :mod:`urllib.request` (no progress, but pure stdlib).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    try:
        try:
            import requests  # type: ignore

            with requests.get(
                url, stream=True, timeout=timeout,
                headers={"User-Agent": user_agent},
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length") or 0)
                downloaded = 0
                last_print = 0
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress and total:
                            pct = int(100 * downloaded / total)
                            if pct >= last_print + 10:
                                logger.info(
                                    "  %s%% (%s / %s)",
                                    pct, _format_bytes(downloaded), _format_bytes(total),
                                )
                                last_print = pct
        except ImportError:
            from urllib.request import Request, urlopen
            req = Request(url, headers={"User-Agent": user_agent})
            with urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:  # noqa: S310
                shutil.copyfileobj(resp, f)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        raise


def _verify_asset(
    path: Path,
    expected_sha256: Optional[str],
) -> Tuple[bool, Optional[bool], int]:
    """Return (present_and_nonempty, sha_ok_or_None, size_bytes)."""
    if not path.exists() or not path.is_file():
        return False, None, 0
    size = path.stat().st_size
    if size == 0:
        return False, None, 0
    if not expected_sha256:
        return True, None, size
    actual = _sha256_file(path)
    return True, actual.lower() == expected_sha256.lower(), size


def _ensure_one(
    name: str,
    url: str,
    target: Path,
    *,
    expected_sha256: Optional[str] = None,
    force: bool = False,
    skip_download: bool = False,
) -> AssetStatus:
    """Ensure one asset is present at ``target``, downloading if necessary."""
    status = AssetStatus(name=name, path=target)
    present, sha_ok, size = _verify_asset(target, expected_sha256)
    if present and (sha_ok is True or sha_ok is None) and not force:
        status.present = True
        status.size_bytes = size
        status.sha256_ok = sha_ok
        logger.info("[%s] cached at %s (%s)", name, target, _format_bytes(size))
        return status

    if present and sha_ok is False:
        logger.warning("[%s] sha256 mismatch, re-downloading", name)
        try:
            target.unlink()
        except Exception:
            pass

    if skip_download:
        status.error = "asset missing and skip_download=True"
        logger.warning("[%s] missing and download disabled", name)
        return status

    if not url:
        status.error = "no URL configured"
        logger.warning("[%s] no URL configured (path=%s)", name, target)
        return status

    logger.info("[%s] downloading %s -> %s", name, url, target)
    try:
        _download_url(url, target)
        status.downloaded = True
    except Exception as e:  # noqa: BLE001
        status.error = f"download failed: {e}"
        logger.error("[%s] download failed: %s", name, e)
        return status

    present, sha_ok, size = _verify_asset(target, expected_sha256)
    status.present = present
    status.size_bytes = size
    status.sha256_ok = sha_ok
    if expected_sha256 and sha_ok is False:
        status.error = "sha256 mismatch after download"
        logger.error("[%s] %s", name, status.error)
    elif present:
        logger.info("[%s] ready (%s)", name, _format_bytes(size))
    else:
        status.error = status.error or "asset missing after download"
    return status


# ---------------------------------------------------------------------------
# Public ensure_* helpers
# ---------------------------------------------------------------------------
def ensure_models(
    cfg: Optional[Config] = None,
    *,
    only: Optional[Iterable[str]] = None,
    force: bool = False,
    skip_download: bool = False,
) -> List[AssetStatus]:
    """Ensure every model declared in ``cfg.model_urls`` is on disk.

    Pass ``only=["arcface", "yunet"]`` to limit which keys are processed.
    """
    cfg = cfg or get_config()
    models_dir = cfg.models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    keys = set(only) if only else set(cfg.model_urls.keys())
    out: List[AssetStatus] = []
    for name, entry in cfg.model_urls.items():
        if name not in keys:
            continue
        url = str(entry.get("url") or "")
        filename = str(entry.get("filename") or f"{name}.bin")
        sha256 = entry.get("sha256") or None
        target = models_dir / filename
        out.append(_ensure_one(
            f"model:{name}", url, target,
            expected_sha256=sha256, force=force, skip_download=skip_download,
        ))
    return out


def ensure_test_videos(
    cfg: Optional[Config] = None,
    *,
    only: Optional[Iterable[str]] = None,
    force: bool = False,
    skip_download: bool = False,
) -> List[AssetStatus]:
    """Ensure every test video declared in ``cfg.test_videos`` is on disk."""
    cfg = cfg or get_config()
    videos_dir = cfg.test_videos_dir()
    videos_dir.mkdir(parents=True, exist_ok=True)
    keys = set(only) if only else set(cfg.test_videos.keys())
    out: List[AssetStatus] = []
    for name, entry in cfg.test_videos.items():
        if name not in keys:
            continue
        url = str(entry.get("url") or "")
        filename = str(entry.get("filename") or f"{name}.mp4")
        sha256 = entry.get("sha256") or None
        target = videos_dir / filename
        out.append(_ensure_one(
            f"video:{name}", url, target,
            expected_sha256=sha256, force=force, skip_download=skip_download,
        ))
    return out


def bootstrap_all(
    cfg: Optional[Config] = None,
    *,
    require_ffmpeg: bool = True,
    only_models: Optional[Iterable[str]] = None,
    only_videos: Optional[Iterable[str]] = None,
    force: bool = False,
    skip_download: bool = False,
) -> BootstrapReport:
    """Run the full bootstrap: system check + models + videos.

    Returns a :class:`BootstrapReport`. The ``ok`` flag is False if any
    *required* asset is missing (currently: ffmpeg if ``require_ffmpeg``,
    plus any model/video flagged ``required: true`` in the config).
    """
    cfg = cfg or get_config()
    report = BootstrapReport()

    # 1) system deps
    report.system = ensure_system_deps(
        ("ffmpeg", "ffprobe") if require_ffmpeg else (),
        raise_on_missing=False,
    )
    if require_ffmpeg and not report.system.get("ffmpeg", {}).get("present"):
        report.ok = False
        report.warnings.append("ffmpeg missing — video I/O will be limited")

    # 2) models
    report.models = ensure_models(
        cfg, only=only_models, force=force, skip_download=skip_download
    )
    for m in report.models:
        if not m.present:
            entry = cfg.model_urls.get(m.name.split(":", 1)[-1], {})
            if entry.get("required"):
                report.ok = False
                report.warnings.append(f"required model missing: {m.name}")

    # 3) videos
    report.videos = ensure_test_videos(
        cfg, only=only_videos, force=force, skip_download=skip_download
    )
    for v in report.videos:
        if not v.present:
            entry = cfg.test_videos.get(v.name.split(":", 1)[-1], {})
            if entry.get("required"):
                report.ok = False
                report.warnings.append(f"required video missing: {v.name}")

    # 4) wire face_detection MODELS_DIR if available
    try:
        from . import face_detection as _fd
        _fd.set_models_dir(cfg.models_dir())
    except Exception:
        pass

    return report


def print_report(report: BootstrapReport) -> None:
    """Pretty-print a :class:`BootstrapReport` to stdout."""
    print("=" * 60)
    print("CineInfini bootstrap report")
    print("=" * 60)
    print("System:")
    for name, info in report.system.items():
        ok = "OK " if info.get("present") else "MISS"
        print(f"  [{ok}] {name}: {info.get('version') or 'not found'}")
        if not info.get("present") and info.get("hint"):
            for line in str(info["hint"]).splitlines():
                print("        " + line)
    print("Models:")
    for m in report.models:
        ok = "OK " if m.present else "MISS"
        sha = ""
        if m.sha256_ok is True:
            sha = " sha256:ok"
        elif m.sha256_ok is False:
            sha = " sha256:MISMATCH"
        size = f" ({_format_bytes(m.size_bytes)})" if m.size_bytes else ""
        print(f"  [{ok}] {m.name}{size}{sha} -> {m.path}")
        if m.error:
            print(f"        error: {m.error}")
    print("Test videos:")
    for v in report.videos:
        ok = "OK " if v.present else "MISS"
        size = f" ({_format_bytes(v.size_bytes)})" if v.size_bytes else ""
        print(f"  [{ok}] {v.name}{size} -> {v.path}")
        if v.error:
            print(f"        error: {v.error}")
    print("-" * 60)
    print(f"Overall OK: {report.ok}")
    for w in report.warnings:
        print(f"  warn: {w}")


__all__ = [
    "AssetStatus",
    "BootstrapReport",
    "check_system_dependency",
    "ensure_system_deps",
    "ensure_models",
    "ensure_test_videos",
    "bootstrap_all",
    "print_report",
    "resolve_github_release_asset",
    "github_release_url",
    "resolve_optional_model_url",
]
