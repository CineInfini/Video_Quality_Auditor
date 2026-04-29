#!/usr/bin/env python3
"""CineInfini release & deploy orchestrator (v0.4.8.8).

Steps performed:

1.  Verify clean git tree, branch, and that the working version matches
    ``--bump`` if supplied.
2.  Sync ``__version__`` across ``__init__.py``, ``CITATION.cff`` and the
    BibTeX block in ``README.md`` (``pyproject.toml`` reads it dynamically,
    so it doesn't need rewriting).
3.  Validate ``cfg/config.yaml`` against the live :class:`Config` schema.
4.  Run pytest, capturing stdout to ``test_output.log``.
5.  Build sdist + wheel.
6.  Build source ``cineinfini-v{NEW_VERSION}.zip`` archive of the working
    tree (excluding ``.git``, ``__pycache__``, ``dist``, ``build``,
    ``.pytest_cache``).
7.  Generate a ``MANIFEST.txt`` with the list of files changed since the
    previous tag (``git diff --name-only <prev_tag>..HEAD``).
8.  Generate API docs via ``scripts/generate_docs.py`` (silent skip if
    ``pdoc`` isn't available).
9.  Tag, push, create a GitHub release, upload sdist + wheel + the source
    ZIP as additional assets.
10. Upload to PyPI via twine.
11. Create a Zenodo deposition draft, upload sdist, publish.

Every step is mirrored to ``deploy.log``. Auth is read from environment
variables — never from prompts:

    GITHUB_KEY        : GitHub PAT with repo + workflow scope
    PYPI_API_TOKEN    : PyPI API token (starts with ``pypi-``)
    ZENODO_TOKEN      : Zenodo API token (optional; skip Zenodo if absent)

Usage::

    python deploy_cineinfini.py --dry-run              # plan only
    python deploy_cineinfini.py --bump 0.4.8.8         # full release

``--dry-run`` shows every step without executing network or write actions.
``--force`` skips the clean-tree check.
``--skip-tests`` is honoured but logs a loud warning.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
SRC_INIT = REPO_ROOT / "src" / "cineinfini" / "__init__.py"
CITATION = REPO_ROOT / "CITATION.cff"
README = REPO_ROOT / "README.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
CONFIG_YAML = REPO_ROOT / "cfg" / "config.yaml"
PAPER_DRAFT = REPO_ROOT / "docs" / "PAPER_DRAFT.md"
PAPER_GUARD_SCRIPT = REPO_ROOT / "scripts" / "check_paper_no_placeholder.py"
DIST_DIR = REPO_ROOT / "dist"
DEPLOY_LOG = REPO_ROOT / "deploy.log"
TEST_LOG = REPO_ROOT / "test_output.log"
MANIFEST_TXT = REPO_ROOT / "MANIFEST.txt"
GENERATE_DOCS_SCRIPT = REPO_ROOT / "scripts" / "generate_docs.py"

GITHUB_OWNER = "CineInfini"
GITHUB_REPO = "Video_Quality_Auditor"

# Patterns excluded from the source ZIP archive (see step 6).
ZIP_EXCLUDE_DIRS = {
    ".git", "__pycache__", "dist", "build", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "node_modules", ".venv", "venv", ".tox",
    ".cineinfini",  # local cache, never shipped
}
ZIP_EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so"}


# ===========================================================================
# Logging plumbing  (NEW v0.4.8.8)
# ===========================================================================
def _setup_logging() -> logging.Logger:
    """Configure root + 'deploy' logger to mirror to console + deploy.log."""
    fmt = "%(asctime)s | %(levelname)-7s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("deploy")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    # File
    fh = logging.FileHandler(DEPLOY_LOG, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Console
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


log = _setup_logging()


# ===========================================================================
# Plumbing
# ===========================================================================
class DeployError(RuntimeError):
    """Raised when a deploy step fails (and we don't want to continue)."""


@dataclass
class Step:
    name: str
    fn: Any
    skip_if_dry_run: bool = False


def run(cmd, *, check=True, capture=False, dry_run=False, cwd=None) -> str:
    if isinstance(cmd, str):
        cmd_str = cmd
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = list(cmd)
        cmd_str = " ".join(shlex.quote(c) for c in cmd_list)
    log.info("$ %s", cmd_str)
    if dry_run:
        return ""
    res = subprocess.run(
        cmd_list, cwd=cwd or REPO_ROOT,
        capture_output=capture, text=True, check=False,
    )
    if capture and res.stdout:
        for line in res.stdout.splitlines():
            log.info("    %s", line)
    if capture and res.stderr:
        for line in res.stderr.splitlines():
            log.info("    [stderr] %s", line)
    if check and res.returncode != 0:
        raise DeployError(
            f"command failed ({res.returncode}): {cmd_str}\n"
            f"stdout: {res.stdout}\n"
            f"stderr: {res.stderr}"
        )
    return (res.stdout or "").strip()


def env_or_die(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise DeployError(
            f"missing environment variable: {key} "
            f"(set it before running this script)"
        )
    return val


def assert_secrets_present(*, skip_pypi: bool, skip_github: bool) -> None:
    """Fail fast if required secrets are missing; warn for optional ones."""
    if not skip_github and not os.environ.get("GITHUB_KEY"):
        raise DeployError("GITHUB_KEY missing — set it or pass --skip-github")
    if not skip_pypi and not os.environ.get("PYPI_API_TOKEN"):
        raise DeployError("PYPI_API_TOKEN missing — set it or pass --skip-pypi")
    if not os.environ.get("ZENODO_TOKEN"):
        log.warning("ZENODO_TOKEN not set — Zenodo step will be skipped silently")


# ===========================================================================
# Version helpers
# ===========================================================================
def read_current_version() -> str:
    text = SRC_INIT.read_text(encoding="utf-8")
    m = re.search(r"__version__\s*=\s*[\"']([^\"']+)[\"']", text)
    if not m:
        raise DeployError("Could not find __version__ in src/cineinfini/__init__.py")
    return m.group(1)


def write_init_version(new_version: str, *, dry_run: bool) -> None:
    text = SRC_INIT.read_text(encoding="utf-8")
    new = re.sub(
        r"__version__\s*=\s*[\"'][^\"']+[\"']",
        f'__version__ = "{new_version}"',
        text, count=1,
    )
    log.info("  __version__ -> %s", new_version)
    if not dry_run:
        SRC_INIT.write_text(new, encoding="utf-8")


def write_citation_version(new_version: str, *, dry_run: bool) -> None:
    text = CITATION.read_text(encoding="utf-8")
    new = re.sub(r"^version:\s*.+$", f"version: {new_version}", text, count=1, flags=re.M)
    log.info("  CITATION.cff version -> %s", new_version)
    if not dry_run:
        CITATION.write_text(new, encoding="utf-8")


def write_readme_bibtex_version(new_version: str, *, dry_run: bool) -> None:
    text = README.read_text(encoding="utf-8")
    new = re.sub(
        r"version\s*=\s*\{[^}]*\}",
        f"version = {{{new_version}}}",
        text, count=1,
    )
    log.info("  README.md BibTeX version -> %s", new_version)
    if not dry_run:
        README.write_text(new, encoding="utf-8")


def sync_versions(target: str, *, dry_run: bool) -> None:
    write_init_version(target, dry_run=dry_run)
    write_citation_version(target, dry_run=dry_run)
    write_readme_bibtex_version(target, dry_run=dry_run)


# ===========================================================================
# Git
# ===========================================================================
def assert_clean_tree(*, force: bool, dry_run: bool) -> None:
    if force:
        log.info("  --force: skipping clean-tree check")
        return
    out = run("git status --porcelain", capture=True, dry_run=False, check=False)
    if out:
        raise DeployError(
            "git working tree is not clean. Commit, stash, or pass --force.\n" + out
        )


def assert_on_branch(branch: str = "main", *, dry_run: bool) -> None:
    if dry_run:
        return
    out = run("git rev-parse --abbrev-ref HEAD", capture=True, check=False)
    if out and out != branch:
        log.warning("  not on %s (current: %s)", branch, out)


def commit_version_bump(version: str, *, dry_run: bool) -> None:
    run(["git", "add", "src/cineinfini/__init__.py", "CITATION.cff", "README.md"],
        dry_run=dry_run)
    run(["git", "commit", "-m", f"chore(release): bump to v{version}"], dry_run=dry_run)


def tag_and_push(version: str, *, dry_run: bool) -> None:
    tag = f"v{version}"
    run(["git", "tag", "-a", tag, "-m", f"Release {tag}"], dry_run=dry_run)
    run(["git", "push", "origin", "HEAD"], dry_run=dry_run)
    run(["git", "push", "origin", tag], dry_run=dry_run)


def previous_tag() -> Optional[str]:
    """Return the tag immediately before HEAD, or None if no tags exist."""
    out = run("git tag --sort=-creatordate", capture=True, check=False)
    tags = [t for t in out.splitlines() if t.strip()]
    return tags[0] if tags else None


# ===========================================================================
# Config validation step (NEW v0.4.8.8)
# ===========================================================================
def validate_config_yaml(*, dry_run: bool) -> None:
    """Load cfg/config.yaml and run Config.validate() to fail fast on a
    structural mistake. Uses the freshly-bumped source tree via PYTHONPATH."""
    if dry_run:
        log.info("  (dry-run) would validate %s against Config schema", CONFIG_YAML)
        return
    if not CONFIG_YAML.exists():
        log.warning("  cfg/config.yaml not found — skipping schema validation")
        return
    src = REPO_ROOT / "src"
    cmd = [
        sys.executable, "-c",
        "import sys; sys.path.insert(0, %r); "
        "from cineinfini.core.config import load_config; "
        "cfg = load_config(%r); cfg.validate(); "
        "print('config OK', cfg.summary())" % (str(src), str(CONFIG_YAML)),
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    for line in (res.stdout or "").splitlines():
        log.info("    %s", line)
    if res.returncode != 0:
        log.error("    [stderr] %s", res.stderr)
        raise DeployError(
            f"cfg/config.yaml failed Config.validate():\n{res.stderr or res.stdout}"
        )


# ===========================================================================
# Paper placeholder guard (NEW v0.4.8.9)
# ===========================================================================
def validate_paper_no_placeholder(*, dry_run: bool) -> None:
    """Run scripts/check_paper_no_placeholder.py — fail the deploy if the
    paper still contains any unfilled {{TBD-XXX}} slot, 'placeholder',
    'TODO', 'XXX', or 'FIXME' marker.

    The CI guard exists to enforce, mechanically, the project rule that
    no placeholder numbers ever ship in a tagged release.
    """
    if not PAPER_DRAFT.exists():
        log.info("  no PAPER_DRAFT.md — skipping placeholder guard")
        return
    if not PAPER_GUARD_SCRIPT.exists():
        log.warning("  scripts/check_paper_no_placeholder.py missing — "
                    "cannot enforce placeholder guard")
        return
    if dry_run:
        log.info("  (dry-run) would run paper placeholder guard on %s",
                 PAPER_DRAFT.name)
        return
    res = subprocess.run(
        [sys.executable, str(PAPER_GUARD_SCRIPT), str(PAPER_DRAFT)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    for line in (res.stdout or "").splitlines():
        log.info("    %s", line)
    for line in (res.stderr or "").splitlines():
        log.info("    [stderr] %s", line)
    if res.returncode != 0:
        raise DeployError(
            "paper placeholder guard failed: docs/PAPER_DRAFT.md still "
            "contains unfilled {{TBD-XXX}} or placeholder text. Run "
            "`python scripts/fill_paper.py <results.json> "
            "docs/PAPER_DRAFT.md` to fill from real data."
        )


# ===========================================================================
# Tests (with stdout capture to test_output.log)  (NEW v0.4.8.8)
# ===========================================================================
def run_tests(*, dry_run: bool, force: bool) -> None:
    if dry_run:
        log.info("  (dry-run) would run pytest")
        return
    cmd = [sys.executable, "-m", "pytest", "tests", "-v",
           "-m", "not integration and not slow", "--tb=short"]
    log.info("$ %s   (output -> %s)", " ".join(cmd), TEST_LOG.name)
    with TEST_LOG.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=REPO_ROOT,
                              stdout=logf, stderr=subprocess.STDOUT)
    # Echo summary lines back to console + deploy.log
    tail_lines = TEST_LOG.read_text(encoding="utf-8").splitlines()[-20:]
    for ln in tail_lines:
        log.info("    %s", ln)
    if proc.returncode != 0:
        if force:
            log.warning("  pytest FAILED but --force given — continuing anyway")
        else:
            raise DeployError(
                f"pytest failed ({proc.returncode}); see {TEST_LOG.name} "
                "for the full output. Re-run with --force to override."
            )


# ===========================================================================
# Build sdist + wheel
# ===========================================================================
def build_artifacts(*, dry_run: bool) -> None:
    if DIST_DIR.exists() and not dry_run:
        shutil.rmtree(DIST_DIR)
    run([sys.executable, "-m", "pip", "install", "--upgrade",
         "build", "twine"], dry_run=dry_run)
    run([sys.executable, "-m", "build"], dry_run=dry_run)


def list_artifacts() -> List[Path]:
    if not DIST_DIR.exists():
        return []
    return sorted([p for p in DIST_DIR.iterdir() if p.is_file()])


# ===========================================================================
# Source ZIP archive  (NEW v0.4.8.8)
# ===========================================================================
def build_source_zip(version: str, *, dry_run: bool) -> Optional[Path]:
    """Build cineinfini-v{version}.zip of the working tree (filtered).

    Returns the path to the archive, or None on dry run.
    """
    zip_name = f"cineinfini-v{version}.zip"
    zip_path = REPO_ROOT / zip_name
    if dry_run:
        log.info("  (dry-run) would build %s", zip_name)
        return None
    if zip_path.exists():
        zip_path.unlink()

    n_files = 0
    total_bytes = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(REPO_ROOT):
            # Filter dirs in-place to skip excluded subtrees
            dirs[:] = [d for d in dirs if d not in ZIP_EXCLUDE_DIRS]
            rel_root = Path(root).relative_to(REPO_ROOT)
            for fname in files:
                if Path(fname).suffix in ZIP_EXCLUDE_SUFFIXES:
                    continue
                # Don't ship deploy.log / test_output.log / the zip itself
                if fname in {DEPLOY_LOG.name, TEST_LOG.name, zip_name,
                             MANIFEST_TXT.name}:
                    continue
                src = Path(root) / fname
                arc = Path(f"cineinfini-v{version}") / rel_root / fname
                zf.write(src, arcname=str(arc))
                n_files += 1
                total_bytes += src.stat().st_size
    log.info("  wrote %s (%d files, %.2f MB)",
             zip_name, n_files, total_bytes / (1024 * 1024))
    return zip_path


# ===========================================================================
# MANIFEST.txt — files changed since previous tag  (NEW v0.4.8.8)
# ===========================================================================
def write_manifest(version: str, *, dry_run: bool) -> Optional[Path]:
    if dry_run:
        log.info("  (dry-run) would write %s", MANIFEST_TXT.name)
        return None
    prev = previous_tag()
    if prev:
        out = run(["git", "diff", "--name-only", f"{prev}..HEAD"],
                  capture=True, check=False)
    else:
        log.info("  no previous tag found — listing every tracked file")
        out = run(["git", "ls-files"], capture=True, check=False)
    body = (
        f"# CineInfini v{version} — change manifest\n"
        f"# Previous tag: {prev or '(none)'}\n"
        f"# Files: {len([l for l in out.splitlines() if l.strip()])}\n\n"
        f"{out.strip()}\n"
    )
    MANIFEST_TXT.write_text(body, encoding="utf-8")
    log.info("  wrote %s", MANIFEST_TXT.name)
    return MANIFEST_TXT


# ===========================================================================
# pdoc API documentation  (NEW v0.4.8.8 — invocation)
# ===========================================================================
def generate_api_docs(*, dry_run: bool) -> None:
    if not GENERATE_DOCS_SCRIPT.exists():
        log.info("  scripts/generate_docs.py not present — skipping pdoc step")
        return
    if dry_run:
        log.info("  (dry-run) would run scripts/generate_docs.py")
        return
    res = subprocess.run(
        [sys.executable, str(GENERATE_DOCS_SCRIPT), "--clean"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    for line in (res.stdout or "").splitlines():
        log.info("    %s", line)
    if res.returncode != 0:
        log.warning("  pdoc generation failed (non-fatal): %s",
                    (res.stderr or "").strip().splitlines()[-1] if res.stderr else "")


# ===========================================================================
# GitHub release
# ===========================================================================
def github_release(version: str, *, dry_run: bool,
                   extra_assets: Optional[List[Path]] = None) -> None:
    if dry_run:
        log.info("  (dry-run) would create GitHub release v%s", version)
        return
    token = env_or_die("GITHUB_KEY")
    import urllib.error  # noqa: F401
    from urllib.request import Request, urlopen

    api = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases"
    notes = _extract_changelog_notes(version)
    payload = json.dumps({
        "tag_name": f"v{version}",
        "name": f"v{version}",
        "body": notes,
        "draft": False,
        "prerelease": False,
    }).encode("utf-8")
    req = Request(api, data=payload, method="POST", headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    })
    try:
        with urlopen(req, timeout=30) as r:  # noqa: S310
            release = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise DeployError(f"GitHub release creation failed: {e}") from e

    upload_url = release.get("upload_url", "").split("{", 1)[0]
    if not upload_url:
        raise DeployError("GitHub release: missing upload_url")

    assets: List[Path] = list(list_artifacts())
    for extra in (extra_assets or []):
        if extra and extra.exists() and extra not in assets:
            assets.append(extra)

    for artifact in assets:
        log.info("  uploading %s -> GitHub release", artifact.name)
        with artifact.open("rb") as f:
            data = f.read()
        url = f"{upload_url}?name={artifact.name}"
        ureq = Request(url, data=data, method="POST", headers={
            "Authorization": f"token {token}",
            "Content-Type": "application/octet-stream",
        })
        try:
            urlopen(ureq, timeout=120)  # noqa: S310
        except Exception as e:  # noqa: BLE001
            raise DeployError(f"upload of {artifact.name} failed: {e}") from e


# ===========================================================================
# PyPI
# ===========================================================================
def pypi_upload(*, dry_run: bool) -> None:
    if dry_run:
        log.info("  (dry-run) would `twine upload dist/*`")
        return
    token = env_or_die("PYPI_API_TOKEN")
    env = os.environ.copy()
    env["TWINE_USERNAME"] = "__token__"
    env["TWINE_PASSWORD"] = token
    artifacts = [str(p) for p in list_artifacts()]
    if not artifacts:
        raise DeployError("no dist/ artifacts to upload")
    res = subprocess.run(
        [sys.executable, "-m", "twine", "upload", *artifacts],
        env=env, cwd=REPO_ROOT,
    )
    if res.returncode != 0:
        raise DeployError("twine upload failed")


# ===========================================================================
# Zenodo
# ===========================================================================
def zenodo_publish(version: str, *, dry_run: bool) -> Optional[str]:
    if dry_run:
        log.info("  (dry-run) would create Zenodo deposition for v%s", version)
        return None
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        log.info("  ZENODO_TOKEN not set — skipping Zenodo upload")
        return None

    import urllib.error  # noqa: F401
    from urllib.request import Request, urlopen

    api = "https://zenodo.org/api/deposit/depositions"
    metadata: Dict[str, Any] = {
        "metadata": {
            "title": f"CineInfini v{version}",
            "upload_type": "software",
            "description": (
                f"CineInfini Adaptive Multi-Stage Video Quality Audit Pipeline, "
                f"release v{version}. See CHANGELOG.md."
            ),
            "creators": [{
                "name": "BENBRAHIM, Salah-Eddine",
                "affiliation": "Independent Researcher",
            }],
            "version": version,
            "license": "MIT",
        }
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    req = Request(api, data=b"{}", method="POST", headers=headers)
    try:
        with urlopen(req, timeout=30) as r:  # noqa: S310
            dep = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise DeployError(f"Zenodo: failed to create deposition: {e}") from e

    dep_id = dep["id"]
    bucket_url = dep["links"].get("bucket")
    if not bucket_url:
        raise DeployError("Zenodo: deposition response missing bucket URL")

    sdists = [p for p in list_artifacts() if p.suffix == ".gz"]
    if not sdists:
        raise DeployError("no sdist found in dist/")
    sdist = sdists[0]
    put_url = f"{bucket_url}/{sdist.name}"
    log.info("  Zenodo: uploading %s ...", sdist.name)
    put_req = Request(
        put_url, data=sdist.read_bytes(), method="PUT",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/octet-stream"},
    )
    try:
        urlopen(put_req, timeout=300)  # noqa: S310
    except Exception as e:  # noqa: BLE001
        raise DeployError(f"Zenodo: file upload failed: {e}") from e

    meta_url = f"{api}/{dep_id}"
    meta_req = Request(
        meta_url, data=json.dumps(metadata).encode("utf-8"),
        method="PUT", headers=headers,
    )
    try:
        urlopen(meta_req, timeout=30)  # noqa: S310
    except Exception as e:  # noqa: BLE001
        raise DeployError(f"Zenodo: metadata update failed: {e}") from e

    publish_url = f"{api}/{dep_id}/actions/publish"
    pub_req = Request(publish_url, data=b"", method="POST",
                      headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(pub_req, timeout=30) as r:  # noqa: S310
            pub = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise DeployError(f"Zenodo: publish failed: {e}") from e

    doi = pub.get("doi")
    log.info("  Zenodo DOI: %s", doi)
    return doi


# ===========================================================================
# Changelog parsing
# ===========================================================================
def _extract_changelog_notes(version: str) -> str:
    if not CHANGELOG.exists():
        return f"Release v{version}"
    text = CHANGELOG.read_text(encoding="utf-8")
    m = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        text, re.M | re.S,
    )
    return m.group(1).strip() if m else f"Release v{version}"


# ===========================================================================
# Main
# ===========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CineInfini deploy script")
    parser.add_argument("--bump", help="Target version (e.g. 0.4.8.8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only — no writes, commits, or network calls")
    parser.add_argument("--force", action="store_true",
                        help="Skip clean-tree check; allow tests to fail")
    parser.add_argument("--branch", default="main",
                        help="Expected git branch (default: main)")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-validate-config", action="store_true",
                        help="Skip the cfg/config.yaml schema validation step")
    parser.add_argument("--skip-paper-guard", action="store_true",
                        help="Skip the paper-placeholder CI guard")
    parser.add_argument("--skip-zip", action="store_true",
                        help="Skip the source ZIP archive step")
    parser.add_argument("--skip-manifest", action="store_true",
                        help="Skip MANIFEST.txt generation")
    parser.add_argument("--skip-docs", action="store_true",
                        help="Skip pdoc API doc generation")
    parser.add_argument("--skip-github", action="store_true")
    parser.add_argument("--skip-pypi", action="store_true")
    parser.add_argument("--skip-zenodo", action="store_true")
    args = parser.parse_args(argv)

    log.info("=" * 64)
    log.info("CineInfini deploy  —  dry_run=%s  force=%s",
             args.dry_run, args.force)
    log.info("repo: %s", REPO_ROOT)
    log.info("logs: %s, %s", DEPLOY_LOG.name, TEST_LOG.name)
    log.info("=" * 64)

    current = read_current_version()
    log.info("current __version__: %s", current)
    target = args.bump or current
    if not re.match(r"^\d+(\.\d+){2,}$", target):
        log.error("invalid version: %s", target)
        return 1
    log.info("target  __version__: %s", target)

    extra_assets: List[Path] = []

    try:
        # -- 0. secrets fail-fast (skipped during dry-run since they're not used)
        if not args.dry_run:
            assert_secrets_present(skip_pypi=args.skip_pypi,
                                   skip_github=args.skip_github)

        log.info("\n--- 1. git checks ---")
        assert_clean_tree(force=args.force, dry_run=args.dry_run)
        assert_on_branch(args.branch, dry_run=args.dry_run)

        log.info("\n--- 2. sync versions -> %s ---", target)
        if target != current:
            sync_versions(target, dry_run=args.dry_run)
            commit_version_bump(target, dry_run=args.dry_run)
        else:
            log.info("  version unchanged, skipping sync")

        if not args.skip_validate_config:
            log.info("\n--- 3. validate cfg/config.yaml ---")
            validate_config_yaml(dry_run=args.dry_run)
        else:
            log.info("\n--- 3. validate cfg/config.yaml SKIPPED ---")

        if not args.skip_paper_guard:
            log.info("\n--- 3.5 paper placeholder guard ---")
            validate_paper_no_placeholder(dry_run=args.dry_run)
        else:
            log.warning("\n--- 3.5 paper placeholder guard SKIPPED ---")

        if not args.skip_tests:
            log.info("\n--- 4. tests ---")
            run_tests(dry_run=args.dry_run, force=args.force)
        else:
            log.warning("\n--- 4. tests SKIPPED (--skip-tests) ---")

        log.info("\n--- 5. build sdist + wheel ---")
        build_artifacts(dry_run=args.dry_run)
        if not args.dry_run:
            log.info("  built: %s", [p.name for p in list_artifacts()])

        if not args.skip_zip:
            log.info("\n--- 6. source ZIP archive ---")
            zip_path = build_source_zip(target, dry_run=args.dry_run)
            if zip_path:
                extra_assets.append(zip_path)
        else:
            log.info("\n--- 6. source ZIP archive SKIPPED ---")

        if not args.skip_manifest:
            log.info("\n--- 7. MANIFEST.txt ---")
            mf = write_manifest(target, dry_run=args.dry_run)
            if mf:
                extra_assets.append(mf)
        else:
            log.info("\n--- 7. MANIFEST.txt SKIPPED ---")

        if not args.skip_docs:
            log.info("\n--- 8. API docs (pdoc) ---")
            generate_api_docs(dry_run=args.dry_run)
        else:
            log.info("\n--- 8. API docs SKIPPED ---")

        if target != current:
            log.info("\n--- 9. tag + push ---")
            tag_and_push(target, dry_run=args.dry_run)
        else:
            log.info("\n--- 9. tag SKIPPED (no version bump) ---")

        if not args.skip_github and target != current:
            log.info("\n--- 10. GitHub release ---")
            github_release(target, dry_run=args.dry_run,
                           extra_assets=extra_assets)
        else:
            log.info("\n--- 10. GitHub release SKIPPED ---")

        if not args.skip_pypi:
            log.info("\n--- 11. PyPI upload ---")
            pypi_upload(dry_run=args.dry_run)
        else:
            log.info("\n--- 11. PyPI upload SKIPPED ---")

        if not args.skip_zenodo and target != current:
            log.info("\n--- 12. Zenodo publish ---")
            zenodo_publish(target, dry_run=args.dry_run)
        else:
            log.info("\n--- 12. Zenodo publish SKIPPED ---")

        log.info("\n=== DONE — see %s ===", DEPLOY_LOG.name)
        return 0
    except DeployError as e:
        log.error("\nERROR: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
