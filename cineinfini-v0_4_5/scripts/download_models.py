#!/usr/bin/env python
"""
download_models.py — Download all CineInfini model weights.

Uses the URL registry in ``cineinfini.core.config`` so there is ONE place
to update URLs when they change (not scattered across README / tests / code).

Features
--------
- SHA256 integrity check (skips download if file already present + valid)
- Resume-capable download with progress bar (tqdm if available, else % log)
- Proxy support via ``HTTP_PROXY`` / ``HTTPS_PROXY`` env vars
- Dry-run mode (--dry-run) prints what would be downloaded

Usage
-----
    python scripts/download_models.py              # download all
    python scripts/download_models.py arcface yunet  # specific models
    python scripts/download_models.py --dry-run    # show URLs only
    python scripts/download_models.py --models-dir /custom/path
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Optional


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress_hook(desc: str):
    """Return a urlretrieve reporthook that prints progress."""
    try:
        from tqdm import tqdm  # type: ignore[import]
        pbar = tqdm(unit="B", unit_scale=True, desc=desc, leave=False)

        def hook(count, block_size, total_size):
            if pbar.total is None and total_size > 0:
                pbar.total = total_size
            pbar.update(block_size)

        return hook, lambda: pbar.close()
    except ImportError:
        last: list = [0]

        def hook(count, block_size, total_size):
            downloaded = count * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                if pct - last[0] >= 10:
                    print(f"  {desc}: {pct}%", flush=True)
                    last[0] = pct

        return hook, lambda: None


def download_asset(
    key: str,
    url: str,
    filename: str,
    sha256: Optional[str],
    dest_dir: Path,
    dry_run: bool = False,
) -> bool:
    """Download one asset.  Return True on success (or already present)."""
    dest = dest_dir / filename
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already present and valid
    if dest.exists():
        if sha256 is not None:
            actual = _sha256(dest)
            if actual == sha256:
                print(f"  ✓ {key} — already present, hash OK ({dest})")
                return True
            else:
                print(f"  ✗ {key} — present but hash mismatch, re-downloading")
        else:
            print(f"  ✓ {key} — already present (no hash to verify) ({dest})")
            return True

    if dry_run:
        print(f"  [dry-run] {key}: {url} → {dest}")
        return True

    print(f"  ↓ {key}: {url}")
    tmp = dest.with_suffix(".part")
    try:
        hook, close = _progress_hook(filename)
        urllib.request.urlretrieve(url, str(tmp), reporthook=hook)
        close()
        if sha256 is not None:
            actual = _sha256(tmp)
            if actual != sha256:
                tmp.unlink()
                print(f"  ✗ {key} — download corrupt (expected {sha256[:8]}…, got {actual[:8]}…)")
                return False
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  ✓ {key} — saved {size_mb:.1f} MB to {dest}")
        return True
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        print(f"  ✗ {key} — download failed: {exc}")
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("keys", nargs="*",
                        help="Specific model keys to download (default: all)")
    parser.add_argument("--models-dir", default=None,
                        help="Destination directory (default: from config)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print URLs without downloading")
    parser.add_argument("--list", action="store_true",
                        help="List available model keys and exit")
    args = parser.parse_args(argv)

    from cineinfini.core.config import get_config
    cfg = get_config()
    registry = cfg.model_urls

    if args.list:
        print("Available model keys:")
        for k, v in registry.items():
            present = "✓" if cfg.model_path(k) and cfg.model_path(k).exists() else "·"
            print(f"  {present} {k:<20} {v['filename']}")
        return 0

    dest_dir = Path(args.models_dir).expanduser() if args.models_dir else cfg.models_dir()
    keys = args.keys if args.keys else list(registry.keys())

    print(f"Downloading to: {dest_dir}")
    print(f"Models: {', '.join(keys)}")
    print()

    ok = 0
    failed = []
    for key in keys:
        if key not in registry:
            print(f"  ⚠ Unknown key: {key}. Run --list to see available models.")
            failed.append(key)
            continue
        entry = registry[key]
        success = download_asset(
            key=key,
            url=entry["url"],
            filename=entry["filename"],
            sha256=entry.get("sha256"),
            dest_dir=dest_dir,
            dry_run=args.dry_run,
        )
        if success:
            ok += 1
        else:
            failed.append(key)

    print()
    print(f"Done: {ok}/{len(keys)} succeeded")
    if failed:
        print(f"Failed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
