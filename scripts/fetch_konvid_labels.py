#!/usr/bin/env python3
"""Fetch the KoNViD-1k MOS labels CSV from the cnn-tlvqm GitHub mirror.

The full KoNViD-1k attributes file (`KoNViD_1k_attributes.csv`) is gated
behind a registration form at http://database.mmsp-kn.de/. However,
Korhonen's CNN-TLVQM repository on GitHub publishes a 2-column variant
(`KoNViD_mos_fr.csv`, columns: `file_name, mos`) under a permissive
license. This helper downloads that file so calibration can proceed
without registration.

Usage:
    python scripts/fetch_konvid_labels.py --out-dir data/

If you have already obtained the official `KoNViD_1k_attributes.csv`,
prefer it — it contains additional metadata (DMOS, framerate, content
labels). The cnn-tlvqm CSV is sufficient for Spearman / Pearson
calibration but not for content-aware splits.
"""
from __future__ import annotations
import argparse
import sys
import urllib.request
from pathlib import Path

# Multiple URLs in fallback order — in case one mirror is down.
URLS = [
    # CNN-TLVQM repo, master branch
    "https://github.com/jarikorhonen/cnn-tlvqm/raw/refs/heads/master/KoNViD_mos_fr.csv",
    # CNN-TLVQM via raw.githubusercontent
    "https://raw.githubusercontent.com/jarikorhonen/cnn-tlvqm/master/KoNViD_mos_fr.csv",
]

USER_AGENT = "cineinfini-fetcher/0.4.9.0 (+https://github.com/CineInfini/Video_Quality_Auditor)"


def download(url: str, dest: Path, timeout: int = 30) -> bool:
    """Try to download `url` to `dest`. Returns True on success."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 100:  # too small to be the real CSV
            print(f"  ! suspiciously small response from {url} ({len(data)} bytes)")
            return False
        dest.write_bytes(data)
        print(f"  ✓ {url}")
        print(f"    → {dest} ({len(data):,} bytes)")
        return True
    except Exception as e:
        print(f"  ✗ {url}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data"),
                    help="Output directory (created if missing)")
    ap.add_argument("--filename", default="KoNViD_mos_fr.csv",
                    help="Output filename")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / args.filename

    if dest.exists() and dest.stat().st_size > 100:
        print(f"  already present: {dest} ({dest.stat().st_size:,} bytes)")
        print("  delete it first to force re-download")
        return 0

    print(f"Fetching KoNViD-1k labels into {dest} …")
    for url in URLS:
        if download(url, dest):
            # Quick sanity check: count rows
            n = sum(1 for _ in dest.open(encoding="utf-8", errors="replace"))
            print(f"\nDownloaded {n} lines (expected ~1200).")
            return 0

    print("\nALL MIRRORS FAILED.", file=sys.stderr)
    print("Manual fallback: download from", file=sys.stderr)
    for u in URLS:
        print(f"  - {u}", file=sys.stderr)
    print(f"and place at {dest}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
