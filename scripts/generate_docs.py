#!/usr/bin/env python3
"""Generate pdoc HTML API documentation for CineInfini.

Usage:
    python scripts/generate_docs.py [--output docs/api]

Requires `pip install pdoc>=14.0` (NOT the older pdoc3). Generates
HTML under the output directory; safe to commit on a `gh-pages`
branch or upload as a release artifact.

The deploy script (`deploy_cineinfini.py`) calls this in dry-run-safe
mode as part of the release process.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/api",
                        help="Output directory for generated HTML")
    parser.add_argument("--clean", action="store_true",
                        help="Remove the output directory first")
    args = parser.parse_args()

    out_dir = REPO / args.output
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check pdoc is installed
    try:
        subprocess.run([sys.executable, "-m", "pdoc", "--version"],
                       capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("error: pdoc is not installed.", file=sys.stderr)
        print("install with: pip install 'pdoc>=14.0'", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, "-m", "pdoc",
        "cineinfini",
        "--output-directory", str(out_dir),
        "--docformat", "google",
    ]
    print(f"$ {' '.join(cmd)}")
    env_paths = str(REPO / "src")
    proc = subprocess.run(
        cmd,
        env={**__import__("os").environ, "PYTHONPATH": env_paths},
    )
    if proc.returncode != 0:
        return proc.returncode

    # Index page
    index = out_dir / "cineinfini.html"
    if index.exists():
        print(f"\n✓ HTML docs generated at {out_dir}")
        print(f"  open: file://{index.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
