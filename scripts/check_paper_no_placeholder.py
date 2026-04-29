#!/usr/bin/env python3
"""Fail the build if any unfilled placeholder slot remains in PAPER_DRAFT.md.

This is wired into ``deploy_cineinfini.py`` as step 3.5 (after
``validate_config_yaml``, before tests). It is also runnable
stand-alone:

    python scripts/check_paper_no_placeholder.py docs/PAPER_DRAFT.md

The patterns below match every form of placeholder we use:

  * ``{{TBD-...}}``   — typed-slot markers (the official mechanism)
  * ``placeholder``   — anywhere (case-insensitive)
  * ``XXX``, ``TODO`` — common ad-hoc placeholders
  * ``TBD`` standalone (preceded/followed by space or punctuation)

The script returns:

  exit 0 — no placeholders found
  exit 1 — placeholders found, lists every offending line + position
  exit 2 — file not found / unreadable
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

# Regex set — case-insensitive where appropriate
PATTERNS = [
    (re.compile(r"\{\{TBD-[A-Za-z0-9_-]+\}\}"), "typed-slot {{TBD-...}}"),
    (re.compile(r"placeholder", re.IGNORECASE), "literal 'placeholder'"),
    (re.compile(r"\bXXX\b"), "literal 'XXX'"),
    (re.compile(r"\bTODO\b"), "literal 'TODO'"),
    (re.compile(r"\bTBD\b"), "literal 'TBD'"),
    (re.compile(r"FIXME", re.IGNORECASE), "literal 'FIXME'"),
]

# Lines that are *allowed* to mention these terms (e.g. policy statements)
ALLOWLIST_PATTERNS = [
    # The "no placeholder ever appears" promise itself mentions placeholder
    re.compile(r"no placeholder numbers? ever appears?", re.IGNORECASE),
    re.compile(r"no placeholder numbers? in tagged releases", re.IGNORECASE),
    re.compile(r"if any unfilled slot remains", re.IGNORECASE),
    re.compile(r"build pipeline.*fails", re.IGNORECASE),
    # The CI guard line in the appendix
    re.compile(r"check_paper_no_placeholder\.py", re.IGNORECASE),
    # Status banner block
    re.compile(r"^>\s", re.MULTILINE),
]


def is_allowlisted(line: str) -> bool:
    return any(p.search(line) for p in ALLOWLIST_PATTERNS)


def scan(path: Path) -> list[tuple[int, str, str, str]]:
    """Return list of (line_no, line, matched_text, pattern_label)."""
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        sys.exit(2)

    issues: list[tuple[int, str, str, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if is_allowlisted(line):
            continue
        for rx, label in PATTERNS:
            for m in rx.finditer(line):
                issues.append((lineno, line.strip(), m.group(0), label))
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["docs/PAPER_DRAFT.md"],
                    help="Files to scan (default: docs/PAPER_DRAFT.md)")
    ap.add_argument("--strict-tbd-only", action="store_true",
                    help="Only flag {{TBD-...}} slots (skip XXX/TODO/etc)")
    args = ap.parse_args()

    if args.strict_tbd_only:
        global PATTERNS
        PATTERNS = [PATTERNS[0]]

    total = 0
    for p in args.paths:
        path = Path(p)
        issues = scan(path)
        if not issues:
            print(f"  ✓ {path}: no placeholders")
            continue
        print(f"  ✗ {path}: {len(issues)} placeholder(s) found")
        for lineno, line, matched, label in issues:
            print(f"    L{lineno:>4}  [{label}]  '{matched}'")
            print(f"          → {line[:120]}")
        total += len(issues)

    if total > 0:
        print(f"\n=== {total} placeholder(s) in {len(args.paths)} file(s) ===",
              file=sys.stderr)
        print("Fill them via: python scripts/fill_paper.py "
              "<results.json> docs/PAPER_DRAFT.md", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
