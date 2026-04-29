#!/usr/bin/env python3
"""Fill the {{TBD-XXX}} slots in PAPER_DRAFT.md with REAL numbers from
calibration results JSON.

Usage:
    python scripts/fill_paper.py \\
        out/t2vqa_calibration_paper.json \\
        out/videofeedback_calibration_paper.json \\
        out/competitor_results.json \\
        docs/PAPER_DRAFT.md

Every JSON file is a flat dict ``{slot_key: value}`` — see
``run_t2vqa_calibration.py`` for the canonical schema. Missing slots
are reported but do NOT cause a fill error; instead the slot remains
in the output, so the CI guard
(``scripts/check_paper_no_placeholder.py``) catches it at release time.

This separates concerns: filling is best-effort; the *gate* is the
guard. We never silently substitute a fake number.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

SLOT_RX = re.compile(r"\{\{TBD-([A-Za-z0-9_-]+)\}\}")


def format_value(v: Any, slot_key: str) -> str:
    """Format a value for inclusion in the paper.

    Heuristics:
      * RHO / PEARSON keys → 3-decimal signed
      * P (p-value) keys   → 4-decimal
      * CI keys            → 3-decimal signed
      * N / count keys     → integer
      * RATIO keys         → 1-decimal
      * other floats       → general repr
    """
    if v is None:
        return f"{{{{TBD-{slot_key}}}}}"  # leave unfilled
    s = slot_key.upper().replace("-", "_")
    try:
        if any(t in s for t in ("CI_LO", "CI_HI", "DELTA")):
            return f"{float(v):+.3f}"
        if (s.endswith("_P") or "_P_" in s or s.endswith("_P_VALUE")):
            fv = float(v)
            return f"{fv:.4f}" if fv >= 1e-4 else "<10⁻⁴"
        if any(t in s for t in ("RHO", "PEARSON")):
            return f"{float(v):+.3f}"
        if "RATIO" in s:
            return f"{float(v):.1f}"
        if any(t in s for t in ("_N", "NBOOT", "COUNT", "_COMPETITORS")):
            return str(int(v))
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)
    except (ValueError, TypeError):
        return str(v)


def fill(text: str, mapping: Dict[str, Any]) -> tuple[str, List[str], List[str]]:
    """Replace {{TBD-X}} with mapping[X] formatted; return (new_text,
    filled, missing). Lookup tries both ``X`` and ``X.replace('-','_')``."""
    filled: List[str] = []
    missing: List[str] = []

    def lookup(key: str) -> Any:
        # Try original key, then dash→underscore, then upper variants
        for candidate in (key, key.replace("-", "_"),
                          key.upper(), key.upper().replace("-", "_")):
            if candidate in mapping and mapping[candidate] is not None:
                return mapping[candidate]
        return None

    def repl(m: re.Match) -> str:
        key = m.group(1)
        v = lookup(key)
        if v is not None:
            filled.append(key)
            return format_value(v, key)
        missing.append(key)
        return m.group(0)

    return SLOT_RX.sub(repl, text), filled, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", help="One or more results JSONs")
    ap.add_argument("paper", help="docs/PAPER_DRAFT.md (modified in place)")
    ap.add_argument("-o", "--output", help="Output path; default = overwrite paper")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be filled, do not write")
    args = ap.parse_args()

    # Merge all result JSONs into one mapping
    mapping: Dict[str, Any] = {}
    for r in args.results:
        path = Path(r)
        if not path.exists():
            print(f"warning: {r} not found, skipping", file=sys.stderr)
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            print(f"warning: {r} is not a dict, skipping", file=sys.stderr)
            continue
        mapping.update(data)
        print(f"  loaded {len(data)} keys from {r}")

    paper_path = Path(args.paper)
    text = paper_path.read_text(encoding="utf-8")
    new_text, filled, missing = fill(text, mapping)

    print(f"\n  filled  : {len(filled)} slot(s)")
    if filled:
        for k in sorted(set(filled)):
            # Use same lookup as fill() to find the value
            for cand in (k, k.replace("-", "_"), k.upper(),
                         k.upper().replace("-", "_")):
                if cand in mapping:
                    print(f"    ✓ {k:<35} = {format_value(mapping[cand], k)}")
                    break
    print(f"  missing : {len(missing)} slot(s)")
    if missing:
        for k in sorted(set(missing)):
            print(f"    ⚠ {k}")

    if args.dry_run:
        return 0

    out = Path(args.output) if args.output else paper_path
    out.write_text(new_text, encoding="utf-8")
    print(f"\n  wrote -> {out}")
    print("  next: python scripts/check_paper_no_placeholder.py "
          f"{out} (must exit 0 before release)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
