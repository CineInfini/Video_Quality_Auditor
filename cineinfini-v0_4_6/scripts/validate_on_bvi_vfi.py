#!/usr/bin/env python
"""
validate_on_bvi_vfi.py — Partial validation of CineInfini metrics on BVI-VFI.

Workshop paper enabler: correlate CineInfini composite scores with human DMOS
on the BVI-VFI Database from University of Bristol.

Dataset
-------
Full archive (~5 GB):
    https://data.bris.ac.uk/datasets/tar/k8bfn0qsj9fs1rwnc2x75z6t7.zip

Strategy
--------
This script supports PARTIAL download — only fetches the videos and DMOS
files you specify, not the full 5 GB archive. Uses HTTP range requests
via the `remotezip` library when available, or downloads the central
directory only and prints a manifest.

Usage
-----
    # Step 1: list contents (no download yet)
    python scripts/validate_on_bvi_vfi.py --list

    # Step 2: download a small subset (5 videos + DMOS scores)
    python scripts/validate_on_bvi_vfi.py --subset 5 --output ~/.cineinfini/bvi_vfi/

    # Step 3: run audit on each video, correlate with DMOS
    python scripts/validate_on_bvi_vfi.py --analyse ~/.cineinfini/bvi_vfi/ \\
        --csv-out bvi_vfi_results.csv \\
        --report-out bvi_vfi_report.md

Output
------
A markdown report with:
  - Spearman + Pearson correlations between each metric and DMOS
  - Per-video metric values
  - Recommended threshold adjustments via core.calibrate

References
----------
Danier, D., Zhang, F., & Bull, D. (2022). BVI-VFI: A video quality database
for video frame interpolation. IEEE TIP.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional


BVI_VFI_URL = "https://data.bris.ac.uk/datasets/tar/k8bfn0qsj9fs1rwnc2x75z6t7.zip"


def list_contents(zip_url: str = BVI_VFI_URL) -> list[str]:
    """Use remotezip to list files in the archive without downloading it."""
    try:
        from remotezip import RemoteZip
    except ImportError:
        print("Install remotezip first: pip install remotezip")
        sys.exit(1)
    print(f"Reading archive index from {zip_url}...")
    print("(This downloads only the ZIP central directory, ~few KB)")
    with RemoteZip(zip_url) as zf:
        names = zf.namelist()
    print(f"\n{len(names)} entries in archive:")
    for n in names[:30]:
        print(f"  {n}")
    if len(names) > 30:
        print(f"  ... and {len(names) - 30} more")
    return names


def download_subset(
    output_dir: Path,
    subset_size: int = 5,
    zip_url: str = BVI_VFI_URL,
) -> list[Path]:
    """Download a subset of videos (and any DMOS file) from the archive."""
    try:
        from remotezip import RemoteZip
    except ImportError:
        print("Install remotezip: pip install remotezip")
        sys.exit(1)

    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening remote archive {zip_url}...")
    downloaded: list[Path] = []

    with RemoteZip(zip_url) as zf:
        names = zf.namelist()
        # Find video files (.mp4, .mov, .yuv) and DMOS metadata
        videos = [n for n in names if n.lower().endswith((".mp4", ".mov", ".yuv", ".y4m"))]
        metadata = [n for n in names if "dmos" in n.lower() or "score" in n.lower()
                    or n.endswith((".csv", ".xlsx", ".txt"))]

        # Always download metadata (small)
        for m in metadata[:10]:
            print(f"  ↓ metadata: {m}")
            zf.extract(m, str(output_dir))
            downloaded.append(output_dir / m)

        # Download the first N videos
        for v in videos[:subset_size]:
            print(f"  ↓ video: {v}")
            zf.extract(v, str(output_dir))
            downloaded.append(output_dir / v)

    print(f"\nDownloaded {len(downloaded)} files to {output_dir}")
    return downloaded


def analyse(
    bvi_dir: Path,
    csv_out: Path,
    report_out: Path,
):
    """Run CineInfini audit on each video, correlate with DMOS scores."""
    try:
        import numpy as np
        import pandas as pd
        from scipy.stats import spearmanr, pearsonr
    except ImportError as e:
        print(f"Missing dep: {e}. pip install pandas scipy")
        sys.exit(1)

    from cineinfini import audit_video, set_global_paths
    from cineinfini.core.face_detection import set_models_dir

    bvi_dir = Path(bvi_dir).expanduser()
    if not bvi_dir.exists():
        print(f"Directory not found: {bvi_dir}")
        sys.exit(1)

    # Find DMOS scores file
    dmos_files = list(bvi_dir.rglob("*dmos*")) + list(bvi_dir.rglob("*DMOS*"))
    dmos_files = [f for f in dmos_files if f.suffix in (".csv", ".xlsx", ".txt")]
    if not dmos_files:
        print(f"No DMOS file found in {bvi_dir}.")
        print(f"Expected a .csv with columns: video_name, dmos_score")
        sys.exit(1)
    dmos_file = dmos_files[0]
    print(f"Reading DMOS from {dmos_file}")
    if dmos_file.suffix == ".csv":
        dmos_df = pd.read_csv(dmos_file)
    else:
        dmos_df = pd.read_excel(dmos_file)
    print(f"  {len(dmos_df)} DMOS rows")

    # Find video files
    videos = list(bvi_dir.rglob("*.mp4")) + list(bvi_dir.rglob("*.mov")) \
           + list(bvi_dir.rglob("*.y4m")) + list(bvi_dir.rglob("*.yuv"))
    print(f"Found {len(videos)} videos")

    # Audit each video
    set_models_dir(Path.home() / ".cineinfini" / "models")
    output_root = bvi_dir / "audit_reports"
    output_root.mkdir(exist_ok=True)
    set_global_paths(output_root, output_root / "benchmark")

    results = []
    for v in videos:
        print(f"\n📹 Auditing {v.name}")
        try:
            metrics, _ = audit_video(
                str(v),
                video_params={"max_duration_s": 10, "n_frames_per_shot": 8},
            )
        except Exception as e:
            print(f"  ⚠ failed: {e}")
            continue

        # Aggregate per-video means
        gates = metrics.get("gates", {})
        if not gates:
            continue
        row = {"video_name": v.stem}
        for key in ["motion_peak_div", "ssim3d_self", "flicker", "identity_intra",
                    "ssim_long_range", "clip_temp_consistency", "composite"]:
            vals = [g.get(key) for g in gates.values() if g.get(key) is not None]
            row[key] = float(np.mean(vals)) if vals else None
        results.append(row)

    if not results:
        print("No successful audits.")
        sys.exit(1)

    results_df = pd.DataFrame(results)
    print(f"\n📊 {len(results_df)} videos audited")

    # Try to merge with DMOS
    if "video_name" in dmos_df.columns:
        merged = results_df.merge(dmos_df, on="video_name", how="inner")
    else:
        # Try first column as video name
        first_col = dmos_df.columns[0]
        dmos_df2 = dmos_df.rename(columns={first_col: "video_name"})
        merged = results_df.merge(dmos_df2, on="video_name", how="inner")

    print(f"  {len(merged)} videos matched with DMOS")
    if len(merged) == 0:
        print("⚠ No videos matched DMOS. Check column naming.")
        results_df.to_csv(csv_out, index=False)
        return

    # Find DMOS column (auto-detect)
    dmos_col = None
    for col in merged.columns:
        if "dmos" in col.lower() or "mos" in col.lower():
            dmos_col = col
            break
    if dmos_col is None:
        print(f"Columns available: {list(merged.columns)}")
        sys.exit(1)
    print(f"  Using DMOS column: {dmos_col}")

    # Correlations
    print(f"\n📈 Correlations with {dmos_col}:")
    correlations = []
    for metric in ["motion_peak_div", "ssim3d_self", "flicker", "identity_intra",
                   "ssim_long_range", "clip_temp_consistency", "composite"]:
        sub = merged.dropna(subset=[metric, dmos_col])
        if len(sub) < 3:
            continue
        rho, p_rho = spearmanr(sub[metric], sub[dmos_col])
        r, p_r = pearsonr(sub[metric], sub[dmos_col])
        correlations.append({
            "metric": metric, "n": len(sub),
            "spearman": float(rho), "p_spearman": float(p_rho),
            "pearson": float(r), "p_pearson": float(p_r),
        })
        print(f"  {metric:<25} ρ={rho:+.3f} (p={p_rho:.3f})  r={r:+.3f} (p={p_r:.3f})")

    # Save results
    merged.to_csv(csv_out, index=False)
    print(f"\nSaved metrics + DMOS to {csv_out}")

    # Markdown report
    report = ["# CineInfini × BVI-VFI Validation\n"]
    report.append(f"**N videos**: {len(merged)}\n")
    report.append(f"**DMOS column**: `{dmos_col}`\n\n")
    report.append("## Correlations (Spearman / Pearson)\n\n")
    report.append("| Metric | N | Spearman ρ | p | Pearson r | p |\n")
    report.append("|---|---|---|---|---|---|\n")
    for c in correlations:
        report.append(
            f"| {c['metric']} | {c['n']} | {c['spearman']:+.3f} | {c['p_spearman']:.3f} "
            f"| {c['pearson']:+.3f} | {c['p_pearson']:.3f} |\n"
        )
    report.append("\n## Per-video data\n\n")
    report.append(merged.to_markdown(index=False))
    Path(report_out).write_text("".join(report))
    print(f"Saved report to {report_out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true",
                   help="List archive contents (only downloads central directory)")
    p.add_argument("--subset", type=int, default=0,
                   help="Download N videos + metadata from archive")
    p.add_argument("--output", default="~/.cineinfini/bvi_vfi/", type=str,
                   help="Where to extract subset")
    p.add_argument("--analyse", type=str, default=None,
                   help="Run audit + correlation on a directory of BVI-VFI files")
    p.add_argument("--csv-out", default="bvi_vfi_results.csv", type=str)
    p.add_argument("--report-out", default="bvi_vfi_report.md", type=str)
    args = p.parse_args()

    if args.list:
        list_contents()
        return 0
    if args.subset > 0:
        download_subset(Path(args.output), subset_size=args.subset)
        return 0
    if args.analyse:
        analyse(Path(args.analyse), Path(args.csv_out), Path(args.report_out))
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
