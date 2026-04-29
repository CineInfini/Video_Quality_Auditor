#!/usr/bin/env python3
"""Build a labels CSV (and optionally download MP4s) for a calibration run.

Usage:
    python scripts/build_calibration_csv.py videofeedback \
        --label-field "visual quality" \
        --download-videos 50 \
        --output ~/.cineinfini/datasets/videofeedback/labels.csv

After running this you can directly call:

    cineinfini calibrate --labels-csv <output_path>

This script handles the gap between what `cineinfini datasets --fetch` gives
you (the HF dataset, with labels and frame thumbnails) and what
`cineinfini calibrate` needs (a CSV with `video_path,mos` rows pointing at
real MP4 files on disk).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_LABEL_FIELDS = {
    "videofeedback": "visual quality",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_key",
                   help="Dataset key from cfg.datasets (e.g. 'videofeedback')")
    p.add_argument("--split", default="test",
                   help="HF split to read from (default: test)")
    p.add_argument("--config-name", default=None,
                   help="HF dataset config name (default: same as fetch_command)")
    p.add_argument("--label-field", default=None,
                   help="Which MOS field to use (default: dataset's first label_field)")
    p.add_argument("--download-videos", type=int, default=0,
                   help="Download up to N MP4 files from videos_mirror (default: 0)")
    p.add_argument("--output", "-o", default=None,
                   help="Output CSV path (default: <dataset_dir>/labels.csv)")
    p.add_argument("--video-dir", default=None,
                   help="Where to store/find MP4s (default: <dataset_dir>/videos)")
    args = p.parse_args()

    try:
        from cineinfini.core.config import get_config
        from datasets import load_from_disk, load_dataset
    except ImportError as e:
        print(f"ImportError: {e}", file=sys.stderr)
        print("Run from a CineInfini install with `datasets` available:")
        print("  pip install datasets")
        return 2

    cfg = get_config()
    entry = cfg.datasets.get(args.dataset_key)
    if not entry:
        print(f"Unknown dataset key: {args.dataset_key}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(cfg.datasets.keys()))}", file=sys.stderr)
        return 1

    target = cfg.dataset_dir(args.dataset_key)
    video_dir = Path(args.video_dir) if args.video_dir else target / "videos"
    out_csv = Path(args.output) if args.output else target / "labels.csv"

    # Load: prefer cached on-disk version
    if (target / "dataset_info.json").exists() or (target / "test").exists():
        print(f"Loading from disk: {target}")
        try:
            ds = load_from_disk(str(target))
            split = ds[args.split] if hasattr(ds, "keys") and args.split in ds else ds
        except Exception:
            # Fall back to live download
            split = None
    else:
        split = None

    if split is None:
        cmd = entry.get("fetch_command", "")
        import re
        m = re.match(r"load_dataset\(['\"]([^'\"]+)['\"](?:,\s*name=['\"]([^'\"]+)['\"])?", cmd)
        repo = m.group(1) if m else f"TIGER-Lab/{args.dataset_key}"
        name = args.config_name or (m.group(2) if m else None)
        print(f"Loading from HF: {repo} (config: {name}, split: {args.split})")
        ds = load_dataset(repo, name=name, split=args.split) if name else load_dataset(repo, split=args.split)
        split = ds

    # Pick label field
    label_field = (args.label_field
                   or DEFAULT_LABEL_FIELDS.get(args.dataset_key)
                   or (entry.get("label_fields") or [None])[0])
    if not label_field:
        print("No label-field given and dataset has no label_fields entry", file=sys.stderr)
        return 1
    print(f"Label field: {label_field}")

    video_dir.mkdir(parents=True, exist_ok=True)

    # Optional: download N MP4s
    if args.download_videos > 0:
        try:
            import requests
        except ImportError:
            print("requests is needed for --download-videos. pip install requests")
            return 2
        print(f"Downloading up to {args.download_videos} MP4s -> {video_dir}")
        downloaded = 0
        for i, sample in enumerate(split):
            if downloaded >= args.download_videos:
                break
            url = sample.get("video link") or sample.get("video_link")
            if not url:
                continue
            fname = url.rsplit("/", 1)[-1]
            target_mp4 = video_dir / fname
            if target_mp4.exists():
                downloaded += 1
                continue
            try:
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()
                with open(target_mp4, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                downloaded += 1
                if downloaded % 10 == 0:
                    print(f"  downloaded {downloaded} / {args.download_videos}")
            except Exception as e:
                print(f"  skip {fname}: {e}")
        print(f"Downloaded {downloaded} MP4s")

    # Build CSV from samples that have a local MP4
    n_written = 0
    n_skipped = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_path", "mos"])
        for sample in split:
            url = sample.get("video link") or sample.get("video_link") or ""
            fname = url.rsplit("/", 1)[-1] if url else ""
            mp4_path = video_dir / fname if fname else None
            mos = sample.get(label_field)
            if mp4_path and mp4_path.exists() and mos is not None:
                w.writerow([str(mp4_path), mos])
                n_written += 1
            else:
                n_skipped += 1
    print(f"\n✅ Wrote {n_written} rows to {out_csv}")
    if n_skipped:
        print(f"   Skipped {n_skipped} samples (no local MP4 yet — run with --download-videos N)")
    print(f"\nNext step:")
    print(f"  cineinfini calibrate --labels-csv {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
