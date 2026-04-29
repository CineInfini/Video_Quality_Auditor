#!/usr/bin/env python3
"""Chunked runner for advanced pure-CV features (BRISQUE NSS + LBP + HOG +
DCT + Canny + multi-scale Sobel) on a video dataset.

Each invocation processes up to --max-this-call videos and exits within
--time-budget-s seconds. Resumable per-video.

Usage:
    python scripts/run_advanced_chunked.py \
        --videos-dir <dir> --labels-csv <file> \
        --labels-format <konvid|t2vqa|videofeedback|csv2> \
        --regime <natural|aigc> \
        --state-dir <dir> --max-this-call 50 --time-budget-s 42
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make src/cineinfini importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cineinfini.metrics.advanced_pure_cv import all_advanced_features


def load_labels(labels_csv: Path, fmt: str, regime: str) -> dict[str, dict]:
    if not labels_csv.exists():
        return {}
    text = labels_csv.read_text(encoding="utf-8", errors="replace")
    out = {}
    if fmt == "t2vqa":
        for ln in text.splitlines():
            parts = ln.strip().split("|")
            if len(parts) == 3:
                try:
                    out[parts[0]] = {"mos": float(parts[2]), "prompt": parts[1],
                                     "regime": regime}
                except ValueError:
                    continue
    elif fmt == "konvid":
        for ln in text.splitlines():
            parts = ln.split(",")
            if parts and parts[0].isdigit():
                try:
                    out[parts[0]] = {"mos": float(parts[1]), "prompt": "",
                                     "regime": regime}
                except (ValueError, IndexError):
                    continue
    elif fmt == "videofeedback":
        import io
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            if "video_path" in row and "mos" in row:
                try:
                    out[Path(row["video_path"]).name] = {
                        "mos": float(row["mos"]),
                        "prompt": row.get("prompt", ""),
                        "regime": regime,
                    }
                except (ValueError, TypeError):
                    continue
    elif fmt == "csv2":
        for ln in text.splitlines()[1:]:  # assume header
            parts = ln.split(",")
            if len(parts) >= 2:
                try:
                    out[parts[0].strip()] = {"mos": float(parts[1]),
                                              "prompt": "", "regime": regime}
                except ValueError:
                    continue
    return out


def build_workset(args, state_dir: Path) -> list[dict]:
    cache = state_dir / "workset_cache.json"
    if cache.exists():
        return json.loads(cache.read_text())
    state_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.labels_csv, args.labels_format, args.regime)
    items = []
    if not args.videos_dir.exists():
        cache.write_text("[]")
        return items
    all_paths = list(args.videos_dir.rglob("*.mp4"))
    index = {p.name: p for p in all_paths}
    konvid_index = {}
    if args.labels_format == "konvid":
        for p in all_paths:
            fid = p.name.split("_")[0]
            konvid_index.setdefault(fid, p)
    for vid, meta in labels.items():
        vp = (index.get(vid) or index.get(f"{vid}.mp4")
              or (konvid_index.get(vid) if args.labels_format == "konvid" else None))
        if vp is None:
            continue
        items.append({
            "video_id": vid, "video_path": str(vp),
            "regime": meta["regime"], "mos": meta["mos"], "prompt": meta["prompt"],
        })
    cache.write_text(json.dumps(items))
    return items


def sample_frames(video_path: str, n: int = 4) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idx = np.linspace(0, total - 1, min(n, total)).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok and fr is not None:
            frames.append(cv2.resize(fr, (320, 320)))
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", required=True, type=Path)
    ap.add_argument("--labels-csv", required=True, type=Path)
    ap.add_argument("--labels-format", required=True,
                    choices=["t2vqa", "konvid", "videofeedback", "csv2"])
    ap.add_argument("--regime", required=True, choices=["natural", "aigc"])
    ap.add_argument("--state-dir", required=True, type=Path)
    ap.add_argument("--max-this-call", type=int, default=40)
    ap.add_argument("--time-budget-s", type=float, default=42.0)
    args = ap.parse_args()

    workset = build_workset(args, args.state_dir)
    n_total = len(workset)
    print(f"Workset: {n_total} videos")
    results_jsonl = args.state_dir / "results.jsonl"

    done_ids = set()
    if results_jsonl.exists():
        for line in results_jsonl.read_text().splitlines():
            try:
                done_ids.add(json.loads(line)["video_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    pending = [w for w in workset if w["video_id"] not in done_ids]
    print(f"Done: {len(done_ids)}  Pending: {len(pending)}")
    if not pending:
        print("All done.")
        return 0

    t0 = time.time()
    processed = 0
    with results_jsonl.open("a") as out:
        for w in pending[:args.max_this_call]:
            if time.time() - t0 > args.time_budget_s:
                print(f"  budget reached at {processed}, exiting")
                break
            try:
                frames = sample_frames(w["video_path"], n=4)
                if len(frames) < 2:
                    continue
                feats = all_advanced_features(frames)
                feats["video_id"] = w["video_id"]
                feats["regime"] = w["regime"]
                feats["mos"] = w["mos"]
                feats["prompt"] = w["prompt"]
                out.write(json.dumps(feats) + "\n")
                out.flush()
                processed += 1
            except Exception as e:
                print(f"  ! {w['video_id']}: {e}")
            gc.collect()

    elapsed = time.time() - t0
    print(f"This call: {processed} videos in {elapsed:.1f}s "
          f"({elapsed / max(processed, 1):.2f}s/vid)")
    total = len(done_ids) + processed
    print(f"Cumulative: {total}/{n_total} ({100 * total / n_total:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
