#!/usr/bin/env python3
"""Resumable pure-CV calibration on KoNViD-1k. Each call processes
N videos and exits. State is checkpointed to disk so consecutive
calls resume where the previous left off, even if the parent bash
session was killed.

Usage (one chunk per call; re-invoke until the script reports 100 %):
    python scripts/run_konvid_chunked.py \
        --labels-csv data/KoNViD_mos_fr.csv \
        --videos-dir data/KoNViD-1k/videos \
        --state-dir out/konvid_chunks \
        --max-this-call 50 --time-budget-s 47

Files:
    <state-dir>/state.json    — progress state
    <state-dir>/results.jsonl — one row per processed video
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# CLI args set these — see main()
LABELS_CSV: Path | None = None
VIDEOS_DIR: Path | None = None
STATE_DIR: Path | None = None
STATE_FILE: Path | None = None
RESULTS_FILE: Path | None = None


def build_workset():
    """Return ordered list of (video_path, mos, flickr_id) tuples."""
    labels = {}
    for ln in LABELS_CSV.read_text().splitlines():
        parts = ln.split(",")
        if parts[0].isdigit():
            labels[parts[0]] = float(parts[1])
    items = []
    for v in sorted(VIDEOS_DIR.glob("*.mp4")):
        fid = v.name.split("_")[0]
        if fid in labels:
            items.append({"path": str(v), "mos": labels[fid], "flickr_id": fid})
    return items


def metrics(video_path: str, n_frames: int = 4) -> dict | None:
    """Pure-CV per-video metrics. n_frames=4 keeps it fast."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idx = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.resize(fr, (256, 144)))
    cap.release()
    if len(frames) < 2:
        return None
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames]
    return {
        "sharpness": float(np.mean([cv2.Laplacian(x, cv2.CV_64F).var() for x in g])),
        "brightness": float(np.mean([h[..., 2].mean() for h in hsv])),
        "saturation": float(np.mean([h[..., 1].mean() for h in hsv])),
        "contrast": float(np.mean([x.std() for x in g])),
        "flicker": float(np.mean([cv2.absdiff(frames[i], frames[i + 1]).mean()
                                   for i in range(len(frames) - 1)])),
        "flicker_var": float(np.var([cv2.absdiff(frames[i], frames[i + 1]).mean()
                                      for i in range(len(frames) - 1)])),
        "motion_proxy": float(np.mean([np.abs(g[i].astype(float) - g[i + 1].astype(float)).mean()
                                        for i in range(len(g) - 1)])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-csv", required=True, type=Path,
                    help="Path to KoNViD labels CSV (e.g. KoNViD_mos_fr.csv)")
    ap.add_argument("--videos-dir", required=True, type=Path,
                    help="Directory containing the KoNViD MP4 files")
    ap.add_argument("--state-dir", required=True, type=Path,
                    help="Directory for checkpoint state + results.jsonl")
    ap.add_argument("--max-this-call", type=int, default=25,
                    help="Max videos to process in this invocation")
    ap.add_argument("--time-budget-s", type=float, default=42.0,
                    help="Soft time budget; exit cleanly when reached")
    args = ap.parse_args()

    global LABELS_CSV, VIDEOS_DIR, STATE_DIR, STATE_FILE, RESULTS_FILE
    LABELS_CSV = args.labels_csv
    VIDEOS_DIR = args.videos_dir
    STATE_DIR = args.state_dir
    STATE_FILE = STATE_DIR / "state.json"
    RESULTS_FILE = STATE_DIR / "results.jsonl"

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    workset = build_workset()
    n_total = len(workset)

    # Load completed IDs from results.jsonl
    done_ids = set()
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text().splitlines():
            try:
                done_ids.add(json.loads(line)["flickr_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    pending = [w for w in workset if w["flickr_id"] not in done_ids]
    print(f"Total: {n_total}  done: {len(done_ids)}  pending: {len(pending)}", flush=True)

    if not pending:
        print("All done. Run aggregator next.", flush=True)
        return 0

    t0 = time.time()
    processed_this_call = 0
    with RESULTS_FILE.open("a") as out:
        for w in pending[:args.max_this_call]:
            elapsed = time.time() - t0
            if elapsed > args.time_budget_s:
                print(f"  time budget reached ({elapsed:.1f}s) — saving and exiting",
                      flush=True)
                break
            try:
                m = metrics(w["path"])
                if m is None:
                    continue
                m["mos"] = w["mos"]
                m["flickr_id"] = w["flickr_id"]
                out.write(json.dumps(m) + "\n")
                out.flush()
                processed_this_call += 1
            except Exception as e:
                print(f"  ! {w['flickr_id']} failed: {e}", flush=True)
            # Free memory between videos
            gc.collect()

    elapsed = time.time() - t0
    print(f"This call: processed {processed_this_call} videos in {elapsed:.1f}s "
          f"({elapsed/max(processed_this_call,1):.2f}s/video)", flush=True)
    total_done = len(done_ids) + processed_this_call
    print(f"Cumulative: {total_done}/{n_total} ({100*total_done/n_total:.1f}%)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
