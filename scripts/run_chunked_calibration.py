#!/usr/bin/env python3
"""Generic resumable chunked pure-CV calibration for any video dataset.

Supersedes the dataset-specific run_konvid_chunked.py from v0.4.9.1 by
accepting an arbitrary labels source. State is checkpointed per-video
to disk; if the parent bash session times out, the next invocation
resumes where the previous left off.

Supported labels sources:
  * 't2vqa'       — info.txt format ``video_id.mp4|prompt|MOS``
  * 'konvid'      — KoNViD_mos_fr.csv format (3-col headerless: id,mos,fr)
  * 'videofeedback' — labels_with_videos.csv (video_path,mos)
  * 'csv2'        — generic 2-column CSV (id_or_path, mos), optional header

Usage (one chunk per call; re-invoke until the script reports 100 %):
    python scripts/run_chunked_calibration.py \\
        --labels-source t2vqa \\
        --labels-csv /path/to/info.txt \\
        --videos-dir /path/to/videos \\
        --state-dir out/t2vqa_chunks \\
        --max-this-call 50 --time-budget-s 47

Then aggregate:
    python scripts/aggregate_chunked.py \\
        --results-jsonl out/t2vqa_chunks/results.jsonl \\
        --out-json docs/figures/t2vqa_real_results.json
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


def load_labels(source: str, path: Path) -> dict[str, float]:
    """Return {video_id_or_basename: mos} from labels source."""
    out: dict[str, float] = {}
    text = path.read_text(encoding="utf-8", errors="replace")

    if source == "t2vqa":
        for ln in text.splitlines():
            parts = ln.strip().split("|")
            if len(parts) != 3:
                continue
            try:
                out[parts[0]] = float(parts[2])
            except ValueError:
                continue
        return out

    if source == "konvid":
        for ln in text.splitlines():
            parts = ln.split(",")
            if parts and parts[0].isdigit():
                try:
                    out[parts[0]] = float(parts[1])
                except (ValueError, IndexError):
                    continue
        return out

    if source == "videofeedback":
        import io
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            if "video_path" in row and "mos" in row:
                try:
                    out[Path(row["video_path"]).name] = float(row["mos"])
                except (ValueError, TypeError):
                    continue
        return out

    if source == "csv2":
        # Auto-detect header: if first non-empty cell of first row is numeric → no header
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return out
        first = lines[0].split(",")
        try:
            float(first[1])
            has_header = not first[0].replace(".", "").replace("-", "").isdigit() \
                         and not Path(first[0]).suffix
            # If first cell is a "raw" id (digits or path) and second is numeric,
            # treat as headerless
        except (ValueError, IndexError):
            has_header = True
        rows = lines[1:] if has_header else lines
        for ln in rows:
            parts = ln.split(",")
            if len(parts) < 2:
                continue
            try:
                out[parts[0].strip()] = float(parts[1])
            except ValueError:
                continue
        return out

    raise ValueError(f"unknown labels source: {source!r}")


def find_video(video_id: str, videos_dir: Path,
               konvid_pattern: bool = False) -> Path | None:
    """Resolve video_id (either a basename like '00000_07.mp4' or an id like
    '8536919744') to an actual file under videos_dir."""
    # Direct hit
    p = videos_dir / video_id
    if p.exists():
        return p
    if not video_id.endswith(".mp4"):
        p2 = videos_dir / f"{video_id}.mp4"
        if p2.exists():
            return p2
    # Recursive search by basename
    candidates = list(videos_dir.rglob(video_id))
    if candidates:
        return candidates[0]
    if not video_id.endswith(".mp4"):
        candidates = list(videos_dir.rglob(f"{video_id}.mp4"))
        if candidates:
            return candidates[0]
    # KoNViD fuzzy: <flickr_id>_<type>_centercrop_960x540_8s.mp4
    if konvid_pattern:
        stem = video_id.replace(".mp4", "")
        candidates = list(videos_dir.rglob(f"{stem}_*_8s.mp4"))
        if candidates:
            return candidates[0]
    return None


def metrics(video_path: str, n_frames: int = 4) -> dict | None:
    """Pure-CV per-video metrics. n_frames=4 keeps it fast (~0.85s)."""
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
    flicks = [cv2.absdiff(frames[i], frames[i + 1]).mean()
              for i in range(len(frames) - 1)]
    motions = [np.abs(g[i].astype(float) - g[i + 1].astype(float)).mean()
               for i in range(len(g) - 1)]
    return {
        "sharpness": float(np.mean([cv2.Laplacian(x, cv2.CV_64F).var() for x in g])),
        "brightness": float(np.mean([h[..., 2].mean() for h in hsv])),
        "saturation": float(np.mean([h[..., 1].mean() for h in hsv])),
        "contrast": float(np.mean([x.std() for x in g])),
        "flicker": float(np.mean(flicks)),
        "flicker_var": float(np.var(flicks)),
        "motion_proxy": float(np.mean(motions)),
        "motion_var": float(np.var(motions)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-source", required=True,
                    choices=["t2vqa", "konvid", "videofeedback", "csv2"])
    ap.add_argument("--labels-csv", required=True, type=Path)
    ap.add_argument("--videos-dir", required=True, type=Path)
    ap.add_argument("--state-dir", required=True, type=Path)
    ap.add_argument("--max-this-call", type=int, default=50)
    ap.add_argument("--time-budget-s", type=float, default=42.0)
    args = ap.parse_args()

    args.state_dir.mkdir(parents=True, exist_ok=True)
    results_file = args.state_dir / "results.jsonl"
    konvid_fuzzy = (args.labels_source == "konvid")

    # Load labels and resolve video paths
    labels = load_labels(args.labels_source, args.labels_csv)
    workset = []
    for vid, mos in labels.items():
        vp = find_video(vid, args.videos_dir, konvid_pattern=konvid_fuzzy)
        if vp is not None:
            workset.append({"path": str(vp), "mos": mos, "video_id": vid})

    n_total = len(workset)
    print(f"Labels: {len(labels)}  matched videos: {n_total}", flush=True)

    # Load completed IDs
    done_ids = set()
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            try:
                done_ids.add(json.loads(line)["video_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    pending = [w for w in workset if w["video_id"] not in done_ids]
    print(f"Done: {len(done_ids)}  Pending: {len(pending)}", flush=True)

    if not pending:
        print("All done. Run aggregator next.", flush=True)
        return 0

    t0 = time.time()
    processed = 0
    with results_file.open("a") as f:
        for w in pending[:args.max_this_call]:
            if time.time() - t0 > args.time_budget_s:
                print(f"  time budget reached, exiting cleanly", flush=True)
                break
            try:
                m = metrics(w["path"])
                if m is None:
                    continue
                m["mos"] = w["mos"]
                m["video_id"] = w["video_id"]
                f.write(json.dumps(m) + "\n")
                f.flush()
                processed += 1
            except Exception as e:
                print(f"  ! {w['video_id']}: {e}", flush=True)
            gc.collect()

    elapsed = time.time() - t0
    print(f"This call: {processed} videos in {elapsed:.1f}s "
          f"({elapsed / max(processed, 1):.2f}s/video)", flush=True)
    total = len(done_ids) + processed
    print(f"Cumulative: {total}/{n_total} ({100 * total / n_total:.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
