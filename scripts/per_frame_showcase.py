#!/usr/bin/env python3
"""Extract DENSE per-frame pure-CV metrics on a sample of showcase videos.

Used to generate `docs/figures/showcase_perframe.json` consumed by
`scripts/build_comparison_report.py` for intra-video and inter-shot
analyses in the comparison report.

Usage:
    python scripts/per_frame_showcase.py \
        --picks-json picks.json \
        --konvid-dir data/KoNViD-1k/videos \
        --t2vqa-dir data/T2VQA-DB \
        --out-json docs/figures/showcase_perframe.json \
        [--n-frames 24]

picks.json structure:
    {"konvid": [["flickr_id", mos], ...], "t2vqa": [["video_id.mp4", mos], ...]}
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def find_konvid_video(flickr_id: str, base: Path) -> Path | None:
    for p in base.glob(f"{flickr_id}_*_8s.mp4"):
        return p
    return None


def find_t2vqa_video(name: str, base: Path) -> Path | None:
    for p in base.rglob(name):
        return p
    return None


def per_frame_metrics(video_path: str, n_frames: int = 24) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    out = []
    prev_g = None
    for k, i in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        fr_small = cv2.resize(fr, (256, 144))
        g = cv2.cvtColor(fr_small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(fr_small, cv2.COLOR_BGR2HSV)
        m = {
            "frame_idx": int(i),
            "sample_pos": int(k),
            "sharpness": float(cv2.Laplacian(g, cv2.CV_64F).var()),
            "brightness": float(hsv[..., 2].mean()),
            "saturation": float(hsv[..., 1].mean()),
            "contrast": float(g.std()),
        }
        if prev_g is not None:
            m["flicker_to_prev"] = float(np.abs(g.astype(float) - prev_g.astype(float)).mean())
        prev_g = g
        out.append(m)
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--picks-json", required=True, type=Path)
    ap.add_argument("--konvid-dir", type=Path)
    ap.add_argument("--t2vqa-dir", type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--n-frames", type=int, default=24)
    args = ap.parse_args()

    picks = json.loads(args.picks_json.read_text())
    out = {}
    for fid, mos in picks.get("konvid", []):
        if args.konvid_dir is None:
            continue
        vp = find_konvid_video(fid, args.konvid_dir)
        if vp is None:
            continue
        out[f"konvid:{fid}"] = {
            "dataset": "konvid", "mos": float(mos),
            "video_path": f"<konvid-dir>/{vp.name}",
            "frames": per_frame_metrics(str(vp), args.n_frames),
        }
        print(f"  konvid {fid}: {len(out[f'konvid:{fid}']['frames'])} frames")
    for vname, mos in picks.get("t2vqa", []):
        if args.t2vqa_dir is None:
            continue
        vp = find_t2vqa_video(vname, args.t2vqa_dir)
        if vp is None:
            continue
        out[f"t2vqa:{vname}"] = {
            "dataset": "t2vqa", "mos": float(mos),
            "video_path": f"<t2vqa-dir>/{vp.name}",
            "frames": per_frame_metrics(str(vp), args.n_frames),
        }
        print(f"  t2vqa {vname}: {len(out[f't2vqa:{vname}']['frames'])} frames")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {len(out)} videos to {args.out_json}")


if __name__ == "__main__":
    main()
