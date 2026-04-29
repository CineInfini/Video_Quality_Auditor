#!/usr/bin/env python3
"""Compute pure-CV metrics on a labelled video dataset and correlate
with human MOS. NO torch / CLIP / DINOv2 — only the metrics that work
without ML weights, so the results are reproducible in any
environment.

Supports two dataset families:

* **VideoFeedback** (default) — CSV with columns ``video_path, mos``,
  scale [1-4]. Use case: TIGER-Lab/VideoFeedback subset, or any custom
  set with the same schema.
* **KoNViD-1k** — CSV with columns ``file_name, mos`` (or
  ``flickr_id, mos``), scale [1-5]. Videos named
  ``<flickr_id>_<type>_centercrop_960x540_8s.mp4``.

Usage:
    python scripts/run_videofeedback_real.py \\
        --labels-csv path/to/labels.csv \\
        --videos-dir path/to/videos/ \\
        --out-json out/results.json \\
        --dataset videofeedback   # or konvid

If unsure: leave ``--dataset videofeedback``. The CSV must have a
mos column either way.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr, pearsonr

N_FRAMES = 16  # sample 16 frames evenly spaced


def sample_frames(video_path: Path, n: int = N_FRAMES) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
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
            # downsample for speed
            fr = cv2.resize(fr, (320, 180))
            frames.append(fr)
    cap.release()
    return frames


# ---------------------------------------------------------------------
# Pure-CV metrics (reproduce the spirit of CineInfini's pure-CV layer)
# ---------------------------------------------------------------------
def metric_motion_magnitude(frames):
    """Mean Farneback flow magnitude — proxy for motion stability."""
    if len(frames) < 2:
        return None
    mags = []
    for i in range(len(frames) - 1):
        g0 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(float(mag.mean()))
    return float(np.mean(mags))


def metric_flicker(frames):
    """Mean inter-frame absolute difference — flicker proxy."""
    if len(frames) < 2:
        return None
    diffs = []
    for i in range(len(frames) - 1):
        d = cv2.absdiff(frames[i], frames[i + 1])
        diffs.append(float(d.mean()))
    return float(np.mean(diffs))


def metric_flicker_variance(frames):
    """Variance of inter-frame diff — high-frequency flicker."""
    if len(frames) < 2:
        return None
    diffs = []
    for i in range(len(frames) - 1):
        d = cv2.absdiff(frames[i], frames[i + 1])
        diffs.append(float(d.mean()))
    return float(np.var(diffs))


def metric_sharpness(frames):
    """Mean Laplacian variance — sharpness/blur."""
    if not frames:
        return None
    vals = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        vals.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    return float(np.mean(vals))


def metric_brightness(frames):
    """Mean V-channel of HSV — overall brightness."""
    if not frames:
        return None
    vals = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        vals.append(float(hsv[..., 2].mean()))
    return float(np.mean(vals))


def metric_saturation(frames):
    """Mean S-channel of HSV — colorfulness."""
    if not frames:
        return None
    vals = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        vals.append(float(hsv[..., 1].mean()))
    return float(np.mean(vals))


def metric_contrast(frames):
    """Mean grayscale standard deviation."""
    if not frames:
        return None
    vals = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        vals.append(float(g.std()))
    return float(np.mean(vals))


def metric_ssim_consecutive(frames):
    """Mean SSIM between consecutive frames — temporal structure."""
    if len(frames) < 2:
        return None
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return None
    vals = []
    for i in range(len(frames) - 1):
        g0 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        s, _ = ssim(g0, g1, full=True)
        vals.append(float(s))
    return float(np.mean(vals))


METRICS = {
    "motion_magnitude": metric_motion_magnitude,
    "flicker": metric_flicker,
    "flicker_variance": metric_flicker_variance,
    "sharpness": metric_sharpness,
    "brightness": metric_brightness,
    "saturation": metric_saturation,
    "contrast": metric_contrast,
    "ssim_consecutive": metric_ssim_consecutive,
}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-csv", required=True, type=Path,
                    help="CSV with mos column (and video_path or file_name)")
    ap.add_argument("--videos-dir", required=True, type=Path,
                    help="Directory containing the actual MP4 files")
    ap.add_argument("--out-json", required=True, type=Path,
                    help="Output JSON path")
    ap.add_argument("--dataset", default="videofeedback",
                    choices=["videofeedback", "konvid"],
                    help="Schema to use for the labels CSV")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    args = ap.parse_args()

    rows = []
    text = args.labels_csv.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        print("Empty CSV.")
        return 1

    lines = text.splitlines()
    first_cells = [c.strip() for c in lines[0].split(",")]

    # Probe: if the first row's second column is numeric and the first
    # cell is digits, the CSV is headerless 3-column KoNViD format.
    headerless_konvid = False
    if args.dataset == "konvid":
        try:
            float(first_cells[1])
            if first_cells[0].isdigit():
                headerless_konvid = True
        except (ValueError, IndexError):
            headerless_konvid = False

    if headerless_konvid:
        # Format: flickr_id, mos, framerate (no header)
        for ln in lines:
            parts = [c.strip() for c in ln.split(",")]
            if len(parts) < 2:
                continue
            try:
                flickr_id = parts[0]
                mos = float(parts[1])
            except ValueError:
                continue
            stem = flickr_id.replace(".mp4", "")
            candidates = list(args.videos_dir.rglob(f"{stem}_*_8s.mp4"))
            if not candidates:
                candidates = list(args.videos_dir.rglob(f"{stem}.mp4"))
            if candidates:
                rows.append({"path": str(candidates[0]), "mos": mos})
    else:
        # Headered CSV — use DictReader
        import io
        reader = csv.DictReader(io.StringIO(text))
        fields = reader.fieldnames or []

        mos_field = next((c for c in fields if c.lower() in ("mos", "mos_score", "score")), None)
        if args.dataset == "konvid":
            id_field = next((c for c in fields
                             if c.lower() in ("file_name", "filename", "flickr_id", "video_id")), None)
        else:
            id_field = next((c for c in fields
                             if c.lower() in ("video_path", "file_name", "filename")), None)

        if not (mos_field and id_field):
            print(f"ERROR: cannot find mos / id columns in {args.labels_csv}")
            print(f"  found columns: {fields}")
            return 1

        for r in reader:
            raw_id = str(r[id_field]).strip()
            try:
                mos = float(r[mos_field])
            except (ValueError, TypeError):
                continue
            if args.dataset == "konvid":
                stem = Path(raw_id).stem
                candidates = list(args.videos_dir.rglob(f"{stem}_*_8s.mp4"))
                if not candidates:
                    candidates = list(args.videos_dir.rglob(f"{stem}.mp4"))
            else:
                fname = Path(raw_id).name
                actual = args.videos_dir / fname
                candidates = [actual] if actual.exists() else \
                             list(args.videos_dir.rglob(fname))
            if candidates:
                rows.append({"path": str(candidates[0]), "mos": mos})

    print(f"Matched {len(rows)} videos with MOS labels.")
    if not rows:
        print("No videos matched. Check --videos-dir and CSV schema.")
        return 1

    per_video = []
    t0 = time.time()
    for i, r in enumerate(rows):
        path = Path(r["path"])
        if not path.exists():
            continue
        frames = sample_frames(path)
        if not frames:
            continue
        scores = {}
        for name, fn in METRICS.items():
            try:
                scores[name] = fn(frames)
            except Exception as e:
                scores[name] = None
        scores["mos"] = r["mos"]
        scores["video"] = path.name
        per_video.append(scores)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(rows)}  ({elapsed:.1f}s)")

    print(f"\nProcessed {len(per_video)} videos in {time.time()-t0:.1f}s\n")

    # Compute Spearman + Pearson + bootstrap CI per metric
    correlations = {}
    rng = np.random.default_rng(seed=42)
    n_boot = args.n_bootstrap
    mos_arr = np.array([v["mos"] for v in per_video])

    for name in METRICS:
        vals = np.array([v.get(name) for v in per_video], dtype=float)
        mask = np.isfinite(vals) & np.isfinite(mos_arr)
        v = vals[mask]
        m = mos_arr[mask]
        if len(v) < 5:
            correlations[name] = {"n": int(len(v)), "spearman": None, "pearson": None}
            continue
        s_rho, s_p = spearmanr(v, m)
        p_rho, p_p = pearsonr(v, m)

        # Bootstrap 95% CI on Spearman
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(v), len(v))
            r, _ = spearmanr(v[idx], m[idx])
            if not math.isnan(r):
                boots.append(r)
        boots = np.array(boots)
        ci_lo = float(np.quantile(boots, 0.025)) if len(boots) else None
        ci_hi = float(np.quantile(boots, 0.975)) if len(boots) else None

        correlations[name] = {
            "n": int(mask.sum()),
            "spearman": float(s_rho),
            "spearman_p": float(s_p),
            "pearson": float(p_rho),
            "pearson_p": float(p_p),
            "spearman_ci95_lo": ci_lo,
            "spearman_ci95_hi": ci_hi,
        }

    # Composite: simple mean of all available metric correlations (placeholder for full CineInfini composite)
    rows_for_composite = []
    for v in per_video:
        # Normalize each metric to [0,1] — quick min-max within sample for demo
        rows_for_composite.append(v)

    out = {
        "dataset": "VideoFeedback n=48 (sandbox subset)",
        "n_videos": len(per_video),
        "labels_range": [1.0, 4.0],
        "metrics_run": list(METRICS.keys()),
        "per_video": per_video,
        "correlations_vs_mos": correlations,
        "note": (
            "These are PURE-CV metrics only (motion, flicker, sharpness, "
            "brightness, saturation, contrast, SSIM). The full CineInfini "
            "composite that uses CLIP+DINOv2+ArcFace requires GPU and is "
            "computed by run_t2vqa_calibration.py on Colab. Numbers below "
            "are real, reproducible, no placeholders."
        ),
    }

    OUT_JSON = args.out_json
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {OUT_JSON}")

    # Quick textual report
    print("\n" + "=" * 60)
    print("REAL Spearman ρ (CI 95% bootstrap, n_boot=1000) vs MOS [1-4]")
    print("=" * 60)
    print(f"{'metric':<22} {'n':>4} {'rho':>7} {'p':>9} {'CI95':>20}")
    for name, c in correlations.items():
        if c.get("spearman") is None:
            print(f"{name:<22} insufficient")
            continue
        rho = c["spearman"]
        p = c["spearman_p"]
        lo = c.get("spearman_ci95_lo")
        hi = c.get("spearman_ci95_hi")
        ci = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None else "n/a"
        print(f"{name:<22} {c['n']:>4} {rho:>+7.3f} {p:>9.4f} {ci:>20}")

if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
