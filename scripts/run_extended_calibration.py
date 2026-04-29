#!/usr/bin/env python3
"""Extended pure-CV metric extractor — adds 8 metrics on top of the v0.4.10.0
baseline, computed in a SINGLE pass per video. Chunked + resumable.

New metrics (cv2/numpy only, no ML weights):

  spatial_info       — ITU-R BT.500: std(Sobel-filtered frame), per-frame mean
  temporal_info      — ITU-R BT.500: std(inter-frame difference), per-frame mean
  edge_density       — Canny edge pixel ratio, per-frame mean
  color_richness     — unique colors after 4-bit quantization, per-frame mean
  block_artifact     — DCT 8×8 block boundary discontinuity score
  noise_level        — high-pass Laplacian residual std
  hog_consistency    — mean inter-frame cosine on HOG descriptors
  face_count         — total Haar-cascade frontal face detections across frames

Output: <state-dir>/results.jsonl with one row per video.
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import cv2
import numpy as np

DEFAULT_KEYS = [
    "sharpness", "brightness", "saturation", "contrast",
    "flicker", "flicker_var", "motion_proxy", "motion_var",
    "spatial_info", "temporal_info", "edge_density",
    "color_richness", "block_artifact", "noise_level",
    "hog_consistency", "face_count",
]


def load_labels(source: str, path: Path) -> dict[str, float]:
    out = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    if source == "t2vqa":
        for ln in text.splitlines():
            parts = ln.strip().split("|")
            if len(parts) == 3:
                try: out[parts[0]] = float(parts[2])
                except ValueError: continue
    elif source == "konvid":
        for ln in text.splitlines():
            parts = ln.split(",")
            if parts and parts[0].isdigit():
                try: out[parts[0]] = float(parts[1])
                except (ValueError, IndexError): continue
    elif source == "videofeedback":
        import csv, io
        for row in csv.DictReader(io.StringIO(text)):
            if "video_path" in row and "mos" in row:
                try: out[Path(row["video_path"]).name] = float(row["mos"])
                except (ValueError, TypeError): continue
    return out


def find_video(vid: str, base: Path, konvid: bool = False) -> Path | None:
    if (base / vid).exists():
        return base / vid
    if not vid.endswith(".mp4") and (base / f"{vid}.mp4").exists():
        return base / f"{vid}.mp4"
    cands = list(base.rglob(vid)) + list(base.rglob(f"{vid}.mp4"))
    if cands: return cands[0]
    if konvid:
        cands = list(base.rglob(f"{vid.replace('.mp4', '')}_*_8s.mp4"))
        if cands: return cands[0]
    return None


# Initialize HOG once globally for speed
_HOG = cv2.HOGDescriptor()
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def block_artifact_score(gray: np.ndarray) -> float:
    """Mean DCT high-frequency energy at 8×8 boundaries — proxy for blockiness."""
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 < 16 or w8 < 16:
        return 0.0
    g = gray[:h8, :w8].astype(np.float32)
    # Compare pixels just inside vs just outside each 8-pixel boundary
    # Vertical boundaries: cols at 7, 15, 23, ... vs 8, 16, 24, ...
    cols_in  = list(range(7, w8 - 1, 8))   # 7, 15, 23, ...
    cols_out = [c + 1 for c in cols_in]    # 8, 16, 24, ...
    if not cols_in: return 0.0
    v_bound = float(np.abs(g[:, cols_in] - g[:, cols_out]).mean())
    rows_in  = list(range(7, h8 - 1, 8))
    rows_out = [r + 1 for r in rows_in]
    h_bound = 0.0
    if rows_in:
        h_bound = float(np.abs(g[rows_in, :] - g[rows_out, :]).mean())
    return (v_bound + h_bound) / 2


def hog_descriptor(gray_64x128: np.ndarray) -> np.ndarray:
    """Standard 64×128 HOG; returns flat descriptor (3780-dim)."""
    return _HOG.compute(gray_64x128).flatten()


def metrics_extended(video_path: str, n_frames: int = 6) -> dict | None:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return None

    # Sequential sampling (faster than seeking on H.264)
    stride = max(1, total // n_frames)
    frames, gs, hsvs = [], [], []
    pos = 0
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok or fr is None: break
        if fi == pos:
            f256 = cv2.resize(fr, (256, 144))
            frames.append(f256)
            gs.append(cv2.cvtColor(f256, cv2.COLOR_BGR2GRAY))
            hsvs.append(cv2.cvtColor(f256, cv2.COLOR_BGR2HSV))
            pos += stride
            if len(frames) >= n_frames: break
        fi += 1
    cap.release()
    if len(frames) < 2: return None

    # Baseline metrics (same as v0.4.10.0)
    flicks = [cv2.absdiff(frames[i], frames[i + 1]).mean() for i in range(len(frames) - 1)]
    motions = [np.abs(gs[i].astype(float) - gs[i + 1].astype(float)).mean()
               for i in range(len(gs) - 1)]
    out = {
        "sharpness": float(np.mean([cv2.Laplacian(g, cv2.CV_64F).var() for g in gs])),
        "brightness": float(np.mean([h[..., 2].mean() for h in hsvs])),
        "saturation": float(np.mean([h[..., 1].mean() for h in hsvs])),
        "contrast": float(np.mean([g.std() for g in gs])),
        "flicker": float(np.mean(flicks)),
        "flicker_var": float(np.var(flicks)),
        "motion_proxy": float(np.mean(motions)),
        "motion_var": float(np.var(motions)),
    }

    # NEW: Spatial Information (Sobel-based, BT.500)
    sobel_stds = []
    for g in gs:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        sobel_stds.append(float(mag.std()))
    out["spatial_info"] = float(np.mean(sobel_stds))

    # NEW: Temporal Information (BT.500: std of inter-frame difference)
    if len(gs) > 1:
        ti_stds = [(gs[i + 1].astype(float) - gs[i].astype(float)).std()
                   for i in range(len(gs) - 1)]
        out["temporal_info"] = float(np.mean(ti_stds))
    else:
        out["temporal_info"] = 0.0

    # NEW: Edge density (Canny)
    canny_ratios = [cv2.Canny(g, 100, 200).mean() / 255.0 for g in gs]
    out["edge_density"] = float(np.mean(canny_ratios))

    # NEW: Color richness — unique 4-bit-quantized colors per frame
    rich = []
    for f in frames:
        q = (f >> 4).astype(np.uint8)  # 4-bit quantization per channel
        # Pack to 12-bit code per pixel
        codes = (q[..., 0].astype(np.int32) * 256 +
                 q[..., 1].astype(np.int32) * 16 +
                 q[..., 2].astype(np.int32))
        rich.append(len(np.unique(codes)))
    out["color_richness"] = float(np.mean(rich))

    # NEW: Block artifact
    out["block_artifact"] = float(np.mean([block_artifact_score(g) for g in gs]))

    # NEW: Noise level — high-pass residual std
    noise_levels = []
    for g in gs:
        blur = cv2.GaussianBlur(g, (5, 5), 1.5)
        residual = g.astype(float) - blur.astype(float)
        noise_levels.append(float(residual.std()))
    out["noise_level"] = float(np.mean(noise_levels))

    # NEW: HOG-based subject consistency (mean inter-frame cosine)
    try:
        hog_descs = []
        for g in gs:
            g_resized = cv2.resize(g, (64, 128))
            d = hog_descriptor(g_resized)
            n = np.linalg.norm(d) + 1e-9
            hog_descs.append(d / n)
        if len(hog_descs) >= 2:
            cosines = []
            for i in range(len(hog_descs)):
                for j in range(i + 1, len(hog_descs)):
                    cosines.append(float(np.dot(hog_descs[i], hog_descs[j])))
            out["hog_consistency"] = float(np.mean(cosines))
        else:
            out["hog_consistency"] = 0.0
    except Exception:
        out["hog_consistency"] = 0.0

    # NEW: Face count via Haar cascade
    n_faces = 0
    for g in gs:
        g_eq = cv2.equalizeHist(g)
        faces = _FACE_CASCADE.detectMultiScale(g_eq, scaleFactor=1.2,
                                                minNeighbors=4, minSize=(20, 20))
        n_faces += len(faces)
    out["face_count"] = int(n_faces)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-source", required=True,
                    choices=["t2vqa", "konvid", "videofeedback"])
    ap.add_argument("--labels-csv", required=True, type=Path)
    ap.add_argument("--videos-dir", required=True, type=Path)
    ap.add_argument("--state-dir", required=True, type=Path)
    ap.add_argument("--max-this-call", type=int, default=30)
    ap.add_argument("--time-budget-s", type=float, default=42.0)
    args = ap.parse_args()

    args.state_dir.mkdir(parents=True, exist_ok=True)
    results_file = args.state_dir / "results.jsonl"
    konvid = args.labels_source == "konvid"

    labels = load_labels(args.labels_source, args.labels_csv)
    workset = []

    # Pre-index video files by basename for fast lookup
    all_videos = list(args.videos_dir.rglob("*.mp4"))
    index = {p.name: p for p in all_videos}
    konvid_idx = {}
    if konvid:
        for p in all_videos:
            konvid_idx.setdefault(p.name.split("_")[0], p)

    for vid, mos in labels.items():
        vp = None
        if vid in index: vp = index[vid]
        elif f"{vid}.mp4" in index: vp = index[f"{vid}.mp4"]
        elif konvid and vid in konvid_idx: vp = konvid_idx[vid]
        if vp is not None:
            workset.append({"path": str(vp), "mos": mos, "video_id": vid})

    print(f"Workset: {len(workset)} videos", flush=True)

    done_ids = set()
    if results_file.exists():
        for ln in results_file.read_text().splitlines():
            try: done_ids.add(json.loads(ln)["video_id"])
            except (json.JSONDecodeError, KeyError): continue
    pending = [w for w in workset if w["video_id"] not in done_ids]
    print(f"Done: {len(done_ids)}  Pending: {len(pending)}", flush=True)
    if not pending:
        print("All done."); return 0

    t0 = time.time()
    processed = 0
    with results_file.open("a") as f:
        for w in pending[:args.max_this_call]:
            if time.time() - t0 > args.time_budget_s:
                print(f"  time budget at {processed}, exiting"); break
            try:
                m = metrics_extended(w["path"])
                if m is None: continue
                m["mos"] = w["mos"]
                m["video_id"] = w["video_id"]
                f.write(json.dumps(m) + "\n"); f.flush()
                processed += 1
            except Exception as e:
                print(f"  ! {w['video_id']}: {e}")
            gc.collect()

    elapsed = time.time() - t0
    print(f"This call: {processed} in {elapsed:.1f}s ({elapsed / max(processed,1):.2f}s/vid)")
    total = len(done_ids) + processed
    print(f"Cumulative: {total}/{len(workset)} ({100 * total / len(workset):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
