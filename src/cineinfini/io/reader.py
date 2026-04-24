import cv2
import numpy as np
import time
from pathlib import Path

def detect_shot_boundaries(video_path, max_duration_s, shot_threshold, min_shot_duration_s, downsample_to,
                           adaptive_threshold=True, threshold_percentile=85, step=2):
    t_total = time.time()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, int(max_duration_s * fps))
    min_gap = max(1, int(min_shot_duration_s * fps))

    hist_resize = (160, 90)
    bins_hue, bins_sat = 20, 20

    # First pass
    first_pass = []
    prev_hist = None
    idx = 0
    print(f"  First pass: computing histogram differences (step={step})...")
    t_first_start = time.time()
    last_print = 0
    while idx < limit:
        ok, frame = cap.read()
        if not ok: break
        if idx % step == 0:
            frame_small = cv2.resize(frame, hist_resize)
            hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0,1], None, [bins_hue, bins_sat], [0,180,0,256])
            hist = hist.flatten()
            if prev_hist is not None:
                diff = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA)
                first_pass.append(diff)
            prev_hist = hist
        idx += 1
        if idx - last_print >= 1000 * step or idx == limit:
            elapsed = time.time() - t_first_start
            print(f"      First pass: {idx}/{limit} frames processed (elapsed {elapsed:.1f}s)")
            last_print = idx
    cap.release()
    t_first_end = time.time()
    print(f"  [TIMING] First pass: {t_first_end - t_first_start:.1f}s, {len(first_pass)} diffs")

    if adaptive_threshold and first_pass:
        final_threshold = np.percentile(first_pass, threshold_percentile)
        print(f"  Adaptive threshold: {final_threshold:.3f}")
    else:
        final_threshold = shot_threshold

    # Second pass
    cap = cv2.VideoCapture(str(video_path))
    boundaries = [0]
    prev_hist = None
    idx = 0
    print("  Second pass: detecting boundaries...")
    t_second_start = time.time()
    last_print = 0
    while idx < limit:
        ok, frame = cap.read()
        if not ok: break
        if idx % step == 0:
            frame_small = cv2.resize(frame, hist_resize)
            hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0,1], None, [bins_hue, bins_sat], [0,180,0,256])
            hist = hist.flatten()
            if prev_hist is not None:
                diff = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > final_threshold and (idx - boundaries[-1]) >= min_gap:
                    boundaries.append(idx)
            prev_hist = hist
        idx += 1
        if idx - last_print >= 1000 * step or idx == limit:
            elapsed = time.time() - t_second_start
            print(f"      Second pass: {idx}/{limit} frames processed (found {len(boundaries)-1} boundaries, elapsed {elapsed:.1f}s)")
            last_print = idx
    cap.release()
    boundaries.append(idx)
    t_second_end = time.time()
    print(f"  [TIMING] Second pass: {t_second_end - t_second_start:.1f}s, found {len(boundaries)-1} cut candidates")

    shots = []
    for i in range(len(boundaries)-1):
        s, e = boundaries[i], boundaries[i+1]-1
        if e - s >= min_gap:
            shots.append((s, e, fps))
    if not shots and limit > 0:
        segment_frames = max(1, int(min_shot_duration_s * fps))
        for start in range(0, limit, segment_frames):
            end = min(start + segment_frames - 1, limit - 1)
            if end - start >= min_gap:
                shots.append((start, end, fps))
    if not shots and limit > 0:
        shots = [(0, limit-1, fps)]
    print(f"  [TIMING] Shot detection total: {time.time()-t_total:.1f}s, {len(shots)} shots generated")
    return shots

def extract_shot_frames_global(video_path, shots, n_frames_per_shot, frame_resize):
    t_start = time.time()
    needed = set()
    for (s,e,_) in shots:
        n = min(n_frames_per_shot, e - s + 1)
        idxs = np.linspace(s, e, n, dtype=int)
        needed.update(idxs)
    needed_sorted = sorted(needed)
    if not needed_sorted:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    frames_dict = {}
    current = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, needed_sorted[0])
    pos = 0
    print(f"  Extracting {len(needed_sorted)} required frames...")
    last_print = 0
    while cap.isOpened() and current <= needed_sorted[-1]:
        ret, frame = cap.read()
        if not ret:
            break
        if current in needed:
            frames_dict[current] = cv2.resize(frame, frame_resize)
            pos += 1
            if pos - last_print >= 100 or pos == len(needed_sorted):
                print(f"      Extraction progress: {pos}/{len(needed_sorted)} frames extracted")
                last_print = pos
        current += 1
    cap.release()
    print(f"  [TIMING] extract_shot_frames_global: {time.time()-t_start:.3f}s, {len(frames_dict)} unique frames")
    return frames_dict
