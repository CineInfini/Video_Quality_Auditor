"""Video reading, shot detection, and frame extraction"""
import cv2
import numpy as np
from pathlib import Path

def detect_shot_boundaries(video_path, max_duration_s, shot_threshold, min_shot_duration_s, downsample_to, adaptive_threshold=True, threshold_percentile=85):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, int(max_duration_s * fps))
    min_gap = max(1, int(min_shot_duration_s * fps))

    # First pass: collect histogram differences
    prev_hist = None
    idx = 0
    hist_diffs = []
    while idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, downsample_to)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [30,32], [0,180,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev_hist is not None:
            diff = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA)
            hist_diffs.append(diff)
        prev_hist = hist
        idx += 1
    cap.release()

    if adaptive_threshold and hist_diffs:
        final_threshold = np.percentile(hist_diffs, threshold_percentile)
        print(f"  Adaptive threshold: {final_threshold:.3f}")
    else:
        final_threshold = shot_threshold

    # Second pass: detect boundaries
    cap = cv2.VideoCapture(str(video_path))
    boundaries = [0]
    prev_hist = None
    idx = 0
    while idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, downsample_to)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [30,32], [0,180,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev_hist is not None:
            diff = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > final_threshold and (idx - boundaries[-1]) >= min_gap:
                boundaries.append(idx)
        prev_hist = hist
        idx += 1
    cap.release()
    boundaries.append(idx)

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
    return shots

def extract_shot_frames_global(video_path, shots, n_frames_per_shot, frame_resize):
    """Extract all required frames in one sequential pass."""
    # Collect all needed frame indices
    needed_indices = set()
    for (s, e, _) in shots:
        n = min(n_frames_per_shot, e - s + 1)
        idxs = np.linspace(s, e, n, dtype=int)
        needed_indices.update(idxs)
    needed_sorted = sorted(needed_indices)
    if not needed_sorted:
        return {}

    cap = cv2.VideoCapture(str(video_path))
    frames_dict = {}
    current_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, needed_sorted[0])
    pos = 0
    while cap.isOpened() and current_idx <= needed_sorted[-1]:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in needed_indices:
            frames_dict[current_idx] = cv2.resize(frame, frame_resize)
        current_idx += 1
    cap.release()
    return frames_dict
