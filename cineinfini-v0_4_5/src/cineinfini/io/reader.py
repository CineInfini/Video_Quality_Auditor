"""
Video reading, shot boundary detection, and frame extraction.

v0.4.0 refactor
---------------
``detect_shot_boundaries`` (CC=26) decomposed into 3 functions:

1. ``_compute_histogram_diffs``  — single video pass, returns diff array
2. ``_apply_threshold``          — pure function, picks cut threshold
3. ``_boundaries_to_shots``      — pure function, converts boundaries to shots

``detect_shot_boundaries`` is now a thin orchestrator (CC ≤ 6).
All sub-functions are unit-testable without a real video.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIST_RESIZE = (160, 90)
_BINS_HUE    = 20
_BINS_SAT    = 20


# ---------------------------------------------------------------------------
# Stage 1 — histogram differences (single video pass)
# ---------------------------------------------------------------------------

def _compute_histogram_diffs(
    video_path: Path,
    limit: int,
    step: int = 2,
    verbose: bool = True,
) -> tuple[list[float], float]:
    """Compute Bhattacharyya distances between consecutive HSV histograms.

    Parameters
    ----------
    video_path : Path
    limit : int
        Maximum frame index to process.
    step : int
        Process every `step`-th frame (speed vs precision).
    verbose : bool

    Returns
    -------
    (diffs, fps) : (list[float], float)
        ``diffs[i]`` is the distance between histogram i and i-1.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 24.0

    diffs: list[float] = []
    prev_hist: Optional[np.ndarray] = None
    idx = 0
    last_print = 0
    t0 = time.time()

    while idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.resize(frame, _HIST_RESIZE)
            hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist  = cv2.calcHist(
                [hsv], [0, 1], None,
                [_BINS_HUE, _BINS_SAT], [0, 180, 0, 256]
            ).flatten()
            if prev_hist is not None:
                diffs.append(cv2.compareHist(hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA))
            prev_hist = hist
        idx += 1
        if verbose and (idx - last_print >= 1000 * step or idx == limit):
            print(f"    Pass: {idx}/{limit} frames  ({time.time()-t0:.1f}s)", flush=True)
            last_print = idx

    cap.release()
    return diffs, fps


# ---------------------------------------------------------------------------
# Stage 2 — pick threshold (pure function, no I/O)
# ---------------------------------------------------------------------------

def _apply_threshold(
    diffs: list[float],
    shot_threshold: float,
    adaptive: bool,
    percentile: int,
) -> float:
    """Return the cut threshold to use.

    Parameters
    ----------
    diffs : list[float]
        Histogram differences from Stage 1.
    shot_threshold : float
        Fixed threshold (used when adaptive=False or diffs is empty).
    adaptive : bool
        If True, compute the threshold from the distribution of `diffs`.
    percentile : int
        Percentile of diffs to use as the adaptive threshold.
    """
    if adaptive and diffs:
        thresh = float(np.percentile(diffs, percentile))
        return thresh
    return shot_threshold


# ---------------------------------------------------------------------------
# Stage 3 — find boundaries and convert to shots (pure function)
# ---------------------------------------------------------------------------

def _boundaries_to_shots(
    diffs: list[float],
    threshold: float,
    limit: int,
    fps: float,
    min_gap: int,
    step: int,
) -> list[tuple[int, int, float]]:
    """Convert a diff array to a list of (start_frame, end_frame, fps) shots.

    Parameters
    ----------
    diffs : list[float]
        Histogram diff per sampled frame pair.
    threshold : float
    limit : int
        Maximum frame index analysed.
    fps : float
    min_gap : int
        Minimum number of frames between cuts.
    step : int
        Sampling step used in Stage 1 (needed to convert diff-index → frame-index).

    Returns
    -------
    list of (start, end, fps) tuples.
    """
    boundaries = [0]
    for diff_idx, diff in enumerate(diffs):
        frame_idx = (diff_idx + 1) * step
        if diff > threshold and (frame_idx - boundaries[-1]) >= min_gap:
            boundaries.append(frame_idx)
    boundaries.append(limit)

    shots = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1] - 1
        if e - s >= min_gap:
            shots.append((s, e, fps))

    # Fallback: no cuts detected → split into fixed-length segments
    if not shots and limit > 0:
        seg = max(1, min_gap)
        for start in range(0, limit, seg):
            end = min(start + seg - 1, limit - 1)
            if end - start >= min_gap:
                shots.append((start, end, fps))

    # Absolute fallback: entire clip as one shot
    if not shots and limit > 0:
        shots = [(0, limit - 1, fps)]

    return shots


# ---------------------------------------------------------------------------
# Public API — thin orchestrator
# ---------------------------------------------------------------------------

def detect_shot_boundaries(
    video_path: str | Path,
    max_duration_s: float,
    shot_threshold: float,
    min_shot_duration_s: float,
    downsample_to: tuple,
    adaptive_threshold: bool = True,
    threshold_percentile: int = 85,
    step: int = 2,
) -> list[tuple[int, int, float]]:
    """Detect shot boundaries in a video.

    Parameters
    ----------
    video_path : str or Path
    max_duration_s : float
        Maximum duration to analyse (seconds).
    shot_threshold : float
        Fixed Bhattacharyya threshold (used when adaptive_threshold=False).
    min_shot_duration_s : float
        Minimum shot duration; shorter segments are merged.
    downsample_to : tuple
        (width, height) — unused (kept for backward compatibility; the
        implementation uses its own internal resolution _HIST_RESIZE).
    adaptive_threshold : bool
        If True, compute the threshold from the histogram diff distribution.
    threshold_percentile : int
        Which percentile of diffs to use as the adaptive threshold.
    step : int
        Process every ``step``-th frame (2 = half the frames, 2× faster).

    Returns
    -------
    list of (start_frame, end_frame, fps) tuples, one per detected shot.
    """
    t0 = time.time()
    video_path = Path(video_path)

    # Probe fps and total frames
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    limit = min(total, int(max_duration_s * fps))
    min_gap = max(1, int(min_shot_duration_s * fps))

    # Stage 1
    diffs, fps = _compute_histogram_diffs(video_path, limit, step)

    # Stage 2
    threshold = _apply_threshold(diffs, shot_threshold, adaptive_threshold, threshold_percentile)

    # Stage 3
    shots = _boundaries_to_shots(diffs, threshold, limit, fps, min_gap, step)

    elapsed = time.time() - t0
    print(f"  Shot detection: {len(shots)} shots in {elapsed:.1f}s"
          f"  (threshold={threshold:.3f}, step={step})")
    return shots


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_shot_frames_global(
    video_path: str | Path,
    shots: list[tuple[int, int, float]],
    n_frames_per_shot: int,
    frame_resize: tuple[int, int],
) -> dict[int, np.ndarray]:
    """Extract and resize frames needed by all shots in one video pass.

    Parameters
    ----------
    video_path : str or Path
    shots : list of (start, end, fps)
    n_frames_per_shot : int
    frame_resize : (width, height)

    Returns
    -------
    dict[frame_index → resized BGR frame]
    """
    if not shots:
        return {}

    # Compute the union of all required frame indices
    needed: set[int] = set()
    for s, e, _ in shots:
        n = min(n_frames_per_shot, e - s + 1)
        for idx in np.linspace(s, e, n, dtype=int):
            needed.add(int(idx))

    cap = cv2.VideoCapture(str(video_path))
    frames: dict[int, np.ndarray] = {}
    idx = 0
    max_needed = max(needed)

    while idx <= max_needed:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in needed:
            frames[idx] = cv2.resize(frame, frame_resize)
        idx += 1

    cap.release()
    return frames
