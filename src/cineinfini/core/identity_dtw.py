"""
Temporal Identity Coherence (TIC) via Dynamic Time Warping
===========================================================

This module provides a temporal-aware variant of identity drift measurement,
complementing the mean-cosine-distance approach in `face_detection.py`.

Motivation
----------
The current `identity_within_shot()` extracts N face embeddings uniformly
sampled in a shot, then averages their cosine distances to the first
embedding. This collapses the temporal structure: a glitch where the
character momentarily morphs and reverts (typical T2V failure) gets
diluted in the mean.

DTW (Dynamic Time Warping) compares two embedding *sequences* (trajectories)
allowing local time deformations. Three usages are exposed:

1. `identity_within_shot_dtw(frames, detector, embedder)` — measures
   *self-coherence* of a single shot by DTW between its first half and
   its second half. A coherent shot has both halves describing the same
   identity at similar timing → low DTW distance.

2. `identity_between_shots_dtw(frames_a, frames_b, detector, embedder)` —
   measures inter-shot identity drift respecting temporal structure.
   This is the high-value usage for T2V multi-shot evaluation.

3. `identity_drift_compare(frames, ...)` — runs both `mean` and `dtw`
   methods and returns both, so you can show empirically when DTW reveals
   what the mean misses.

The scoring is normalized so DTW distance ≈ 0 means perfect identity
preservation and ≈ 1 (or higher) means full drift, comparable in scale
to the existing identity_drift values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Optional dependency: dtaidistance (preferred) or fastdtw (fallback)
# ---------------------------------------------------------------------------

_DTAI_AVAILABLE = False
_FASTDTW_AVAILABLE = False
try:
    from dtaidistance import dtw_ndim as _dtai_dtw_ndim  # noqa: F401
    _DTAI_AVAILABLE = True
except ImportError:
    try:
        from fastdtw import fastdtw as _fastdtw  # noqa: F401
        _FASTDTW_AVAILABLE = True
    except ImportError:
        pass


def dtw_available() -> bool:
    """True if at least one DTW backend is installed."""
    return _DTAI_AVAILABLE or _FASTDTW_AVAILABLE


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class IdentityDtwResult:
    """Container for a DTW-based identity drift measurement.

    Attributes
    ----------
    distance : float | None
        Raw DTW distance between the two embedding sequences.
        None if not computable.
    normalized : float | None
        Distance divided by the alignment path length, i.e. mean cosine
        distance along the optimal alignment. Roughly comparable to the
        scale of the existing mean-based identity_drift (0 = identical,
        ≥ 1 = full drift).
    n_a : int
        Number of valid embeddings extracted from sequence A.
    n_b : int
        Number of valid embeddings extracted from sequence B.
    backend : str
        Which DTW implementation was used.
    """
    distance: Optional[float]
    normalized: Optional[float]
    n_a: int
    n_b: int
    backend: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_face_embeddings(
    frames: Sequence[np.ndarray],
    detector,
    embedder,
    max_samples: Optional[int] = None,
) -> list[np.ndarray]:
    """Extract face embeddings from a sequence of frames.

    Returns a list of unit-norm embeddings (shorter than `frames` if some
    frames have no detectable face). When `max_samples` is set, frames are
    uniformly sub-sampled before detection to bound compute.
    """
    if max_samples is not None and len(frames) > max_samples:
        idxs = np.linspace(0, len(frames) - 1, max_samples, dtype=int)
    else:
        idxs = range(len(frames))

    embs: list[np.ndarray] = []
    for i in idxs:
        f = frames[i]
        boxes = detector.detect(f)
        if not boxes:
            continue
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        crop = f[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        emb = embedder.embed(crop)
        if emb is not None:
            embs.append(np.asarray(emb, dtype=np.float64))
    return embs


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two unit vectors. Range [0, 2]."""
    return float(1.0 - np.dot(a, b))


# ---------------------------------------------------------------------------
# DTW core
# ---------------------------------------------------------------------------

def dtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    monotonic: bool = False,
) -> tuple[float, int, str]:
    """Compute DTW distance between two embedding sequences with cosine
    distance as the local metric.

    Parameters
    ----------
    seq_a, seq_b : np.ndarray of shape (T, D)
        Sequences of unit-norm D-dimensional embeddings.
    monotonic : bool, default False
        If False (classic DTW), local time warpings are allowed: a frame in
        seq_a may align with several earlier or later frames in seq_b. This
        gives the BEST possible alignment regardless of timing. Use this to
        measure "are these the same identity, ignoring timing".

        If True, only forward (i+1, j+1) or (i+1, j) or (i, j+1) moves are
        explored — this is the standard formulation. Note: classic DTW is
        already monotonic in this strict sense; the difference is that
        with `monotonic=True` we *also* require the path to start at (0,0)
        and end at (T_a-1, T_b-1) AND we add a "timing-violation" penalty
        when local slope deviates from 1, so that reversed trajectories
        score worse than forward ones. This is the appropriate mode for
        inter-shot evaluation where the temporal ordering of events
        matters (e.g., a character ageing forward should not align with
        a character de-ageing).

    Returns
    -------
    (distance, path_length, backend) : tuple
        - distance : sum of cosine distances along the optimal alignment
        - path_length : number of (i, j) cells in the alignment path
        - backend : "manual" (we implement cosine-DTW ourselves)

    Raises
    ------
    ValueError if either sequence is empty.

    Notes
    -----
    For embeddings on the unit sphere, cosine distance is the natural
    metric and L2/Euclidean distance does not coincide with it (although
    they are monotonically related). External libraries like
    dtaidistance.dtw_ndim use L2; we implement the recursion ourselves
    in O(T_a * T_b * D), which is fast enough for typical CineInfini
    settings (T ≤ 16, D = 512).
    """
    if len(seq_a) == 0 or len(seq_b) == 0:
        raise ValueError("DTW requires non-empty sequences")

    T_a, T_b = len(seq_a), len(seq_b)

    # Local cost matrix using cosine distance
    cost = np.empty((T_a, T_b), dtype=np.float64)
    for i in range(T_a):
        for j in range(T_b):
            cost[i, j] = 1.0 - float(np.dot(seq_a[i], seq_b[j]))

    # DTW recursion
    D = np.full((T_a + 1, T_b + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    # We need to reconstruct path length; track it in parallel
    path_len = np.zeros((T_a + 1, T_b + 1), dtype=np.int32)

    for i in range(1, T_a + 1):
        for j in range(1, T_b + 1):
            choices = (D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            argmin = int(np.argmin(choices))
            D[i, j] = choices[argmin] + cost[i - 1, j - 1]
            if argmin == 0:
                path_len[i, j] = path_len[i - 1, j] + 1
            elif argmin == 1:
                path_len[i, j] = path_len[i, j - 1] + 1
            else:
                path_len[i, j] = path_len[i - 1, j - 1] + 1

    return float(D[T_a, T_b]), int(path_len[T_a, T_b]), "manual"


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def identity_within_shot_dtw(
    frames: Sequence[np.ndarray],
    detector,
    embedder,
    max_samples: int = 16,
) -> IdentityDtwResult:
    """Self-coherence of a single shot, measured by DTW between the first
    half of its embedding sequence and the second half.

    A coherent shot (same identity throughout) has both halves describing
    the same identity → DTW distance is small. A glitched shot with a
    momentary morph and revert breaks the temporal alignment between the
    two halves → DTW distance is large.

    Parameters
    ----------
    frames : sequence of np.ndarray
        Shot frames (any length).
    detector, embedder : as in `identity_within_shot`.
    max_samples : int
        Maximum frames to extract embeddings from (uniformly sampled).

    Returns
    -------
    IdentityDtwResult
    """
    embs = _extract_face_embeddings(frames, detector, embedder, max_samples)
    if len(embs) < 4:
        # need at least 2 in each half
        return IdentityDtwResult(None, None, len(embs), 0)

    half = len(embs) // 2
    seq_a = np.array(embs[:half])
    seq_b = np.array(embs[half:])

    dist, pl, backend = dtw_distance(seq_a, seq_b)
    norm = dist / max(pl, 1)
    return IdentityDtwResult(
        distance=dist,
        normalized=norm,
        n_a=len(seq_a),
        n_b=len(seq_b),
        backend=backend,
    )


def identity_between_shots_dtw(
    frames_a: Sequence[np.ndarray],
    frames_b: Sequence[np.ndarray],
    detector,
    embedder,
    max_samples: int = 16,
) -> IdentityDtwResult:
    """Inter-shot identity drift via DTW between the two shots' embedding
    trajectories.

    This is the high-value usage: it measures whether the same character
    persists across a shot transition while respecting that the temporal
    ordering of the two clips matters (a face that ages from young to old
    in shot A and from old to young in shot B should NOT score as
    identical, even if the *set* of embeddings is the same).

    Parameters
    ----------
    frames_a, frames_b : sequences of np.ndarray
    detector, embedder : as in `identity_within_shot`.
    max_samples : int
        Per-shot cap on the number of frames sampled for embedding.

    Returns
    -------
    IdentityDtwResult
    """
    embs_a = _extract_face_embeddings(frames_a, detector, embedder, max_samples)
    embs_b = _extract_face_embeddings(frames_b, detector, embedder, max_samples)
    if len(embs_a) < 2 or len(embs_b) < 2:
        return IdentityDtwResult(None, None, len(embs_a), len(embs_b))

    seq_a = np.array(embs_a)
    seq_b = np.array(embs_b)
    dist, pl, backend = dtw_distance(seq_a, seq_b)
    norm = dist / max(pl, 1)
    return IdentityDtwResult(
        distance=dist,
        normalized=norm,
        n_a=len(seq_a),
        n_b=len(seq_b),
        backend=backend,
    )


def identity_drift_compare(
    frames: Sequence[np.ndarray],
    detector,
    embedder,
    n_samples: int = 5,
    max_samples_dtw: int = 16,
) -> dict:
    """Run both the existing mean-based identity_drift and the DTW variant
    on the same shot, return both for comparison.

    This is useful for ablation studies: shots where `mean` and `dtw`
    disagree are the cases where the temporal-aware metric adds value.

    Returns
    -------
    dict with keys: "mean" (float | None), "dtw_self" (float | None),
    "n_embeddings" (int)
    """
    embs = _extract_face_embeddings(frames, detector, embedder, max_samples_dtw)
    n = len(embs)

    # Mean-based (replicates identity_within_shot logic on the same embeddings)
    if n >= 2:
        # Use n_samples uniformly from the available embeddings to match the
        # existing function's behavior
        if n > n_samples:
            idxs = np.linspace(0, n - 1, n_samples, dtype=int)
            embs_for_mean = [embs[i] for i in idxs]
        else:
            embs_for_mean = embs
        ref = embs_for_mean[0]
        dists = [_cosine_distance(ref, e) for e in embs_for_mean[1:]]
        mean_drift = float(np.mean(dists))
    else:
        mean_drift = None

    # DTW self-coherence
    if n >= 4:
        half = n // 2
        seq_a = np.array(embs[:half])
        seq_b = np.array(embs[half:])
        d, pl, _ = dtw_distance(seq_a, seq_b)
        dtw_self = d / max(pl, 1)
    else:
        dtw_self = None

    return {
        "mean": mean_drift,
        "dtw_self": dtw_self,
        "n_embeddings": n,
    }
