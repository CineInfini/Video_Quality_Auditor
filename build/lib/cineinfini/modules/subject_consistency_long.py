"""Long-term subject consistency.

Wraps the existing ``identity_dtw`` machinery for the long-term audit.
Computes pairwise DTW distance between identity-embedding sequences of
non-adjacent shots; high distance over a long window = subject identity
has drifted across the video.

Returns a video-level coherence score in [0, 1] where higher is better.

Reuses existing ArcFace embedder + DTW routines, so no new ML deps.
Disabled by default.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.subject_consistency_long")
MOD_ID = "subject_consistency_long"
VERSION = "0.4.8.1.4"


def _shot_identity_sequence(
    frames: List[np.ndarray], detector, embedder, max_samples: int = 8,
) -> Optional[np.ndarray]:
    if not frames:
        return None
    idxs = np.linspace(0, len(frames) - 1, min(max_samples, len(frames)), dtype=int)
    embs: List[np.ndarray] = []
    for i in idxs:
        f = frames[int(i)]
        try:
            boxes = detector.detect(f)
        except Exception:
            continue
        if not boxes:
            continue
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        crop = f[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        emb = embedder.embed(crop)
        if emb is not None:
            embs.append(emb)
    if len(embs) < 2:
        return None
    return np.stack(embs, axis=0)


def _dtw_pair(seq_a: np.ndarray, seq_b: np.ndarray) -> Optional[float]:
    try:
        from ..core.identity_dtw import dtw_distance
        return float(dtw_distance(seq_a, seq_b))
    except Exception as e:  # noqa: BLE001
        logger.debug("identity_dtw unavailable: %s", e)
        # Fallback: cosine distance between sequence means
        if seq_a.size == 0 or seq_b.size == 0:
            return None
        a, b = seq_a.mean(axis=0), seq_b.mean(axis=0)
        return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@register_module(MOD_ID, requires=["arcface"],
                 description="Long-term subject identity via DTW across non-adjacent shots.",
                 version=VERSION)
def run(context: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    window_size = int(mod_cfg.get("window_size", 30))
    use_dtw = bool(mod_cfg.get("dtw_enabled", True))

    # Build per-shot identity sequences (lazy import to avoid torch on bootstrap)
    try:
        from ..core.face_detection import get_face_detector, get_face_embedder
        detector = get_face_detector()
        embedder = get_face_embedder()
    except Exception as e:  # noqa: BLE001
        return {"module": MOD_ID, "version": VERSION,
                "available": False, "error": f"face stack unavailable: {e}"}

    sequences: Dict[int, np.ndarray] = {}
    for sid, frames in context.shot_frames.items():
        seq = _shot_identity_sequence(frames, detector, embedder)
        if seq is not None:
            sequences[sid] = seq

    shot_ids = sorted(sequences.keys())
    if len(shot_ids) < 2:
        return {"module": MOD_ID, "version": VERSION,
                "available": True, "n_shots_with_identity": len(shot_ids),
                "long_term_coherence": None, "pair_distances": []}

    pairs: List[Dict[str, Any]] = []
    for i, a in enumerate(shot_ids):
        for b in shot_ids[i + 1:]:
            if abs(a - b) < 2:
                continue  # skip adjacent (covered by inter_shot_loss)
            d = _dtw_pair(sequences[a], sequences[b]) if use_dtw else None
            if d is not None and abs(a - b) <= window_size:
                pairs.append({"a": a, "b": b, "dtw_distance": d})
    if not pairs:
        return {"module": MOD_ID, "version": VERSION, "available": True,
                "long_term_coherence": None, "pair_distances": []}
    distances = [p["dtw_distance"] for p in pairs]
    coherence = float(max(0.0, 1.0 - np.mean(distances)))
    return {
        "module": MOD_ID, "version": VERSION, "available": True,
        "long_term_coherence": coherence,
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
        "n_pairs": len(pairs),
        "pair_distances": pairs[:50],
    }
