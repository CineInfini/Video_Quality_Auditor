"""Long-term narrative coherence.

Splits the video into N equal segments, encodes each with DINOv2 (mean
embedding), and measures the pairwise cosine distance between segment
embeddings. A coherent narrative has gradually-evolving embeddings
(adjacent segments similar, far segments diverge slowly); a fragmented
one shows sharp jumps.

Falls back to a colour-histogram embedding if DINOv2 is not loadable.
Disabled by default.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.long_term_narrative")
MOD_ID = "long_term_narrative"
VERSION = "0.4.8.1.4"


def _color_histogram(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()
    out = np.concatenate([h, s, v])
    if out.sum() > 0:
        out = out / out.sum()
    return out.astype(np.float32)


def _segment_embedding_dino(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    try:
        from ..core.embedding import _DinoV2State, load_dinov2
        if not _DinoV2State.is_loaded():
            load_dinov2(device="cpu")
        if not _DinoV2State.is_loaded():
            return None
        import torch
        from PIL import Image
        import cv2
        proc = _DinoV2State.processor
        model = _DinoV2State.model
        embs = []
        for f in frames:
            pil = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            inputs = proc(images=pil, return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embs.append(emb)
        return np.mean(np.stack(embs), axis=0) if embs else None
    except Exception as e:  # noqa: BLE001
        logger.info("DINOv2 unavailable, falling back to colour histogram: %s", e)
        return None


def _segment_embedding(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    if not frames:
        return None
    dino = _segment_embedding_dino(frames)
    if dino is not None:
        n = float(np.linalg.norm(dino))
        return dino / n if n > 1e-9 else None
    # Fallback: mean colour histogram
    hists = [_color_histogram(f) for f in frames]
    mean = np.mean(np.stack(hists), axis=0)
    n = float(np.linalg.norm(mean))
    return mean / n if n > 1e-9 else None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@register_module(MOD_ID, requires=["dinov2_vitb14"],
                 description="Long-range narrative cohesion via segment embeddings.",
                 version=VERSION)
def run(context: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    seg_size = int(mod_cfg.get("segment_size", 16))

    # Concatenate all shot frames in order
    shot_ids = sorted(context.shot_frames.keys())
    all_frames: List = []
    for sid in shot_ids:
        all_frames.extend(context.shot_frames[sid])
    if len(all_frames) < 2 * seg_size:
        return {"module": MOD_ID, "version": VERSION, "available": True,
                "long_term_narrative": None, "n_segments": 0}

    segments: List[List] = []
    for i in range(0, len(all_frames) - seg_size + 1, seg_size):
        segments.append(all_frames[i:i + seg_size])
    embs: List[np.ndarray] = []
    for seg in segments:
        e = _segment_embedding(seg)
        if e is not None:
            embs.append(e)
    if len(embs) < 2:
        return {"module": MOD_ID, "version": VERSION, "available": True,
                "long_term_narrative": None, "n_segments": len(embs)}

    # Adjacent similarity (should be high) and far-apart similarity (allowed to drop)
    adj_sims = [_cosine(embs[i], embs[i + 1]) for i in range(len(embs) - 1)]
    far_sims = [_cosine(embs[0], embs[-1])]
    return {
        "module": MOD_ID, "version": VERSION, "available": True,
        "long_term_narrative": float(np.mean(adj_sims)),
        "adjacent_similarity_mean": float(np.mean(adj_sims)),
        "first_to_last_similarity": float(np.mean(far_sims)),
        "n_segments": len(embs),
        "segment_size": seg_size,
    }
