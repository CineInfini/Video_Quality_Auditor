"""Origin detection (real footage vs AI-generated).

Linear classifier over DINOv2 features, scoring how AI-like each shot
looks. Requires a trained ``origin_classifier.npz`` (weight + bias) in
``cfg.models_dir()``; until that file is present the module reports
``available: false`` so the audit pipeline keeps running.

The training data is not bundled (it would mix many AI generators'
outputs which is logistically painful); ship a small classifier or
write your own with::

    from cineinfini.core.embedding import load_dinov2, get_dinov2
    # collect DINOv2 features over labelled frames, fit sklearn LogReg,
    # save weights as np.savez(models_dir / 'origin_classifier.npz',
    #   coef=clf.coef_, intercept=clf.intercept_)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.origin_detection")
MOD_ID = "origin_detection"
VERSION = "0.4.8.1.4"


def _classifier_path() -> Path:
    return get_config().models_dir() / "origin_classifier.npz"


def _classify_features(feats: np.ndarray) -> Optional[float]:
    p = _classifier_path()
    if not p.exists():
        return None
    try:
        data = np.load(str(p))
        coef = data["coef"].ravel()
        bias = float(data["intercept"]) if "intercept" in data else 0.0
        if feats.ndim == 1:
            feats = feats[None, :]
        if feats.shape[1] != coef.shape[0]:
            return None
        logits = feats @ coef + bias
        return float(1.0 / (1.0 + np.exp(-logits.mean())))
    except Exception as e:  # noqa: BLE001
        logger.debug("origin_classifier failed: %s", e)
        return None


@register_module(MOD_ID, requires=["dinov2_vitb14"],
                 description="Linear classifier (DINOv2 → AI-likelihood).",
                 version=VERSION)
def run(context: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    if not _classifier_path().exists():
        return {"module": MOD_ID, "version": VERSION, "available": False,
                "error": f"trained classifier not found at {_classifier_path()} — "
                         "see module docstring to train one"}

    try:
        from ..core.embedding import _segment_embedding_dino  # type: ignore
    except ImportError:
        try:
            from .long_term_narrative import _segment_embedding_dino
        except Exception:
            return {"module": MOD_ID, "version": VERSION, "available": False,
                    "error": "DINOv2 backbone unavailable"}

    threshold = float(cfg.get_module_config(MOD_ID).get("confidence_threshold", 0.65))
    per_shot: Dict[int, Dict[str, Any]] = {}
    for sid, frames in context.shot_frames.items():
        feats = _segment_embedding_dino(frames[:8])
        if feats is None:
            per_shot[sid] = {"ai_likelihood": None, "verdict": "unknown"}
            continue
        score = _classify_features(np.asarray(feats))
        if score is None:
            per_shot[sid] = {"ai_likelihood": None, "verdict": "unknown"}
            continue
        per_shot[sid] = {
            "ai_likelihood": score,
            "verdict": "ai_generated" if score >= threshold else "real",
        }
    return {"module": MOD_ID, "version": VERSION, "available": True,
            "threshold": threshold, "per_shot": per_shot}
