"""Fine-grained prompt alignment.

Audits how well each shot matches a user-supplied prompt or per-shot
description. When a VLM (BLIP-2) is available, computes
caption-prompt similarity. Falls back to CLIP zero-shot when only CLIP
is loaded.

Reads the description from ``context.cache['prompts']`` (a dict mapping
shot_id → str). When that's absent, returns ``available: false`` —
without prompts the module has nothing to align *against*.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.prompt_alignment_fine")
MOD_ID = "prompt_alignment_fine"
VERSION = "0.4.8.1.4"


def _clip_alignment(scorer, frames: List[np.ndarray], prompt: str,
                    n_samples: int = 6) -> Optional[float]:
    if not getattr(scorer, "available", False) or not frames or not prompt:
        return None
    out = scorer.score(frames, prompt, n_samples=n_samples)
    return float(out.get("mean", 0.0))


@register_module(MOD_ID, requires=["clip_vit_b32"],
                 description="Per-shot prompt alignment (CLIP zero-shot).",
                 version=VERSION)
def run(context: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    prompts = context.cache.get("prompts") or {}
    if not prompts:
        return {"module": MOD_ID, "version": VERSION, "available": False,
                "error": "no per-shot prompts in context.cache['prompts']; "
                         "use cineinfini.build_all_prompts()"}

    try:
        from ..core.embedding import CLIPSemanticScorer
        scorer = CLIPSemanticScorer(device=cfg.effective_device())
    except Exception as e:  # noqa: BLE001
        return {"module": MOD_ID, "version": VERSION, "available": False,
                "error": f"CLIP unavailable: {e}"}
    if not scorer.available:
        return {"module": MOD_ID, "version": VERSION, "available": False,
                "error": "CLIP weights missing; run `cineinfini bootstrap`"}

    per_shot: Dict[int, Dict[str, Any]] = {}
    for sid, frames in context.shot_frames.items():
        prompt = prompts.get(sid) or prompts.get(str(sid)) or ""
        score = _clip_alignment(scorer, frames, prompt) if prompt else None
        per_shot[sid] = {"prompt": prompt, "alignment_score": score}
    return {"module": MOD_ID, "version": VERSION, "available": True,
            "per_shot": per_shot}
