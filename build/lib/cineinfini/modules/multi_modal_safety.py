"""Multi-modal safety: NSFW / violence detection.

Wraps a CLIP-based zero-shot classifier (template prompts: "a safe-for-
work scene" vs "an NSFW scene", "a peaceful scene" vs "a violent
scene"). Returns per-shot probabilities and a binary flag. Requires
the CLIP weights to be present (downloaded by ``cineinfini bootstrap``);
otherwise reports ``available: false``.

Disabled by default. Enable explicitly only after reviewing the
classifier's documented false-positive characteristics — zero-shot
safety classification is *not* a substitute for proper moderation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.multi_modal_safety")
MOD_ID = "multi_modal_safety"
VERSION = "0.4.8.1.4"

_PROMPTS_NSFW = [
    "a safe-for-work scene",
    "an explicit nsfw scene",
]
_PROMPTS_VIOLENCE = [
    "a peaceful scene",
    "a violent scene with weapons or fighting",
]


def _clip_zero_shot_score(scorer, frames: List[np.ndarray],
                          positive_prompt: str, negative_prompt: str,
                          n_samples: int = 4) -> Optional[float]:
    if not getattr(scorer, "available", False) or not frames:
        return None
    pos = scorer.score(frames, positive_prompt, n_samples=n_samples)
    neg = scorer.score(frames, negative_prompt, n_samples=n_samples)
    p = float(pos.get("mean", 0.0))
    n = float(neg.get("mean", 0.0))
    # softmax over the two
    e_p = np.exp(p)
    e_n = np.exp(n)
    return float(e_p / (e_p + e_n))


@register_module(MOD_ID, requires=["clip_vit_b32"],
                 description="Zero-shot NSFW / violence classification via CLIP prompts.",
                 version=VERSION)
def run(context: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    nsfw_thr = float(mod_cfg.get("nsfw_threshold", 0.8))
    violence_thr = float(mod_cfg.get("violence_threshold", 0.7))

    try:
        from ..core.embedding import CLIPSemanticScorer
        scorer = CLIPSemanticScorer(device=cfg.effective_device())
    except Exception as e:  # noqa: BLE001
        return {"module": MOD_ID, "version": VERSION, "available": False,
                "error": f"CLIP unavailable: {e} — run `cineinfini bootstrap`"}
    if not scorer.available:
        return {"module": MOD_ID, "version": VERSION, "available": False,
                "error": "CLIP weights not loaded; run `cineinfini bootstrap`"}

    per_shot: Dict[int, Dict[str, Any]] = {}
    for sid, frames in context.shot_frames.items():
        nsfw_p = _clip_zero_shot_score(scorer, frames, _PROMPTS_NSFW[1], _PROMPTS_NSFW[0])
        viol_p = _clip_zero_shot_score(scorer, frames, _PROMPTS_VIOLENCE[1], _PROMPTS_VIOLENCE[0])
        per_shot[sid] = {
            "nsfw_probability": nsfw_p,
            "violence_probability": viol_p,
            "flag_nsfw": (nsfw_p is not None and nsfw_p >= nsfw_thr),
            "flag_violence": (viol_p is not None and viol_p >= violence_thr),
        }
    return {"module": MOD_ID, "version": VERSION, "available": True,
            "thresholds": {"nsfw": nsfw_thr, "violence": violence_thr},
            "per_shot": per_shot}
