from __future__ import annotations
import logging
from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module
logger = logging.getLogger("cineinfini.modules.semantic_consistency")
MOD_ID = "semantic_consistency"
def _load_clip(device: str):
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        return {"model": model, "preprocess": preprocess, "device": device}
    except Exception as e:
        logger.warning("semantic_consistency: CLIP unavailable (%s)", e)
        return None
@register_module(MOD_ID, requires=["clip_vit_b32"])
def run(context: VideoContext):
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    threshold = float(mod_cfg.get("threshold", cfg.thresholds.get("clip_temp",0.25)))
    bundle = context.pool.get_or_load("clip_vit_b32", lambda: _load_clip(context.device))
    per_shot = {}
    if bundle is None:
        for sid in context.shot_frames:
            per_shot[sid] = {"clip_temp_consistency": None, "available": False}
        return {"module": MOD_ID, "version": "0.4.7", "threshold": threshold, "per_shot": per_shot, "available": False}
    try:
        from ..core.embedding import clip_semantic_consistency
    except Exception as e:
        logger.warning("semantic_consistency: import failed (%s)", e)
        return {"module": MOD_ID, "version": "0.4.7", "threshold": threshold, "per_shot": {}, "available": False}
    for sid, frames in context.shot_frames.items():
        entry = {"clip_temp_consistency": None, "available": True}
        if frames:
            try:
                entry["clip_temp_consistency"] = clip_semantic_consistency(frames, bundle["model"], bundle["preprocess"], bundle["device"])
            except Exception as e:
                logger.debug("clip_semantic_consistency failed on shot %s: %s", sid, e)
        per_shot[sid] = entry
    return {"module": MOD_ID, "version": "0.4.7", "threshold": threshold, "per_shot": per_shot, "available": True}
