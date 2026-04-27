from __future__ import annotations
import logging, numpy as np
from ..core.config import get_config
from ..core.context import VideoContext
from ..core.device_utils import autocast_context, inference_mode
from ..core.registry import register_module
logger = logging.getLogger("cineinfini.modules.semantic_consistency")
MOD_ID = "semantic_consistency"
def _load_clip(device: str):
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device); model.eval()
        return {"model": model, "preprocess": preprocess, "device": device}
    except Exception as e: logger.warning("semantic_consistency: CLIP unavailable (%s)", e); return None
def _encode_frames_batched(frames, bundle):
    import cv2, torch
    from PIL import Image
    model, preprocess, device = bundle["model"], bundle["preprocess"], bundle["device"]
    pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    tensor = torch.stack([preprocess(p) for p in pil_imgs], dim=0).to(device, non_blocking=True)
    with inference_mode(), autocast_context():
        feats = model.encode_image(tensor); feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return feats.float().cpu().numpy()
def _temporal_consistency(features): return float(np.mean((features[:-1] * features[1:]).sum(axis=1))) if features.shape[0]>=2 else float("nan")
@register_module(MOD_ID, requires=["clip_vit_b32"], description="CLIP-based temporal consistency within shot (batched, AMP-aware).", version="0.4.8")
def run(context: VideoContext):
    cfg = get_config(); mod_cfg = cfg.get_module_config(MOD_ID); threshold = float(mod_cfg.get("threshold", cfg.thresholds.get("clip_temp",0.25)))
    bundle = context.pool.get_or_load("clip_vit_b32", lambda: _load_clip(context.device))
    per_shot = {}
    if bundle is None:
        for sid in context.shot_frames: per_shot[sid] = {"clip_temp_consistency": None, "available": False}
        return {"module": MOD_ID, "version": "0.4.8", "threshold": threshold, "per_shot": per_shot, "available": False}
    for sid, frames in context.shot_frames.items():
        entry = {"clip_temp_consistency": None, "n_frames": len(frames), "available": True}
        if frames:
            try:
                feats = _encode_frames_batched(frames, bundle)
                entry["clip_temp_consistency"] = _temporal_consistency(feats)
            except Exception:
                try:
                    from ..core.embedding import clip_semantic_consistency
                    entry["clip_temp_consistency"] = clip_semantic_consistency(frames, bundle["model"], bundle["preprocess"], bundle["device"])
                except: pass
        per_shot[sid] = entry
    return {"module": MOD_ID, "version": "0.4.8", "threshold": threshold, "per_shot": per_shot, "available": True}
