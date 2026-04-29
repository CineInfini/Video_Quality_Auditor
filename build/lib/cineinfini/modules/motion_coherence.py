from __future__ import annotations
import logging, numpy as np
from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module
logger = logging.getLogger("cineinfini.modules.motion_coherence")
MOD_ID = "motion_coherence"
def _to_gray_stack(frames):
    import cv2
    grays = []
    for f in frames:
        if f.ndim == 3: grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        else: grays.append(f)
    return np.stack(grays, axis=0)
def _flicker_hf_var_vectorised(stack):
    if stack.shape[0] < 3: return float("nan")
    luma = stack.mean(axis=(1,2)); diff = np.diff(luma, n=2); return float(np.var(diff) / 255.0**2)
def _safe(metric_fn, frames):
    try: return metric_fn(frames)
    except Exception as e: logger.debug("metric %s failed: %s", getattr(metric_fn,"__name__","?"), e); return None
@register_module(MOD_ID, requires=[], description="Motion peak/divergence, flicker, SSIM3D (vectorised, no model).", version="0.4.8")
def run(context: VideoContext):
    from ..core.metrics import motion_peak_div, ssim3d_self, flicker_score, ssim_long_range, flicker_highfreq_variance
    cfg = get_config(); mod_cfg = cfg.get_module_config(MOD_ID); threshold = float(mod_cfg.get("threshold", cfg.thresholds.get("motion",25.0)))
    use_fast_hf = bool(mod_cfg.get("vectorised_hf", True))
    per_shot = {}
    for sid, frames in context.shot_frames.items():
        if not frames: continue
        hf_var = None
        if use_fast_hf:
            try: gray_stack = _to_gray_stack(frames); hf_var = _flicker_hf_var_vectorised(gray_stack)
            except: hf_var = _safe(flicker_highfreq_variance, frames)
        else: hf_var = _safe(flicker_highfreq_variance, frames)
        per_shot[sid] = {
            "motion_peak_div": _safe(motion_peak_div, frames), "ssim3d_self": _safe(ssim3d_self, frames),
            "flicker": _safe(flicker_score, frames), "ssim_long_range": _safe(ssim_long_range, frames),
            "flicker_hf_var": hf_var,
        }
    return {"module": MOD_ID, "version": "0.4.8", "threshold": threshold, "per_shot": per_shot}
