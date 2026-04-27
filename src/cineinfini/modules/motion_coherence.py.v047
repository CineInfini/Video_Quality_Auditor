from __future__ import annotations
import logging
from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module
logger = logging.getLogger("cineinfini.modules.motion_coherence")
MOD_ID = "motion_coherence"
@register_module(MOD_ID, requires=[])
def run(context: VideoContext):
    from ..core.metrics import motion_peak_div, ssim3d_self, flicker_score, ssim_long_range, flicker_highfreq_variance
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    th = float(mod_cfg.get("threshold", cfg.thresholds.get("motion", 25.0)))
    per_shot = {}
    for sid, frames in context.shot_frames.items():
        if not frames: continue
        try:
            per_shot[sid] = {"motion_peak_div": motion_peak_div(frames), "ssim3d_self": ssim3d_self(frames), "flicker": flicker_score(frames), "ssim_long_range": ssim_long_range(frames), "flicker_hf_var": flicker_highfreq_variance(frames)}
        except Exception as e:
            logger.warning("motion_coherence failed on shot %s: %s", sid, e)
            per_shot[sid] = {k: None for k in ["motion_peak_div","ssim3d_self","flicker","ssim_long_range","flicker_hf_var"]}
    return {"module": MOD_ID, "version": "0.4.7", "threshold": th, "per_shot": per_shot}
