from __future__ import annotations
import logging
from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module
logger = logging.getLogger("cineinfini.modules.identity_consistency")
MOD_ID = "identity_consistency"
def _load_detector():
    try:
        from ..core.face_detection import CascadeFaceDetector
        return CascadeFaceDetector()
    except Exception as e:
        logger.warning("identity_consistency: detector unavailable (%s)", e)
        return None
def _load_embedder():
    try:
        from ..core.face_detection import ArcFaceEmbedder
        return ArcFaceEmbedder()
    except Exception as e:
        logger.warning("identity_consistency: embedder unavailable (%s)", e)
        return None
@register_module(MOD_ID, requires=["yunet","arcface"])
def run(context: VideoContext):
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    n_samples = int(mod_cfg.get("n_samples",5))
    use_dtw = bool(mod_cfg.get("use_dtw",True))
    threshold = float(mod_cfg.get("threshold", cfg.thresholds.get("identity_drift",0.6)))
    detector = context.pool.get_or_load("face_detector", _load_detector)
    embedder = context.pool.get_or_load("arcface", _load_embedder)
    per_shot = {}
    if detector is None or embedder is None:
        for sid in context.shot_frames:
            per_shot[sid] = {"identity_intra": None, "identity_intra_dtw": None, "identity_intra_dtw_n": 0, "available": False}
        return {"module": MOD_ID, "version": "0.4.7", "threshold": threshold, "per_shot": per_shot, "available": False}
    try:
        from ..core.face_detection import identity_within_shot
    except Exception as e:
        logger.warning("identity_consistency: identity_within_shot import failed (%s)", e)
        identity_within_shot = None
    for sid, frames in context.shot_frames.items():
        entry = {"identity_intra": None, "identity_intra_dtw": None, "identity_intra_dtw_n": 0, "available": True}
        if frames:
            try:
                if identity_within_shot is not None:
                    entry["identity_intra"] = identity_within_shot(frames, detector, embedder, n_samples=n_samples)
            except Exception as e:
                logger.debug("identity_within_shot failed on shot %s: %s", sid, e)
            if use_dtw:
                try:
                    from ..core.identity_dtw import identity_within_shot_dtw
                    res = identity_within_shot_dtw(frames, detector, embedder, max_samples=int(cfg.processing.get("dtw_max_samples",8)))
                    entry["identity_intra_dtw"] = res.normalized
                    entry["identity_intra_dtw_n"] = res.n_a + res.n_b
                except Exception as e:
                    logger.debug("identity_within_shot_dtw failed on shot %s: %s", sid, e)
        per_shot[sid] = entry
    return {"module": MOD_ID, "version": "0.4.7", "threshold": threshold, "per_shot": per_shot, "available": True}
