"""World-model surprise: frame-level surprise via flow variance.

A "surprise" is a frame whose predicted state (obtained by linearly
interpolating the optical flow forward from the previous frame) differs
strongly from the observed frame. AI-generated videos accumulate these
surprises around object pop-ins, morphing artifacts, and abrupt scene
changes within a shot (which by definition shouldn't happen).

Implementation: for each frame triplet (t-1, t, t+1), compute the flow
t-1 → t, warp t-1 forward by that flow to predict t (linear motion
prior), and measure the squared error between predicted-t and observed-t.
The per-shot summary is the mean and 95th percentile of those errors.

Pure NumPy + OpenCV. Disabled by default.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.world_model_surprise")
MOD_ID = "world_model_surprise"
VERSION = "0.4.8.1.4"


def _per_frame_surprise(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    import cv2
    if len(frames) < 3:
        return None
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]
    surprises: List[float] = []
    for i in range(1, len(grays) - 1):
        try:
            flow = cv2.calcOpticalFlowFarneback(
                grays[i - 1], grays[i], None, 0.5, 3, 15, 3, 5, 1.2, 0,
            )
            h, w = grays[i].shape
            # Warp grays[i-1] forward by flow -> predicted grays[i]
            map_y, map_x = np.indices((h, w), dtype=np.float32)
            map_x_warp = map_x + flow[..., 0]
            map_y_warp = map_y + flow[..., 1]
            predicted = cv2.remap(
                grays[i - 1], map_x_warp, map_y_warp,
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )
            err = np.abs(predicted.astype(np.float32) - grays[i].astype(np.float32))
            surprises.append(float(err.mean()) / 255.0)
        except Exception as e:  # noqa: BLE001
            logger.debug("surprise %d failed: %s", i, e)
    return np.array(surprises, dtype=np.float64) if surprises else None


def surprise_summary(frames: List[np.ndarray]) -> Dict[str, Any]:
    s = _per_frame_surprise(frames)
    if s is None or s.size == 0:
        return {
            "surprise_mean": None, "surprise_p95": None,
            "surprise_max": None, "n_frames": 0,
        }
    return {
        "surprise_mean": float(s.mean()),
        "surprise_p95": float(np.percentile(s, 95)),
        "surprise_max": float(s.max()),
        "n_frames": int(s.size),
    }


@register_module(MOD_ID, requires=[], description="Per-frame surprise via flow-warp prediction error.", version=VERSION)
def run(context: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    threshold = float(mod_cfg.get("surprise_threshold", 0.7))
    per_shot: Dict[int, Dict[str, Any]] = {
        sid: surprise_summary(frames) for sid, frames in context.shot_frames.items()
    }
    p95s = [v["surprise_p95"] for v in per_shot.values() if v["surprise_p95"] is not None]
    summary = {
        "n_shots": len(per_shot),
        "mean_p95": float(np.mean(p95s)) if p95s else None,
        "n_above_threshold": sum(1 for p in p95s if p > threshold),
    }
    return {"module": MOD_ID, "version": VERSION, "threshold": threshold,
            "per_shot": per_shot, "summary": summary}
