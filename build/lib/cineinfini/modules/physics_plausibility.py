"""Physics plausibility: centroid trajectory smoothness.

Tracks the centroid of the moving region (via flow magnitude > threshold)
across a shot and measures whether the trajectory is *smooth*. Real-world
motion has bounded jerk (third derivative of position); AI-generated
videos often produce centroid trajectories with sudden teleportation,
manifesting as high acceleration variance.

Returns three sub-scores per shot, all in [0, 1]:

* ``trajectory_smoothness``: 1 - (normalised acceleration std-dev). Higher = smoother.
* ``object_persistence``: fraction of frames where motion is detected.
  Low values can mean teleportation / pop-ins / pop-outs.
* ``jerk_score``: peak third-derivative magnitude, normalised. High = bad.

Pure NumPy + OpenCV. Disabled by default.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.physics_plausibility")
MOD_ID = "physics_plausibility"
VERSION = "0.4.8.1.4"


def _track_centroids(
    frames: List[np.ndarray], motion_threshold: float = 0.5
) -> List[Optional[Tuple[float, float]]]:
    import cv2
    if len(frames) < 2:
        return []
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]
    out: List[Optional[Tuple[float, float]]] = []
    for i in range(len(grays) - 1):
        try:
            flow = cv2.calcOpticalFlowFarneback(
                grays[i], grays[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mask = mag > motion_threshold
            if mask.sum() < 50:
                out.append(None)
                continue
            ys, xs = np.indices(mask.shape)
            cx = float(xs[mask].mean())
            cy = float(ys[mask].mean())
            out.append((cx, cy))
        except Exception as e:  # noqa: BLE001
            logger.debug("flow %d failed: %s", i, e)
            out.append(None)
    return out


def trajectory_smoothness(
    centroids: List[Optional[Tuple[float, float]]]
) -> Optional[float]:
    valid = [c for c in centroids if c is not None]
    if len(valid) < 4:
        return None
    arr = np.array(valid, dtype=np.float64)
    accel = np.diff(arr, n=2, axis=0)
    if accel.shape[0] == 0:
        return None
    std = float(np.linalg.norm(accel, axis=1).std())
    # Normalise: std<5 px/frame² is smooth; std>30 is jittery
    return float(max(0.0, 1.0 - std / 30.0))


def object_persistence(
    centroids: List[Optional[Tuple[float, float]]]
) -> Optional[float]:
    if not centroids:
        return None
    valid = sum(1 for c in centroids if c is not None)
    return float(valid / len(centroids))


def jerk_score(centroids: List[Optional[Tuple[float, float]]]) -> Optional[float]:
    valid = [c for c in centroids if c is not None]
    if len(valid) < 5:
        return None
    arr = np.array(valid, dtype=np.float64)
    jerk = np.diff(arr, n=3, axis=0)
    if jerk.shape[0] == 0:
        return None
    peak = float(np.linalg.norm(jerk, axis=1).max())
    return float(min(1.0, peak / 50.0))


@register_module(MOD_ID, requires=[], description="Centroid trajectory smoothness / object permanence.", version=VERSION)
def run(context: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    motion_threshold = float(mod_cfg.get("motion_threshold", 0.5))
    per_shot: Dict[int, Dict[str, Any]] = {}
    for sid, frames in context.shot_frames.items():
        cents = _track_centroids(frames, motion_threshold)
        per_shot[sid] = {
            "trajectory_smoothness": trajectory_smoothness(cents),
            "object_persistence": object_persistence(cents),
            "jerk_score": jerk_score(cents),
            "n_frames": len(cents),
        }
    return {"module": MOD_ID, "version": VERSION, "per_shot": per_shot}
