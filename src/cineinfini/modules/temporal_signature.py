"""Temporal signature: cyclic coherence of optical flow.

Real videos with periodic motion (walking, breathing, vehicle traffic)
produce strong autocorrelation peaks in the optical-flow magnitude
signal. AI-generated videos often lack this rhythmic structure or
produce artificial repeating patterns. We measure two things:

* ``flow_periodicity``: peak-to-mean ratio of the autocorrelation of the
  per-frame flow magnitude (excluding lag 0). High = strong rhythm.
* ``flow_signature_entropy``: Shannon entropy of the normalised
  autocorrelation. Low entropy = monotonic / artificial.

Pure NumPy + OpenCV. Disabled by default.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.temporal_signature")
MOD_ID = "temporal_signature"
VERSION = "0.4.8.1.4"


def _flow_magnitude_series(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    import cv2
    if len(frames) < 4:
        return None
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]
    mags: List[float] = []
    for i in range(len(grays) - 1):
        try:
            flow = cv2.calcOpticalFlowFarneback(
                grays[i], grays[i + 1], None,
                0.5, 3, 15, 3, 5, 1.2, 0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mags.append(float(mag.mean()))
        except Exception as e:  # noqa: BLE001
            logger.debug("flow %d failed: %s", i, e)
    return np.array(mags, dtype=np.float64) if mags else None


def autocorrelation(series: np.ndarray) -> np.ndarray:
    s = series - series.mean()
    if s.std() < 1e-9:
        return np.zeros_like(s)
    out = np.correlate(s, s, mode="full")
    out = out[out.size // 2:]
    return out / (out[0] + 1e-9)


def flow_periodicity(series: np.ndarray) -> Optional[float]:
    if series is None or series.size < 4:
        return None
    ac = autocorrelation(series)
    if ac.size < 3:
        return None
    body = ac[1:]
    peak = float(body.max())
    mean = float(np.abs(body).mean())
    if mean < 1e-9:
        return None
    return peak / (peak + mean)


def flow_signature_entropy(series: np.ndarray, bins: int = 16) -> Optional[float]:
    if series is None or series.size < 4:
        return None
    ac = np.abs(autocorrelation(series)[1:])
    if ac.size < 2 or ac.sum() < 1e-9:
        return None
    hist, _ = np.histogram(ac, bins=bins, range=(0, 1))
    p = hist / hist.sum()
    p = p[p > 0]
    h = float(-(p * np.log2(p)).sum())
    return h / np.log2(bins)  # normalise to [0,1]


@register_module(MOD_ID, requires=[], description="Cyclic coherence of optical flow.", version=VERSION)
def run(context: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    max_frames = int(mod_cfg.get("max_frames", 30))
    per_shot: Dict[int, Dict[str, Any]] = {}
    for sid, frames in context.shot_frames.items():
        sub = frames[:max_frames]
        series = _flow_magnitude_series(sub)
        per_shot[sid] = {
            "flow_periodicity": flow_periodicity(series) if series is not None else None,
            "flow_signature_entropy": flow_signature_entropy(series) if series is not None else None,
            "n_pairs": int(series.size) if series is not None else 0,
        }
    return {"module": MOD_ID, "version": VERSION, "per_shot": per_shot}
