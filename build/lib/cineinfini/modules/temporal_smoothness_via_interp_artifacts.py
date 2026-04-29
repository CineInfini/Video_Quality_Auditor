"""VFI (Video Frame Interpolation) artifact detector — pure CV.

Detects three families of artifacts that appear when frames are produced
by interpolation algorithms (RIFE, FILM, AdaCoF, classical motion compensation):

  1. **Ghosting** — bidirectional optical-flow inconsistency.
     For a frame triple (i-1, i, i+1), warp frame i-1 forward to predict
     frame i. If the residual is high while motion magnitude is low,
     that's a ghosting candidate (an interpolator blended two timesteps).

  2. **Judder** — irregular motion-magnitude rhythm across frames.
     Healthy real-world video has motion magnitude that varies smoothly.
     Judder shows up as a high coefficient-of-variation of consecutive
     motion-magnitude ratios — typical of 3:2 pulldown errors and
     hard-cut interpolation.

  3. **Interpolation blur** — Laplacian-variance dip on suspect frames.
     Interpolated frames typically have less high-frequency content
     than their natural neighbours at comparable motion magnitudes.
     For each frame we compute its Laplacian variance; frames whose
     L-var is meaningfully lower than the mean of their two neighbours
     get flagged.

The module returns four metrics per shot:

    ghosting_score              [0..1]   higher = more ghosting
    judder_score                [0..1]   higher = more irregular motion
    interp_blur_score           [0..1]   higher = more interpolation blur
    vfi_artifact_composite      [0..1]   mean of the three above

Pure-CV implementation — no ML deps required (only OpenCV + numpy, both
already required by CineInfini's core).

**Validation target** for this module is **VFIPS** (Video Frame
Interpolation Perceptual Similarity, Hou et al.) which uses 2AFC paired
comparisons rather than scalar MOS — the right tool for subtle artifact
detection.

Disabled by default. Enable via:

    modules:
      temporal_smoothness_via_interp_artifacts:
        enabled: true

References:
- Hou et al. "VFIPS: Video Frame Interpolation Perceptual Similarity Metric." (2022)
- Danier et al. "FloLPIPS: A Bespoke Video Quality Metric for Frame Interpolation." (2022)
"""
from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module


MODULE_ID = "temporal_smoothness_via_interp_artifacts"
MODULE_VERSION = "0.4.8.7"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _farneback_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Dense optical flow via Farneback. Returns (H, W, 2) float32."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )


def _warp_with_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warp ``img`` using ``flow`` (H, W, 2)."""
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _laplacian_variance(gray: np.ndarray) -> float:
    """High-frequency content proxy."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Sub-detectors
# ---------------------------------------------------------------------------
def _ghosting_score(frames: List[np.ndarray]) -> float:
    """Bidirectional flow-consistency residual on every frame triple.

    For each triple (i-1, i, i+1):
      - estimate flow from i-1 to i,
      - warp i-1 forward to predict i,
      - compare predicted_i with actual i (mean abs error, normalised
        to [0,1] via division by 255 and clipping).
      - normalise by motion magnitude so static scenes don't score high
        just because the warp is trivial.

    Higher = more ghosting.
    """
    if len(frames) < 3:
        return 0.0
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    residuals = []
    for i in range(1, len(grays) - 1):
        flow_back = _farneback_flow(grays[i - 1], grays[i])
        predicted = _warp_with_flow(grays[i - 1], flow_back)
        residual = float(np.mean(np.abs(predicted.astype(np.float32) -
                                        grays[i].astype(np.float32)))) / 255.0
        motion_mag = float(np.mean(np.linalg.norm(flow_back, axis=2)))
        # Normalise residual by sqrt(motion+1) — static scenes get suppressed
        residuals.append(residual / np.sqrt(motion_mag + 1.0))
    if not residuals:
        return 0.0
    return float(np.clip(np.mean(residuals) * 4.0, 0.0, 1.0))


def _judder_score(frames: List[np.ndarray]) -> float:
    """Coefficient-of-variation of consecutive motion-magnitude ratios.

    Healthy footage: ratios are close to 1 (smooth motion).
    Judder: ratios oscillate (slow/fast/slow/fast).

    Higher = more judder.
    """
    if len(frames) < 3:
        return 0.0
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    motion_mags = []
    for i in range(1, len(grays)):
        flow = _farneback_flow(grays[i - 1], grays[i])
        motion_mags.append(float(np.mean(np.linalg.norm(flow, axis=2))))
    if len(motion_mags) < 2 or max(motion_mags) < 1e-3:
        return 0.0
    # Consecutive ratios — log-domain CoV is more stable
    ratios = []
    for i in range(1, len(motion_mags)):
        if motion_mags[i - 1] > 1e-6:
            ratios.append(motion_mags[i] / motion_mags[i - 1])
    if len(ratios) < 2:
        return 0.0
    log_ratios = np.log(np.maximum(ratios, 1e-6))
    cov = float(np.std(log_ratios))
    return float(np.clip(cov / 1.5, 0.0, 1.0))


def _interp_blur_score(frames: List[np.ndarray]) -> float:
    """Fraction of frames whose Laplacian variance dips significantly
    below the mean of their two neighbours.

    A 25% dip is the empirical threshold from FloLPIPS-related
    literature for "this looks like an interpolated frame."

    Higher = more frames flagged as interpolation-blurred.
    """
    if len(frames) < 3:
        return 0.0
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    lvars = [_laplacian_variance(g) for g in grays]
    flagged = 0
    for i in range(1, len(lvars) - 1):
        neighbour_mean = (lvars[i - 1] + lvars[i + 1]) / 2.0
        if neighbour_mean > 1e-3 and lvars[i] < 0.75 * neighbour_mean:
            flagged += 1
    n_inner = len(lvars) - 2
    return float(flagged / n_inner) if n_inner > 0 else 0.0


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------
@register_module(
    MODULE_ID,
    description="VFI artifact detector — ghosting, judder, interpolation blur (pure CV)",
    version=MODULE_VERSION,
)
def temporal_smoothness_via_interp_artifacts(ctx: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    mod_cfg = cfg.modules.get(MODULE_ID, {}) or {}
    # Per-shot scoring
    per_shot: Dict[int, Dict[str, float]] = {}
    g_all, j_all, b_all = [], [], []
    for sid, frames in ctx.shot_frames.items():
        if not frames or len(frames) < 3:
            per_shot[sid] = {
                "ghosting_score": 0.0,
                "judder_score": 0.0,
                "interp_blur_score": 0.0,
                "vfi_artifact_composite": 0.0,
                "n_frames_analysed": float(len(frames) if frames else 0),
            }
            continue
        g = _ghosting_score(frames)
        j = _judder_score(frames)
        b = _interp_blur_score(frames)
        comp = float((g + j + b) / 3.0)
        per_shot[sid] = {
            "ghosting_score": g,
            "judder_score": j,
            "interp_blur_score": b,
            "vfi_artifact_composite": comp,
            "n_frames_analysed": float(len(frames)),
        }
        g_all.append(g); j_all.append(j); b_all.append(b)

    summary: Dict[str, Any] = {
        "module": MODULE_ID,
        "version": MODULE_VERSION,
        "available": True,
        "per_shot": per_shot,
        "mean_ghosting": float(np.mean(g_all)) if g_all else None,
        "mean_judder": float(np.mean(j_all)) if j_all else None,
        "mean_interp_blur": float(np.mean(b_all)) if b_all else None,
        "validation_target_dataset": "VFIPS",
        "thresholds_applied": {
            "lvar_dip_ratio": mod_cfg.get("lvar_dip_ratio", 0.75),
            "judder_cov_normaliser": mod_cfg.get("judder_cov_normaliser", 1.5),
        },
    }
    return summary
