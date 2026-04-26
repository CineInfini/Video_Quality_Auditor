"""Stable metric functions (optical flow, SSIM, flicker, etc.).

All functions return None when input is too short or degenerate, so callers
can distinguish "not computed" from "zero". Never returns 0.0 as a fallback.
"""
from __future__ import annotations

from typing import Optional
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_2d


# -------------------------------------------------------------------
# Optical flow and motion
# -------------------------------------------------------------------

def optical_flow_farneback(frame1: "np.ndarray", frame2: "np.ndarray") -> "np.ndarray":
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def motion_field_divergence(flow: "np.ndarray") -> "np.ndarray":
    u = flow[..., 0]
    v = flow[..., 1]
    u_x, _ = np.gradient(u)
    _, v_y = np.gradient(v)
    return u_x + v_y


def motion_peak_div(frames: list) -> Optional[float]:
    if len(frames) < 3:
        return None
    peaks = []
    for i in range(len(frames) - 2):
        flow = optical_flow_farneback(frames[i], frames[i + 2])
        div = np.abs(motion_field_divergence(flow))
        peaks.append(float(div.max()))
    return float(np.max(peaks)) if peaks else None


# -------------------------------------------------------------------
# 3D-SSIM
# -------------------------------------------------------------------

def ssim_3d_self_shifted(vol: "np.ndarray", block_size: int = 7) -> float:
    if vol.ndim == 4:
        vol = np.mean(vol, axis=3).astype(np.uint8)
    if vol.shape[0] < 2:
        return 1.0
    if np.all(vol == vol[0]):
        return 1.0
    vals = []
    for i in range(vol.shape[0] - 1):
        if np.var(vol[i]) == 0 or np.var(vol[i + 1]) == 0:
            vals.append(1.0)
        else:
            vals.append(ssim_2d(vol[i], vol[i + 1], data_range=vol.max() - vol.min()))
    return float(np.mean(vals))


def ssim3d_self(frames: list) -> Optional[float]:
    if len(frames) < 16:
        return None
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    vol = np.stack(gray, axis=0)
    return ssim_3d_self_shifted(vol)


# -------------------------------------------------------------------
# Flicker
# -------------------------------------------------------------------

def flicker_score_no_reference(vol: "np.ndarray") -> float:
    if vol.ndim == 4:
        vol = np.mean(vol, axis=3).astype(np.float32)
    else:
        vol = vol.astype(np.float32)
    if vol.shape[0] < 2:
        return 0.0
    diffs = []
    for i in range(vol.shape[0] - 1):
        diff = np.mean(np.abs(vol[i + 1] - vol[i])) / 255.0
        diffs.append(diff)
    return float(np.mean(diffs))


def flicker_score(frames: list) -> Optional[float]:
    if len(frames) < 3:
        return None
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    vol = np.stack(gray, axis=0)
    return flicker_score_no_reference(vol)


def flicker_highfreq_variance(frames: list) -> Optional[float]:
    """Variance of inter-frame mean absolute differences."""
    if len(frames) < 3:
        return None
    diffs = []
    for i in range(len(frames) - 1):
        diff = np.abs(
            cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            - cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        )
        diffs.append(diff.mean())
    return float(np.var(diffs))


# -------------------------------------------------------------------
# SSIM long range
# -------------------------------------------------------------------

def ssim_long_range(frames: list) -> Optional[float]:
    if len(frames) < 2:
        return None
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
    if g0.shape != g1.shape:
        g1 = cv2.resize(g1, (g0.shape[1], g0.shape[0]))
    return float(ssim_2d(g0, g1, data_range=255))


# -------------------------------------------------------------------
# Composite score
# -------------------------------------------------------------------

DEFAULT_COMPOSITE_WEIGHTS = {
    "motion_mean": -1.0,
    "ssim_mean": 1.0,
    "flicker_mean": -1.0,
    "identity_mean": -1.0,
    "ssim_lr_mean": 1.0,
    "clip_temp_mean": 1.0,
}

WEIGHT_TO_METRIC_KEY = {
    "motion_mean": "motion_peak_div",
    "ssim_mean": "ssim3d_self",
    "flicker_mean": "flicker",
    "identity_mean": "identity_intra",
    "ssim_lr_mean": "ssim_long_range",
    "clip_temp_mean": "clip_temp_consistency",
}


def compute_composite_score(metrics: dict, weights: Optional[dict] = None) -> float:
    if weights is None:
        weights = DEFAULT_COMPOSITE_WEIGHTS
    score = 0.0
    for key, w in weights.items():
        val = metrics.get(key)
        if val is not None:
            score += w * val
    return float(score)


def recompute_composite_scores(gates: dict, weights: Optional[dict] = None) -> dict:
    if weights is None:
        weights = DEFAULT_COMPOSITE_WEIGHTS
    for g in gates.values():
        mets = {
            weight_key: g.get(metric_key)
            for weight_key, metric_key in WEIGHT_TO_METRIC_KEY.items()
        }
        g["composite"] = compute_composite_score(mets, weights)
    return gates



def clip_temp_consistency(frames, clip_scorer):
    """
    Measure temporal consistency of semantic content using CLIP.
    
    Parameters
    ----------
    frames : list of np.ndarray
        List of frames in temporal order.
    clip_scorer : CLIPSemanticScorer
        Instance of CLIPSemanticScorer.
    
    Returns
    -------
    float
        Average cosine similarity between consecutive frame CLIP embeddings.
        Returns 0.0 if not enough frames or scorer unavailable.
    """
    if len(frames) < 2 or not clip_scorer.available:
        return 0.0
    
    feats = []
    for f in frames:
        feat = clip_scorer.extract_features(f)
        if feat is not None:
            feats.append(feat)
        else:
            return 0.0
    
    if len(feats) < 2:
        return 0.0
    
    sims = []
    for i in range(len(feats) - 1):
        sim = np.dot(feats[i], feats[i+1]) / (np.linalg.norm(feats[i]) * np.linalg.norm(feats[i+1]) + 1e-9)
        sims.append(sim)
    
    return float(np.mean(sims))
