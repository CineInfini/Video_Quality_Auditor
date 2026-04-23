"""Stable metric functions (optical flow, SSIM, flicker, etc.)"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_2d

# -------------------------------------------------------------------
# Optical flow and motion
# -------------------------------------------------------------------
def optical_flow_farneback(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def motion_field_divergence(flow):
    u = flow[...,0]
    v = flow[...,1]
    u_x, u_y = np.gradient(u)
    v_x, v_y = np.gradient(v)
    return u_x + v_y

def motion_peak_div(frames):
    if len(frames) < 3:
        return None
    peaks = []
    for i in range(len(frames)-2):
        flow = optical_flow_farneback(frames[i], frames[i+2])
        div = np.abs(motion_field_divergence(flow))
        peaks.append(float(div.max()))
    return float(np.max(peaks)) if peaks else None

# -------------------------------------------------------------------
# 3D-SSIM
# -------------------------------------------------------------------
def ssim_3d_self_shifted(vol, block_size=7):
    if vol.ndim == 4:
        vol = np.mean(vol, axis=3).astype(np.uint8)
    if vol.shape[0] < 2:
        return 1.0
    if np.all(vol == vol[0]):
        return 1.0
    vals = []
    for i in range(vol.shape[0]-1):
        if np.var(vol[i]) == 0 or np.var(vol[i+1]) == 0:
            vals.append(1.0)
        else:
            vals.append(ssim_2d(vol[i], vol[i+1], data_range=vol.max()-vol.min()))
    return float(np.mean(vals))

def ssim3d_self(frames):
    if len(frames) < 16:
        return None
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    vol = np.stack(gray, axis=0)
    return ssim_3d_self_shifted(vol)

# -------------------------------------------------------------------
# Flicker
# -------------------------------------------------------------------
def flicker_score_no_reference(vol):
    if vol.ndim == 4:
        vol = np.mean(vol, axis=3).astype(np.float32)
    else:
        vol = vol.astype(np.float32)
    if vol.shape[0] < 2:
        return 0.0
    diffs = []
    for i in range(vol.shape[0]-1):
        diff = np.mean(np.abs(vol[i+1] - vol[i])) / 255.0
        diffs.append(diff)
    return float(np.mean(diffs))

def flicker_score(frames):
    if len(frames) < 3:
        return None
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    vol = np.stack(gray, axis=0)
    return flicker_score_no_reference(vol)

def flicker_highfreq_variance(frames):
    if len(frames) < 3:
        return None
    diffs = []
    for i in range(len(frames)-1):
        diff = np.abs(cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY).astype(np.float32) -
                      cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32))
        diffs.append(diff.mean())
    return float(np.var(diffs))

# -------------------------------------------------------------------
# SSIM long range
# -------------------------------------------------------------------
def ssim_long_range(frames):
    if len(frames) < 2:
        return None
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
    return ssim_2d(g0, g1, data_range=255)

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

def compute_composite_score(metrics, weights=None):
    if weights is None:
        weights = DEFAULT_COMPOSITE_WEIGHTS
    score = 0.0
    for key, w in weights.items():
        if key in metrics and metrics[key] is not None:
            score += w * metrics[key]
    return score
