"""Trustworthiness: SSIM stability under additive Gaussian noise.

A trustworthy video gives consistent SSIM scores when small amounts of
noise are added — its quality assessment is robust to perturbation. AI-
generated content often has fragile features that collapse under noise.

We perturb a sample of frames N times and report the standard deviation
of the SSIM-3d-self metric. Low std-dev = high trust.

Pure NumPy + scikit-image. Disabled by default.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.trustworthiness")
MOD_ID = "trustworthiness"
VERSION = "0.4.8.1.4"


def _ssim3d_self_safe(frames: List[np.ndarray]) -> Optional[float]:
    try:
        from ..core.metrics import ssim3d_self
        return ssim3d_self(frames)
    except Exception as e:  # noqa: BLE001
        logger.debug("ssim3d_self failed: %s", e)
        return None


def _add_gaussian_noise(
    frames: List[np.ndarray], sigma: float, rng: np.random.Generator
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for f in frames:
        noise = rng.normal(0, sigma * 255.0, f.shape)
        noisy = np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        out.append(noisy)
    return out


def trust_score(
    frames: List[np.ndarray], noise_level: float = 0.05, n_samples: int = 5,
) -> Dict[str, Any]:
    if len(frames) < 3:
        return {"trust_score": None, "ssim_baseline": None, "ssim_perturbed_std": None}
    baseline = _ssim3d_self_safe(frames)
    if baseline is None:
        return {"trust_score": None, "ssim_baseline": None, "ssim_perturbed_std": None}

    rng = np.random.default_rng(42)
    perturbed_scores: List[float] = []
    for _ in range(n_samples):
        noisy = _add_gaussian_noise(frames, noise_level, rng)
        s = _ssim3d_self_safe(noisy)
        if s is not None:
            perturbed_scores.append(s)
    if not perturbed_scores:
        return {"trust_score": None, "ssim_baseline": baseline, "ssim_perturbed_std": None}
    std = float(np.std(perturbed_scores))
    # Trust = 1 - normalised std. Std > 0.1 -> 0; std < 0.01 -> 1
    trust = float(max(0.0, 1.0 - std / 0.1))
    return {
        "trust_score": trust,
        "ssim_baseline": float(baseline),
        "ssim_perturbed_std": std,
        "ssim_perturbed_mean": float(np.mean(perturbed_scores)),
        "n_samples": len(perturbed_scores),
    }


@register_module(MOD_ID, requires=[], description="SSIM robustness to additive Gaussian noise.", version=VERSION)
def run(context: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    noise_level = float(mod_cfg.get("noise_level", 0.05))
    n_samples = int(mod_cfg.get("n_samples", 5))
    per_shot: Dict[int, Dict[str, Any]] = {}
    for sid, frames in context.shot_frames.items():
        per_shot[sid] = trust_score(frames, noise_level=noise_level, n_samples=n_samples)
    return {"module": MOD_ID, "version": VERSION, "per_shot": per_shot}
