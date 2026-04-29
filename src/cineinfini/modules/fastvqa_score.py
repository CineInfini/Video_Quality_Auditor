"""FAST-VQA fragment-sampling video-quality scorer (ECCV 2022, opt-in).

FAST-VQA produces a single quality score in [0, 1] using fragment
sampling over a Swin-T backbone. We pin the v0.3 paper weights (see
``cfg.optional_models['fastvqa']``) and surface its score in CineInfini's
report alongside our 19 native metrics.

Activation:
  1. ``cfg.modules['fastvqa_score']['enabled'] = True``
  2. Weight present (run ``cineinfini bootstrap --include-optional``)
  3. ``pip install torch torchvision``
  4. The FAST-VQA model definition importable as ``fastvqa`` —
     ``pip install fast-vqa`` or clone
     https://github.com/VQAssessment/FAST-VQA-and-FasterVQA and
     ``pip install -e .``

Same graceful-degradation policy as ``dover_score`` and
``origin_detection``: when prerequisites are missing the module reports
``available: false`` with a precise reason rather than producing fake
scores.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.fastvqa_score")

MODULE_ID = "fastvqa_score"
METRIC_VERSION = "fastvqa@v0.3"


def _check_availability() -> Tuple[bool, str]:
    cfg = get_config()
    opt = (cfg.optional_models or {}).get("fastvqa")
    if not opt:
        return False, "cfg.optional_models['fastvqa'] not configured"
    weight_path = cfg.models_dir() / opt.get("filename", "fast-vqa_v0_3.pth")
    if not weight_path.exists():
        return False, (
            f"weights missing at {weight_path}; run "
            f"`cineinfini bootstrap --include-optional` to fetch"
        )
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch not installed"
    try:
        import fastvqa  # noqa: F401
    except ImportError:
        return False, (
            "FAST-VQA model definition not importable; install with "
            "`pip install fast-vqa` or clone "
            "https://github.com/VQAssessment/FAST-VQA-and-FasterVQA"
        )
    return True, "OK"


def _run_fastvqa_inference(frames: List[np.ndarray]) -> float:
    """Plug the real FAST-VQA call here; return a quality score in [0, 1].

    Real-world implementation (from FAST-VQA's README):

        from fastvqa import deep_end_to_end_vqa
        import torch
        video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float()
        scorer = deep_end_to_end_vqa(pretrained=True, model_type="FAST-VQA")
        return float(scorer(video))
    """
    raise NotImplementedError(
        "Plug your FAST-VQA inference call here; see module docstring."
    )


@register_module(
    MODULE_ID,
    description="FAST-VQA fragment-sampling VQA scorer (ECCV 2022)",
    requires=("fastvqa",),
)
def fastvqa_score(ctx: VideoContext) -> Dict[str, Any]:
    available, reason = _check_availability()
    if not available:
        logger.info("fastvqa_score not available: %s", reason)
        return {
            "module": MODULE_ID,
            "version": METRIC_VERSION,
            "available": False,
            "reason": reason,
            "per_shot": {},
        }

    per_shot: Dict[int, Dict[str, float]] = {}
    scores: List[float] = []
    for sid, frames in ctx.shot_frames.items():
        try:
            s = _run_fastvqa_inference(frames)
        except NotImplementedError as e:
            return {
                "module": MODULE_ID, "version": METRIC_VERSION,
                "available": False, "reason": str(e),
                "per_shot": {},
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("FAST-VQA inference failed on shot %d: %s", sid, e)
            continue
        per_shot[sid] = {"fastvqa_score": float(s)}
        scores.append(float(s))

    summary: Dict[str, Any] = {
        "module": MODULE_ID,
        "version": METRIC_VERSION,
        "available": True,
        "per_shot": per_shot,
    }
    if scores:
        summary["mean_fastvqa"] = float(np.mean(scores))
    return summary


__all__ = ["fastvqa_score", "MODULE_ID", "METRIC_VERSION"]
