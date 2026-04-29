"""DOVER aesthetic + technical scoring (ICCV 2023, opt-in wrapper).

This module surfaces DOVER's two-branch UGC-VQA scores inside CineInfini's
report so a single audit run produces both our 19 native metrics *and*
DOVER's outputs. That's the "subsume" strategy — instead of competing
with DOVER, we include it.

Activation requires three pieces:
  1. ``cfg.modules['dover_score']['enabled'] = True`` (config gate)
  2. ``cfg.optional_models['dover']`` weight present locally (run
     ``cineinfini bootstrap --include-optional`` to fetch).
  3. ``pip install torch torchvision`` (DOVER uses torch — we don't
     vendor the model code; users who want the real scores either
     install ``pip install dover-vqa`` or copy the model definition
     from https://github.com/QualityAssessment/DOVER).

When any prerequisite is missing the module reports ``available: False``
with the specific reason — exactly the same pattern as
``origin_detection`` and ``multi_modal_safety``. **No fake science.**

The wrapper is structured so that swapping in the real inference is a
one-function change (``_run_dover_inference``); everything else —
shot-level aggregation, normalisation, gate registration — is handled
here and won't change.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.dover_score")

MODULE_ID = "dover_score"
METRIC_VERSION = "dover@v0.1.0"


# ---------------------------------------------------------------------------
# Availability detection — honest reporting
# ---------------------------------------------------------------------------
def _check_availability() -> Tuple[bool, str]:
    """Return (available, reason). reason is informative when available=False."""
    cfg = get_config()
    # Weight file?
    opt = (cfg.optional_models or {}).get("dover")
    if not opt:
        return False, "cfg.optional_models['dover'] not configured"
    weight_name = opt.get("filename", "DOVER.pth")
    weight_path = cfg.models_dir() / weight_name
    if not weight_path.exists():
        return False, (
            f"weights missing at {weight_path}; run "
            f"`cineinfini bootstrap --include-optional` to fetch"
        )
    # torch?
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch not installed (`pip install torch torchvision`)"
    # DOVER model code? Try the official package, then a local copy.
    has_dover_pkg = False
    try:
        import dover  # noqa: F401
        has_dover_pkg = True
    except ImportError:
        pass
    if not has_dover_pkg:
        return False, (
            "DOVER model definition not importable; install with "
            "`pip install dover-vqa` or clone "
            "https://github.com/QualityAssessment/DOVER and pip install -e ."
        )
    return True, "OK"


def _run_dover_inference(frames: List[np.ndarray]) -> Tuple[float, float]:
    """Call DOVER on a list of frames; return (aesthetic, technical) in [0, 1].

    Implementation kept thin so users with the real DOVER package can plug
    in without touching the rest of this module. Falls back to NotImplemented
    when called inside the dummy stub path.
    """
    cfg = get_config()
    weight_path = cfg.models_dir() / cfg.optional_models["dover"]["filename"]
    # The real implementation looks like:
    #
    #     from dover import DOVER  # or whatever the package exposes
    #     model = DOVER.from_pretrained(weight_path)
    #     aesthetic, technical = model.score(frames)
    #     return float(aesthetic), float(technical)
    #
    # We don't ship the model code, so this raises and the module reports
    # "available: false" until the user wires it.
    raise NotImplementedError(
        "Plug your DOVER inference call here; see module docstring."
    )


# ---------------------------------------------------------------------------
# Module entry
# ---------------------------------------------------------------------------
@register_module(
    MODULE_ID,
    description="DOVER aesthetic + technical UGC scorer (ICCV 2023)",
    requires=("dover",),
)
def dover_score(ctx: VideoContext) -> Dict[str, Any]:
    cfg = ctx.cfg
    available, reason = _check_availability()
    if not available:
        logger.info("dover_score not available: %s", reason)
        return {
            "module": MODULE_ID,
            "version": METRIC_VERSION,
            "available": False,
            "reason": reason,
            "per_shot": {},
        }

    per_shot: Dict[int, Dict[str, float]] = {}
    aesthetics: List[float] = []
    technicals: List[float] = []

    for sid, frames in ctx.shot_frames.items():
        try:
            a, t = _run_dover_inference(frames)
        except NotImplementedError as e:
            return {
                "module": MODULE_ID, "version": METRIC_VERSION,
                "available": False, "reason": str(e),
                "per_shot": {},
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("DOVER inference failed on shot %d: %s", sid, e)
            continue
        per_shot[sid] = {
            "dover_aesthetic": float(a),
            "dover_technical": float(t),
        }
        aesthetics.append(float(a))
        technicals.append(float(t))

    summary: Dict[str, Any] = {
        "module": MODULE_ID,
        "version": METRIC_VERSION,
        "available": True,
        "per_shot": per_shot,
    }
    if aesthetics:
        summary["mean_aesthetic"] = float(np.mean(aesthetics))
        summary["mean_technical"] = float(np.mean(technicals))
        summary["mean_dover_fused"] = 0.5 * (
            summary["mean_aesthetic"] + summary["mean_technical"]
        )
    return summary


__all__ = ["dover_score", "MODULE_ID", "METRIC_VERSION"]
