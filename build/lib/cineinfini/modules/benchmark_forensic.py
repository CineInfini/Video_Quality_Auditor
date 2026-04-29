"""Forensic robustness benchmark.

Re-encode each shot at multiple compression levels (CRF) and measure the
SSIM drop. Real footage degrades gradually; AI artifacts often collapse
abruptly. Returns the SSIM trajectory across CRFs and a "robustness"
score equal to 1 - normalised area-under-the-curve of the SSIM drop.

Pure FFmpeg subprocess + scikit-image. Disabled by default.
Requires ffmpeg on PATH.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module

logger = logging.getLogger("cineinfini.modules.benchmark_forensic")
MOD_ID = "benchmark_forensic"
VERSION = "0.4.8.1.4"


def _frames_to_video(frames: List[np.ndarray], path: Path, fps: float = 24.0) -> bool:
    import cv2
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        return False
    for f in frames:
        writer.write(f)
    writer.release()
    return True


def _reencode(src: Path, dst: Path, crf: int) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-crf", str(crf),
        "-preset", "ultrafast", str(dst),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=60)
        return res.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    except Exception as e:  # noqa: BLE001
        logger.debug("ffmpeg failed at CRF %d: %s", crf, e)
        return False


def _read_all_frames(path: Path, max_frames: int = 30) -> List[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    finally:
        cap.release()
    return frames


def _ssim_pair(a: List[np.ndarray], b: List[np.ndarray]) -> Optional[float]:
    try:
        from skimage.metrics import structural_similarity as ssim
        import cv2
        n = min(len(a), len(b))
        if n == 0:
            return None
        ssims: List[float] = []
        for i in range(n):
            ga = cv2.cvtColor(a[i], cv2.COLOR_BGR2GRAY) if a[i].ndim == 3 else a[i]
            gb = cv2.cvtColor(b[i], cv2.COLOR_BGR2GRAY) if b[i].ndim == 3 else b[i]
            if ga.shape != gb.shape:
                gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
            ssims.append(float(ssim(ga, gb)))
        return float(np.mean(ssims))
    except Exception as e:  # noqa: BLE001
        logger.debug("ssim failed: %s", e)
        return None


@register_module(MOD_ID, requires=[],
                 description="SSIM under successive re-encoding (forensic robustness).",
                 version=VERSION)
def run(context: VideoContext) -> Dict[str, Any]:
    cfg = get_config()
    mod_cfg = cfg.get_module_config(MOD_ID)
    crfs = list(mod_cfg.get("compression_crfs", [23, 28, 35]))
    if shutil.which("ffmpeg") is None:
        return {"module": MOD_ID, "version": VERSION, "available": False,
                "error": "ffmpeg not on PATH; run `cineinfini bootstrap` for instructions"}

    fps = float(getattr(context.video, "fps", 24.0)) or 24.0
    per_shot: Dict[int, Dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="forensic_") as td:
        td_path = Path(td)
        for sid, frames in context.shot_frames.items():
            src = td_path / f"shot_{sid}.mp4"
            if not _frames_to_video(frames, src, fps):
                per_shot[sid] = {"available": False, "error": "encode failed"}
                continue
            ssim_curve: Dict[str, Optional[float]] = {}
            for crf in crfs:
                dst = td_path / f"shot_{sid}_crf{crf}.mp4"
                if not _reencode(src, dst, crf):
                    ssim_curve[str(crf)] = None
                    continue
                reencoded = _read_all_frames(dst, max_frames=len(frames))
                ssim_curve[str(crf)] = _ssim_pair(frames, reencoded)
            valid = [v for v in ssim_curve.values() if v is not None]
            if valid:
                # Robustness = mean SSIM across CRFs (1 = perfect, 0 = collapse)
                robustness = float(np.mean(valid))
            else:
                robustness = None
            per_shot[sid] = {"ssim_curve": ssim_curve, "robustness": robustness,
                             "n_crfs": len(crfs)}
    return {"module": MOD_ID, "version": VERSION,
            "compression_crfs": crfs, "per_shot": per_shot}
