from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from ..core.config import get_config

CONFIG = get_config().to_audit_config()
REPORTS_DIR: Optional[Path] = None
BENCHMARK_DIR: Optional[Path] = None

def set_global_paths(reports_dir: str|Path, benchmark_dir: str|Path) -> None:
    global REPORTS_DIR, BENCHMARK_DIR
    REPORTS_DIR = Path(reports_dir)
    BENCHMARK_DIR = Path(benchmark_dir)
    cfg = get_config()
    cfg.paths["reports_dir"] = str(REPORTS_DIR)
    cfg.paths["benchmark_dir"] = str(BENCHMARK_DIR)

@dataclass
class VideoInfo:
    path: Path; fps: float; total_frames: int; duration_s: float
    @property
    def name(self) -> str: return self.path.stem

@dataclass
class ModelBundle:
    detector: Optional[Any] = None; embedder: Optional[Any] = None; clip_scorer: Optional[Any] = None
    clip_model: Optional[Any] = None; clip_preprocess: Optional[Any] = None; dinov2_model: Optional[Any] = None
    dinov2_processor: Optional[Any] = None; device: str = "cpu"
    @property
    def clip_available(self) -> bool: return self.clip_model is not None
    @property
    def dinov2_available(self) -> bool: return self.dinov2_model is not None

@dataclass
class AuditTiming:
    video_info: float = 0.0; shot_detection: float = 0.0; frame_extraction: float = 0.0; model_init: float = 0.0
    shot_processing: float = 0.0; inter_coherence: float = 0.0; narrative: float = 0.0; composite: float = 0.0; persist: float = 0.0
    @property
    def total(self) -> float: return sum(vars(self).values())

def _load_video_info(video_path: Path, timing: AuditTiming) -> VideoInfo:
    t0 = time.time()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    timing.video_info = time.time() - t0
    return VideoInfo(Path(video_path), fps, total, total / max(fps,1e-6))

def _init_models(device: str, cfg: dict, timing: AuditTiming) -> ModelBundle:
    return ModelBundle(device=device)

def _process_shots(shots, frames_dict, models, cfg, timing):
    raise RuntimeError("_process_shots deprecated in v0.4.7. Use orchestrator.run_audit() or cineinfini.audit_video().")

def _compute_inter_coherence(shot_frames, models, timing):
    from ..core.coherence import compute_inter_shot_coherence
    t0 = time.time()
    res = compute_inter_shot_coherence(shot_frames, models.detector, models.embedder, models.clip_scorer, 5)
    timing.inter_coherence = time.time() - t0
    return res

def _compute_composite(gates, timing):
    from ..core.metrics import compute_composite_score
    t0 = time.time()
    cfg = get_config()
    for sid, gate in gates.items():
        try: gate["composite"] = compute_composite_score(gate, cfg.thresholds)
        except: pass
    timing.composite = time.time() - t0

def _persist_results(audit_data: Dict[str, Any], out_dir: Path, timing: AuditTiming) -> None:
    import json
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(audit_data, indent=2, default=str))
    timing.persist = time.time() - t0

def audit_video(video_path: str|Path, video_params: Optional[Dict[str,Any]] = None, force_full_video: bool = False) -> Tuple[Dict[str,Any],Path]:
    from .orchestrator import run_audit
    cfg = get_config()
    if video_params:
        for k,v in video_params.items():
            if k == "thresholds" and isinstance(v,dict): cfg.thresholds.update(v)
            elif k in cfg.processing: cfg.processing[k] = v
    out_dir = REPORTS_DIR / Path(video_path).stem if REPORTS_DIR else None
    return run_audit(video_path, output_dir=out_dir, force_full_video=force_full_video)

def adaptive_multi_stage_audit(video_path, force_full_video=False):
    return audit_video(video_path, force_full_video=force_full_video)

def generate_synthetic_video(path, duration_s=2, fps=24, size=(320,240), shape="circle"):
    path = str(path); w,h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w,h))
    n_frames = int(duration_s * fps)
    for i in range(n_frames):
        frame = np.full((h,w,3), 30, dtype=np.uint8)
        cx = int(w/2 + (w/4)*np.sin(2*np.pi*i/max(n_frames,1)))
        cy = h//2
        if shape == "circle": cv2.circle(frame, (cx,cy), 25, (200,200,50), -1)
        else: cv2.rectangle(frame, (cx-25,cy-25), (cx+25,cy+25), (200,200,50), -1)
        writer.write(frame)
    writer.release()
    return path
