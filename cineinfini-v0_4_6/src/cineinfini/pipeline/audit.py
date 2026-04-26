"""
CineInfini audit pipeline (refactored in v0.4.0).

Changes from v0.3.0
-------------------
- audit_video split into 6 single-responsibility functions.
- cv2.cuda.getCudaEnabledDeviceCount() no longer called at import time
  (was crashing CI without CUDA-enabled OpenCV).
- audit_todelete.py removed.
- Full type annotations on all public functions.
- AuditTiming dataclass replaces scattered timing variables.
- VideoInfo and ModelBundle dataclasses isolate I/O concerns.
- CONFIG and path globals kept for backward compatibility.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ..core.metrics import (
    motion_peak_div, ssim3d_self, flicker_score,
    ssim_long_range, flicker_highfreq_variance,
    compute_composite_score, recompute_composite_scores,
)
from ..core.face_detection import (
    CascadeFaceDetector, ArcFaceEmbedder, identity_within_shot,
)
from ..core.embedding import (
    CLIPSemanticScorer, clip_semantic_consistency, load_dinov2, get_dinov2,
)
from ..core.coherence import (
    compute_inter_shot_coherence, compute_narrative_coherence,
)
from ..io.reader import detect_shot_boundaries, extract_shot_frames_global
from ..io.report import generate_intra_report


# ---------------------------------------------------------------------------
# GPU detection — safe, never raises at import time
# ---------------------------------------------------------------------------

def _detect_gpu() -> str:
    """Return 'cuda' if a GPU is usable, else 'cpu'. Never raises."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Global configuration — backward compatible
# ---------------------------------------------------------------------------

CONFIG: dict[str, Any] = {
    "max_duration_s": 60,
    "shot_threshold": 0.2,
    "min_shot_duration_s": 0.5,
    "downsample_to": (320, 180),
    "n_frames_per_shot": 16,
    "frame_resize": (320, 180),
    "embedder": "arcface_onnx",
    "semantic_scorer": "clip",
    "thresholds": {
        "motion": 25.0, "ssim3d": 0.45, "flicker": 0.1,
        "identity_drift": 0.6, "ssim_long_range": 0.45,
        "clip_temp": 0.25, "flicker_hf": 0.01, "narrative_coherence": 0.7,
    },
    "parallel_shots": True,
    "max_workers": 4,
    "narrative_coherence": True,
    "benchmark_mode": True,
    "gpu_device": _detect_gpu(),   # FIX: was cv2.cuda.* at module level
    "enable_animal_face_detection": False,
    "adaptive_threshold": True,
    "threshold_percentile": 85,
    "compute_dtw_self": True,
    "compute_dtw_inter": True,
}

REPORTS_DIR: Optional[Path] = None
BENCHMARK_DIR: Optional[Path] = None


def set_global_paths(reports_dir: str | Path, benchmark_dir: str | Path) -> None:
    """Set output directories. Kept for backward compatibility."""
    global REPORTS_DIR, BENCHMARK_DIR
    REPORTS_DIR = Path(reports_dir)
    BENCHMARK_DIR = Path(benchmark_dir)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VideoInfo:
    path: Path
    fps: float
    total_frames: int
    duration_s: float

    @property
    def name(self) -> str:
        return self.path.stem


@dataclass
class ModelBundle:
    detector: Optional[Any]
    embedder: Optional[Any]
    clip_scorer: Optional[Any]
    clip_model: Any
    clip_preprocess: Any
    dinov2_model: Any
    dinov2_processor: Any
    device: str

    @property
    def clip_available(self) -> bool:
        return self.clip_model is not None

    @property
    def dinov2_available(self) -> bool:
        return self.dinov2_model is not None


@dataclass
class AuditTiming:
    video_info: float = 0.0
    shot_detection: float = 0.0
    frame_extraction: float = 0.0
    model_init: float = 0.0
    shot_processing: float = 0.0
    inter_coherence: float = 0.0
    narrative: float = 0.0
    composite: float = 0.0
    persist: float = 0.0

    @property
    def total(self) -> float:
        return sum(vars(self).values())

    def report(self, video_name: str) -> str:
        lines = [f"\n📊 TIMING SUMMARY for {video_name}:"]
        for k, v in vars(self).items():
            lines.append(f"   {k:<22}: {v:6.2f}s")
        lines.append(f"   {'TOTAL':<22}: {self.total:6.2f}s  ({self.total/60:.2f} min)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline stages — one responsibility each, fully testable
# ---------------------------------------------------------------------------

def _load_video_info(video_path: Path, timing: AuditTiming) -> VideoInfo:
    """Stage 1: read video metadata without extracting any frame."""
    t0 = time.time()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    timing.video_info = time.time() - t0
    return VideoInfo(
        path=video_path,
        fps=fps,
        total_frames=total_frames,
        duration_s=total_frames / fps,
    )


def _init_models(device: str, cfg: dict, timing: AuditTiming) -> ModelBundle:
    """Stage 2: load all ML models. Gracefully degrades on missing weights."""
    t0 = time.time()
    detector = embedder = None
    try:
        detector = CascadeFaceDetector()
        embedder = ArcFaceEmbedder()
    except Exception as exc:
        print(f"  ⚠️  Face models unavailable: {exc}")

    scorer = CLIPSemanticScorer(device=device)
    clip_model = scorer.model if scorer.available else None
    clip_preprocess = scorer.preprocess if scorer.available else None

    dinov2_model = dinov2_proc = None
    if cfg.get("narrative_coherence", True):
        if load_dinov2(device):
            dinov2_proc, dinov2_model = get_dinov2()
            print("  ✅ DINOv2 loaded")
        else:
            print("  ⚠️  DINOv2 unavailable — narrative coherence disabled")
            cfg["narrative_coherence"] = False

    timing.model_init = time.time() - t0
    return ModelBundle(
        detector=detector, embedder=embedder,
        clip_scorer=scorer,
        clip_model=clip_model, clip_preprocess=clip_preprocess,
        dinov2_model=dinov2_model, dinov2_processor=dinov2_proc,
        device=device,
    )


def _process_shots(
    shots: list,
    frames_dict: dict[int, np.ndarray],
    models: ModelBundle,
    cfg: dict,
    timing: AuditTiming,
) -> tuple[dict[int, dict], dict[int, list]]:
    """Stage 3: compute 7 intra-shot metrics. Returns gates + shot_frames."""
    t0 = time.time()
    gates: dict[int, dict] = {}
    shot_frames: dict[int, list] = {}
    n_frames = cfg.get("n_frames_per_shot", 16)

    for i, (s, e, _) in enumerate(shots, 1):
        n = min(n_frames, e - s + 1)
        idxs = np.linspace(s, e, n, dtype=int)
        frames = [frames_dict[idx] for idx in idxs if idx in frames_dict]
        if not frames:
            continue

        g: dict[str, Any] = {
            "motion_peak_div": motion_peak_div(frames),
            "ssim3d_self": ssim3d_self(frames),
            "flicker": flicker_score(frames),
            "ssim_long_range": ssim_long_range(frames),
            "flicker_hf_var": flicker_highfreq_variance(frames),
            "identity_intra": None,
            "identity_intra_dtw": None,
            "identity_intra_dtw_n": 0,
            "clip_temp_consistency": None,
        }

        if models.detector and models.embedder:
            g["identity_intra"] = identity_within_shot(
                frames, models.detector, models.embedder, n_samples=5
            )
            if cfg.get("compute_dtw_self", True):
                try:
                    from ..core.identity_dtw import identity_within_shot_dtw
                    res = identity_within_shot_dtw(
                        frames, models.detector, models.embedder, max_samples=8
                    )
                    g["identity_intra_dtw"] = res.normalized
                    g["identity_intra_dtw_n"] = res.n_a + res.n_b
                except Exception:
                    pass

        if models.clip_available:
            g["clip_temp_consistency"] = clip_semantic_consistency(
                frames, models.clip_model, models.clip_preprocess, models.device
            )

        gates[i] = g
        shot_frames[i] = frames

    timing.shot_processing = time.time() - t0
    return gates, shot_frames


def _compute_inter_coherence(
    shot_frames: dict[int, list],
    models: ModelBundle,
    timing: AuditTiming,
) -> list[dict]:
    """Stage 4: structure / style / semantic coherence between adjacent shots."""
    t0 = time.time()
    results = compute_inter_shot_coherence(
        shot_frames,
        models.detector,
        models.embedder,
        models.clip_scorer,
        subsample=5,
    )
    timing.inter_coherence = time.time() - t0
    return results


def _compute_composite(gates: dict[int, dict], timing: AuditTiming) -> None:
    """Stage 5: compute composite quality score for each shot (in-place)."""
    t0 = time.time()
    WEIGHTS = {
        "motion_peak_div": -1.0,
        "ssim3d_self": 1.0,
        "flicker": -1.0,
        "identity_intra": -1.0,
        "ssim_long_range": 1.0,
        "clip_temp_consistency": 1.0,
    }
    for g in gates.values():
        score = sum(
            w * v for k, w in WEIGHTS.items()
            if (v := g.get(k)) is not None
        )
        g["composite"] = float(score)
    timing.composite = time.time() - t0


def _persist_results(
    video_info: VideoInfo,
    metrics_data: dict,
    thresholds: dict,
    output_dir: Path,
    benchmark_dir: Optional[Path],
    benchmark_mode: bool,
    timing: AuditTiming,
) -> Path:
    """Stage 6: write data.json, dashboard.md, optional benchmark JSON."""
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize None → 0.0 for dashboard only (metrics_data keeps Nones)
    sanitised = json.loads(
        json.dumps(metrics_data, default=lambda x: 0.0 if x is None else x)
    )
    try:
        generate_intra_report(
            video_info.name, sanitised, output_dir.parent, thresholds
        )
    except Exception as exc:
        print(f"  ⚠️  Dashboard error: {exc}")
        (output_dir / "dashboard.md").write_text(
            f"# {video_info.name}\nError: {exc}", encoding="utf-8"
        )

    (output_dir / "data.json").write_text(
        json.dumps(metrics_data, indent=2, default=str), encoding="utf-8"
    )

    if benchmark_mode and benchmark_dir is not None:
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        gates = metrics_data.get("gates", {})
        bm = {
            "video_name": video_info.name,
            "duration_s": video_info.duration_s,
            "n_shots": len(gates),
            "composite_mean": float(np.mean(
                [g.get("composite", 0.0) or 0.0 for g in gates.values()]
            )) if gates else 0.0,
            "timestamp": time.time(),
        }
        (benchmark_dir / f"{video_info.name}_benchmark.json").write_text(
            json.dumps(bm, indent=2)
        )

    timing.persist = time.time() - t0
    return output_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def audit_video(
    video_path: str | Path,
    video_params: Optional[dict] = None,
    force_full_video: bool = False,
) -> tuple[dict, Path]:
    """Run the full CineInfini quality audit on a single video.

    Parameters
    ----------
    video_path : str or Path
    video_params : dict, optional
        Per-call overrides for CONFIG keys.
    force_full_video : bool
        If True, ignore max_duration_s.

    Returns
    -------
    metrics_data : dict
    report_dir : Path
    """
    timing = AuditTiming()
    video_path = Path(video_path)

    cfg: dict[str, Any] = {**CONFIG}
    cfg["thresholds"] = dict(CONFIG["thresholds"])
    if video_params:
        cfg.update({k: v for k, v in video_params.items() if k != "thresholds"})
        if "thresholds" in video_params:
            cfg["thresholds"].update(video_params["thresholds"])

    reports_root = REPORTS_DIR or Path.cwd() / "reports"
    output_dir = reports_root / "intra" / video_path.stem

    video_info = _load_video_info(video_path, timing)
    if force_full_video:
        cfg["max_duration_s"] = video_info.duration_s + 10

    t0 = time.time()
    shots = detect_shot_boundaries(
        video_path, cfg["max_duration_s"], cfg["shot_threshold"],
        cfg["min_shot_duration_s"], cfg["downsample_to"],
        cfg["adaptive_threshold"], cfg["threshold_percentile"], step=2,
    )
    timing.shot_detection = time.time() - t0

    t0 = time.time()
    frames_dict = extract_shot_frames_global(
        video_path, shots, cfg["n_frames_per_shot"], cfg["frame_resize"]
    )
    timing.frame_extraction = time.time() - t0
    print(f"  {len(shots)} shots | {len(frames_dict)} frames")

    models = _init_models(cfg["gpu_device"], cfg, timing)
    gates, shot_frames = _process_shots(shots, frames_dict, models, cfg, timing)
    inter_results = _compute_inter_coherence(shot_frames, models, timing)

    narrative_scores: Optional[list] = None
    if cfg.get("narrative_coherence") and models.dinov2_available:
        t0 = time.time()
        narrative_scores = compute_narrative_coherence(
            shot_frames, models.dinov2_model, models.dinov2_processor, models.device
        )
        timing.narrative = time.time() - t0

    _compute_composite(gates, timing)

    metrics_data: dict = {
        "gates": {str(k): v for k, v in gates.items()},
        "inter_results": inter_results,
        "narrative_coherence": narrative_scores,
        "n_shots": len(gates),
        "video": str(video_path),
        "params_used": cfg,
        "video_info": {
            "duration": video_info.duration_s,
            "resolution": list(cfg["frame_resize"]),
            "fps": video_info.fps,
        },
    }

    report_dir = _persist_results(
        video_info, metrics_data, cfg["thresholds"],
        output_dir, BENCHMARK_DIR,
        cfg.get("benchmark_mode", False), timing,
    )

    print(timing.report(video_info.name))
    return metrics_data, report_dir


def adaptive_multi_stage_audit(
    video_path: str | Path,
    force_full_video: bool = False,
) -> tuple[dict, Path]:
    """Two-pass adaptive audit. Full two-stage logic planned for v0.5.0."""
    return audit_video(video_path, force_full_video=force_full_video)


def generate_synthetic_video(
    name: str | Path,
    duration_s: float,
    fps: int,
    resolution: tuple[int, int],
    pattern: str = "circle",
) -> Path:
    """Generate a synthetic test video (circle / color_switch / noise)."""
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = resolution
    out = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    total = int(duration_s * fps)
    rng = np.random.default_rng(42)
    for idx in range(total):
        if pattern == "circle":
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            cx = int(w * (0.2 + 0.6 * idx / max(total - 1, 1)))
            cv2.circle(frame, (cx, h // 2), min(w, h) // 8, (0, 200, 255), -1)
        elif pattern == "color_switch":
            c = (0, 0, 200) if (idx // max(fps // 2, 1)) % 2 == 0 else (0, 200, 0)
            frame = np.full((h, w, 3), c, dtype=np.uint8)
        else:
            frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return path
