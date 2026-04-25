"""
Core audit pipeline for CineInfini.
"""
import cv2
import numpy as np
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Tuple, List

from ..io.reader import detect_shot_boundaries, extract_shot_frames_global
from ..core.metrics import (
    motion_peak_div,
    ssim3d_self,
    flicker_score,
    ssim_long_range,
    flicker_highfreq_variance,
    clip_temp_consistency,
    compute_composite_score,
    recompute_composite_scores,
)
from ..core.face_detection import identity_within_shot

from ..core.coherence import compute_inter_shot_coherence
from ..core.embedding import CLIPSemanticScorer
from ..core.face_detection import get_face_detector, get_face_embedder, set_models_dir as set_fd_models_dir
from ..core.identity_dtw import identity_within_shot_dtw, dtw_available
from ..io.report import generate_intra_report

# Global configuration dictionary
CONFIG = {
    "shot_threshold": 0.25,
    "min_shot_duration_s": 0.5,
    "downsample_to": (320, 240),
    "adaptive_threshold": True,
    "threshold_percentile": 85,
    "n_frames_per_shot": 5,
    "frame_resize": (224, 224),
    "max_duration_s": 60,
    "inter_shot_subsample": 5,
    "narrative_coherence": True,
    "compute_dtw_self": True,
    "dtw_max_samples": 16,
    "num_workers": 4,
    "thresholds": {
        "motion": 25.0,
        "ssim3d": 0.45,
        "flicker": 0.1,
        "identity_drift": 0.6,
        "ssim_long_range": 0.45,
        "flicker_hf": 0.01,
        "clip_temp": 0.25,
    },
}

# Global paths (set by set_global_paths)
MODELS_DIR = None
REPORTS_DIR = None
BENCHMARK_DIR = None

def set_global_paths(models_dir: Path, reports_dir: Path, benchmark_dir: Path):
    global MODELS_DIR, REPORTS_DIR, BENCHMARK_DIR
    MODELS_DIR = Path(models_dir)
    REPORTS_DIR = Path(reports_dir)
    BENCHMARK_DIR = Path(benchmark_dir)
    set_fd_models_dir(MODELS_DIR)

def set_models_dir(models_dir: Path):
    global MODELS_DIR
    MODELS_DIR = Path(models_dir)
    set_fd_models_dir(MODELS_DIR)

def _process_shot(shot_idx, shot_frames, detector, embedder, clip_scorer, config):
    """Process a single shot: compute all intra-shot metrics."""
    # Mean-based identity drift
    identity_intra = identity_within_shot(shot_frames, detector, embedder, config["n_frames_per_shot"])

    # DTW-based self-coherence (new in 0.2.0)
    identity_intra_dtw = None
    if config.get("compute_dtw_self", False):
        dtw_res = identity_within_shot_dtw(shot_frames, detector, embedder, config["dtw_max_samples"])
        identity_intra_dtw = dtw_res.normalized if dtw_res.normalized is not None else 0.0

    metrics = {
        "motion_peak_div": motion_peak_div(shot_frames),
        "ssim3d_self": ssim3d_self(shot_frames),
        "flicker": flicker_score(shot_frames),
        "identity_intra": identity_intra,
        "identity_intra_dtw": identity_intra_dtw,
        "ssim_long_range": ssim_long_range(shot_frames),
        "flicker_hf_var": flicker_highfreq_variance(shot_frames),
        "clip_temp_consistency": clip_temp_consistency(shot_frames, clip_scorer),
    }
    return shot_idx, metrics

def audit_video(
    video_path: str,
    video_params: Optional[Dict[str, Any]] = None,
    force_full_video: bool = False,
) -> Tuple[Dict[str, Any], Path]:
    """
    Run a full quality audit on a video.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    video_params : dict, optional
        Override configuration parameters for this audit.
    force_full_video : bool
        If True, ignore max_duration_s and analyze the whole video.

    Returns
    -------
    metrics_data : dict
        Dictionary containing per-shot metrics, inter-shot coherence, etc.
    report_dir : Path
        Directory where the report (dashboard.md, data.json, figures) is saved.
    """
    if video_params:
        cfg = {**CONFIG, **video_params}
    else:
        cfg = CONFIG.copy()

    if force_full_video:
        cfg["max_duration_s"] = 999999

    # 1. Shot detection
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        fps = 24.0
    shots = detect_shot_boundaries(
        video_path,
        cfg["max_duration_s"],
        cfg["shot_threshold"],
        cfg["min_shot_duration_s"],
        cfg["downsample_to"],
        adaptive_threshold=cfg["adaptive_threshold"],
        threshold_percentile=cfg["threshold_percentile"],
        step=2,
    )

    # 2. Extract frames for each shot
    frame_resize = cfg["frame_resize"]
    n_frames_per_shot = cfg["n_frames_per_shot"]
    shot_frames_dict = extract_shot_frames_global(video_path, shots, n_frames_per_shot, frame_resize)

    # 3. Initialize models
    detector = get_face_detector()
    embedder = get_face_embedder()
    clip_scorer = CLIPSemanticScorer()

    # 4. Process shots in parallel
    metrics_by_shot = {}
    with ThreadPoolExecutor(max_workers=cfg["num_workers"]) as executor:
        futures = {}
        for idx, (s, e, _) in enumerate(shots):
            frames = shot_frames_dict.get(idx, [])
            if not frames:
                continue
            fut = executor.submit(_process_shot, idx, frames, detector, embedder, clip_scorer, cfg)
            futures[fut] = idx
        for fut in as_completed(futures):
            idx, metrics = fut.result()
            metrics_by_shot[idx] = metrics

    # Build gates dictionary (with DTW field)
    gates = {}
    for idx, (s, e, _) in enumerate(shots):
        m = metrics_by_shot.get(idx, {})
        gates[str(idx)] = {
            "start_frame": s,
            "end_frame": e,
            "motion_peak_div": m.get("motion_peak_div"),
            "ssim3d_self": m.get("ssim3d_self"),
            "flicker": m.get("flicker"),
            "identity_intra": m.get("identity_intra"),
            "identity_intra_dtw": m.get("identity_intra_dtw"),
            "ssim_long_range": m.get("ssim_long_range"),
            "flicker_hf_var": m.get("flicker_hf_var"),
            "clip_temp_consistency": m.get("clip_temp_consistency"),
        }

    # 5. Inter-shot coherence (only if at least 2 shots)
    inter_results = []
    if len(shots) >= 2:
        shot_repr_frames = []
        for idx, (s, e, _) in enumerate(shots):
            frames = shot_frames_dict.get(idx, [])
            if frames:
                shot_repr_frames.append(frames[0])
            else:
                shot_repr_frames.append(None)
        inter_results = compute_inter_shot_coherence(
            shot_repr_frames, detector, embedder, clip_scorer, cfg["inter_shot_subsample"]
        )

    # 6. Narrative coherence (optional)
    narrative_coherence = None
    if cfg["narrative_coherence"]:
        from ..core.embedding import get_dinov2
        dinov2 = get_dinov2()
        if dinov2:
            embeddings = []
            for idx, (s, e, _) in enumerate(shots):
                frames = shot_frames_dict.get(idx, [])
                if frames:
                    emb = dinov2.extract(frames[0])
                    embeddings.append(emb)
                else:
                    embeddings.append(None)
            if len(embeddings) >= 2:
                from sklearn.metrics.pairwise import cosine_similarity
                sims = []
                for i in range(len(embeddings)-1):
                    if embeddings[i] is not None and embeddings[i+1] is not None:
                        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0,0]
                        sims.append(float(sim))
                    else:
                        sims.append(0.0)
                narrative_coherence = np.mean(sims) if sims else 0.0

    # 7. Composite scores (mean and DTW)
    composite_mean = compute_composite_score(gates, cfg["thresholds"], identity_key="identity_intra")
    composite_dtw = compute_composite_score(gates, cfg["thresholds"], identity_key="identity_intra_dtw") if cfg.get("compute_dtw_self", False) else None

    # 8. Prepare final metrics data
    video_name = Path(video_path).stem
    metrics_data = {
        "video_info": {
            "path": str(video_path),
            "duration": cfg["max_duration_s"],
            "fps": fps,
            "num_shots": len(shots),
        },
        "params_used": cfg,
        "gates": gates,
        "inter_results": inter_results,
        "narrative_coherence": narrative_coherence,
        "composite_scores": {
            "mean_based": composite_mean,
            "dtw_based": composite_dtw,
        },
    }

    # 9. Generate report
    report_dir = REPORTS_DIR / "intra" / video_name
    generate_intra_report(video_name, metrics_data, REPORTS_DIR / "intra", cfg["thresholds"])

    return metrics_data, report_dir / video_name

def adaptive_multi_stage_audit(video_path: str, force_full_video: bool = False) -> Path:
    """Two-stage optimisation placeholder."""
    metrics, report_dir = audit_video(video_path, force_full_video=force_full_video)
    return report_dir


def generate_synthetic_video(name, duration_s, fps, resolution, pattern="circle"):
    """Generate a synthetic test video."""
    import cv2
    import numpy as np
    from pathlib import Path
    out_path = Path(name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, resolution)
    w, h = resolution
    for t in range(int(duration_s * fps)):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        if pattern == "circle":
            center = (int(w//2 + 100*np.sin(t*0.1)), int(h//2 + 50*np.cos(t*0.2)))
            cv2.circle(frame, center, 30, (0,255,0), -1)
        elif pattern == "color_switch":
            color = [(255,0,0), (0,255,0), (0,0,255)][(t//30) % 3]
            frame[:] = color
        elif pattern == "noise":
            frame = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8) + 100
        out.write(frame)
    out.release()
    return out_path
