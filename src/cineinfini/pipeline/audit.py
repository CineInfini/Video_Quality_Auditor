# ===================================================================
# Main audit pipeline: shot detection, frame extraction, metric computation,
# report generation, and multi-stage audit.
# ===================================================================
import time
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.metrics import (
    motion_peak_div,
    ssim3d_self,
    flicker_score,
    ssim_long_range,
    flicker_highfreq_variance,
    compute_composite_score,
    recompute_composite_scores,
)
from ..core.face_detection import (
    CascadeFaceDetector,
    ArcFaceEmbedder,
    identity_within_shot,
)
from ..core.embedding import (
    CLIPSemanticScorer,
    clip_semantic_consistency,
    load_dinov2,
    get_dinov2,
)
from ..core.coherence import (
    compute_inter_shot_coherence,
    compute_narrative_coherence,
)
from ..io.reader import detect_shot_boundaries, extract_shot_frames_global
from ..io.report import generate_intra_report

# Global configuration (to be set by the user or loaded from CONFIG)
CONFIG = {
    "max_duration_s": 60,
    "shot_threshold": 0.2,
    "min_shot_duration_s": 0.5,
    "downsample_to": (320, 180),
    "n_frames_per_shot": 16,
    "frame_resize": (320, 180),
    "embedder": "arcface_onnx",
    "semantic_scorer": "clip",
    "thresholds": {
        "motion": 25.0, "ssim3d": 0.45, "flicker": 0.1, "identity_drift": 0.6,
        "ssim_long_range": 0.45, "clip_temp": 0.25, "flicker_hf": 0.01,
        "narrative_coherence": 0.7,
    },
    "parallel_shots": True,
    "max_workers": 4,
    "narrative_coherence": True,
    "benchmark_mode": True,
    "gpu_device": "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
    "enable_animal_face_detection": False,
    "adaptive_threshold": True,
    "threshold_percentile": 85,
}

# Global paths (to be set by set_global_paths)
REPORTS_DIR = None
BENCHMARK_DIR = None


def set_global_paths(reports_dir, benchmark_dir):
    global REPORTS_DIR, BENCHMARK_DIR
    REPORTS_DIR = Path(reports_dir)
    BENCHMARK_DIR = Path(benchmark_dir)


def audit_video(video_path, video_params=None, force_full_video=False):
    """
    Main audit function.

    Parameters
    ----------
    video_path : str or Path
        Path to the input video.
    video_params : dict, optional
        Override default parameters (max_duration_s, thresholds, etc.).
    force_full_video : bool
        If True, analyse the entire video (ignoring max_duration_s).

    Returns
    -------
    metrics_data : dict
        Dictionary with per-shot metrics, coherence scores, etc.
    report_dir : Path
        Directory where the JSON and Markdown dashboard were saved.
    """
    start_global = time.time()
    params = {
        "max_duration_s": CONFIG["max_duration_s"],
        "shot_threshold": CONFIG["shot_threshold"],
        "min_shot_duration_s": CONFIG["min_shot_duration_s"],
        "downsample_to": CONFIG["downsample_to"],
        "n_frames_per_shot": CONFIG["n_frames_per_shot"],
        "frame_resize": CONFIG["frame_resize"],
        "embedder": CONFIG["embedder"],
        "semantic_scorer": CONFIG["semantic_scorer"],
        "thresholds": CONFIG["thresholds"].copy(),
    }
    if video_params:
        params.update(video_params)
        if "thresholds" in video_params:
            params["thresholds"].update(video_params["thresholds"])

    video_path = Path(video_path)
    video_name = video_path.stem
    output_dir = REPORTS_DIR / "intra" / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Video duration (real) -----
    t0 = time.time()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    real_duration = total_frames / fps
    cap.release()
    video_info_time = time.time() - t0
    print(f"  [TIMING] Video info read: {video_info_time:.2f}s")

    if force_full_video:
        params["max_duration_s"] = real_duration + 10
        print(f"   Forced full video analysis: {params['max_duration_s']:.1f}s")

    # ----- Shot detection -----
    print("   Detecting shot boundaries...")
    t0 = time.time()
    shots = detect_shot_boundaries(
        video_path,
        params["max_duration_s"],
        params["shot_threshold"],
        params["min_shot_duration_s"],
        params["downsample_to"],
        CONFIG["adaptive_threshold"],
        CONFIG["threshold_percentile"],
        step=2,  # optimisation
    )
    shot_detection_time = time.time() - t0
    print(f"  [TIMING] Shot detection: {shot_detection_time:.2f}s for {len(shots)} shots")

    # ----- Global frame extraction -----
    print("   Extracting frames...")
    t0 = time.time()
    frames_dict = extract_shot_frames_global(
        video_path, shots, params["n_frames_per_shot"], params["frame_resize"]
    )
    frame_extraction_time = time.time() - t0
    print(f"  [TIMING] Frame extraction: {frame_extraction_time:.2f}s, {len(frames_dict)} unique frames")

    # ----- Model initialization (face, CLIP) -----
    device = CONFIG["gpu_device"]
    t0 = time.time()
    detector = CascadeFaceDetector()
    embedder = ArcFaceEmbedder()
    scorer = CLIPSemanticScorer(device=device)
    clip_model = scorer.model if scorer.available else None
    clip_preprocess = scorer.preprocess if scorer.available else None
    model_init_time = time.time() - t0
    print(f"  [TIMING] Model initialization: {model_init_time:.2f}s")

    # ----- Initialize DINOv2 if requested -----
    dinov2_proc = None
    dinov2_mod = None
    if CONFIG["narrative_coherence"]:
        if load_dinov2(device):
            dinov2_proc, dinov2_mod = get_dinov2()
            print("  ✅ DINOv2 loaded")
        else:
            print("  ⚠️ DINOv2 could not be loaded. Narrative coherence disabled.")
            CONFIG["narrative_coherence"] = False

    # ----- Process each shot -----
    gates = {}
    shot_frames = {}
    total_shots = len(shots)
    print(f"   Processing {total_shots} shots...")
    start_shots = time.time()
    for i, (s, e, fps_shot) in enumerate(shots, 1):
        t_shot_start = time.time()
        n = min(params["n_frames_per_shot"], e - s + 1)
        idxs = np.linspace(s, e, n, dtype=int)
        frames = [frames_dict[idx] for idx in idxs if idx in frames_dict]
        if not frames:
            continue
        t_extract = time.time() - t_shot_start
        t_metrics_start = time.time()
        g = {
            "motion_peak_div": motion_peak_div(frames),
            "ssim3d_self": ssim3d_self(frames),
            "flicker": flicker_score(frames),
            "identity_intra": identity_within_shot(frames, detector, embedder, n_samples=5),
            "ssim_long_range": ssim_long_range(frames),
            "flicker_hf_var": flicker_highfreq_variance(frames),
        }
        if clip_model:
            g["clip_temp_consistency"] = clip_semantic_consistency(
                frames, clip_model, clip_preprocess, device
            )
        else:
            g["clip_temp_consistency"] = None
        t_metrics = time.time() - t_metrics_start
        gates[i] = g
        shot_frames[i] = frames
        if i % 10 == 0 or i == total_shots:
            print(f"      Shot {i}/{total_shots}: extract={t_extract:.3f}s, metrics={t_metrics:.3f}s, total={time.time()-t_shot_start:.3f}s")
    shot_processing_time = time.time() - start_shots
    print(f"  [TIMING] All shots processed: {shot_processing_time:.2f}s")

    # ----- Inter-shot coherence -----
    t0 = time.time()
    inter_results = compute_inter_shot_coherence(shot_frames, clip_model, clip_preprocess, device)
    inter_time = time.time() - t0
    print(f"  [TIMING] Inter-shot coherence: {inter_time:.2f}s")

    # ----- Narrative coherence (DINOv2) -----
    narrative_scores = None
    narrative_time = 0
    if CONFIG["narrative_coherence"] and dinov2_mod is not None:
        t0 = time.time()
        print("   Computing narrative coherence (DINOv2)...")
        narrative_scores = compute_narrative_coherence(shot_frames, dinov2_mod, dinov2_proc, device)
        narrative_time = time.time() - t0
        print(f"  [TIMING] Narrative coherence: {narrative_time:.2f}s")

    # ----- Composite scores -----
    t0 = time.time()
    DEFAULT_WEIGHTS = {
        "motion_mean": -1.0,
        "ssim_mean": 1.0,
        "flicker_mean": -1.0,
        "identity_mean": -1.0,
        "ssim_lr_mean": 1.0,
        "clip_temp_mean": 1.0,
    }
    for g in gates.values():
        metrics = {
            "motion_mean": g.get("motion_peak_div", 0),
            "ssim_mean": g.get("ssim3d_self", 0),
            "flicker_mean": g.get("flicker", 0),
            "identity_mean": g.get("identity_intra", 0),
            "ssim_lr_mean": g.get("ssim_long_range", 0),
            "clip_temp_mean": g.get("clip_temp_consistency", 0),
        }
        score = 0.0
        for k, w in DEFAULT_WEIGHTS.items():
            if k in metrics and metrics[k] is not None:
                score += w * metrics[k]
        g["composite"] = score
    composite_time = time.time() - t0
    print(f"  [TIMING] Composite scores: {composite_time:.2f}s")

    # ----- Build metrics data structure -----
    metrics_data = {
        "gates": {str(k): v for k, v in gates.items()},
        "inter_results": inter_results,
        "narrative_coherence": narrative_scores,
        "n_shots": len(gates),
        "video": str(video_path),
        "params_used": params,
        "video_info": {
            "duration": params["max_duration_s"],
            "resolution": params["frame_resize"],
            "fps": shots[0][2] if shots else 0,
        },
    }

    # ----- Dashboard generation -----
    t0 = time.time()
    try:
        generate_intra_report(video_name, metrics_data, REPORTS_DIR / "intra", params["thresholds"])
        # Clean None values before dashboard
        for shot in metrics_data["gates"].values():
            for key in ["motion_peak_div", "ssim3d_self", "flicker", "identity_intra",
                         "ssim_long_range", "flicker_hf_var", "clip_temp_consistency"]:
                if shot.get(key) is None:
                    shot[key] = 0.0
        dashboard_time = time.time() - t0
        print(f"  [TIMING] Dashboard generation: {dashboard_time:.2f}s")
    except Exception as e:
        print(f"⚠️ Dashboard generation failed: {e}")
        (output_dir / "dashboard.md").write_text(f"# Rapport minimal pour {video_name}\nErreur: {e}", encoding="utf-8")
        dashboard_time = 0

    # ----- Save JSON -----
    (output_dir / "data.json").write_text(json.dumps(metrics_data, indent=2))

    # ----- Benchmark export (optional) -----
    if CONFIG["benchmark_mode"] and BENCHMARK_DIR is not None:
        benchmark_data = {
            "video_name": video_name,
            "duration": params["max_duration_s"],
            "n_shots": len(gates),
            "composite_score": np.mean([g["composite"] for g in gates.values()]) if gates else 0,
            "timestamp": time.time(),
        }
        benchmark_path = BENCHMARK_DIR / f"{video_name}_benchmark.json"
        benchmark_path.write_text(json.dumps(benchmark_data, indent=2))

    total_time = time.time() - start_global

    # ----- Timing summary -----
    print(f"\n📊 TIMING SUMMARY for {video_name}:")
    print(f"   video_info        : {video_info_time:7.2f}s")
    print(f"   shot detection    : {shot_detection_time:7.2f}s")
    print(f"   frame extraction  : {frame_extraction_time:7.2f}s")
    print(f"   model init        : {model_init_time:7.2f}s")
    print(f"   shot processing   : {shot_processing_time:7.2f}s")
    print(f"   inter coherence   : {inter_time:7.2f}s")
    print(f"   narrative coherence: {narrative_time:7.2f}s")
    print(f"   composite scores  : {composite_time:7.2f}s")
    print(f"   dashboard generation: {dashboard_time:7.2f}s")
    print(f"   TOTAL             : {total_time:7.2f}s ({total_time/60:.2f} minutes)")

    return metrics_data, output_dir


def adaptive_multi_stage_audit(video_path, force_full_video=False):
    """
    Simple wrapper around audit_video. In future versions this will implement
    a proper two‑stage adaptive audit (weight optimisation, parameter tuning).
    """
    total_start = time.time()
    video_name = Path(video_path).stem
    print(f"\n🎬 ADAPTIVE AUDIT START: {video_name}")
    print(f"   force_full_video = {force_full_video}")
    print("   ➤ Calling audit_video()...")
    metrics, report_dir = audit_video(video_path, force_full_video=force_full_video)
    total_duration = time.time() - total_start
    print(f"\n🏁 ADAPTIVE AUDIT COMPLETED in {total_duration:.1f}s ({total_duration/60:.2f} minutes)")
    print(f"   Report saved to: {report_dir}\n")
    return metrics, report_dir

def generate_synthetic_video(name, duration_s, fps, resolution, pattern="circle"):
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
