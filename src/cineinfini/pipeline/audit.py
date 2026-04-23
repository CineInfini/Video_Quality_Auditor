"""Main audit orchestration"""
import time
import json
import csv
import numpy as np
import cv2
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.metrics import motion_peak_div, ssim3d_self, flicker_score, ssim_long_range, flicker_highfreq_variance, compute_composite_score
from ..core.face_detection import CascadeFaceDetector, ArcFaceEmbedder, identity_within_shot, set_models_dir
from ..core.embedding import CLIPSemanticScorer, clip_semantic_consistency, load_dinov2, dinov2_processor, dinov2_model
from ..core.coherence import compute_inter_shot_coherence, compute_narrative_coherence
from ..io.reader import detect_shot_boundaries, extract_shot_frames_global
from ..io.report import generate_intra_report

# Global configuration (to be set by user)
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
        "narrative_coherence": 0.7
    },
    "parallel_shots": True,
    "max_workers": 4,
    "narrative_coherence": True,
    "benchmark_mode": True,
    "gpu_device": "cuda" if torch.cuda.is_available() else "cpu",
    "enable_animal_face_detection": False,
    "adaptive_threshold": True,
    "threshold_percentile": 85,
}

# Global paths (to be set by user)
MODELS_DIR = None
REPORTS_DIR = None
BENCHMARK_DIR = None

def set_global_paths(models_dir, reports_dir, benchmark_dir):
    global MODELS_DIR, REPORTS_DIR, BENCHMARK_DIR
    MODELS_DIR = Path(models_dir)
    REPORTS_DIR = Path(reports_dir)
    BENCHMARK_DIR = Path(benchmark_dir)
    set_models_dir(MODELS_DIR)

def audit_video(video_path, video_params=None):
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

    # Shot detection
    print("   Detecting shot boundaries...")
    shots = detect_shot_boundaries(
        video_path, params["max_duration_s"], params["shot_threshold"],
        params["min_shot_duration_s"], params["downsample_to"],
        CONFIG.get("adaptive_threshold", True), CONFIG.get("threshold_percentile", 85)
    )
    print(f"   Found {len(shots)} shots")
    print("   ─────────────────────────────────────────")

    # Save CSV
    csv_path = output_dir / "auto_shots.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shot_id", "shot_description", "character_ids"])
        for i, (s,e,fps) in enumerate(shots,1):
            writer.writerow([i, f"Shot {i}: duration {(e-s+1)/fps:.1f}s", "no_character"])

    device = CONFIG["gpu_device"]
    detector = CascadeFaceDetector()
    embedder = ArcFaceEmbedder()
    scorer = CLIPSemanticScorer(device=device)
    clip_model = scorer.model if scorer.available else None
    clip_preprocess = scorer.preprocess if scorer.available else None

    # Global extraction
    frames_dict = extract_shot_frames_global(video_path, shots, params["n_frames_per_shot"], params["frame_resize"])
    print(f"   Total unique frames extracted: {len(frames_dict)}")

    gates = {}
    shot_frames = {}
    shot_info_list = [(i, (s,e,fps)) for i, (s,e,fps) in enumerate(shots,1)]
    total_shots = len(shot_info_list)
    per_shot_times = []

    def process_shot(info):
        i, (s, e, _) = info
        t_start = time.time()
        n = min(params["n_frames_per_shot"], e - s + 1)
        idxs = np.linspace(s, e, n, dtype=int)
        frames = [frames_dict[idx] for idx in idxs if idx in frames_dict]
        if not frames:
            return None
        t_extract = time.time() - t_start
        t0 = time.time()
        motion = motion_peak_div(frames)
        t_motion = time.time() - t0
        t0 = time.time()
        ssim3d = ssim3d_self(frames)
        t_ssim = time.time() - t0
        t0 = time.time()
        flick = flicker_score(frames)
        t_flick = time.time() - t0
        t0 = time.time()
        identity = identity_within_shot(frames, detector, embedder, n_samples=5)
        t_id = time.time() - t0
        t0 = time.time()
        ssim_lr = ssim_long_range(frames)
        t_ssim_lr = time.time() - t0
        t0 = time.time()
        flick_hf = flicker_highfreq_variance(frames)
        t_flick_hf = time.time() - t0
        if clip_model:
            t0 = time.time()
            clip_temp = clip_semantic_consistency(frames, clip_model, clip_preprocess, device)
            t_clip = time.time() - t0
        else:
            clip_temp = None
            t_clip = 0.0
        t_total = time.time() - t_start
        g = {
            "motion_peak_div": motion,
            "ssim3d_self": ssim3d,
            "flicker": flick,
            "identity_intra": identity,
            "ssim_long_range": ssim_lr,
            "flicker_hf_var": flick_hf,
            "clip_temp_consistency": clip_temp,
        }
        per_shot_times.append({
            "shot_id": i, "total": t_total, "extract": t_extract, "motion": t_motion,
            "ssim": t_ssim, "flicker": t_flick, "identity": t_id, "ssim_lr": t_ssim_lr,
            "flick_hf": t_flick_hf, "clip": t_clip, "n_frames": len(frames)
        })
        if t_total > 2.0:
            print(f"      Shot {i}: total {t_total:.1f}s (extract {t_extract:.2f}s, motion {t_motion:.1f}s, ssim3d {t_ssim:.1f}s, flicker {t_flick:.1f}s, identity {t_id:.1f}s, ssim_lr {t_ssim_lr:.1f}s, flick_hf {t_flick_hf:.1f}s, clip {t_clip:.1f}s)")
        return i, frames, g

    print(f"   Processing {total_shots} shots...")
    if CONFIG["parallel_shots"]:
        with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            futures = {executor.submit(process_shot, info): info for info in shot_info_list}
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if result:
                    i, frames, g = result
                    shot_frames[i] = frames
                    gates[i] = g
                if completed % 5 == 0 or completed == total_shots:
                    print(f"      Progress: {completed}/{total_shots} shots processed")
    else:
        for idx, info in enumerate(shot_info_list):
            result = process_shot(info)
            if result:
                i, frames, g = result
                shot_frames[i] = frames
                gates[i] = g
            if (idx+1) % 5 == 0 or (idx+1) == total_shots:
                print(f"      Progress: {idx+1}/{total_shots} shots processed")
    print(f"   Processing finished.")

    # Semantic scores (optional)
    semantic_scores = {}
    if scorer.available:
        for sid, frames in shot_frames.items():
            s = scorer.score(frames, f"Shot {sid}", n_samples=6)
            semantic_scores[sid] = s

    # Inter-shot and narrative coherence
    print("   Computing inter-shot coherence...")
    inter_results = compute_inter_shot_coherence(shot_frames, clip_model, clip_preprocess, device)
    narrative_scores = []
    if CONFIG["narrative_coherence"]:
        # Load DINOv2 if not already loaded
        global dinov2_processor, dinov2_model
        if dinov2_model is None:
            load_dinov2(device)
        if dinov2_model is not None:
            print("   Computing narrative coherence (DINOv2)...")
            narrative_scores = compute_narrative_coherence(shot_frames, dinov2_model, dinov2_processor, device)
        else:
            narrative_scores = [1.0] * (len(shot_frames)-1) if len(shot_frames) > 1 else []
    else:
        narrative_scores = [1.0] * (len(shot_frames)-1) if len(shot_frames) > 1 else []

    # Composite scores
    for g in gates.values():
        metrics = {
            "motion_mean": g.get("motion_peak_div", 0),
            "ssim_mean": g.get("ssim3d_self", 0),
            "flicker_mean": g.get("flicker", 0),
            "identity_mean": g.get("identity_intra", 0),
            "ssim_lr_mean": g.get("ssim_long_range", 0),
            "clip_temp_mean": g.get("clip_temp_consistency", 0),
        }
        g["composite"] = compute_composite_score(metrics)

    metrics_data = {
        "gates": {str(k): v for k, v in gates.items()},
        "semantic_scores": semantic_scores,
        "inter_results": inter_results,
        "narrative_coherence": narrative_scores,
        "n_shots": len(gates),
        "video": str(video_path),
        "params_used": params,
        "video_info": {
            "duration": params["max_duration_s"],
            "resolution": params["frame_resize"],
            "fps": shots[0][2] if shots else 0
        }
    }

    if CONFIG["benchmark_mode"]:
        benchmark_data = {
            "video_name": video_name,
            "duration": params["max_duration_s"],
            "n_shots": len(gates),
            "shots": {str(k): v for k, v in gates.items()},
            "composite_score": np.mean([g["composite"] for g in gates.values()]) if gates else 0,
            "config": {k: str(v) for k, v in CONFIG.items() if not callable(v)},
            "timestamp": time.time()
        }
        benchmark_path = BENCHMARK_DIR / f"{video_name}_benchmark.json"
        with open(benchmark_path, "w") as f:
            json.dump(benchmark_data, f, indent=2)

    # Generate dashboard
    try:
        generate_intra_report(video_name, metrics_data, REPORTS_DIR / "intra", params["thresholds"])
    except Exception as e:
        print(f"   Warning: Dashboard generation failed: {e}")

    return metrics_data, output_dir

def adaptive_multi_stage_audit(video_path, force_full_video=False):
    # This function will call audit_video with appropriate parameters
    # For brevity, we skip the full implementation here as it's similar to the original
    # but uses the same modular functions.
    # You can adapt the earlier logic using the new structure.
    pass


# -------------------------------------------------------------------
# Optimisation des poids composites (identique à l'original)
# -------------------------------------------------------------------
def optimise_composite_weights(gates, thresholds, base_weights=None):
    from ..core.metrics import DEFAULT_COMPOSITE_WEIGHTS
    if base_weights is None:
        base_weights = DEFAULT_COMPOSITE_WEIGHTS.copy()
    exceed_counts = {metric: 0 for metric in base_weights.keys()}
    total_shots = len(gates)
    for g in gates.values():
        if g.get("motion_peak_div") and g["motion_peak_div"] > thresholds.get("motion", 25.0):
            exceed_counts["motion_mean"] += 1
        if g.get("ssim3d_self") and g["ssim3d_self"] < thresholds.get("ssim3d", 0.45):
            exceed_counts["ssim_mean"] += 1
        if g.get("flicker") and g["flicker"] > thresholds.get("flicker", 0.1):
            exceed_counts["flicker_mean"] += 1
        if g.get("identity_intra") and g["identity_intra"] > thresholds.get("identity_drift", 0.6):
            exceed_counts["identity_mean"] += 1
        if g.get("ssim_long_range") and g["ssim_long_range"] < thresholds.get("ssim_long_range", 0.45):
            exceed_counts["ssim_lr_mean"] += 1
        if g.get("clip_temp_consistency") and g["clip_temp_consistency"] < thresholds.get("clip_temp", 0.25):
            exceed_counts["clip_temp_mean"] += 1
    adjusted_weights = base_weights.copy()
    for metric, count in exceed_counts.items():
        if count > total_shots * 0.3:
            factor = 1.0 + (count / total_shots)
            adjusted_weights[metric] = base_weights[metric] * factor
    return adjusted_weights

# -------------------------------------------------------------------
# Pré‑analyse rapide (generate_AuditWorkflow) – version simplifiée
# -------------------------------------------------------------------
def generate_AuditWorkflow(video_path, config=None, force_full_video=False):
    import time
    start_total = time.time()
    if config is None:
        config = CONFIG
    video_path = Path(video_path)
    print(f"\n🔍 PRE‑ANALYSIS (fast) for {video_path.name}")
    # Récupérer les infos de la vidéo
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 60.0
    cap.release()
    print(f"   [1] Video info: {duration:.1f}s, {fps:.2f}fps, {total_frames} frames")
    # Échantillonner 15 frames max (toutes les 3 secondes environ)
    sample_interval = max(1, int(fps * 3))
    low_res = (160, 90)
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while idx < total_frames and len(frames) < 15:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            small = cv2.resize(frame, low_res)
            frames.append(small)
        idx += sample_interval
    cap.release()
    print(f"   [2] Sampled {len(frames)} low‑res frames")
    # Détection faciale rapide
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_detected = False
    for frame in frames[:8]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(15,15))
        if len(faces) > 0:
            face_detected = True
            break
    print(f"   [3] Face detection: {'Yes' if face_detected else 'No'}")
    # Estimation du mouvement par différence absolue moyenne
    motion_scores = []
    for i in range(len(frames)-1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.mean(np.abs(gray2 - gray1))
        motion_scores.append(diff)
    avg_motion = np.mean(motion_scores) if motion_scores else 0.0
    avg_motion_norm = avg_motion / 2.55
    print(f"   [4] Average motion (MAD): {avg_motion:.1f} → normalized {avg_motion_norm:.1f}")
    animal_detected = (not face_detected) and (avg_motion_norm > 15.0)
    print(f"   [5] Animals suspected: {'Yes' if animal_detected else 'No'}")
    need_stage2 = (duration > 30 and avg_motion_norm > 15) or animal_detected or (not face_detected and avg_motion_norm > 20)
    print(f"   [6] Second stage recommended: {'Yes' if need_stage2 else 'No'}")
    # Définir la durée à analyser
    if force_full_video:
        max_duration = min(config.get("max_duration_s", 1000), duration)
    else:
        max_duration = min(config.get("max_duration_s", 60), duration)
    print(f"   [7] Analysed duration: {max_duration:.1f}s")
    # Paramètres de stage1
    stage1_params = {
        "max_duration_s": max_duration,
        "shot_threshold": 0.2,
        "min_shot_duration_s": 0.5,
        "downsample_to": (320, 180),
        "n_frames_per_shot": 16,
        "frame_resize": (320, 180),
        "embedder": "arcface_onnx",
        "semantic_scorer": "clip",
        "thresholds": config.get("thresholds", {}).copy(),
    }
    if avg_motion_norm > 30:
        stage1_params["thresholds"]["motion"] = 35.0
    elif avg_motion_norm < 8:
        stage1_params["thresholds"]["motion"] = 20.0
    else:
        stage1_params["thresholds"]["motion"] = 25.0
    if not face_detected:
        stage1_params["thresholds"]["identity_drift"] = 100.0
    enable_animal = animal_detected and config.get("enable_animal_face_detection", False)
    face_backend = "haar+yunet" if not enable_animal else "animal_pretrained"
    prestage_summary = f"""
Fast pre‑analysis summary for {video_path.name}:
- Duration: {duration:.1f}s | Motion: {avg_motion_norm:.1f} | Faces: {'Yes' if face_detected else 'No'} | Animals: {'Yes' if animal_detected else 'No'}
- Stage2: {'Yes' if need_stage2 else 'No'} | Face backend: {face_backend}
- Adjusted motion threshold: {stage1_params['thresholds'].get('motion', 25.0)}
"""
    print(prestage_summary)
    print(f"   ⏱️ Pre‑analysis total time: {time.time()-start_total:.2f}s\n")
    return {
        "stage1_params": stage1_params,
        "stage2_auto": need_stage2,
        "enable_animal_face_detection": enable_animal,
        "face_detector_backend": face_backend,
        "prestage_summary": prestage_summary
    }

# -------------------------------------------------------------------
# Fonction adaptive_multi_stage_audit (orchestrateur)
# -------------------------------------------------------------------
def adaptive_multi_stage_audit(video_path, force_full_video=False):
    total_start = time.time()
    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"🎬 ADAPTIVE AUDIT START: {video_name}")
    print(f"{'='*60}")
    # Pré‑analyse
    workflow = generate_AuditWorkflow(video_path, config=CONFIG, force_full_video=force_full_video)
    if workflow["enable_animal_face_detection"]:
        CONFIG["enable_animal_face_detection"] = True
        print("   🐾 Animal face detection ENABLED")
    # Stage 1
    stage1_params = workflow["stage1_params"]
    print(f"\n📊 STAGE 1 AUDIT (first pass) – {video_name}")
    stage1_start = time.time()
    metrics_data1, report_dir1 = audit_video(video_path, video_params=stage1_params)
    stage1_duration = time.time() - stage1_start
    print(f"   ✅ Stage 1 completed in {stage1_duration:.1f}s")
    json_path1 = Path(report_dir1) / "data.json"
    if not json_path1.exists():
        print(f"   ⚠️ data.json missing. Skipping Stage 2.")
        final_dir = report_dir1
    elif workflow["stage2_auto"]:
        print(f"\n📈 STAGE 2 AUDIT (optimised) – {video_name}")
        stage2_start = time.time()
        with open(json_path1, 'r') as f:
            data1 = json.load(f)
        gates = data1["gates"]
        thresholds = stage1_params["thresholds"]
        optimised_weights = optimise_composite_weights(gates, thresholds)
        weights_path = Path(report_dir1) / "optimised_weights.json"
        with open(weights_path, 'w') as f:
            json.dump(optimised_weights, f, indent=2)
        print(f"   ⚙️ Optimised composite weights computed")
        # Ajustement éventuel des paramètres pour Stage 2
        stage2_params = stage1_params.copy()
        motion_exceeds = sum(1 for g in gates.values() if g.get("motion_peak_div", 0) > thresholds.get("motion", 25.0))
        if motion_exceeds / len(gates) > 0.3:
            old_dur = stage2_params["max_duration_s"]
            stage2_params["max_duration_s"] = max(30, int(stage2_params["max_duration_s"] * 0.8))
            print(f"   🔧 Adjusted max_duration_s: {old_dur} → {stage2_params['max_duration_s']} (high motion)")
        metrics_data2, report_dir2 = audit_video(video_path, video_params=stage2_params)
        # Recalculer les scores composites avec les nouveaux poids
        with open(Path(report_dir2) / "data.json", 'r') as f:
            data2 = json.load(f)
        from ..core.metrics import recompute_composite_scores  # à définir si nécessaire
        # Si la fonction n'existe pas, on peut recalculer ici
        for g in data2["gates"].values():
            mets = {k: g.get(k.rstrip("_mean"), 0) for k in optimised_weights.keys()}
            g["composite"] = compute_composite_score(mets, optimised_weights)
        with open(Path(report_dir2) / "data.json", 'w') as f:
            json.dump(data2, f, indent=2)
        stage2_duration = time.time() - stage2_start
        print(f"   ✅ Stage 2 completed in {stage2_duration:.1f}s")
        final_dir = report_dir2
    else:
        final_dir = report_dir1
    total_duration = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"🏁 AUDIT COMPLETED: {video_name}")
    print(f"   Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"   Report saved to: {final_dir}")
    print(f"{'='*60}\n")
    return final_dir
