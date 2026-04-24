"""
CineInfini – Benchmark and multi-video audit utilities.
"""
import json, time, numpy as np
from pathlib import Path
from typing import List, Union, Dict, Any
from cineinfini import audit_video, generate_synthetic_video
from cineinfini.io.report import generate_inter_report
from cineinfini.pipeline.audit import CONFIG, REPORTS_DIR

def audit_multiple_videos(video_paths, output_subdir="multi_audit", max_duration_s=10, force_full_video=False) -> Path:
    reports = []
    for i, vp in enumerate(video_paths):
        vp = Path(vp)
        print(f"[{i+1}/{len(video_paths)}] Auditing {vp.name}...")
        _, rd = audit_video(str(vp), video_params={} if force_full_video else {"max_duration_s": max_duration_s}, force_full_video=force_full_video)
        reports.append(rd)
    inter_root = (REPORTS_DIR / "inter") if REPORTS_DIR else Path.cwd() / "reports/inter"
    inter_root.mkdir(parents=True, exist_ok=True)
    generate_inter_report(reports, inter_root, CONFIG["thresholds"], output_subdir)
    return inter_root / output_subdir

def run_benchmark(video_path, output_file=None, repeats=3) -> Dict[str, Any]:
    video_path = Path(video_path)
    times = []
    for i in range(repeats):
        print(f"  Run {i+1}/{repeats}...")
        start = time.time()
        audit_video(str(video_path), video_params={"max_duration_s": 5}, force_full_video=False)
        times.append(time.time()-start)
    mean_time, std_time = float(np.mean(times)), float(np.std(times))
    result = {"video": str(video_path), "repeats": repeats, "mean_duration_s": mean_time, "std_duration_s": std_time, "min_s": min(times), "max_s": max(times), "times": times}
    print(f"⏱️  Mean time: {mean_time:.2f}s (±{std_time:.2f})")
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f: json.dump(result, f, indent=2)
    return result

def generate_test_dataset(output_dir, num_videos=5, duration_s=2, fps=24, resolution=(320,240), patterns=None) -> List[Path]:
    patterns = patterns or ["circle", "color_switch", "noise"]
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    videos = []
    for i in range(num_videos):
        pattern = patterns[i % len(patterns)]
        name = output_dir / f"test_video_{i+1}_{pattern}.mp4"
        generate_synthetic_video(str(name), duration_s, fps, resolution, pattern)
        videos.append(name)
        print(f"✅ Created: {name}")
    return videos

def benchmark_models(video_path, model_names=None, output_file=None) -> Dict[str, float]:
    from cineinfini.core.embedding import load_dinov2, get_dinov2, CLIPSemanticScorer
    from cineinfini.core.face_detection import ArcFaceEmbedder, CascadeFaceDetector
    model_names = model_names or ["dinov2", "clip", "arcface", "yunet"]
    results = {}
    for model in model_names:
        print(f"Benchmarking {model}..."); start = time.time()
        try:
            if model == "dinov2": load_dinov2("cpu"); get_dinov2()
            elif model == "clip": scorer = CLIPSemanticScorer(device="cpu"); scorer.score([], "dummy")
            elif model == "arcface": embedder = ArcFaceEmbedder()
            elif model == "yunet": detector = CascadeFaceDetector()
            else: continue
        except Exception as e:
            print(f"  ⚠ {model} failed: {e}"); results[model] = float("nan"); continue
        results[model] = time.time()-start
        print(f"  -> {results[model]:.2f}s")
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f: json.dump(result, f, indent=2)
    return results

def compare_multiple_videos(video_list, output_subdir="comparison_group", max_duration_s=10) -> Path:
    return audit_multiple_videos(video_list, output_subdir, max_duration_s)
