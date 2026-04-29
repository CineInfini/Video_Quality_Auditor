"""CineInfini – Benchmark and multi-video audit utilities.

Wires the CLI ``cineinfini benchmark`` command into the modern
``BenchmarkRenderer`` (multi-video aggregation, HTML/MD/CSV/JSON outputs).
Per-video audits are full-power: the same orchestrator that drives
single-video audits, with all enabled modules and renderers.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cineinfini.core.config import get_config

logger = logging.getLogger("cineinfini.benchmark")


def audit_multiple_videos(
    video_paths: List[Union[str, Path]],
    output_subdir: str = "multi_audit",
    max_duration_s: int = 10,
    force_full_video: bool = False,
) -> Path:
    """Audit several videos and write an aggregated benchmark report.

    Each video gets its own per-video sub-directory with the standard set
    of renderers (data.json, dashboard.html, ...). At the top level we
    emit the cross-video report (``benchmark.html``, ``benchmark.json``,
    ``benchmark.csv``, ``benchmark.md``) which ranks videos by composite
    score and aggregates VideoScore axes / verdict distribution / module
    health across the whole batch.
    """
    from cineinfini.io.renderers.benchmark_renderer import run_benchmark_audit

    cfg = get_config()
    if not force_full_video:
        cfg.processing["max_duration_s"] = max_duration_s
    out_root = cfg.reports_dir() / output_subdir
    audits, bench_dir = run_benchmark_audit(
        video_paths, output_dir=out_root, force_full_video=force_full_video,
    )
    print(f"\n=== Benchmark complete ===")
    print(f"Audited {len(audits)} videos")
    print(f"Cross-video report:  {bench_dir}/benchmark.html")
    print(f"Per-video reports:   {out_root}/<video_stem>/dashboard.html")
    return bench_dir


def run_benchmark(video_path, output_file=None, repeats=3) -> Dict[str, Any]:
    """Time a single audit pipeline N times for steady-state measurement."""
    from cineinfini.pipeline.audit import audit_video

    video_path = Path(video_path)
    times = []
    for i in range(repeats):
        print(f"  Run {i+1}/{repeats}...")
        start = time.time()
        audit_video(str(video_path), video_params={"max_duration_s": 5},
                    force_full_video=False)
        times.append(time.time() - start)
    result = {
        "video": str(video_path),
        "repeats": repeats,
        "times": times,
        "mean_s": sum(times) / len(times),
        "min_s": min(times),
        "max_s": max(times),
    }
    if output_file:
        import json
        Path(output_file).write_text(json.dumps(result, indent=2))
    return result


def generate_test_dataset(output_dir, n_videos: int = 3, duration_s: int = 2,
                          fps: int = 24) -> list:
    """Generate small synthetic test videos for benchmarking.

    Used by tests and notebooks that need deterministic synthetic input.
    Each video has a moving square on a coloured background.
    """
    from cineinfini.pipeline.audit import generate_synthetic_video
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_videos):
        p = out / f"synth_{i:02d}.mp4"
        generate_synthetic_video(str(p), duration_s=duration_s, fps=fps,
                                  shape="circle" if i % 2 == 0 else "square")
        paths.append(p)
    return paths


def benchmark_models(*args, **kwargs):
    """Stub kept for backward-compatibility with older imports.

    Real model benchmarking is now in
    ``notebooks/05_benchmarking_competitors.ipynb``.
    """
    print("benchmark_models() is deprecated. Use the "
          "notebooks/05_benchmarking_competitors.ipynb notebook instead.")
    return {}


def compare_multiple_videos(*args, **kwargs):
    """Backward-compat alias. Delegates to ``audit_multiple_videos``."""
    return audit_multiple_videos(*args, **kwargs)
