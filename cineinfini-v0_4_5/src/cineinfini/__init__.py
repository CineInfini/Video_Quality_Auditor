"""CineInfini – Video Quality Audit Pipeline (v0.4.0)."""
from __future__ import annotations

__version__ = "0.4.0"
__author__ = "Salah-Eddine BENBRAHIM"
__license__ = "MIT"

from .pipeline.audit import (
    audit_video, adaptive_multi_stage_audit, generate_synthetic_video,
    CONFIG, set_global_paths,
    VideoInfo, ModelBundle, AuditTiming,
    _load_video_info, _init_models, _process_shots,
    _compute_inter_coherence, _compute_composite, _persist_results,
)
from .io.report import generate_intra_report, generate_inter_report
from .io.report_extended import (
    generate_extended_intra_report,
    figure_heatmap_shot_metric, figure_identity_trajectory,
    figure_verdict_timeline, figure_dtw_vs_mean,
    figure_inter_shot_matrix, figure_sparkline_grid,
)
from .core.metrics import compute_composite_score, recompute_composite_scores
from .core.embedding import load_dinov2, get_dinov2
from .core.identity_dtw import (
    identity_within_shot_dtw, identity_between_shots_dtw,
    identity_drift_compare, dtw_distance, dtw_available, IdentityDtwResult,
)
from .compare import compare_videos
from .benchmark import (
    audit_multiple_videos, run_benchmark, generate_test_dataset,
    benchmark_models, compare_multiple_videos,
)

def get_config() -> dict:
    return CONFIG.copy()

def set_config(key: str, value) -> None:
    if key not in CONFIG:
        raise KeyError(f"Unknown key: '{key}'")
    CONFIG[key] = value

__all__ = [
    "audit_video", "adaptive_multi_stage_audit", "generate_synthetic_video",
    "CONFIG", "set_global_paths", "get_config", "set_config",
    "VideoInfo", "ModelBundle", "AuditTiming",
    "_load_video_info", "_init_models", "_process_shots",
    "_compute_inter_coherence", "_compute_composite", "_persist_results",
    "generate_intra_report", "generate_inter_report",
    "generate_extended_intra_report",
    "figure_heatmap_shot_metric", "figure_identity_trajectory",
    "figure_verdict_timeline", "figure_dtw_vs_mean",
    "figure_inter_shot_matrix", "figure_sparkline_grid",
    "compute_composite_score", "recompute_composite_scores",
    "load_dinov2", "get_dinov2",
    "identity_within_shot_dtw", "identity_between_shots_dtw",
    "identity_drift_compare", "dtw_distance", "dtw_available", "IdentityDtwResult",
    "compare_videos", "audit_multiple_videos", "run_benchmark",
    "generate_test_dataset", "benchmark_models", "compare_multiple_videos",
]
