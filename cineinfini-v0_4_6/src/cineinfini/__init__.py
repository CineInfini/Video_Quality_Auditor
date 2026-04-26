"""CineInfini – Video Quality Audit Pipeline (v0.4.6)."""
from __future__ import annotations

__version__ = "0.4.6"
__author__ = "Salah-Eddine BENBRAHIM"
__license__ = "MIT"

from .pipeline.audit import (
    audit_video, adaptive_multi_stage_audit, generate_synthetic_video,
    CONFIG, set_global_paths,
    VideoInfo, ModelBundle, AuditTiming,
    _load_video_info, _init_models, _process_shots,
    _compute_inter_coherence, _compute_composite, _persist_results,
)
from .io.report import (
    generate_intra_report, generate_inter_report,
    _load_audit_data, _aggregate_per_video_metrics,
)
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
from .core.config import (
    get_config, set_config, reset_config, default_config, test_config,
    load_config, save_config, Config,
)
from .core.calibrate import (
    calibrate_from_csv, grid_search_thresholds,
    logistic_regression_weights, bayesian_optimize_thresholds,
    CalibrationResult,
)
from .core.phase4_aggregator import (
    aggregate_shot_verdict, build_phase4_report,
    GateThresholds, ShotVerdict,
)
from .core.inter_shot_loss import (
    InterShotCoherenceLoss, InterShotLossResult,
)
from .core.prompt_engineering import (
    ShotPrompt, build_prompt, build_all_prompts,
)
from .core.shot_registry import ShotMetadata
from .compare import compare_videos
from .benchmark import (
    audit_multiple_videos, run_benchmark, generate_test_dataset,
    benchmark_models, compare_multiple_videos,
)


def get_config_dict() -> dict:
    return CONFIG.copy()


def set_config_key(key: str, value) -> None:
    if key not in CONFIG:
        raise KeyError(f"Unknown CONFIG key: '{key}'")
    CONFIG[key] = value


__all__ = [
    "__version__",
    # Pipeline
    "audit_video", "adaptive_multi_stage_audit", "generate_synthetic_video",
    "CONFIG", "set_global_paths",
    "VideoInfo", "ModelBundle", "AuditTiming",
    "_load_video_info", "_init_models", "_process_shots",
    "_compute_inter_coherence", "_compute_composite", "_persist_results",
    # Reports
    "generate_intra_report", "generate_inter_report",
    "_load_audit_data", "_aggregate_per_video_metrics",
    "generate_extended_intra_report",
    "figure_heatmap_shot_metric", "figure_identity_trajectory",
    "figure_verdict_timeline", "figure_dtw_vs_mean",
    "figure_inter_shot_matrix", "figure_sparkline_grid",
    # Metrics
    "compute_composite_score", "recompute_composite_scores",
    # Embedding
    "load_dinov2", "get_dinov2",
    # DTW
    "identity_within_shot_dtw", "identity_between_shots_dtw",
    "identity_drift_compare", "dtw_distance", "dtw_available", "IdentityDtwResult",
    # Config (v0.4.5)
    "get_config", "set_config", "reset_config", "default_config", "test_config",
    "load_config", "save_config", "Config",
    "get_config_dict", "set_config_key",
    # Calibration (v0.4.5)
    "calibrate_from_csv", "grid_search_thresholds",
    "logistic_regression_weights", "bayesian_optimize_thresholds",
    "CalibrationResult",
    # Verdicts (v0.4.6)
    "aggregate_shot_verdict", "build_phase4_report",
    "GateThresholds", "ShotVerdict",
    # Inter-shot loss (v0.4.6)
    "InterShotCoherenceLoss", "InterShotLossResult",
    # Prompt engineering (v0.4.6)
    "ShotPrompt", "build_prompt", "build_all_prompts", "ShotMetadata",
    # Multi-video
    "compare_videos", "audit_multiple_videos", "run_benchmark",
    "generate_test_dataset", "benchmark_models", "compare_multiple_videos",
]
