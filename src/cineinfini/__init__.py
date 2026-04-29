"""CineInfini – Adaptive Multi-Stage Video Quality Audit Pipeline."""
from __future__ import annotations

import logging as _logging

__version__ = "0.4.10.2"
__author__ = "Salah-Eddine BENBRAHIM"
__license__ = "MIT"

_log = _logging.getLogger("cineinfini")

# Lightweight imports first — these MUST work on any Python install with
# numpy + opencv + pyyaml. They are what `cineinfini bootstrap` needs.
from .core.config import (
    get_config, set_config, reset_config,
    default_config, test_config, load_config, save_config, Config,
    ConfigValidationError, merge_configs,
)
from .core.bootstrap import (
    bootstrap_all, ensure_models, ensure_test_videos, ensure_system_deps,
    check_system_dependency, print_report,
    AssetStatus, BootstrapReport,
)
from .core.calibrate import (
    calibrate_from_csv, grid_search_thresholds,
    logistic_regression_weights, bayesian_optimize_thresholds,
    CalibrationResult,
)
from .core.phase4_aggregator import (
    aggregate_shot_verdict, build_phase4_report, GateThresholds, ShotVerdict,
)
from .core.inter_shot_loss import InterShotCoherenceLoss, InterShotLossResult
from .core.prompt_engineering import ShotPrompt, build_prompt, build_all_prompts
from .core.shot_registry import ShotMetadata

# Heavy imports — depend on torch / open_clip / onnxruntime / weasyprint.
# Wrapped so a half-installed environment can still bootstrap.
try:
    from .pipeline.audit import (
        audit_video, adaptive_multi_stage_audit, generate_synthetic_video,
        CONFIG, set_global_paths, VideoInfo, ModelBundle, AuditTiming,
        _load_video_info, _init_models, _process_shots,
        _compute_inter_coherence, _compute_composite, _persist_results,
    )
    from .pipeline.orchestrator import run_audit
    from .pipeline.render_dispatch import dispatch as render_dispatch
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
    from .core.device_utils import (
        resolve_dtype, amp_enabled, autocast_context, inference_mode,
        release_vram, vram_usage_mb, batch_iter, effective_batch_size,
    )
    from .core.context import VideoContext, VideoInfoLite, ModelPool
    from .core.registry import (
        register_module, get_registry, get_active_modules,
        all_modules, reset_registry, ModuleEntry,
    )
    from .core.ui_registry import (
        register_renderer, get_ui_registry, get_active_renderers,
        all_renderers, reset_ui_registry, RendererEntry,
    )
    from .compare import compare_videos
    from .benchmark import (
        audit_multiple_videos, run_benchmark, generate_test_dataset,
        benchmark_models, compare_multiple_videos,
    )
    from . import modules
    from .io import renderers
    _PIPELINE_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    _log.warning(
        "CineInfini pipeline disabled: %s. "
        "`cineinfini bootstrap` still works; install torch / open_clip / "
        "onnxruntime to enable audits.", _e,
    )
    CONFIG = {"max_duration_s": 60}  # placeholder so set_global_paths doesn't crash
    _PIPELINE_AVAILABLE = False


def get_config_dict() -> dict:
    return CONFIG.copy() if isinstance(CONFIG, dict) else {}


def set_config_key(key: str, value) -> None:
    if not isinstance(CONFIG, dict) or key not in CONFIG:
        raise KeyError(f"Unknown CONFIG key: '{key}'")
    CONFIG[key] = value


__all__ = [
    "__version__",
    # Config
    "get_config", "set_config", "reset_config",
    "default_config", "test_config", "load_config", "save_config", "Config",
    "ConfigValidationError", "merge_configs",
    # Bootstrap (always available)
    "bootstrap_all", "ensure_models", "ensure_test_videos", "ensure_system_deps",
    "check_system_dependency", "print_report", "AssetStatus", "BootstrapReport",
    # Calibration
    "calibrate_from_csv", "grid_search_thresholds",
    "logistic_regression_weights", "bayesian_optimize_thresholds",
    "CalibrationResult",
    # Phase 4
    "aggregate_shot_verdict", "build_phase4_report",
    "GateThresholds", "ShotVerdict",
    "InterShotCoherenceLoss", "InterShotLossResult",
    "ShotPrompt", "build_prompt", "build_all_prompts", "ShotMetadata",
]
