"""
CineInfini – Video Quality Audit Pipeline
"""

__version__ = "0.4.6"
__author__ = "Salah-Eddine BENBRAHIM"
__license__ = "MIT"

from .pipeline.audit import (
    audit_video,
    adaptive_multi_stage_audit,
    generate_synthetic_video,
    CONFIG,
    set_global_paths,
)

from .io.report import (
    generate_intra_report,
    generate_inter_report,
)

from .core.metrics import (
    compute_composite_score,
    recompute_composite_scores,
)

from .core.embedding import (
    load_dinov2,
    get_dinov2,
)

# Advanced utilities
from .compare import compare_videos
from .benchmark import (
    audit_multiple_videos,
    run_benchmark,
    generate_test_dataset,
    benchmark_models,
    compare_multiple_videos,
)

# DTW identity coherence (added in 0.2.0)
from .core.identity_dtw import (
    identity_within_shot_dtw,
    identity_between_shots_dtw,
    identity_drift_compare,
    dtw_distance,
    dtw_available,
    IdentityDtwResult,
)

def get_config():
    return CONFIG.copy()

def set_config(key: str, value):
    if key in CONFIG:
        CONFIG[key] = value
    else:
        raise KeyError(f"Unknown configuration key: {key}")

__all__ = [
    "audit_video",
    "adaptive_multi_stage_audit",
    "generate_synthetic_video",
    "generate_intra_report",
    "generate_inter_report",
    "get_config",
    "set_config",
    "set_global_paths",
    "compute_composite_score",
    "recompute_composite_scores",
    "load_dinov2",
    "get_dinov2",
    "CONFIG",
    "compare_videos",
    "audit_multiple_videos",
    "run_benchmark",
    "generate_test_dataset",
    "benchmark_models",
    "compare_multiple_videos",
    "identity_within_shot_dtw",
    "identity_between_shots_dtw",
    "identity_drift_compare",
    "dtw_distance",
    "dtw_available",
    "IdentityDtwResult",
]
