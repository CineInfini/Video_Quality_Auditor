"""
CineInfini - Adaptive Multi‑Stage Video Quality Audit Pipeline

This module provides tools for automatic video quality auditing,
including shot detection, metric computation, and report generation.
"""

__version__ = "0.1.0"
__author__ = "CineInfini Contributors"
__license__ = "MIT"

# -------------------------------------------------------------------
# Expose high‑level functions
# -------------------------------------------------------------------
from .pipeline.audit import (
    audit_video,
    adaptive_multi_stage_audit,
    set_global_paths,
    CONFIG,
)

from .io.report import (
    generate_intra_report,
    generate_inter_report,
)

from .core.metrics import compute_composite_score

from .core.face_detection import (
    CascadeFaceDetector,
    ArcFaceEmbedder,
    identity_within_shot,
)

from .core.embedding import (
    CLIPSemanticScorer,
    clip_semantic_consistency,
    load_dinov2,
)

from .core.coherence import (
    compute_inter_shot_coherence,
    compute_narrative_coherence,
)

from .io.download import download_video

# -------------------------------------------------------------------
# Define what is available when importing *
# -------------------------------------------------------------------
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core functions
    "audit_video",
    "adaptive_multi_stage_audit",
    "set_global_paths",
    "CONFIG",
    "generate_intra_report",
    "generate_inter_report",
    "compute_composite_score",
    # Detection & embedding
    "CascadeFaceDetector",
    "ArcFaceEmbedder",
    "identity_within_shot",
    "CLIPSemanticScorer",
    "clip_semantic_consistency",
    "load_dinov2",
    # Coherence
    "compute_inter_shot_coherence",
    "compute_narrative_coherence",
    # Utilities
    "download_video",
]

# -------------------------------------------------------------------
# Optional: set a default logger (placeholder for now)
# -------------------------------------------------------------------
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
