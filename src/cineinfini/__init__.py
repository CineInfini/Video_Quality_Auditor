"""CineInfini - Video Quality Audit Pipeline"""

__version__ = "0.1.0"

from .pipeline.audit import audit_video, adaptive_multi_stage_audit
from .io.report import generate_intra_report, generate_inter_report
from .core.metrics import compute_composite_score

__all__ = [
    "audit_video",
    "adaptive_multi_stage_audit",
    "generate_intra_report",
    "generate_inter_report",
    "compute_composite_score",
]
