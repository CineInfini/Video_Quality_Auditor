"""CineInfini – Video Quality Audit Pipeline"""

__version__ = "0.1.1"  # incrémenter
__author__ = "Salah-Eddine BENBRAHIM"
__license__ = "MIT"

from .pipeline.audit import audit_video, adaptive_multi_stage_audit
from .io.report import generate_intra_report, generate_inter_report
from .core.metrics import compute_composite_score, recompute_composite_scores
from .core.face_detection import CascadeFaceDetector, ArcFaceEmbedder, identity_within_shot, set_models_dir
from .core.embedding import CLIPSemanticScorer, clip_semantic_consistency, load_dinov2, get_dinov2
from .core.coherence import compute_inter_shot_coherence, compute_narrative_coherence

__all__ = [
    "__version__", "__author__", "__license__",
    "audit_video", "adaptive_multi_stage_audit",
    "generate_intra_report", "generate_inter_report",
    "compute_composite_score", "recompute_composite_scores",
    "CascadeFaceDetector", "ArcFaceEmbedder", "identity_within_shot", "set_models_dir",
    "CLIPSemanticScorer", "clip_semantic_consistency", "load_dinov2", "get_dinov2",
    "compute_inter_shot_coherence", "compute_narrative_coherence",
]
