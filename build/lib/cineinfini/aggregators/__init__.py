"""Cross-tool aggregation (VideoScore fusion, composite scoring)."""
from .videoscore_fusion import (
    VIDEOSCORE_AXES, compute_videoscore_axes, compute_global_score,
    attach_videoscore_to_audit,
)
__all__ = ["VIDEOSCORE_AXES", "compute_videoscore_axes",
           "compute_global_score", "attach_videoscore_to_audit"]
