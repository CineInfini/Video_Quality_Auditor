from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from ...core.context import VideoContext
from ...core.ui_registry import register_renderer
from .base import BaseRenderer
def _default(o):
    if isinstance(o,(np.floating,)): return float(o)
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,np.ndarray): return o.tolist()
    if isinstance(o,Path): return str(o)
    raise TypeError(f"Type not JSON-serializable: {type(o).__name__}")
@register_renderer("json", description="Raw JSON dump of audit_data.")
class JSONRenderer(BaseRenderer):
    renderer_id = "json"
    def render(self, audit_data: Dict[str, Any], output_dir: Path, context: Optional[VideoContext] = None) -> Optional[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "data.json"
        out_path.write_text(json.dumps(audit_data, indent=2, default=_default), encoding="utf-8")
        return out_path
