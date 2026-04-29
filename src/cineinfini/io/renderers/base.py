from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
from ...core.config import get_config
from ...core.context import VideoContext
class BaseRenderer(ABC):
    renderer_id: str = ""
    description: str = ""
    version: str = "0.4.7"
    def __init__(self, renderer_id: Optional[str] = None) -> None:
        if renderer_id is not None: self.renderer_id = renderer_id
        if not self.renderer_id: raise ValueError(f"{type(self).__name__} must define `renderer_id`")
    @property
    def cfg(self) -> Dict[str, Any]:
        rep = get_config().reporting
        return rep.get(self.renderer_id, {}) if isinstance(rep, dict) else {}
    @abstractmethod
    def render(self, audit_data: Dict[str, Any], output_dir: Path, context: Optional[VideoContext] = None) -> Optional[Path]: ...
    def __call__(self, audit_data: Dict[str, Any], output_dir: Path, context: Optional[VideoContext] = None) -> Optional[Path]: return self.render(audit_data, output_dir, context)
