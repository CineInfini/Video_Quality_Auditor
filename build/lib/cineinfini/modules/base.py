from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import get_registry
class BaseModule(ABC):
    mod_id: str = ""
    requires: List[str] = []
    description: str = ""
    version: str = "0.4.7"
    def __init__(self, mod_id: Optional[str] = None) -> None:
        if mod_id is not None: self.mod_id = mod_id
        if not self.mod_id: raise ValueError(f"{type(self).__name__} must define `mod_id`")
    @property
    def cfg(self) -> Dict[str, Any]: return get_config().get_module_config(self.mod_id)
    def is_enabled(self) -> bool: return get_config().is_module_enabled(self.mod_id)
    @abstractmethod
    def run(self, context: VideoContext) -> Dict[str, Any]: ...
    def __call__(self, context: VideoContext) -> Dict[str, Any]: return self.run(context)
    @classmethod
    def register(cls) -> None:
        instance = cls()
        get_registry().register(instance.mod_id, instance, requires=list(cls.requires), description=cls.description, version=cls.version)
