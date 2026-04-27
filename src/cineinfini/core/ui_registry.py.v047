from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Type
from .config import get_config
logger = logging.getLogger("cineinfini.ui_registry")
@dataclass
class RendererEntry: renderer_id: str; cls: Type; description: str = ""; version: str = "0.4.7"
class _UIRegistry:
    def __init__(self): self._renderers: Dict[str, RendererEntry] = {}
    def register(self,renderer_id:str,cls:Type,*,description:str="",version:str="0.4.7")->Type:
        if renderer_id in self._renderers: logger.debug("Re-registering renderer '%s'",renderer_id)
        self._renderers[renderer_id]=RendererEntry(renderer_id=renderer_id,cls=cls,description=description,version=version)
        return cls
    def unregister(self,renderer_id:str)->None: self._renderers.pop(renderer_id,None)
    def clear(self)->None: self._renderers.clear()
    def all_renderers(self)->Dict[str,RendererEntry]: return dict(self._renderers)
    def get(self,renderer_id:str)->Optional[RendererEntry]: return self._renderers.get(renderer_id)
    def get_active_renderers(self)->List[RendererEntry]:
        cfg=get_config()
        active_ids=cfg.active_renderers()
        out=[]
        for rid in active_ids:
            entry=self._renderers.get(rid)
            if entry is None: logger.warning("Renderer '%s' is active in config but not registered",rid); continue
            out.append(entry)
        return out
_ui_registry=_UIRegistry()
def register_renderer(renderer_id:str,*,description:str="",version:str="0.4.7"):
    def deco(cls:Type)->Type: _ui_registry.register(renderer_id,cls,description=description,version=version); return cls
    return deco
def get_ui_registry()->_UIRegistry: return _ui_registry
def get_active_renderers()->List[RendererEntry]: return _ui_registry.get_active_renderers()
def all_renderers()->Dict[str,RendererEntry]: return _ui_registry.all_renderers()
def reset_ui_registry()->None: _ui_registry.clear()
