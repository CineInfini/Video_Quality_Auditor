from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from .config import get_config
logger = logging.getLogger("cineinfini.registry")
ModuleFn = Callable[..., Dict[str, Any]]
@dataclass
class ModuleEntry: mod_id: str; func: ModuleFn; requires: List[str]; description: str = ""; version: str = "0.4.7"
class _AuditRegistry:
    def __init__(self): self._modules: Dict[str, ModuleEntry] = {}
    def register(self,mod_id:str,func:ModuleFn,*,requires:Optional[List[str]]=None,description:str="",version:str="0.4.7")->ModuleFn:
        if mod_id in self._modules: logger.debug("Re-registering module '%s'",mod_id)
        self._modules[mod_id]=ModuleEntry(mod_id=mod_id,func=func,requires=list(requires or []),description=description,version=version)
        return func
    def unregister(self,mod_id:str)->None: self._modules.pop(mod_id,None)
    def clear(self)->None: self._modules.clear()
    def all_modules(self)->Dict[str,ModuleEntry]: return dict(self._modules)
    def get(self,mod_id:str)->Optional[ModuleEntry]: return self._modules.get(mod_id)
    def get_active_modules(self)->List[ModuleEntry]:
        cfg=get_config()
        return [entry for mod_id,entry in self._modules.items() if cfg.is_module_enabled(mod_id)]
    def required_models(self)->List[str]:
        active=self.get_active_modules()
        out=[]
        for entry in active:
            for key in entry.requires:
                if key not in out: out.append(key)
        return out
_registry=_AuditRegistry()
def register_module(mod_id:str,*,requires:Optional[List[str]]=None,description:str="",version:str="0.4.7")->Callable[[ModuleFn],ModuleFn]:
    def deco(func:ModuleFn)->ModuleFn: _registry.register(mod_id,func,requires=requires,description=description,version=version); return func
    return deco
def get_registry()->_AuditRegistry: return _registry
def get_active_modules()->List[ModuleEntry]: return _registry.get_active_modules()
def all_modules()->Dict[str,ModuleEntry]: return _registry.all_modules()
def reset_registry()->None: _registry.clear()
