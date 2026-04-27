from __future__ import annotations
import logging, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from .config import Config, get_config
logger = logging.getLogger("cineinfini.context")
class ModelPool:
    def __init__(self,device:str="cpu")->None:
        self.device=device
        self._cache: Dict[str,Any]={}
        self._timing: Dict[str,float]={}
    def get_or_load(self,key:str,loader:Callable[[],Any])->Optional[Any]:
        if key in self._cache: return self._cache[key]
        try:
            t0=time.time()
            obj=loader()
            self._timing[key]=time.time()-t0
            self._cache[key]=obj
            return obj
        except Exception as e:
            logger.warning("ModelPool: failed to load '%s': %s",key,e)
            self._cache[key]=None
            return None
    def has(self,key:str)->bool: return key in self._cache and self._cache[key] is not None
    def get(self,key:str)->Optional[Any]: return self._cache.get(key)
    def keys(self)->List[str]: return list(self._cache.keys())
    def timings(self)->Dict[str,float]: return dict(self._timing)
    def clear(self)->None:
        self._cache.clear()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except: pass
@dataclass
class VideoInfoLite:
    path: Path
    fps: float = 24.0
    total_frames: int = 0
    duration_s: float = 0.0
    @property
    def name(self)->str: return self.path.stem
@dataclass
class VideoContext:
    video: VideoInfoLite
    shots: List[Tuple[int,int,float]] = field(default_factory=list)
    frames_dict: Dict[int,Any] = field(default_factory=dict)
    shot_frames: Dict[int,List[Any]] = field(default_factory=dict)
    inter_shot_results: List[Dict[str,Any]] = field(default_factory=list)
    cfg: Config = field(default_factory=get_config)
    pool: ModelPool = field(default_factory=lambda: ModelPool())
    cache: Dict[str,Any] = field(default_factory=dict)
    @property
    def device(self)->str: return self.pool.device
    def get_shot_frames(self,shot_id:int)->List[Any]: return self.shot_frames.get(shot_id,[])
    def shot_ids(self)->List[int]: return sorted(self.shot_frames.keys())
