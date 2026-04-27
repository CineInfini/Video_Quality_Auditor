from __future__ import annotations
import logging, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from .config import Config, get_config
logger = logging.getLogger("cineinfini.context")
class ModelPool:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device; self._cache: Dict[str, Any] = {}; self._timing: Dict[str, float] = {}; self._loaded_devices: Dict[str, str] = {}
    def get_or_load(self, key: str, loader: Callable[[], Any]) -> Optional[Any]:
        if key in self._cache: return self._cache[key]
        try:
            t0 = time.time(); obj = loader(); self._timing[key] = time.time() - t0; self._cache[key] = obj; self._loaded_devices[key] = self._infer_device(obj)
            logger.info("ModelPool: loaded '%s' in %.2fs (device=%s)", key, self._timing[key], self._loaded_devices[key]); return obj
        except Exception as e: logger.warning("ModelPool: failed to load '%s': %s", key, e); self._cache[key] = None; return None
    def has(self, key: str) -> bool: return key in self._cache and self._cache[key] is not None
    def get(self, key: str) -> Optional[Any]: return self._cache.get(key)
    def keys(self) -> List[str]: return list(self._cache.keys())
    def timings(self) -> Dict[str, float]: return dict(self._timing)
    def to_device(self, key: str, device: Optional[str] = None) -> Optional[Any]:
        device = device or self.device; obj = self._cache.get(key); if obj is None: return None
        try:
            if hasattr(obj, "to"): obj = obj.to(device); self._cache[key] = obj; self._loaded_devices[key] = device
        except: pass; return obj
    @staticmethod
    def _infer_device(obj: Any) -> str:
        try:
            if hasattr(obj, "device"): return str(obj.device)
            if hasattr(obj, "parameters"): p = next(obj.parameters(), None); return str(p.device) if p is not None else "cpu"
        except: pass; return "cpu"
    def release_one(self, key: str) -> None:
        self._cache.pop(key, None); self._loaded_devices.pop(key, None)
    def clear(self) -> None:
        self._cache.clear(); self._loaded_devices.clear()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except: pass
@dataclass
class VideoInfoLite: path: Path; fps: float = 24.0; total_frames: int = 0; duration_s: float = 0.0; @property def name(self) -> str: return self.path.stem
@dataclass
class VideoContext:
    video: VideoInfoLite; shots: List[Tuple[int,int,float]] = field(default_factory=list); frames_dict: Dict[int,Any] = field(default_factory=dict)
    shot_frames: Dict[int,List[Any]] = field(default_factory=dict); inter_shot_results: List[Dict[str,Any]] = field(default_factory=list)
    cfg: Config = field(default_factory=get_config); pool: ModelPool = field(default_factory=lambda: ModelPool()); cache: Dict[str,Any] = field(default_factory=dict)
    @property def device(self) -> str: return self.pool.device
    def get_shot_frames(self, shot_id: int) -> List[Any]: return self.shot_frames.get(shot_id, [])
    def shot_ids(self) -> List[int]: return sorted(self.shot_frames.keys())
    def iter_shot_batches(self, batch_size: Optional[int] = None) -> Iterator[Tuple[List[int], List[Any]]]:
        from .device_utils import effective_batch_size
        bs = batch_size or effective_batch_size(default=16); ids: List[int] = []; frames: List[Any] = []
        for sid in self.shot_ids():
            for f in self.shot_frames.get(sid, []):
                ids.append(sid); frames.append(f)
                if len(frames) >= bs: yield ids, frames; ids, frames = [], []
        if frames: yield ids, frames
