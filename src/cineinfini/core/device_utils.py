from __future__ import annotations
import contextlib, logging
from typing import Any, Iterable, Iterator, List, Optional
from .config import get_config
logger = logging.getLogger("cineinfini.device_utils")
def resolve_dtype(name: Optional[str] = None) -> Any:
    cfg = get_config()
    name = (name or str(cfg.device.get("torch_dtype", "float32"))).lower()
    try:
        import torch
        table = {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16, "fp16": torch.float16, "half": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
        return table.get(name, torch.float32)
    except: return None
def amp_enabled() -> bool:
    cfg = get_config()
    if not cfg.device.get("use_amp", False): return False
    try:
        import torch
        return torch.cuda.is_available() and cfg.effective_device() == "cuda"
    except: return False
@contextlib.contextmanager
def autocast_context():
    if not amp_enabled(): yield; return
    try:
        import torch
        dtype = resolve_dtype()
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype): yield
    except Exception as e: logger.debug("autocast unavailable (%s)", e); yield
@contextlib.contextmanager
def inference_mode():
    try:
        import torch
        with torch.inference_mode(): yield
    except: yield
def release_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    except: pass
def vram_usage_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available(): return float(torch.cuda.memory_allocated()) / (1024**2)
    except: return None
def batch_iter(items: Iterable[Any], batch_size: int) -> Iterator[List[Any]]:
    batch = []
    for it in items:
        batch.append(it)
        if len(batch) >= batch_size: yield batch; batch = []
    if batch: yield batch
def effective_batch_size(default: int = 16) -> int:
    return int(get_config().processing.get("batch_size", default))
