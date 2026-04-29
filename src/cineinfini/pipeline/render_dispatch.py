from __future__ import annotations
import json, logging, multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from ..core.config import get_config
from ..core.context import VideoContext
from ..core.ui_registry import get_active_renderers
logger = logging.getLogger("cineinfini.render_dispatch")
def _safe_for_pickle(o): return float(o) if isinstance(o,np.floating) else int(o) if isinstance(o,np.integer) else o.tolist() if isinstance(o,np.ndarray) else str(o) if isinstance(o,Path) else o
def _strip_for_workers(audit_data):
    cleaned = json.loads(json.dumps(audit_data, default=_safe_for_pickle))
    cleaned.pop("frames_dict", None); cleaned.pop("shot_frames", None)
    return cleaned
def _worker(payload):
    renderer_id, audit_data, out_dir = payload
    try:
        from ..core.ui_registry import get_ui_registry
        from ..io import renderers
        entry = get_ui_registry().get(renderer_id)
        if entry is None: return renderer_id, None, f"renderer '{renderer_id}' not registered"
        out = entry.cls().render(audit_data, Path(out_dir), context=None)
        return renderer_id, str(out) if out else None, None
    except Exception as e: return renderer_id, None, repr(e)
def dispatch(audit_data: Dict[str, Any], output_dir: Path, context: Optional[VideoContext] = None) -> Dict[str, Optional[str]]:
    cfg = get_config()
    parallel = bool(cfg.reporting.get("parallel_renderers", False))
    if cfg.is_jupyter(): parallel = False
    entries = get_active_renderers()
    rendered = {}
    if not parallel or len(entries) <= 1:
        for entry in entries:
            try:
                out = entry.cls().render(audit_data, output_dir, context=context)
                rendered[entry.renderer_id] = str(out) if out else None
            except Exception as e: logger.exception("renderer '%s' failed: %s", entry.renderer_id, e); rendered[entry.renderer_id] = None
        return rendered
    payloads = [(e.renderer_id, _strip_for_workers(audit_data), str(output_dir)) for e in entries]
    n_workers = min(len(entries), int(cfg.processing.get("num_workers", 4)))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for rid, out, err in pool.imap_unordered(_worker, payloads):
            if err: logger.warning("renderer '%s' worker failed: %s", rid, err)
            rendered[rid] = out
    return rendered
