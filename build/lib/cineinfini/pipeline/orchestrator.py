from __future__ import annotations
import logging, time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from ..core.config import get_config
from ..core.context import ModelPool, VideoContext, VideoInfoLite
logger = logging.getLogger("cineinfini.orchestrator")
def _ensure_registries_populated():
    from .. import modules
    from ..io import renderers
    _ = (modules, renderers)
def _build_video_info(video_path: Path) -> VideoInfoLite:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoInfoLite(path=video_path, fps=fps, total_frames=total, duration_s=total/max(fps,1e-6))
def _detect_and_extract(video_path: Path, max_duration_s: float):
    from ..io.reader import detect_shot_boundaries, extract_shot_frames_global
    cfg = get_config()
    proc = cfg.processing
    shots = detect_shot_boundaries(
        str(video_path),
        max_duration_s=max_duration_s,
        shot_threshold=proc.get("shot_threshold", 0.2),
        min_shot_duration_s=proc.get("min_shot_duration_s", 0.5),
        downsample_to=proc.get("downsample_to", 320),
    )
    frames_dict = extract_shot_frames_global(
        str(video_path), shots,
        n_frames_per_shot=proc.get("n_frames_per_shot", 16),
        frame_resize=proc.get("frame_resize", None),
    )
    return shots, frames_dict
def _build_shot_frames(shots, frames_dict, n_frames):
    out={}
    for i,(s,e,_) in enumerate(shots,1):
        n=min(n_frames, e-s+1)
        idxs=np.linspace(s,e,n,dtype=int)
        frames=[frames_dict[idx] for idx in idxs if idx in frames_dict]
        if frames: out[i]=frames
    return out
def _merge_module_result(audit_data, result):
    gates=audit_data.setdefault("gates",{})
    per_shot=result.get("per_shot",{})
    for sid,fields in per_shot.items():
        gate=gates.setdefault(int(sid),{})
        gate.update(fields)
    audit_data.setdefault("modules",{})[result.get("module")]={k:v for k,v in result.items() if k!="per_shot"}
def _compute_composites(audit_data):
    try:
        from ..core.metrics import compute_composite_score
    except: return
    cfg=get_config()
    for sid,gate in (audit_data.get("gates") or {}).items():
        try: gate["composite"]=compute_composite_score(gate,cfg.thresholds)
        except: pass
def run_audit(video_path: str|Path,*,output_dir:Optional[Path]=None,force_full_video:bool=False)->Tuple[Dict[str,Any],Path]:
    _ensure_registries_populated()
    from ..core.registry import get_active_modules
    from ..core.ui_registry import get_active_renderers
    cfg=get_config()
    video_path=Path(video_path)
    out_dir=Path(output_dir) if output_dir else cfg.reports_dir()/video_path.stem
    out_dir.mkdir(parents=True,exist_ok=True)
    timing={}
    t0=time.time()
    video=_build_video_info(video_path)
    max_duration_s=video.duration_s if force_full_video else float(cfg.processing.get("max_duration_s",60))
    pool=ModelPool(device=cfg.effective_device())
    context=VideoContext(video=video,cfg=cfg,pool=pool)
    t_shot=time.time()
    context.shots,context.frames_dict=_detect_and_extract(video_path,max_duration_s)
    context.shot_frames=_build_shot_frames(context.shots,context.frames_dict,int(cfg.processing.get("n_frames_per_shot",16)))
    timing["shot_extraction"]=time.time()-t_shot
    from .. import __version__ as _CINEINFINI_VERSION
    audit_data={"version":_CINEINFINI_VERSION,"video_path":str(video_path),"video_name":video.name,"fps":video.fps,"duration_s":video.duration_s,"n_shots":len(context.shots),"shots":context.shots,"gates":{},"modules":{},"active_modules":cfg.enabled_modules(),"active_renderers":cfg.active_renderers(),"thresholds":dict(cfg.thresholds)}
    t_compute=time.time()
    for entry in get_active_modules():
        ts=time.time()
        try:
            result=entry.func(context)
            _merge_module_result(audit_data,result)
            logger.info("module '%s' OK in %.2fs",entry.mod_id,time.time()-ts)
        except Exception as e:
            logger.exception("module '%s' failed: %s",entry.mod_id,e)
            audit_data.setdefault("errors",{})[entry.mod_id]=str(e)
    timing["compute"]=time.time()-t_compute
    t_comp=time.time()
    _compute_composites(audit_data)
    timing["composite"]=time.time()-t_comp
    if cfg.processing.get("compute_dtw_inter",True) and len(context.shots)>=2:
        try:
            from ..core.coherence import compute_inter_shot_coherence
            audit_data["inter_shot_coherence"]=compute_inter_shot_coherence(context.shot_frames,context.pool.get("face_detector"),context.pool.get("arcface"),None,int(cfg.processing.get("inter_shot_subsample",5)))
        except: pass
    # Attach VideoScore-style 5-axis fusion + composite global score so
    # all renderers (HTML dashboard, JSON, markdown) can surface them.
    try:
        from ..aggregators import attach_videoscore_to_audit
        attach_videoscore_to_audit(audit_data)
    except Exception as e:
        logger.debug("videoscore attach failed: %s", e)
    # Optionally also write the VBench export inline so it shows up next
    # to dashboard.html as a companion file.
    if cfg.reporting.get("emit_vbench_export", False):
        try:
            from ..io.exporters import export_vbench_json
            export_vbench_json(audit_data, out_dir / "audit.vbench.json")
        except Exception as e:
            logger.debug("vbench export failed: %s", e)
    t_render=time.time()
    rendered={}
    for r_entry in get_active_renderers():
        try:
            renderer=r_entry.cls()
            out=renderer.render(audit_data,out_dir,context=context)
            rendered[r_entry.renderer_id]=str(out) if out else None
        except Exception as e:
            logger.exception("renderer '%s' failed: %s",r_entry.renderer_id,e)
            rendered[r_entry.renderer_id]=None
    timing["render"]=time.time()-t_render
    audit_data["rendered"]=rendered
    timing["total"]=time.time()-t0
    audit_data["timing"]=timing
    if not cfg.processing.get("benchmark_mode",True): pool.clear()
    return audit_data,out_dir
