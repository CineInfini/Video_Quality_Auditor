from __future__ import annotations
import logging
import numpy as np
from ..core.config import get_config
from ..core.context import VideoContext
from ..core.registry import register_module
logger = logging.getLogger("cineinfini.modules.background_consistency")
MOD_ID = "background_consistency"
def _ssim_pair(a,b):
    try:
        from skimage.metrics import structural_similarity as ssim_2d
        import cv2
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        return float(ssim_2d(a,b))
    except: return float("nan")
def _mse_pair(a,b):
    a32 = a.astype(np.float32); b32 = b.astype(np.float32)
    return float(np.mean((a32-b32)**2))
@register_module(MOD_ID, requires=[])
def run(context: VideoContext):
    cfg=get_config()
    mod_cfg=cfg.get_module_config(MOD_ID)
    method=str(mod_cfg.get("method","ssim")).lower()
    threshold=float(mod_cfg.get("threshold",cfg.thresholds.get("background_ssim",0.55)))
    subsample=int(mod_cfg.get("subsample",5))
    per_shot={}
    for sid,frames in context.shot_frames.items():
        if len(frames)<2:
            per_shot[sid]={"score":None,"n_pairs":0,"method":method}
            continue
        idxs=list(range(0,len(frames),max(1,subsample)))
        if len(idxs)<2: idxs=[0,len(frames)-1]
        scores=[]
        ref=frames[idxs[0]]
        for i in idxs[1:]:
            cur=frames[i]
            sc=_ssim_pair(ref,cur) if method!="mse" else _mse_pair(ref,cur)
            scores.append(sc)
        arr=np.asarray([s for s in scores if not np.isnan(s)],dtype=float)
        if arr.size==0:
            per_shot[sid]={"score":None,"n_pairs":0,"method":method}
        else:
            per_shot[sid]={"score":float(arr.mean()),"n_pairs":int(arr.size),"method":method}
    return {"module":MOD_ID,"version":"0.4.7","threshold":threshold,"method":method,"per_shot":per_shot}
