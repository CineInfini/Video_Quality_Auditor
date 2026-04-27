from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional
from ..core.config import get_config
logger = logging.getLogger("cineinfini.viz")
_VALID_FORMATS={"png","svg","jpeg","jpg","pdf"}
def _normalize_formats(formats:Optional[Iterable[str]])->List[str]:
    cfg=get_config()
    if formats is None: formats=[cfg.figure_format()]
    out=[]
    for fmt in formats:
        f=str(fmt).lower().strip().lstrip(".")
        if f=="jpg": f="jpeg"
        if f in _VALID_FORMATS: out.append(f)
        else: logger.warning("viz_utils: ignoring unsupported format '%s'",fmt)
    return out or ["png"]
def save_figure(fig:Any,name:str,output_dir:Path,formats:Optional[Iterable[str]]=None,dpi:Optional[int]=None,close:bool=True)->List[Path]:
    cfg=get_config()
    output_dir.mkdir(parents=True,exist_ok=True)
    fmts=_normalize_formats(formats)
    eff_dpi=int(dpi or cfg.figure_dpi())
    saved=[]
    for fmt in fmts:
        ext="jpg" if fmt=="jpeg" else fmt
        path=output_dir/f"{name}.{ext}"
        try:
            fig.savefig(str(path),format=fmt,dpi=eff_dpi,bbox_inches="tight")
            saved.append(path)
        except Exception as e: logger.warning("viz_utils: failed to save %s as %s: %s",name,fmt,e)
    if cfg.is_jupyter() and cfg.reporting.get("interactive",True):
        try:
            from IPython.display import display
            display(fig)
        except: pass
    if close:
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except: pass
    return saved
def new_figure(figsize=(8,4),theme:Optional[str]=None):
    import matplotlib.pyplot as plt
    cfg=get_config()
    theme=(theme or cfg.theme()).lower()
    style="dark_background" if theme=="dark" else "default"
    with plt.style.context(style):
        fig,ax=plt.subplots(figsize=figsize)
    return fig,ax
