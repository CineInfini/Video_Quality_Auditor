from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Any, Dict, Optional
from ...core.config import get_config
from ...core.context import VideoContext
from ...core.ui_registry import register_renderer
from .base import BaseRenderer
logger = logging.getLogger("cineinfini.renderers.jupyter")
@register_renderer("jupyter", description="IPython.display widgets for notebooks.")
class JupyterRenderer(BaseRenderer):
    renderer_id = "jupyter"
    def render(self, audit_data: Dict[str, Any], output_dir: Path, context: Optional[VideoContext] = None) -> Optional[Path]:
        cfg = get_config()
        if not cfg.is_jupyter(): logger.debug("JupyterRenderer: not in a notebook — no-op"); return None
        try:
            from IPython.display import display, HTML, Markdown
        except: return None
        title = audit_data.get("video_name","audit"); version = audit_data.get("version","0.4.8")
        display(HTML(f"<h2 style='margin-bottom:4px'>CineInfini · {title}</h2><div style='color:#666;font-size:12px'>v{version} · theme: {cfg.theme()}</div>"))
        modules_md = "\n".join(f"- **{name}** {'✅' if cfg.is_module_enabled(name) else '⬜'}" for name in sorted(cfg.modules.keys()))
        display(Markdown("### Modules state\n"+modules_md))
        try:
            import pandas as pd
            gates = audit_data.get("gates") or {}
            if gates:
                df = pd.DataFrame.from_dict(gates, orient="index"); df.index.name = "shot"; display(df)
        except Exception as e:
            logger.debug("JupyterRenderer: pandas display failed (%s)", e)
            display(Markdown(f"```json\n{json.dumps(audit_data.get('gates',{}), indent=2, default=str)[:2000]}\n```"))
        return None
