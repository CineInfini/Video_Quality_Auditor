from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from ...core.config import get_config
from ...core.context import VideoContext
from ...core.ui_registry import register_renderer
from ..viz_utils import save_figure
from .base import BaseRenderer
logger = logging.getLogger("cineinfini.renderers.svg")
@register_renderer("svg", description="Vectorial SVG figures bundle.")
class SVGRenderer(BaseRenderer):
    renderer_id = "svg"
    def render(self, audit_data: Dict[str, Any], output_dir: Path, context: Optional[VideoContext] = None) -> Optional[Path]:
        output_dir.mkdir(parents=True, exist_ok=True); svg_dir = output_dir / "svg"; svg_dir.mkdir(parents=True, exist_ok=True)
        try:
            from ..report_extended import figure_verdict_timeline, figure_dtw_vs_mean
        except Exception as e: logger.warning("SVGRenderer: report_extended unavailable (%s) — minimal mode", e); self._minimal_chart(audit_data, svg_dir); return svg_dir
        cfg = get_config(); thr = dict(cfg.thresholds)
        for name,builder in (("verdict_timeline", lambda: figure_verdict_timeline(audit_data,thr,output_path=None)), ("dtw_vs_mean", lambda: figure_dtw_vs_mean(audit_data,output_path=None))):
            try:
                fig = builder()
                if fig is not None: save_figure(fig, name, svg_dir, formats=["svg"])
            except Exception as e: logger.debug("SVGRenderer: %s failed (%s)", name, e)
        return svg_dir
    @staticmethod
    def _minimal_chart(audit_data, svg_dir):
        try:
            import matplotlib.pyplot as plt
            gates = audit_data.get("gates") or {}
            ids = sorted(int(k) for k in gates.keys())
            comps = [gates[str(i) if str(i) in gates else i].get("composite",0.0) for i in ids]
            if not ids: return
            fig,ax = plt.subplots(figsize=(max(6,0.4*len(ids)+2),3)); ax.bar(ids,comps); ax.set_xlabel("Shot"); ax.set_ylabel("Composite"); ax.set_title("Per-shot composite")
            save_figure(fig, "composite_minimal", svg_dir, formats=["svg"])
        except: pass
