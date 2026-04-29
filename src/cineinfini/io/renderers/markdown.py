from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from ...core.config import get_config
from ...core.context import VideoContext
from ...core.ui_registry import register_renderer
from .base import BaseRenderer
logger = logging.getLogger("cineinfini.renderers.markdown")
@register_renderer("markdown", description="GitHub-friendly markdown dashboard.")
class MarkdownRenderer(BaseRenderer):
    renderer_id = "markdown"
    def render(self, audit_data: Dict[str, Any], output_dir: Path, context: Optional[VideoContext] = None) -> Optional[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "dashboard.md"
        try:
            from ..report import generate_intra_report
            legacy_path = generate_intra_report(audit_data, output_dir)
            if legacy_path: return Path(legacy_path)
        except Exception as e:
            logger.debug("Legacy markdown renderer failed (%s); falling back to v0.4.7 minimal.", e)
        out_path.write_text(self._fallback_markdown(audit_data), encoding="utf-8")
        return out_path
    def _fallback_markdown(self, data):
        cfg = get_config()
        lines = [f"# CineInfini audit — {data.get('video_name', 'unknown')}", f"- Version: **v{data.get('version', '0.4.7')}**", f"- Active modules: {', '.join(cfg.enabled_modules()) or '(none)'}", f"- Theme: {cfg.theme()}", "", "## Per-shot composite", "", "| Shot | Composite | Verdict |", "|------|-----------|---------|"]
        for sid, gate in (data.get("gates") or {}).items():
            comp = gate.get("composite", "—")
            verdict = gate.get("verdict", "—")
            lines.append(f"| {sid} | {comp} | {verdict} |")
        return "\n".join(lines) + "\n"
