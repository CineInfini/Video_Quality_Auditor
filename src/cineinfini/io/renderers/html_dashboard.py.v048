from __future__ import annotations
import html
import json
from pathlib import Path
from typing import Any, Dict, Optional
from ...core.config import get_config
from ...core.context import VideoContext
from ...core.ui_registry import register_renderer
from .base import BaseRenderer
_THEMES = {"dark":{"bg":"#0d1117","fg":"#e6edf3","accent":"#58a6ff","muted":"#8b949e","good":"#3fb950","warn":"#d29922","bad":"#f85149"},"light":{"bg":"#ffffff","fg":"#1f2328","accent":"#0969da","muted":"#656d76","good":"#1a7f37","warn":"#9a6700","bad":"#cf222e"}}
@register_renderer("html", description="Themed standalone HTML dashboard.")
class HTMLDashboardRenderer(BaseRenderer):
    renderer_id = "html"
    def render(self, audit_data: Dict[str, Any], output_dir: Path, context: Optional[VideoContext] = None) -> Optional[Path]:
        cfg = get_config()
        theme = _THEMES.get(cfg.theme(), _THEMES["dark"])
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "dashboard.html"
        page = self._build_page(audit_data, theme, cfg)
        out_path.write_text(page, encoding="utf-8")
        if cfg.is_jupyter() and cfg.reporting.get("interactive", True):
            try:
                from IPython.display import HTML, display
                display(HTML(page))
            except: pass
        return out_path
    def _build_page(self, data, theme, cfg):
        title = html.escape(str(data.get("video_name", "CineInfini audit")))
        version = html.escape(str(data.get("version", "0.4.7")))
        rows = self._rows(data, theme)
        active_modules = ", ".join(html.escape(m) for m in cfg.enabled_modules()) or "(none)"
        return f"""<!doctype html><html><head><meta charset="utf-8"/><title>CineInfini · {title}</title><style>
:root {{ --bg: {theme["bg"]}; --fg: {theme["fg"]}; --accent: {theme["accent"]}; --muted: {theme["muted"]}; --good: {theme["good"]}; --warn: {theme["warn"]}; --bad: {theme["bad"]}; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
.container {{ max-width: 980px; margin: 0 auto; padding: 32px 24px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }}
h2 {{ font-size: 18px; margin: 32px 0 12px; color: var(--accent); }}
.muted {{ color: var(--muted); font-size: 13px; }}
.kv {{ display: grid; grid-template-columns: 180px 1fr; gap: 6px 16px; margin: 16px 0; }}
.kv .k {{ color: var(--muted); }}
table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.08); }}
th {{ color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
.badge.ok {{ background: var(--good); color: #fff; }}
.badge.warn {{ background: var(--warn); color: #000; }}
.badge.bad {{ background: var(--bad); color: #fff; }}
pre {{ background: rgba(255,255,255,0.04); padding: 12px; overflow:auto; border-radius: 6px; font-size: 12px; }}
</style></head><body><div class="container"><h1>CineInfini — {title}</h1><div class="muted">Version v{version}</div><div class="kv"><div class="k">Active modules</div><div>{active_modules}</div><div class="k">Theme</div><div>{html.escape(cfg.theme())}</div><div class="k">Figure format</div><div>{html.escape(cfg.figure_format())}</div></div><h2>Per-shot composite</h2>
<table>
<thead>
<tr><th>Shot</th><th>Composite</th><th>Verdict</th></tr>
</thead>
<tbody>{rows}</tbody>
</table>
<h2>Raw data (truncated)</h2>
<pre>{html.escape(json.dumps(self._truncate(data), indent=2, default=str)[:6000])}</pre>
</div></body></html>"""
    @staticmethod
    def _rows(data, theme):
        rows = []
        for sid, gate in (data.get("gates") or {}).items():
            comp = gate.get("composite")
            verdict = str(gate.get("verdict", "—")).upper()
            cls = ("ok" if verdict=="ACCEPT" else "warn" if verdict=="REVIEW" else "bad" if verdict=="REJECT" else "")
            comp_str = f"{comp:.3f}" if isinstance(comp,(int,float)) else "—"
            rows.append(f"<tr><td>{html.escape(str(sid))}</td><td>{comp_str}</td><td><span class='badge {cls}'>{html.escape(verdict)}</span></td></tr>")
        return "
".join(rows) or "<tr><td colspan='3' class='muted'>No shots</td></tr>"
    @staticmethod
    def _truncate(data):
        copy = dict(data)
        copy.pop("frames_dict", None)
        copy.pop("shot_frames", None)
        return copy
