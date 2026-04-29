"""Themed standalone HTML dashboard renderer with KPI cards, inline SVG
charts, and exhaustive per-video metric tables.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...core.config import get_config
from ...core.context import VideoContext
from ...core.ui_registry import register_renderer
from .base import BaseRenderer

_THEMES = {
    "dark": {
        "bg": "#0d1117", "fg": "#e6edf3", "accent": "#58a6ff",
        "muted": "#8b949e", "good": "#3fb950", "warn": "#d29922",
        "bad": "#f85149", "card": "#161b22", "border": "rgba(255,255,255,0.08)",
        "track": "rgba(255,255,255,0.06)",
    },
    "light": {
        "bg": "#ffffff", "fg": "#1f2328", "accent": "#0969da",
        "muted": "#656d76", "good": "#1a7f37", "warn": "#9a6700",
        "bad": "#cf222e", "card": "#f6f8fa", "border": "rgba(0,0,0,0.10)",
        "track": "rgba(0,0,0,0.06)",
    },
}

# Group every gate metric by category, drives the "Exhaustive" section.
_METRIC_GROUPS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("Motion & temporal", [
        ("motion_peak_div", "Optical-flow peak divergence (lower = smoother)"),
        ("flicker", "Inter-frame flicker score (lower = stabler)"),
        ("flicker_hf_var", "High-frequency flicker variance"),
        ("ssim3d_self", "3D-SSIM self-similarity (higher = more coherent)"),
        ("ssim_long_range", "SSIM first-vs-last frame (higher = stable bg)"),
        ("flow_periodicity", "Cyclic structure of flow (-1..1)"),
        ("flow_signature_entropy", "Entropy of flow signature"),
    ]),
    ("Identity & subject", [
        ("identity_within_shot", "Mean identity drift (cosine, lower = stable)"),
        ("identity_within_shot_dtw", "DTW-aligned identity drift"),
        ("subject_consistency_long", "Long-range identity DTW"),
    ]),
    ("Aesthetic & composition", [
        ("color_harmony", "Color harmony (0..1)"),
        ("color_scheme", "Detected scheme"),
        ("contrast", "Global contrast"),
        ("rule_of_thirds", "Rule-of-thirds adherence (0..1)"),
        ("aesthetic_composite", "Aesthetic composite (0..1)"),
    ]),
    ("Causal & physics", [
        ("causal_violation", "Anti-gravity / unphysical motion (lower = better)"),
        ("vertical_flow_imbalance", "Vertical flow asymmetry"),
        ("upward_flow_ratio", "Fraction of upward flow"),
        ("object_persistence", "Object permanence across frames (0..1)"),
        ("trajectory_smoothness", "Centroid trajectory smoothness"),
        ("trajectory_curvature_violation", "Curvature spike count"),
        ("jerk_score", "Trajectory jerk (lower = smoother)"),
    ]),
    ("Surprise & creativity", [
        ("surprise_mean", "Mean per-frame surprise"),
        ("surprise_max", "Max per-frame surprise"),
        ("surprise_p95", "P95 of surprise"),
    ]),
    ("Semantic & alignment", [
        ("semantic_consistency", "CLIP per-frame coherence (higher = better)"),
        ("clip_temp_consistency", "CLIP temporal coherence"),
        ("clip_alignment_score", "Prompt alignment (CLIP)"),
    ]),
    ("Cross-benchmark scores", [
        ("dover_aesthetic", "DOVER aesthetic branch (0..1)"),
        ("dover_technical", "DOVER technical branch (0..1)"),
        ("fastvqa_score", "FAST-VQA quality score (0..1)"),
    ]),
    ("Verdict", [
        ("composite", "CineInfini per-shot composite (higher = better)"),
        ("verdict", "ACCEPT / REVIEW / REJECT / BLOCKED"),
        ("failed_gates", "Gates that triggered a non-ACCEPT verdict"),
        ("score", "Module-level score (when reported)"),
    ]),
]


@register_renderer("html", description="Themed HTML dashboard with exhaustive metrics.")
class HTMLDashboardRenderer(BaseRenderer):
    renderer_id = "html"

    def render(self, audit_data, output_dir, context=None):
        cfg = get_config()
        theme = _THEMES.get(cfg.theme(), _THEMES["dark"])
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "dashboard.html"
        page = self._build_page(audit_data, theme, cfg, output_dir)
        out_path.write_text(page, encoding="utf-8")
        if cfg.is_jupyter() and cfg.reporting.get("interactive", True):
            try:
                from IPython.display import HTML, display
                display(HTML(page))
            except Exception:
                pass
        return out_path

    def _build_page(self, data, theme, cfg, output_dir):
        title = html.escape(str(data.get("video_name", "CineInfini audit")))
        version = html.escape(str(data.get("version", "unknown")))
        gates = data.get("gates") or {}
        modules_data = data.get("modules") or {}
        kpi = self._compute_kpis(gates, modules_data, data)
        sections = [
            self._section_header(title, version, data),
            self._section_kpis(kpi, theme),
            self._section_videoscore(data, theme),
            self._section_modules(modules_data, theme),
            self._section_exhaustive_metrics(gates, theme),
            self._section_shots(gates, theme),
            self._section_exports(output_dir),
            self._section_raw(data),
        ]
        body = "\n".join(s for s in sections if s)
        return self._wrap_html(title, theme, body)

    @staticmethod
    def _compute_kpis(gates, modules_data, audit_data):
        verdict_counts = {"ACCEPT": 0, "REVIEW": 0, "REJECT": 0, "BLOCKED": 0}
        composites = []
        for gate in gates.values():
            v = str(gate.get("verdict", "")).upper()
            if v in verdict_counts:
                verdict_counts[v] += 1
            c = gate.get("composite")
            if isinstance(c, (int, float)):
                composites.append(float(c))
        avg = sum(composites) / len(composites) if composites else None
        modules_run = len(modules_data)
        modules_failed = sum(
            1 for m in modules_data.values()
            if isinstance(m, dict) and m.get("available") is False
        )
        return {
            "verdict_counts": verdict_counts,
            "avg_composite": avg,
            "global_composite": audit_data.get("composite_score"),
            "modules_run": modules_run,
            "modules_ok": modules_run - modules_failed,
            "modules_failed": modules_failed,
            "n_shots": len(gates),
        }

    @staticmethod
    def _section_header(title, version, data):
        fps = data.get("fps", "—")
        duration = data.get("duration_s", "—")
        n_shots = data.get("n_shots", "—")
        active = data.get("active_modules") or []
        am_str = ", ".join(html.escape(m) for m in active) or "(none)"
        d_str = f"{duration:.2f} s" if isinstance(duration, (int, float)) else str(duration)
        f_str = f"{fps:.1f}" if isinstance(fps, (int, float)) else str(fps)
        return f"""<header>
  <h1>CineInfini — {title}</h1>
  <div class="muted">Version v{version} · {len(active)} active modules</div>
  <div class="meta-grid">
    <div><div class="muted">Duration</div><div class="value">{d_str}</div></div>
    <div><div class="muted">FPS</div><div class="value">{f_str}</div></div>
    <div><div class="muted">Shots detected</div><div class="value">{n_shots}</div></div>
    <div><div class="muted">Modules executed</div><div class="value">{len(data.get('modules') or {})}</div></div>
  </div>
  <details class="modules-list">
    <summary class="muted">show active modules</summary>
    <div class="kv-block">{am_str}</div>
  </details>
</header>"""

    @staticmethod
    def _section_kpis(kpi, theme):
        glb = kpi["global_composite"]
        glb_str = f"{glb:.3f}" if isinstance(glb, (int, float)) else "—"
        avg = kpi["avg_composite"]
        avg_str = f"{avg:.3f}" if isinstance(avg, (int, float)) else "—"
        cls = lambda v: ("good" if v is not None and v >= 0.7
                         else "warn" if v is not None and v >= 0.4
                         else "bad" if v is not None else "muted")
        vc = kpi["verdict_counts"]
        return f"""<section>
  <h2>Key indicators</h2>
  <div class="kpi-grid">
    <div class="kpi"><div class="kpi-label">Composite (global)</div>
        <div class="kpi-value {cls(glb)}">{glb_str}</div></div>
    <div class="kpi"><div class="kpi-label">Composite (mean shot)</div>
        <div class="kpi-value {cls(avg)}">{avg_str}</div></div>
    <div class="kpi"><div class="kpi-label">Shots</div><div class="kpi-value">{kpi['n_shots']}</div></div>
    <div class="kpi"><div class="kpi-label">Accept</div><div class="kpi-value good">{vc['ACCEPT']}</div></div>
    <div class="kpi"><div class="kpi-label">Review</div><div class="kpi-value warn">{vc['REVIEW']}</div></div>
    <div class="kpi"><div class="kpi-label">Reject</div><div class="kpi-value bad">{vc['REJECT']}</div></div>
    <div class="kpi"><div class="kpi-label">Modules OK</div><div class="kpi-value good">{kpi['modules_ok']}</div></div>
    <div class="kpi"><div class="kpi-label">Modules unavail.</div><div class="kpi-value muted">{kpi['modules_failed']}</div></div>
  </div>
</section>"""

    @staticmethod
    def _section_videoscore(data, theme):
        axes = data.get("videoscore_axes") or {}
        composite = data.get("composite_score")
        rows = list(axes.items())
        if composite is not None:
            rows.append(("composite_score", composite))
        if not rows:
            return ""
        bar_h, gap = 28, 12
        total_h = (bar_h + gap) * len(rows) + 16
        out = [f'<svg viewBox="0 0 720 {total_h}" xmlns="http://www.w3.org/2000/svg" class="chart">']
        for i, (label, val) in enumerate(rows):
            y = i * (bar_h + gap) + 8
            disp = f"{val:.3f}" if isinstance(val, (int, float)) else "n/a"
            w = (val * 460) if isinstance(val, (int, float)) else 0
            fill = theme["accent"] if label != "composite_score" else theme["good"]
            out.append(f'<text x="0" y="{y + 18}" fill="currentColor" font-size="13">{html.escape(label)}</text>')
            out.append(f'<rect x="200" y="{y}" width="460" height="{bar_h}" rx="4" fill="{theme["track"]}"/>')
            if isinstance(val, (int, float)):
                out.append(f'<rect x="200" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" fill="{fill}"/>')
            out.append(f'<text x="680" y="{y + 18}" fill="currentColor" font-size="13" text-anchor="end" font-weight="600">{disp}</text>')
        out.append("</svg>")
        return f"""<section>
  <h2>VideoScore axes + global composite</h2>
  {''.join(out)}
</section>"""

    @staticmethod
    def _section_modules(modules_data, theme):
        if not modules_data:
            return ""
        rows = []
        for mod_id, mod in sorted(modules_data.items()):
            available = mod.get("available")
            if available is False:
                cls, label = "muted", "skipped"
            elif available is True:
                cls, label = "good", "ok"
            else:
                cls, label = "good", "ran"
            version = html.escape(str(mod.get("version", "—")))
            reason = html.escape(str(mod.get("reason", ""))) if available is False else ""
            rows.append(
                f"<tr><td><code>{html.escape(mod_id)}</code></td>"
                f"<td>{version}</td>"
                f"<td><span class='badge {cls}'>{label}</span></td>"
                f"<td class='muted small'>{reason}</td></tr>"
            )
        return f"""<section>
  <h2>Module status — {len(modules_data)} executed</h2>
  <table>
    <thead><tr><th>Module</th><th>Version</th><th>Status</th><th>Reason if unavailable</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>"""

    @staticmethod
    def _section_exhaustive_metrics(gates, theme):
        if not gates:
            return ""
        all_metrics = set()
        for g in gates.values():
            all_metrics.update(g.keys())
        categorised = {}
        seen = set()
        for cat, defs in _METRIC_GROUPS:
            present = [(n, d) for n, d in defs if n in all_metrics]
            if present:
                categorised[cat] = present
                seen.update(n for n, _ in present)
        leftovers = sorted(all_metrics - seen)
        if leftovers:
            categorised["Other"] = [(m, "") for m in leftovers]
        sorted_sids = sorted(gates.keys(), key=lambda x: int(x))
        sections = []
        for cat, metrics in categorised.items():
            rows = []
            for name, desc in metrics:
                cells = [f"<td><code>{html.escape(name)}</code></td>"]
                cells.append(f"<td class='muted small'>{html.escape(desc) if desc else '—'}</td>")
                for sid in sorted_sids:
                    val = gates[sid].get(name)
                    if val is None:
                        cells.append("<td class='muted'>—</td>")
                    elif isinstance(val, float):
                        cells.append(f"<td>{val:.4f}</td>")
                    elif isinstance(val, list):
                        cells.append(f"<td class='small'>{', '.join(html.escape(str(v)) for v in val) if val else '—'}</td>")
                    else:
                        cells.append(f"<td>{html.escape(str(val))}</td>")
                rows.append("<tr>" + "".join(cells) + "</tr>")
            shot_headers = "".join(f"<th>shot {sid}</th>" for sid in sorted_sids)
            sections.append(f"""
  <h3>{html.escape(cat)}</h3>
  <table class="metrics">
    <thead><tr><th>Metric</th><th>Description</th>{shot_headers}</tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>""")
        return f"""<section>
  <h2>Exhaustive metrics — {len(all_metrics)} fields across {len(sorted_sids)} shot(s)</h2>
  {''.join(sections)}
</section>"""

    @staticmethod
    def _section_shots(gates, theme):
        if not gates:
            return f"""<section>
  <h2>Per-shot results</h2>
  <p class="muted">No per-shot gate data available.</p>
</section>"""
        rows = []
        for sid in sorted(gates.keys(), key=lambda x: int(x)):
            gate = gates[sid]
            comp = gate.get("composite")
            verdict = str(gate.get("verdict", "—")).upper() or "—"
            cls = ("good" if verdict == "ACCEPT" else "warn" if verdict == "REVIEW"
                   else "bad" if verdict == "REJECT" else "muted")
            comp_str = f"{comp:.3f}" if isinstance(comp, (int, float)) else "—"
            failed = gate.get("failed_gates") or []
            failed_str = ", ".join(html.escape(f) for f in failed) if failed else "—"
            rows.append(
                f"<tr><td>{html.escape(str(sid))}</td>"
                f"<td>{comp_str}</td>"
                f"<td><span class='badge {cls}'>{html.escape(verdict)}</span></td>"
                f"<td class='muted small'>{failed_str}</td></tr>"
            )
        return f"""<section>
  <h2>Per-shot summary</h2>
  <table>
    <thead><tr><th>Shot</th><th>Composite</th><th>Verdict</th><th>Failed gates</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>"""

    @staticmethod
    def _section_exports(output_dir):
        candidates = [n for n in ("audit.vbench.json", "videoscore.json",
                                  "data.json", "dashboard.md")
                      if (output_dir / n).exists()]
        if not candidates:
            return ""
        items = "".join(f'<li><a href="{html.escape(c)}">{html.escape(c)}</a></li>' for c in candidates)
        return f'<section><h2>Companion files</h2><ul class="files">{items}</ul></section>'

    @staticmethod
    def _section_raw(data):
        copy = dict(data)
        copy.pop("frames_dict", None)
        copy.pop("shot_frames", None)
        truncated = html.escape(json.dumps(copy, indent=2, default=str)[:8000])
        return f"""<section>
  <h2>Raw JSON</h2>
  <details>
    <summary class="muted">show data.json (first 8 KB)</summary>
    <pre>{truncated}</pre>
  </details>
</section>"""

    @staticmethod
    def _wrap_html(title, theme, body):
        return f"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>CineInfini · {title}</title>
<style>
:root {{
  --bg: {theme['bg']}; --fg: {theme['fg']}; --accent: {theme['accent']};
  --muted: {theme['muted']}; --good: {theme['good']}; --warn: {theme['warn']};
  --bad: {theme['bad']}; --card: {theme['card']}; --border: {theme['border']};
  --track: {theme['track']};
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--fg);
        font: 14px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 32px 24px; }}
header {{ border-bottom: 1px solid var(--border); padding-bottom: 24px; margin-bottom: 24px; }}
h1 {{ font-size: 28px; margin: 0 0 6px; }}
h2 {{ font-size: 14px; margin: 32px 0 14px; color: var(--accent);
      text-transform: uppercase; letter-spacing: 0.6px; font-weight: 700; }}
h3 {{ font-size: 12px; margin: 20px 0 8px; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}
.muted {{ color: var(--muted); }}
.small {{ font-size: 12px; }}
section {{ margin: 28px 0; }}
.meta-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
              margin-top: 16px; padding: 18px; background: var(--card);
              border-radius: 8px; border: 1px solid var(--border); }}
.meta-grid > div .value {{ font-size: 18px; font-weight: 700; margin-top: 4px; }}
.meta-grid > div .muted {{ font-size: 11px;
                            text-transform: uppercase; letter-spacing: 0.5px; }}
.modules-list {{ margin-top: 12px; }}
.modules-list summary {{ cursor: pointer; padding: 6px 0; font-size: 12px; }}
.modules-list .kv-block {{ padding: 12px; background: var(--card);
                            border-radius: 6px; border: 1px solid var(--border);
                            font-size: 12px; line-height: 1.6; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
             gap: 12px; }}
.kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
        padding: 16px; }}
.kpi-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase;
              letter-spacing: 0.5px; }}
.kpi-value {{ font-size: 28px; font-weight: 700; margin-top: 6px; }}
.kpi-value.good {{ color: var(--good); }}
.kpi-value.warn {{ color: var(--warn); }}
.kpi-value.bad {{ color: var(--bad); }}
.kpi-value.muted {{ color: var(--muted); }}
table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
table.metrics {{ font-size: 12px; }}
th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ color: var(--muted); font-weight: 600; font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.5px; }}
.badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px;
          font-size: 11px; font-weight: 600; text-transform: uppercase;
          letter-spacing: 0.5px; }}
.badge.good {{ background: var(--good); color: #fff; }}
.badge.warn {{ background: var(--warn); color: #000; }}
.badge.bad {{ background: var(--bad); color: #fff; }}
.badge.muted {{ background: var(--track); color: var(--muted); }}
.chart {{ width: 100%; max-width: 720px; }}
pre {{ background: var(--card); padding: 12px; overflow: auto;
       border-radius: 6px; font-size: 12px; border: 1px solid var(--border); }}
code {{ background: var(--track); padding: 1px 6px; border-radius: 3px;
        font-size: 12px; }}
ul.files {{ list-style: none; padding: 0; margin: 0; }}
ul.files li {{ padding: 6px 0; }}
ul.files a {{ color: var(--accent); text-decoration: none; }}
ul.files a:hover {{ text-decoration: underline; }}
details summary {{ cursor: pointer; padding: 8px 0; }}
</style>
</head><body><div class="container">{body}</div></body></html>
"""
