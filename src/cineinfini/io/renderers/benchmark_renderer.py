# src/cineinfini/io/renderers/benchmark_renderer.py
from __future__ import annotations
import csv, html, json, logging, statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from ...core.config import get_config
from ...core.context import VideoContext
from ...core.ui_registry import register_renderer
from .base import BaseRenderer
logger = logging.getLogger("cineinfini.renderers.benchmark")
_THEMES = {"dark":{"bg":"#0d1117","fg":"#e6edf3","accent":"#58a6ff","muted":"#8b949e","good":"#3fb950","warn":"#d29922","bad":"#f85149"},"light":{"bg":"#ffffff","fg":"#1f2328","accent":"#0969da","muted":"#656d76","good":"#1a7f37","warn":"#9a6700","bad":"#cf222e"}}
def _aggregate(values, method="median"):
    clean = [v for v in values if isinstance(v,(int,float)) and v==v]
    if not clean: return None
    method = method.lower()
    if method == "mean": return sum(clean)/len(clean)
    if method == "p10": return sorted(clean)[max(0,int(0.1*(len(clean)-1)))]
    if method == "p90": return sorted(clean)[min(len(clean)-1,int(0.9*(len(clean)-1)))]
    return statistics.median(clean)
def _video_composite(audit_data, method="median"):
    comps = [g.get("composite") for g in (audit_data.get("gates") or {}).values() if isinstance(g.get("composite"),(int,float))]
    return _aggregate(comps, method)
def _module_summary(audit_data):
    gates = audit_data.get("gates") or {}
    metric_keys = set()
    for g in gates.values():
        for k,v in g.items():
            if isinstance(v,(int,float)) and k!="composite": metric_keys.add(k)
    summary = {}
    for key in sorted(metric_keys):
        vals = [g.get(key) for g in gates.values() if isinstance(g.get(key),(int,float))]
        summary[key] = {"median":_aggregate(vals,"median"),"mean":_aggregate(vals,"mean"),"p10":_aggregate(vals,"p10"),"p90":_aggregate(vals,"p90"),"n":float(len(vals))}
    return summary
def _verdict_distribution(audit_data):
    counts = {"ACCEPT":0,"REVIEW":0,"REJECT":0,"UNKNOWN":0}
    for g in (audit_data.get("gates") or {}).values():
        v = str(g.get("verdict","UNKNOWN")).upper()
        counts[v if v in counts else "UNKNOWN"] += 1
    return counts

@register_renderer("benchmark", description="Multi-video comparative report.")
class BenchmarkRenderer(BaseRenderer):
    renderer_id = "benchmark"
    def render(self, audit_data, output_dir, context=None):
        return self.render_many([audit_data], output_dir, context)
    def render_many(self, audit_list, output_dir, context=None):
        cfg = get_config()
        method = str(cfg.reporting.get("benchmark",{}).get("aggregation_method","median")).lower()
        bench_dir = Path(output_dir) / "benchmark"
        bench_dir.mkdir(parents=True, exist_ok=True)
        audits = [a for a in audit_list if isinstance(a,dict)]
        if not audits:
            (bench_dir / "benchmark.md").write_text("# CineInfini Benchmark\n\n_No audits to compare._\n", encoding="utf-8")
            return bench_dir
        rows = []
        for a in audits:
            comp = _video_composite(a, method)
            rows.append({"video_name":a.get("video_name","?"),"video_path":a.get("video_path",""),"n_shots":int(a.get("n_shots",len(a.get("gates") or {}))),"composite":comp,"verdicts":_verdict_distribution(a),"modules":_module_summary(a),"duration_s":float(a.get("duration_s",0.0)),"timing":a.get("timing",{})})
        rows.sort(key=lambda r: (r["composite"] is None, -(r["composite"] or 0.0)))
        for i,r in enumerate(rows,1): r["rank"]=i
        # Aggregate VideoScore axes + global composite across all videos
        agg_axes = {}
        agg_composites = []
        for a in audits:
            for axis, val in (a.get("videoscore_axes") or {}).items():
                if isinstance(val, (int, float)):
                    agg_axes.setdefault(axis, []).append(val)
            gc = a.get("composite_score")
            if isinstance(gc, (int, float)):
                agg_composites.append(gc)
        videoscore_summary = {
            axis: {"mean": _aggregate(vals, "mean"),
                   "median": _aggregate(vals, "median"),
                   "p10": _aggregate(vals, "p10"),
                   "p90": _aggregate(vals, "p90")}
            for axis, vals in agg_axes.items()
        }
        verdict_totals = {"ACCEPT": 0, "REVIEW": 0, "REJECT": 0, "UNKNOWN": 0}
        for r in rows:
            for k, v in (r.get("verdicts") or {}).items():
                verdict_totals[k] = verdict_totals.get(k, 0) + v
        from ... import __version__ as _CINEINFINI_VERSION  # type: ignore
        aggregated = {"version":_CINEINFINI_VERSION,"n_videos":len(rows),
                      "aggregation_method":method,
                      "ranking":rows,
                      "active_modules":cfg.enabled_modules(),
                      "videoscore_summary":videoscore_summary,
                      "global_composite_summary":{
                          "mean": _aggregate(agg_composites, "mean"),
                          "median": _aggregate(agg_composites, "median"),
                          "p10": _aggregate(agg_composites, "p10"),
                          "p90": _aggregate(agg_composites, "p90"),
                          "n": len(agg_composites),
                      },
                      "verdict_totals":verdict_totals,
                      "n_total_shots":sum(r["n_shots"] for r in rows)}
        self._write_json(aggregated, bench_dir)
        self._write_csv(rows, bench_dir)
        md_path = self._write_markdown(aggregated, rows, bench_dir)
        self._write_html(aggregated, rows, bench_dir)
        if cfg.is_jupyter() and cfg.reporting.get("interactive",True):
            try:
                from IPython.display import display, Markdown
                display(Markdown(md_path.read_text(encoding="utf-8")))
            except: pass
        return bench_dir
    @staticmethod
    def _write_json(agg, bench_dir): (bench_dir/"benchmark.json").write_text(json.dumps(agg,indent=2,default=str),encoding="utf-8")
    @staticmethod
    def _write_csv(rows, bench_dir):
        metric_keys = set()
        for r in rows:
            for mk in (r.get("modules") or {}).keys(): metric_keys.add(mk)
        metric_keys = sorted(metric_keys)
        header = ["rank","video_name","n_shots","composite","accept","review","reject","duration_s"]
        for mk in metric_keys:
            header.append(f"{mk}__median"); header.append(f"{mk}__mean")
        path = bench_dir / "benchmark.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for r in rows:
                v = r.get("verdicts") or {}
                row = [r.get("rank"), r.get("video_name"), r.get("n_shots"),
                       r.get("composite") if r.get("composite") is not None else "",
                       v.get("ACCEPT",0), v.get("REVIEW",0), v.get("REJECT",0),
                       r.get("duration_s")]
                for mk in metric_keys:
                    stats = (r.get("modules") or {}).get(mk, {})
                    row.append(stats.get("median") if stats.get("median") is not None else "")
                    row.append(stats.get("mean") if stats.get("mean") is not None else "")
                writer.writerow(row)
        return path
    @staticmethod
    def _write_markdown(agg, rows, bench_dir):
        path = bench_dir / "benchmark.md"
        method = agg["aggregation_method"]
        lines = ["# CineInfini Benchmark", "", f"- **Version:** v{agg['version']}", f"- **Videos compared:** {agg['n_videos']}", f"- **Aggregation method:** `{method}`", f"- **Active modules:** {', '.join(agg['active_modules']) or '(none)'}", "", "## Ranking", "", "| Rank | Video | Shots | Composite | ✅ ACCEPT | ⚠️ REVIEW | ❌ REJECT | Duration (s) |", "|-----:|-------|------:|----------:|---------:|---------:|---------:|-------------:|"]
        for r in rows:
            comp = r.get("composite")
            comp_str = f"{comp:.3f}" if isinstance(comp,(int,float)) else "—"
            v = r.get("verdicts") or {}
            lines.append(f"| {r['rank']} | {r['video_name']} | {r['n_shots']} | {comp_str} | {v.get('ACCEPT',0)} | {v.get('REVIEW',0)} | {v.get('REJECT',0)} | {r.get('duration_s',0):.2f} |")
        all_metrics = sorted({mk for r in rows for mk in (r.get("modules") or {}).keys()})
        if all_metrics:
            lines += ["", f"## Per-metric {method} (one column per video)", "", "| Metric | " + " | ".join(r["video_name"] for r in rows) + " |", "|--------|" + "|".join(["------:"] * len(rows)) + "|"]
            for mk in all_metrics:
                cells = []
                for r in rows:
                    val = (r.get("modules") or {}).get(mk, {}).get(method)
                    cells.append(f"{val:.3f}" if isinstance(val,(int,float)) else "—")
                lines.append(f"| `{mk}` | " + " | ".join(cells) + " |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
    @staticmethod
    def _write_html(agg, rows, bench_dir):
        cfg = get_config()
        theme = _THEMES.get(cfg.theme(), _THEMES["dark"])
        path = bench_dir / "benchmark.html"
        max_comp = max((r["composite"] or 0.0) for r in rows) or 1.0
        bar_w = 28
        chart_w = max(360, len(rows)*(bar_w+16)+60)
        chart_h = 220
        bars = []; labels = []
        for i,r in enumerate(rows):
            comp = r["composite"] or 0.0
            h = int((comp/max_comp)*(chart_h-60))
            x = 30 + i*(bar_w+16)
            y = chart_h-30-h
            color = theme["good"] if comp>=0.7 else theme["warn"] if comp>=0.5 else theme["bad"]
            bars.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}" rx="3"/>'
                        f'<text x="{x+bar_w/2}" y="{y-4}" text-anchor="middle" fill="{theme["fg"]}" font-size="10">{comp:.2f}</text>')
            labels.append(f'<text x="{x+bar_w/2}" y="{chart_h-12}" text-anchor="middle" fill="{theme["muted"]}" font-size="10">{html.escape(r["video_name"][:8])}</text>')
        rank_rows_html = "\n".join([f"<tr><td>{r['rank']}</td><td>{html.escape(str(r['video_name']))}</td><td>{r['n_shots']}</td><td>{(r['composite'] or 0):.3f}</td><td><span class='badge ok'>{(r.get('verdicts') or {}).get('ACCEPT',0)}</span></td><td><span class='badge warn'>{(r.get('verdicts') or {}).get('REVIEW',0)}</span></td><td><span class='badge bad'>{(r.get('verdicts') or {}).get('REJECT',0)}</span></td></tr>" for r in rows])
        # KPI cards
        gcs = agg.get("global_composite_summary", {})
        vt = agg.get("verdict_totals", {})
        gc_mean = gcs.get("mean")
        gc_median = gcs.get("median")
        gc_mean_s = f"{gc_mean:.3f}" if isinstance(gc_mean, (int, float)) else "—"
        gc_med_s = f"{gc_median:.3f}" if isinstance(gc_median, (int, float)) else "—"
        score_class = lambda v: ("good" if isinstance(v,(int,float)) and v>=0.7
                                 else "warn" if isinstance(v,(int,float)) and v>=0.4
                                 else "bad" if isinstance(v,(int,float))
                                 else "muted")
        kpi_html = "".join([
            f"<div class='kpi'><div class='kpi-label'>Videos</div><div class='kpi-value'>{agg['n_videos']}</div></div>",
            f"<div class='kpi'><div class='kpi-label'>Total shots</div><div class='kpi-value'>{agg['n_total_shots']}</div></div>",
            f"<div class='kpi'><div class='kpi-label'>Composite (mean)</div><div class='kpi-value {score_class(gc_mean)}'>{gc_mean_s}</div></div>",
            f"<div class='kpi'><div class='kpi-label'>Composite (median)</div><div class='kpi-value {score_class(gc_median)}'>{gc_med_s}</div></div>",
            f"<div class='kpi'><div class='kpi-label'>Accept</div><div class='kpi-value good'>{vt.get('ACCEPT',0)}</div></div>",
            f"<div class='kpi'><div class='kpi-label'>Review</div><div class='kpi-value warn'>{vt.get('REVIEW',0)}</div></div>",
            f"<div class='kpi'><div class='kpi-label'>Reject</div><div class='kpi-value bad'>{vt.get('REJECT',0)}</div></div>",
        ])
        # VideoScore axes table
        vs = agg.get("videoscore_summary") or {}
        vs_rows_list = []
        def _f(v): return f"{v:.3f}" if isinstance(v, (int, float)) else "—"
        for axis, stats in vs.items():
            vs_rows_list.append(
                f"<tr><td><code>{html.escape(axis)}</code></td>"
                f"<td>{_f(stats.get('mean'))}</td>"
                f"<td>{_f(stats.get('median'))}</td>"
                f"<td>{_f(stats.get('p10'))}</td>"
                f"<td>{_f(stats.get('p90'))}</td></tr>"
            )
        vs_rows = "".join(vs_rows_list) or "<tr><td colspan='5' class='muted'>No VideoScore axes available</td></tr>"
        page = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>CineInfini · Benchmark</title>
<style>
:root {{ --bg: {theme["bg"]}; --fg: {theme["fg"]}; --accent: {theme["accent"]};
         --muted: {theme["muted"]}; --good: {theme["good"]}; --warn: {theme["warn"]}; --bad: {theme["bad"]}; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.5 sans-serif; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
h1 {{ font-size: 26px; margin: 0 0 6px; }}
h2 {{ font-size: 14px; margin: 32px 0 14px; color: var(--accent); text-transform: uppercase; letter-spacing: 0.6px; font-weight: 700; }}
.muted {{ color: var(--muted); font-size: 13px; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }}
.kpi {{ background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 16px; }}
.kpi-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
.kpi-value {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
.kpi-value.good {{ color: var(--good); }}
.kpi-value.warn {{ color: var(--warn); }}
.kpi-value.bad {{ color: var(--bad); }}
.kpi-value.muted {{ color: var(--muted); }}
table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.08); }}
th {{ color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
.badge {{ display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
.badge.ok {{ background: var(--good); color: #fff; }}
.badge.warn {{ background: var(--warn); color: #000; }}
.badge.bad {{ background: var(--bad); color: #fff; }}
code {{ background: rgba(255,255,255,0.06); padding: 1px 6px; border-radius: 3px; font-size: 12px; }}
svg.chart {{ background: rgba(255,255,255,0.03); border-radius: 6px; }}
</style>
</head><body><div class="container">
<h1>CineInfini Benchmark</h1>
<div class="muted">Version v{agg['version']} · {agg['n_videos']} videos · {agg['n_total_shots']} total shots · aggregation: {agg['aggregation_method']}</div>

<h2>Key indicators</h2>
<div class="kpi-grid">
{kpi_html}
</div>

<h2>VideoScore axes — aggregate across videos</h2>
<table>
  <thead><tr><th>Axis</th><th>Mean</th><th>Median</th><th>P10</th><th>P90</th></tr></thead>
  <tbody>{vs_rows}</tbody>
</table>

<h2>Composite ranking</h2>
<svg class="chart" width="{chart_w}" height="{chart_h}" xmlns="http://www.w3.org/2000/svg">
{''.join(bars)}{''.join(labels)}
</svg>
<h2>Detail</h2>
<table><thead><tr><th>Rank</th><th>Video</th><th>Shots</th><th>Composite</th><th>ACCEPT</th><th>REVIEW</th><th>REJECT</th></tr></thead>
<tbody>{rank_rows_html}</tbody></table>
</div></body></html>"""
        path.write_text(page, encoding="utf-8")
        return path

def run_benchmark_audit(video_paths, output_dir=None, *, force_full_video=False):
    from ...pipeline.orchestrator import run_audit
    cfg = get_config()
    out_root = Path(output_dir) if output_dir else cfg.reports_dir() / "_benchmark"
    out_root.mkdir(parents=True, exist_ok=True)
    audits = []
    for vp in video_paths:
        vp = Path(vp)
        per_video_dir = out_root / vp.stem
        try:
            data, _ = run_audit(vp, output_dir=per_video_dir, force_full_video=force_full_video)
            audits.append(data)
        except Exception as e:
            logger.exception("Benchmark: audit failed for %s: %s", vp, e)
    bench_dir = BenchmarkRenderer().render_many(audits, out_root)
    return audits, bench_dir or out_root
