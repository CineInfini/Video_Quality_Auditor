# src/cineinfini/io/renderers/pdf_renderer.py
"""
PDFRenderer — multi-backend, fail-safe PDF report (v0.4.8.1).
Backend order driven by cfg.reporting["pdf_backend_priority"].
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from ...core.config import get_config
from ...core.context import VideoContext
from ...core.ui_registry import register_renderer
from .base import BaseRenderer
logger = logging.getLogger("cineinfini.renderers.pdf")
_DEFAULT_PRIORITY: List[str] = ["reportlab", "weasyprint", "fpdf2", "matplotlib"]

@register_renderer("pdf", description="Scientific PDF report (ReportLab/WeasyPrint/fpdf2/matplotlib).")
class PDFRenderer(BaseRenderer):
    renderer_id = "pdf"
    def render(self, audit_data: Dict[str, Any], output_dir: Path, context: Optional[VideoContext] = None) -> Optional[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "report.pdf"
        rows = self._build_rows(audit_data)
        backends: Dict[str, Callable[..., bool]] = {
            "reportlab":  self._render_reportlab,
            "weasyprint": self._render_weasyprint,
            "fpdf2":      self._render_fpdf2,
            "matplotlib": self._render_matplotlib,
        }
        cfg = get_config()
        priority = list(cfg.reporting.get("pdf_backend_priority") or _DEFAULT_PRIORITY)
        safe_priority = [p for p in priority if p in backends]
        if not safe_priority:
            safe_priority = list(_DEFAULT_PRIORITY)
        for backend in safe_priority:
            try:
                if backends[backend](out_path, audit_data, rows):
                    logger.info("PDFRenderer: used %s backend → %s", backend, out_path)
                    return out_path
            except Exception as e:
                logger.debug("PDFRenderer: %s backend failed (%s)", backend, e)
        logger.warning("PDFRenderer: no PDF backend available — skipping report.pdf")
        return None
    @staticmethod
    def _build_rows(data):
        rows = []
        for sid, gate in (data.get("gates") or {}).items():
            comp = gate.get("composite")
            comp_str = f"{comp:.3f}" if isinstance(comp,(int,float)) else "—"
            rows.append((str(sid), comp_str, str(gate.get("verdict","—")).upper()))
        return rows
    def _render_reportlab(self, out_path, data, rows):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        except Exception:
            return False
        cfg = get_config()
        styles = getSampleStyleSheet()
        story = [Paragraph(f"<b>CineInfini audit — {data.get('video_name','?')}</b>", styles["Title"]), Spacer(1,6),
                 Paragraph(f"Version v{data.get('version','0.4.8.1')}", styles["Normal"]),
                 Paragraph(f"Active modules: {', '.join(cfg.enabled_modules()) or '(none)'}", styles["Normal"]),
                 Paragraph(f"Theme: {cfg.theme()} · Figure format: {cfg.figure_format()}", styles["Normal"]),
                 Spacer(1,12), Paragraph("<b>Per-shot composite</b>", styles["Heading2"])]
        table_data = [["Shot","Composite","Verdict"]] + rows
        tbl = Table(table_data, hAlign="LEFT")
        tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0d1117")),
                                 ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                                 ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                                 ("ALIGN",(0,0),(-1,-1),"LEFT"),
                                 ("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
        story.append(tbl)
        story.append(PageBreak())
        story.append(Paragraph("<b>Active configuration</b>", styles["Heading2"]))
        story.append(Paragraph(f"Thresholds: {dict(cfg.thresholds)}", styles["Code"]))
        SimpleDocTemplate(str(out_path), pagesize=A4).build(story)
        return True
    def _render_weasyprint(self, out_path, data, rows):
        try:
            from weasyprint import HTML, CSS
        except (ImportError, OSError) as e:
            logger.debug("WeasyPrint unavailable (%s)", e); return False
        try:
            from .html_dashboard import HTMLDashboardRenderer, _THEMES
        except Exception:
            return False
        cfg = get_config()
        theme = _THEMES.get(cfg.theme(), _THEMES["dark"])
        html_renderer = HTMLDashboardRenderer()
        html_string = html_renderer._build_page(data, theme, cfg)
        print_css = CSS(string="""
            @page { size: A4; margin: 18mm 14mm 18mm 14mm; }
            body { -weasy-page-break-inside: avoid; }
            h1 { page-break-after: avoid; }
            table { page-break-inside: auto; }
            tr { page-break-inside: avoid; page-break-after: auto; }
            pre { white-space: pre-wrap; word-wrap: break-word; font-size: 9px; }
        """)
        HTML(string=html_string).write_pdf(str(out_path), stylesheets=[print_css])
        return True
    def _render_fpdf2(self, out_path, data, rows):
        try:
            from fpdf import FPDF
        except Exception:
            return False
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica","B",16); pdf.cell(0,10,f"CineInfini audit — {data.get('video_name','?')}",ln=True)
        pdf.set_font("Helvetica","",11); pdf.cell(0,8,f"Version v{data.get('version','0.4.8.1')}",ln=True); pdf.ln(4)
        pdf.set_font("Helvetica","B",12); pdf.cell(30,8,"Shot"); pdf.cell(40,8,"Composite"); pdf.cell(40,8,"Verdict",ln=True)
        pdf.set_font("Helvetica","",11)
        for shot,comp,verdict in rows: pdf.cell(30,8,shot); pdf.cell(40,8,comp); pdf.cell(40,8,verdict,ln=True)
        pdf.output(str(out_path))
        return True
    def _render_matplotlib(self, out_path, data, rows):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except Exception:
            return False
        cfg = get_config()
        with PdfPages(str(out_path)) as pdf:
            fig,ax = plt.subplots(figsize=(8.27,11.69)); ax.axis("off")
            ax.text(0.05,0.95,f"CineInfini audit — {data.get('video_name','?')}",fontsize=18,weight="bold",transform=ax.transAxes)
            ax.text(0.05,0.91,f"Version v{data.get('version','0.4.8.1')}",fontsize=11,transform=ax.transAxes)
            ax.text(0.05,0.88,f"Active modules: {', '.join(cfg.enabled_modules()) or '(none)'}",fontsize=10,transform=ax.transAxes)
            pdf.savefig(fig); plt.close(fig)
            fig,ax = plt.subplots(figsize=(8.27,11.69)); ax.axis("off")
            tbl = ax.table(cellText=rows or [["—","—","—"]], colLabels=["Shot","Composite","Verdict"], loc="upper left", cellLoc="left")
            tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.4)
            pdf.savefig(fig); plt.close(fig)
        return True
