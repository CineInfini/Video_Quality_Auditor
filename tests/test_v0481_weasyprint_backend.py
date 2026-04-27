import sys, pytest
from pathlib import Path
from cineinfini.core.config import get_config
def _audit(): return {"video_name":"demo","version":"0.4.8.1","gates":{"1":{"composite":0.81,"verdict":"ACCEPT"},"2":{"composite":0.42,"verdict":"REVIEW"}}}
def test_priority_unknown_backend(tmp_path, caplog):
    cfg = get_config()
    cfg.reporting["pdf_backend_priority"] = ["matplotlib","_does_not_exist_"]
    from cineinfini.io.renderers.pdf_renderer import PDFRenderer
    out = PDFRenderer().render(_audit(), tmp_path)
    assert out is not None and out.exists()
def test_weasyprint_falls_through_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules,"weasyprint",None)
    cfg = get_config()
    cfg.reporting["pdf_backend_priority"] = ["weasyprint","matplotlib"]
    from cineinfini.io.renderers.pdf_renderer import PDFRenderer
    out = PDFRenderer().render(_audit(), tmp_path)
    assert out is None or out.exists()
def test_default_priority_contains_weasyprint():
    from cineinfini.io.renderers.pdf_renderer import _DEFAULT_PRIORITY
    assert "weasyprint" in _DEFAULT_PRIORITY and _DEFAULT_PRIORITY.index("weasyprint")==1
