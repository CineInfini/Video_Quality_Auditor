"""
Report generation and markdown dashboard.
"""
from pathlib import Path
from cineinfini.io.report import generate_intra_report, safe_format

def test_safe_format():
    assert safe_format(None) == "—"
    assert safe_format(1.234) == "1.234"
    assert safe_format(1.234, ".2f") == "1.23"

def test_generate_intra_report_with_none(tmp_path):
    metrics = {
        "gates": {"1": {"motion_peak_div":12.34, "ssim3d_self":None, "flicker":0.01,
                        "identity_intra":None, "ssim_long_range":0.95, "flicker_hf_var":0.02,
                        "clip_temp_consistency":None}},
        "video_info": {"duration":10},
        "params_used": {"thresholds":{}}
    }
    out = tmp_path / "reports"
    generate_intra_report("test", metrics, out, {"motion":25.0})
    dashboard = out / "test" / "dashboard.md"
    assert dashboard.exists()
    assert "—" in dashboard.read_text()
