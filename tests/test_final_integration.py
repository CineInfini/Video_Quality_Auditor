"""
Final integration: custom config, step=2, adaptive threshold.
"""
import pytest
from pathlib import Path
from cineinfini import generate_synthetic_video, set_config, get_config, audit_video, set_global_paths
from cineinfini.core.face_detection import set_models_dir

@pytest.fixture(scope="module")
def models_dir():
    base = Path("/content/cineinfini-dev")
    models = base / "_models"
    if not models.exists():
        pytest.skip("Models not found")
    return models

@pytest.fixture
def tmp_env(tmp_path, models_dir):
    reports = tmp_path / "reports"
    benchmark = tmp_path / "benchmark"
    reports.mkdir(); benchmark.mkdir()
    set_models_dir(models_dir)
    set_global_paths(reports, benchmark)
    return {"reports": reports, "benchmark": benchmark}

def test_custom_config_audit_with_step2(tmp_env):
    set_config("n_frames_per_shot", 8)
    set_config("max_duration_s", 30)
    config = get_config()
    assert config["n_frames_per_shot"] == 8
    assert config["max_duration_s"] == 30
    video = tmp_env["reports"].parent / "synth.mp4"
    generate_synthetic_video(str(video), 5, 24, (640,480), "circle")
    _, report_dir = audit_video(str(video), force_full_video=False)
    dashboard = Path(report_dir) / "dashboard.md"
    assert dashboard.exists()
    content = dashboard.read_text()
    for h in ["Motion", "3D-SSIM", "Flicker", "Identity drift", "SSIM LR", "CLIP temp"]:
        assert h in content
    import json
    data = json.loads((Path(report_dir) / "data.json").read_text())
    assert "gates" in data and len(data["gates"]) > 0

def test_adaptive_threshold_behavior(tmp_env):
    set_config("adaptive_threshold", True)
    set_config("threshold_percentile", 85)
    video = tmp_env["reports"].parent / "adaptive.mp4"
    generate_synthetic_video(str(video), 3, 24, (320,240), "color_switch")
    _, report_dir = audit_video(str(video), force_full_video=False)
    dashboard = Path(report_dir) / "dashboard.md"
    assert dashboard.exists()
    import json
    data = json.loads((Path(report_dir) / "data.json").read_text())
    assert "params_used" in data and "thresholds" in data["params_used"]
