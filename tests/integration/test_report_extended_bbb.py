"""Integration test for report_extended on Big Buck Bunny."""
import pytest
from pathlib import Path
import urllib.request

BBB_URL = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
CACHE_DIR = Path.home()/".cineinfini"/"test_videos"
CACHED_VIDEO = CACHE_DIR/"BigBuckBunny.mp4"

@pytest.fixture(scope="module")
def bbb_video():
    CACHE_DIR.mkdir(parents=True,exist_ok=True)
    if CACHED_VIDEO.exists() and CACHED_VIDEO.stat().st_size>100*1024*1024:
        return CACHED_VIDEO
    print("Downloading BBB (158 MB)...")
    tmp = CACHED_VIDEO.with_suffix(".part")
    urllib.request.urlretrieve(BBB_URL, tmp)
    tmp.rename(CACHED_VIDEO)
    return CACHED_VIDEO

@pytest.mark.integration
def test_bbb_audit_produces_data_json(bbb_video, tmp_path):
    from cineinfini import audit_video, set_global_paths, CONFIG
    set_global_paths(Path.home()/".cineinfini"/"models", tmp_path/"reports", tmp_path/"benchmark")
    CONFIG["max_duration_s"]=30
    CONFIG["compute_dtw_self"]=True
    metrics, report_dir = audit_video(str(bbb_video))
    data_json = Path(report_dir)/"data.json"
    assert data_json.exists()
    assert len(metrics["gates"])>=1

@pytest.mark.integration
def test_bbb_extended_report_generates_figures(bbb_video, tmp_path):
    from cineinfini import audit_video, set_global_paths, CONFIG
    from cineinfini.io.report_extended import generate_extended_intra_report
    set_global_paths(Path.home()/".cineinfini"/"models", tmp_path/"reports", tmp_path/"benchmark")
    CONFIG["max_duration_s"]=30
    metrics, report_dir = audit_video(str(bbb_video))
    paths = generate_extended_intra_report("BBB", metrics, tmp_path, {"motion":25.0,"ssim3d":0.45}, save_html=True)
    assert "html" in paths and Path(paths["html"]).exists()
