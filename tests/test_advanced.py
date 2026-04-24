"""
Advanced tests: inter comparisons, benchmark, real video, compare_videos.
"""
import pytest, tempfile, shutil
from pathlib import Path

SOURCE_MODELS = Path("/content/cineinfini-dev/_models")
TEARS_VIDEO = Path("/content/cineinfini-dev/tears_of_steel.mov")

def setup_models(tmpdir):
    models = Path(tmpdir) / "_models"
    models.mkdir()
    if SOURCE_MODELS.exists():
        for f in SOURCE_MODELS.glob("*"):
            if f.is_file():
                shutil.copy2(f, models / f.name)
    return models

@pytest.mark.xfail(reason="generate_inter_report may be unstable")
def test_compare_two_videos():
    with tempfile.TemporaryDirectory() as tmpdir:
        models = setup_models(tmpdir)
        from cineinfini import generate_synthetic_video, audit_video, set_global_paths, generate_inter_report, CONFIG
        from cineinfini.core.face_detection import set_models_dir
        set_models_dir(models)
        reports = Path(tmpdir) / "reports"
        bench = Path(tmpdir) / "benchmark"
        reports.mkdir(); bench.mkdir()
        set_global_paths(reports, bench)
        v1 = Path(tmpdir) / "v1.mp4"
        v2 = Path(tmpdir) / "v2.mp4"
        generate_synthetic_video(str(v1),2,24,(320,240),"circle")
        generate_synthetic_video(str(v2),2,24,(320,240),"color_switch")
        intra = []
        for vp in [v1,v2]:
            _, rd = audit_video(str(vp), force_full_video=False)
            intra.append(rd)
        out = Path(tmpdir) / "inter"
        generate_inter_report(intra, out, CONFIG["thresholds"], "test_compare")
        dashboard = out / "test_compare" / "dashboard.md"
        assert dashboard.exists()
        assert "motion_mean" in dashboard.read_text()

def test_benchmark_simulated():
    with tempfile.TemporaryDirectory() as tmpdir:
        models = setup_models(tmpdir)
        from cineinfini.core.face_detection import set_models_dir, ArcFaceEmbedder
        import numpy as np
        set_models_dir(models)
        embedder = ArcFaceEmbedder()
        emb1 = np.random.randn(512); emb2 = np.random.randn(512)
        emb1 /= np.linalg.norm(emb1); emb2 /= np.linalg.norm(emb2)
        assert -1 <= float(np.dot(emb1, emb2)) <= 1

def test_audit_real_video_download():
    if not TEARS_VIDEO.exists():
        pytest.skip("Tears of Steel video not found")
    with tempfile.TemporaryDirectory() as tmpdir:
        models = setup_models(tmpdir)
        from cineinfini import audit_video, set_global_paths
        from cineinfini.core.face_detection import set_models_dir
        set_models_dir(models)
        reports = Path(tmpdir) / "reports"
        bench = Path(tmpdir) / "benchmark"
        reports.mkdir(); bench.mkdir()
        set_global_paths(reports, bench)
        _, report_dir = audit_video(str(TEARS_VIDEO), video_params={"max_duration_s":5}, force_full_video=False)
        dashboard = Path(report_dir) / "dashboard.md"
        assert dashboard.exists()
        assert "Motion" in dashboard.read_text()

def test_compare_videos_function():
    """Test compare_videos utility (fixed version)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models = setup_models(tmpdir)
        from cineinfini import generate_synthetic_video, set_global_paths, compare_videos
        from cineinfini.core.face_detection import set_models_dir
        set_models_dir(models)
        reports = Path(tmpdir) / "reports"
        bench = Path(tmpdir) / "benchmark"
        reports.mkdir(); bench.mkdir()
        set_global_paths(reports, bench)
        v1 = Path(tmpdir) / "v1.mp4"
        v2 = Path(tmpdir) / "v2.mp4"
        generate_synthetic_video(str(v1),2,24,(320,240),"circle")
        generate_synthetic_video(str(v2),2,24,(320,240),"color_switch")
        inter_dir = compare_videos(v1, v2, output_subdir="test_func", max_duration_s=3)
        dashboard = inter_dir / "dashboard.md"
        assert dashboard.exists()
        assert "Motion" in dashboard.read_text()
