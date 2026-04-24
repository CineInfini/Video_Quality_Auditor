"""
Full audit pipeline on synthetic video.
"""
import tempfile, shutil
from pathlib import Path
from cineinfini import generate_synthetic_video, audit_video, set_global_paths
from cineinfini.core.face_detection import set_models_dir

SOURCE_MODELS = Path("/content/cineinfini-dev/_models")

def test_audit_synthetic():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        models_dir = base / "_models"
        reports = base / "reports"
        bench = base / "benchmark"
        models_dir.mkdir(); reports.mkdir(); bench.mkdir()
        if SOURCE_MODELS.exists():
            for f in SOURCE_MODELS.glob("*"):
                if f.is_file():
                    shutil.copy2(f, models_dir / f.name)
        set_models_dir(models_dir)
        set_global_paths(reports, bench)
        video = base / "test.mp4"
        generate_synthetic_video(str(video), 1, 24, (320,240), "circle")
        _, report_dir = audit_video(str(video), force_full_video=False)
        assert (Path(report_dir) / "dashboard.md").exists()
