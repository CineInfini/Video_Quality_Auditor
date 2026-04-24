"""
Face detection tests.
"""
from pathlib import Path
import tempfile
from cineinfini.core.face_detection import set_models_dir, CascadeFaceDetector

def test_detector_initialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        set_models_dir(Path(tmpdir))
        detector = CascadeFaceDetector()
        assert len(detector.names) > 0
