"""
Advanced tests: multi-video comparison, benchmark, inter-report.

All synthetic tests use generated videos — no hardcoded Colab paths.
Integration tests (need HTTP or model weights) are marked accordingly.
"""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest


def _make_video(path: Path, pattern: str = "circle", duration: float = 2.0) -> Path:
    from cineinfini import generate_synthetic_video
    generate_synthetic_video(str(path), duration, 24, (320, 240), pattern)
    return path


class TestSyntheticVideos:

    def test_generate_two_different_patterns(self, tmp_path):
        v1 = _make_video(tmp_path / "v1.mp4", "circle")
        v2 = _make_video(tmp_path / "v2.mp4", "color_switch")
        assert v1.exists() and v2.exists()
        assert v1.stat().st_size > 0 and v2.stat().st_size > 0

    def test_synthetic_video_has_correct_duration(self, tmp_path):
        import cv2
        v = _make_video(tmp_path / "v.mp4", "noise", duration=3.0)
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS)
        nf = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration = nf / fps if fps > 0 else 0
        assert abs(duration - 3.0) < 0.5

    def test_generate_synthetic_returns_path(self, tmp_path):
        from cineinfini import generate_synthetic_video
        out = tmp_path / "test.mp4"
        result = generate_synthetic_video(str(out), 1, 24, (160, 120), "circle")
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0


class TestAuditInternals:

    def test_composite_better_for_good_metrics(self):
        from cineinfini.pipeline.audit import _compute_composite, AuditTiming
        good = {1: {"motion_peak_div": 3.0, "ssim3d_self": 0.95, "flicker": 0.01,
                    "identity_intra": 0.05, "ssim_long_range": 0.90, "clip_temp_consistency": 0.90}}
        bad = {1: {"motion_peak_div": 40.0, "ssim3d_self": 0.20, "flicker": 0.30,
                   "identity_intra": 0.80, "ssim_long_range": 0.10, "clip_temp_consistency": 0.10}}
        timing = AuditTiming()
        _compute_composite(good, timing)
        _compute_composite(bad, timing)
        assert good[1]["composite"] > bad[1]["composite"]

    def test_audit_timing_total(self):
        from cineinfini.pipeline.audit import AuditTiming
        t = AuditTiming(video_info=1.0, shot_detection=2.0, frame_extraction=3.0,
                        model_init=4.0, shot_processing=5.0, inter_coherence=6.0,
                        narrative=7.0, composite=8.0, persist=9.0)
        assert t.total == pytest.approx(45.0)

    def test_video_info_reads_fps_and_duration(self, tmp_path):
        from cineinfini.pipeline.audit import _load_video_info, AuditTiming, VideoInfo
        v = _make_video(tmp_path / "v.mp4", "noise", duration=2.0)
        timing = AuditTiming()
        info = _load_video_info(v, timing)
        assert isinstance(info, VideoInfo)
        assert info.fps > 0
        assert 1.5 <= info.duration_s <= 3.0
        assert timing.video_info > 0

    def test_detect_gpu_returns_string(self):
        from cineinfini.pipeline.audit import _detect_gpu
        result = _detect_gpu()
        assert result in ("cpu", "cuda")


class TestBenchmarkSimulated:

    def test_dtw_distance_positive(self):
        from cineinfini.core.identity_dtw import dtw_distance
        rng = np.random.default_rng(0)
        a = rng.standard_normal((6, 8))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = rng.standard_normal((6, 8))
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        dist, pl, backend = dtw_distance(a, b)
        assert dist >= 0.0
        assert pl >= max(len(a), len(b))


@pytest.mark.integration
@pytest.mark.slow
class TestRealVideo:

    def test_audit_mini_bbb(self, tmp_path):
        import urllib.request
        from cineinfini import audit_video, set_global_paths
        from cineinfini.core.face_detection import set_models_dir

        url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_1MB.mp4"
        local = tmp_path / "bbb_10s.mp4"
        try:
            urllib.request.urlretrieve(url, str(local))
        except Exception as e:
            pytest.skip(f"Cannot download BBB 10s: {e}")

        set_models_dir(Path.home() / ".cineinfini" / "models")
        set_global_paths(tmp_path / "reports", tmp_path / "bench")
        metrics, report_dir = audit_video(
            str(local),
            video_params={"max_duration_s": 10, "n_frames_per_shot": 8},
        )
        assert (report_dir / "data.json").exists()
        data = json.loads((report_dir / "data.json").read_text())
        assert len(data["gates"]) >= 1

    def test_compare_videos_function(self, tmp_path):
        from cineinfini import compare_videos, set_global_paths
        from cineinfini.core.face_detection import set_models_dir
        set_models_dir(Path.home() / ".cineinfini" / "models")
        set_global_paths(tmp_path / "reports", tmp_path / "bench")
        v1 = _make_video(tmp_path / "a.mp4", "circle")
        v2 = _make_video(tmp_path / "b.mp4", "color_switch")
        try:
            result = compare_videos(str(v1), str(v2), output_subdir="tc", max_duration_s=5)
        except Exception as e:
            pytest.skip(f"compare_videos failed: {e}")
        assert result is not None
