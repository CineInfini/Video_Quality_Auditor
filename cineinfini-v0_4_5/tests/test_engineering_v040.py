"""
Tests for v0.4.0 engineering modules:
  - core/config.py     (singleton, test_config, from_dict)
  - core/calibrate.py  (grid_search, logistic, annotations loader)
  - io/reader.py       (decomposed detect_shot_boundaries subfunctions)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ===========================================================================
# Config singleton
# ===========================================================================

class TestConfig:

    def test_default_config_returns_config_instance(self):
        from cineinfini.core.config import default_config, Config
        cfg = default_config()
        assert isinstance(cfg, Config)

    def test_resolve_path_expands_tilde(self):
        from cineinfini.core.config import default_config
        cfg = default_config()
        p = cfg.resolve_path("models_dir")
        assert not str(p).startswith("~")
        assert p.is_absolute()

    def test_models_dir_helper(self):
        from cineinfini.core.config import default_config
        cfg = default_config()
        assert cfg.models_dir() == cfg.resolve_path("models_dir")

    def test_test_config_uses_tmp(self):
        from cineinfini.core.config import test_config
        cfg = test_config()
        # reports and cache go to /tmp, NOT ~/.cineinfini
        r = cfg.resolve_path("reports_dir")
        assert str(r).startswith("/tmp")

    def test_test_config_fast_settings(self):
        from cineinfini.core.config import test_config
        cfg = test_config()
        assert cfg.processing["max_duration_s"] == 10
        assert cfg.processing["n_frames_per_shot"] == 8
        assert cfg.processing["narrative_coherence"] is False

    def test_set_get_config(self):
        from cineinfini.core.config import (
            set_config, get_config, reset_config, test_config, default_config
        )
        original = default_config()
        try:
            tc = test_config()
            set_config(tc)
            assert get_config().processing["max_duration_s"] == 10
        finally:
            set_config(original)

    def test_reset_config(self):
        from cineinfini.core.config import set_config, reset_config, get_config, test_config
        set_config(test_config())
        reset_config()
        # Next get_config() should not raise
        cfg = get_config()
        assert cfg is not None

    def test_from_dict_merges_defaults(self):
        from cineinfini.core.config import Config
        cfg = Config.from_dict({"processing": {"max_duration_s": 42}})
        assert cfg.processing["max_duration_s"] == 42
        assert cfg.processing["shot_threshold"] == 0.2  # default kept

    def test_model_url_returns_dict(self):
        from cineinfini.core.config import default_config
        cfg = default_config()
        entry = cfg.model_url("arcface")
        assert entry is not None
        assert "url" in entry
        assert "filename" in entry

    def test_model_url_unknown_key_returns_none(self):
        from cineinfini.core.config import default_config
        cfg = default_config()
        assert cfg.model_url("nonexistent_model_xyz") is None

    def test_to_audit_config_has_thresholds(self):
        from cineinfini.core.config import default_config
        cfg = default_config()
        audit_cfg = cfg.to_audit_config()
        assert "thresholds" in audit_cfg
        assert "motion" in audit_cfg["thresholds"]

    def test_save_load_roundtrip_yaml(self, tmp_path):
        from cineinfini.core.config import default_config, save_config, load_config
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")
        cfg = default_config()
        path = tmp_path / "test_config.yaml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.processing["max_duration_s"] == cfg.processing["max_duration_s"]
        assert loaded.thresholds["motion"] == cfg.thresholds["motion"]

    def test_replace_creates_new_instance(self):
        from cineinfini.core.config import default_config
        cfg = default_config()
        cfg2 = cfg.replace(device="cuda")
        assert cfg2.device == "cuda"
        assert cfg.device == "cpu"   # original unchanged


# ===========================================================================
# Reader decomposition
# ===========================================================================

class TestReaderDecomposition:
    """Test the three sub-functions of detect_shot_boundaries independently."""

    def test_apply_threshold_adaptive(self):
        from cineinfini.io.reader import _apply_threshold
        diffs = [0.1, 0.2, 0.5, 0.3, 0.9, 0.1]
        th = _apply_threshold(diffs, shot_threshold=0.3, adaptive=True, percentile=80)
        # 80th percentile of the diffs
        expected = float(np.percentile(diffs, 80))
        assert th == pytest.approx(expected)

    def test_apply_threshold_fixed(self):
        from cineinfini.io.reader import _apply_threshold
        diffs = [0.1, 0.2, 0.5]
        th = _apply_threshold(diffs, shot_threshold=0.42, adaptive=False, percentile=85)
        assert th == pytest.approx(0.42)

    def test_apply_threshold_empty_diffs_uses_fixed(self):
        from cineinfini.io.reader import _apply_threshold
        th = _apply_threshold([], shot_threshold=0.25, adaptive=True, percentile=85)
        assert th == pytest.approx(0.25)

    def test_boundaries_to_shots_single_shot(self):
        from cineinfini.io.reader import _boundaries_to_shots
        diffs = [0.1, 0.1, 0.1, 0.1]  # no cut
        shots = _boundaries_to_shots(diffs, threshold=0.5, limit=100, fps=24.0, min_gap=12, step=1)
        assert len(shots) == 1
        assert shots[0][0] == 0
        assert shots[0][2] == 24.0

    def test_boundaries_to_shots_detects_cut(self):
        from cineinfini.io.reader import _boundaries_to_shots
        diffs = [0.1] * 10 + [0.9] + [0.1] * 10  # one cut at index 10
        shots = _boundaries_to_shots(diffs, threshold=0.5, limit=100, fps=24.0, min_gap=2, step=1)
        assert len(shots) >= 2

    def test_boundaries_to_shots_respects_min_gap(self):
        from cineinfini.io.reader import _boundaries_to_shots
        # Two cuts very close together (gap < min_gap) → second one ignored
        diffs = [0.9, 0.9] + [0.1] * 20
        shots = _boundaries_to_shots(diffs, threshold=0.5, limit=50, fps=24.0, min_gap=10, step=1)
        # The two cuts are at frame 1 and 2 — gap=1 < min_gap=10, so second cut ignored
        assert all(e - s >= 10 for s, e, _ in shots)

    def test_detect_shot_boundaries_on_real_synthetic_video(self, tmp_path):
        from cineinfini.io.reader import detect_shot_boundaries
        from cineinfini import generate_synthetic_video
        v = tmp_path / "v.mp4"
        generate_synthetic_video(str(v), 3.0, 24, (320, 240), "color_switch")
        shots = detect_shot_boundaries(str(v), max_duration_s=3, shot_threshold=0.2,
                                       min_shot_duration_s=0.3, downsample_to=(160, 90))
        assert len(shots) >= 1
        for s, e, fps in shots:
            assert s >= 0
            assert e >= s
            assert fps > 0


# ===========================================================================
# Calibrate
# ===========================================================================

class TestCalibrate:

    def _make_df(self, n: int = 100, seed: int = 42):
        """Build a synthetic annotations dataframe."""
        import pandas as pd
        rng = np.random.default_rng(seed)
        good = rng.integers(0, 2, n)  # 0=reject, 1=accept
        df = pd.DataFrame({
            "motion":        np.where(good, rng.uniform(5, 20, n), rng.uniform(20, 45, n)),
            "ssim3d":        np.where(good, rng.uniform(0.5, 0.95, n), rng.uniform(0.1, 0.5, n)),
            "flicker":       np.where(good, rng.uniform(0, 0.08, n), rng.uniform(0.08, 0.3, n)),
            "identity_drift":np.where(good, rng.uniform(0, 0.4, n), rng.uniform(0.4, 0.9, n)),
            "label":         np.where(good, "ACCEPT", "REJECT"),
        })
        return df

    def test_load_annotations_parses_label(self, tmp_path):
        from cineinfini.core.calibrate import load_annotations
        import pandas as pd
        df = self._make_df()
        csv = tmp_path / "ann.csv"
        df.to_csv(csv, index=False)
        loaded = load_annotations(csv)
        assert "label_bin" in loaded.columns
        assert set(loaded["label_bin"].unique()).issubset({0, 1})

    def test_load_annotations_handles_column_aliases(self, tmp_path):
        from cineinfini.core.calibrate import load_annotations
        import pandas as pd
        df = self._make_df().rename(columns={"motion": "motion_peak_div"})
        csv = tmp_path / "ann.csv"
        df.to_csv(csv, index=False)
        loaded = load_annotations(csv)
        # motion_peak_div should be aliased to motion
        assert "motion" in loaded.columns

    def test_grid_search_returns_calibration_result(self, tmp_path):
        from cineinfini.core.calibrate import (
            grid_search_thresholds, load_annotations, CalibrationResult
        )
        df = self._make_df()
        csv = tmp_path / "ann.csv"
        df.to_csv(csv, index=False)
        loaded = load_annotations(csv)
        result = grid_search_thresholds(loaded)
        assert isinstance(result, CalibrationResult)
        assert result.method == "grid_search"
        assert "motion" in result.thresholds

    def test_logistic_returns_weights(self, tmp_path):
        pytest.importorskip("sklearn")
        from cineinfini.core.calibrate import (
            logistic_regression_weights, load_annotations
        )
        df = self._make_df(n=200)
        csv = tmp_path / "ann.csv"
        df.to_csv(csv, index=False)
        loaded = load_annotations(csv)
        result = logistic_regression_weights(loaded)
        assert result.method == "logistic"
        assert result.weights is not None
        assert len(result.weights) > 0
        assert "cv_accuracy" in result.metrics
        assert result.metrics["cv_accuracy"] > 0.5   # should beat random

    def test_calibration_result_summary(self, tmp_path):
        from cineinfini.core.calibrate import grid_search_thresholds, load_annotations
        df = self._make_df()
        csv = tmp_path / "ann.csv"
        df.to_csv(csv, index=False)
        loaded = load_annotations(csv)
        result = grid_search_thresholds(loaded)
        summary = result.summary()
        assert "CalibrationResult" in summary
        assert "motion" in summary

    def test_calibration_result_save_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        from cineinfini.core.calibrate import grid_search_thresholds, load_annotations
        df = self._make_df()
        csv = tmp_path / "ann.csv"
        df.to_csv(csv, index=False)
        result = grid_search_thresholds(load_annotations(csv))
        out = tmp_path / "calibrated.yaml"
        result.save(out)
        assert out.exists()
        content = out.read_text()
        assert "thresholds" in content
        assert "motion" in content
