"""
Tests for v0.4.6 ported modules:
  - core/phase4_aggregator.py  (aggregate_shot_verdict, build_phase4_report)
  - core/inter_shot_loss.py    (InterShotCoherenceLoss with asymmetric weights)
  - core/prompt_engineering.py (ShotMetadata, build_prompt, safety neutralization)
  - face_detection migration to get_config()
  - report.py decomposition
"""
from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pytest


# ===========================================================================
# phase4_aggregator
# ===========================================================================

class TestPhase4Aggregator:

    def test_module_imports(self):
        from cineinfini.core.phase4_aggregator import (
            GateThresholds, ShotVerdict, aggregate_shot_verdict, build_phase4_report
        )
        assert GateThresholds is not None

    def test_default_thresholds(self):
        from cineinfini.core.phase4_aggregator import GateThresholds
        th = GateThresholds()
        assert th.motion_accept_below > 0
        assert th.motion_reject_above > th.motion_accept_below

    def test_aggregate_good_shot_returns_accept(self):
        from cineinfini.core.phase4_aggregator import aggregate_shot_verdict
        v = aggregate_shot_verdict(
            shot_id=1, motion_peak_div=5.0, ssim3d=0.92,
            identity_drift=0.10, safety_status="ok",
        )
        assert v.verdict in ("ACCEPT", "REVIEW")
        assert v.shot_id == 1

    def test_aggregate_bad_shot_returns_reject(self):
        from cineinfini.core.phase4_aggregator import aggregate_shot_verdict
        v = aggregate_shot_verdict(
            shot_id=2, motion_peak_div=80.0, ssim3d=0.10,
            identity_drift=0.95, safety_status="ok",
        )
        assert v.verdict in ("REJECT", "REVIEW")

    def test_aggregate_blocked_shot(self):
        from cineinfini.core.phase4_aggregator import aggregate_shot_verdict
        v = aggregate_shot_verdict(
            shot_id=3, motion_peak_div=5.0, ssim3d=0.92,
            identity_drift=0.10, safety_status="blocked_for_review",
        )
        assert v.verdict == "BLOCKED_SAFETY"

    def test_aggregate_no_data(self):
        from cineinfini.core.phase4_aggregator import aggregate_shot_verdict
        v = aggregate_shot_verdict(
            shot_id=4, motion_peak_div=None, ssim3d=None,
            identity_drift=None, safety_status="ok",
        )
        assert v.verdict == "NO_DATA"

    def test_build_phase4_report_returns_markdown(self):
        from cineinfini.core.phase4_aggregator import (
            aggregate_shot_verdict, build_phase4_report, GateThresholds
        )
        verdicts = [
            aggregate_shot_verdict(1, 5.0, 0.92, 0.10, "ok"),
            aggregate_shot_verdict(2, 80.0, 0.10, 0.95, "ok"),
        ]
        md = build_phase4_report(
            verdicts,
            backend_info={"name": "test", "version": "0.4.6"},
            thresholds=GateThresholds(),
        )
        assert isinstance(md, str)
        assert "ACCEPT" in md or "REJECT" in md or "REVIEW" in md


# ===========================================================================
# inter_shot_loss
# ===========================================================================

class TestInterShotLoss:

    def _frames(self, n=6, h=128, w=192, fill=128):
        return [np.full((h, w, 3), fill, dtype=np.uint8) for _ in range(n)]

    def test_module_imports(self):
        from cineinfini.core.inter_shot_loss import (
            InterShotCoherenceLoss, InterShotLossResult,
            extract_structure_histogram, extract_style_moments,
        )
        assert InterShotCoherenceLoss is not None

    def test_extract_structure_histogram(self):
        from cineinfini.core.inter_shot_loss import extract_structure_histogram
        frames = self._frames()
        hist = extract_structure_histogram(frames)
        assert isinstance(hist, np.ndarray)
        assert hist.shape[0] > 0

    def test_extract_style_moments(self):
        from cineinfini.core.inter_shot_loss import extract_style_moments
        frames = self._frames(fill=128)
        moments = extract_style_moments(frames)
        assert isinstance(moments, np.ndarray)

    def test_loss_compute_returns_result(self):
        from cineinfini.core.inter_shot_loss import (
            InterShotCoherenceLoss, InterShotLossResult,
        )
        loss_fn = InterShotCoherenceLoss()
        a = self._frames(fill=100)
        b = self._frames(fill=200)
        result = loss_fn.compute(a, b)
        assert isinstance(result, InterShotLossResult)
        assert hasattr(result, "structure")
        assert hasattr(result, "style")


# ===========================================================================
# prompt_engineering
# ===========================================================================

class TestPromptEngineering:

    def test_module_imports(self):
        from cineinfini.core.prompt_engineering import (
            ShotPrompt, build_prompt, build_all_prompts,
        )
        from cineinfini.core.shot_registry import ShotMetadata
        assert build_prompt is not None
        assert ShotMetadata is not None

    def test_basic_prompt_no_safety_issue(self):
        from cineinfini.core.prompt_engineering import build_prompt
        from cineinfini.core.shot_registry import ShotMetadata
        meta = ShotMetadata(
            shot_id=1,
            n_words=10,
            characters=["a runner"],
            locations=["sunny park"],
        )
        prompt = build_prompt(meta)
        # Should produce a ShotPrompt-like object
        assert prompt is not None

    def test_prompt_neutralizes_real_person(self):
        from cineinfini.core.prompt_engineering import _apply_real_person_neutralization
        from cineinfini.core.shot_registry import ShotMetadata
        meta = ShotMetadata(
            shot_id=2,
            n_words=15,
            real_people_mentioned=["Elon Musk"],
        )
        notes, warnings = _apply_real_person_neutralization(meta)
        assert isinstance(notes, list)
        assert isinstance(warnings, list)

    def test_build_all_prompts_handles_list(self):
        from cineinfini.core.prompt_engineering import build_all_prompts
        from cineinfini.core.shot_registry import ShotMetadata
        registry = [
            ShotMetadata(shot_id=i, n_words=10, locations=["park"])
            for i in range(3)
        ]
        prompts = build_all_prompts(registry)
        assert len(prompts) == 3


# ===========================================================================
# face_detection migration to get_config()
# ===========================================================================

class TestFaceDetectionConfigMigration:

    def test_resolve_falls_back_to_config(self):
        """Without explicit override, _resolve_models_dir() reads from config."""
        from cineinfini.core import face_detection as fd
        from cineinfini.core.config import get_config

        original_override = fd.MODELS_DIR
        try:
            fd.MODELS_DIR = None
            resolved = fd._resolve_models_dir()
            assert resolved is not None
            assert resolved == get_config().models_dir()
        finally:
            fd.MODELS_DIR = original_override

    def test_explicit_override_takes_priority(self, tmp_path):
        """set_models_dir() overrides config."""
        from cineinfini.core import face_detection as fd
        custom = tmp_path / "custom"
        custom.mkdir()
        original = fd.MODELS_DIR
        try:
            fd.set_models_dir(custom)
            assert fd._resolve_models_dir() == custom
        finally:
            fd.MODELS_DIR = original


# ===========================================================================
# Inter-report decomposition (added in v0.4.6)
# ===========================================================================

class TestInterReportDecomposition:

    def test_load_audit_data_missing(self, tmp_path):
        from cineinfini.io.report import _load_audit_data
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _load_audit_data(empty) is None

    def test_load_audit_data_valid(self, tmp_path):
        from cineinfini.io.report import _load_audit_data
        d = tmp_path / "ad"
        d.mkdir()
        (d / "data.json").write_text(json.dumps({"gates": {}}))
        data = _load_audit_data(d)
        assert data == {"gates": {}}

    def test_load_audit_data_malformed_json(self, tmp_path):
        from cineinfini.io.report import _load_audit_data
        d = tmp_path / "bad"
        d.mkdir()
        (d / "data.json").write_text("not valid json {{{")
        assert _load_audit_data(d) is None

    def test_aggregate_per_video_metrics(self, tmp_path):
        from cineinfini.io.report import _aggregate_per_video_metrics
        data = {
            "gates": {
                "1": {"motion_peak_div": 10.0, "ssim3d_self": 0.5,
                      "flicker": 0.05, "identity_intra": 0.3,
                      "ssim_long_range": 0.7, "clip_temp_consistency": 0.8,
                      "composite": 1.5},
                "2": {"motion_peak_div": 20.0, "ssim3d_self": 0.7,
                      "flicker": 0.10, "identity_intra": 0.4,
                      "ssim_long_range": 0.8, "clip_temp_consistency": 0.9,
                      "composite": 2.5},
            }
        }
        d = tmp_path / "video1"
        d.mkdir()
        result = _aggregate_per_video_metrics(d, data)
        assert result["video"] == "video1"
        assert result["motion_mean"] == 15.0
        assert result["n_shots"] == 2
        assert abs(result["ssim_mean"] - 0.6) < 0.001

    def test_aggregate_handles_none_values(self, tmp_path):
        from cineinfini.io.report import _aggregate_per_video_metrics
        data = {
            "gates": {
                "1": {"motion_peak_div": None, "ssim3d_self": 0.5,
                      "flicker": None, "identity_intra": 0.3},
            }
        }
        d = tmp_path / "v"
        d.mkdir()
        result = _aggregate_per_video_metrics(d, data)
        assert result["motion_mean"] == 0.0   # None values filtered
        assert result["ssim_mean"] == 0.5
