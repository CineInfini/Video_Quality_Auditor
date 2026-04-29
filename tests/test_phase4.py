"""Test the phase-4 aggregator."""
from __future__ import annotations

import pytest

from cineinfini.core.phase4_aggregator import (
    GateThresholds, ShotVerdict,
    aggregate_shot_verdict, build_phase4_report,
)


def _good_metrics():
    return {
        "motion_peak_div": 1.0,
        "ssim3d_self": 0.9,
        "flicker": 0.05,
        "identity_intra": 0.1,
        "ssim_long_range": 0.85,
        "clip_temp_consistency": 0.9,
        "flicker_hf": 0.001,
        "background_ssim": 0.8,
    }


def _bad_metrics():
    return {
        "motion_peak_div": 999.0,
        "ssim3d_self": 0.1,
        "flicker": 0.9,
        "identity_intra": 0.95,
        "ssim_long_range": 0.05,
        "clip_temp_consistency": 0.05,
        "flicker_hf": 0.5,
        "background_ssim": 0.05,
    }


def test_gatethresholds_from_config():
    th = GateThresholds.from_config({"motion": 50.0, "ssim3d": 0.6})
    assert th.motion == 50.0
    assert th.ssim3d == 0.6
    # Defaults preserved
    assert th.flicker == 0.10


def test_aggregate_shot_verdict_accept():
    v = aggregate_shot_verdict(0, _good_metrics(), GateThresholds())
    assert v.verdict == "ACCEPT"
    assert len(v.gates_passed) == 8
    assert v.gates_failed == []
    assert v.composite == 1.0


def test_aggregate_shot_verdict_reject():
    v = aggregate_shot_verdict(1, _bad_metrics(), GateThresholds())
    assert v.verdict == "REJECT"
    assert len(v.gates_failed) == 8
    assert v.composite == 0.0


def test_aggregate_shot_verdict_review_partial():
    metrics = _good_metrics()
    metrics["motion_peak_div"] = 999.0
    metrics["flicker"] = 0.9
    metrics["ssim3d_self"] = 0.1
    metrics["flicker_hf"] = 0.5
    v = aggregate_shot_verdict(2, metrics, GateThresholds())
    # 4/8 gates pass -> composite=0.5 -> REVIEW (>= 0.3 < 0.6)
    assert v.verdict == "REVIEW"
    assert len(v.gates_passed) == 4
    assert len(v.gates_failed) == 4


def test_aggregate_shot_verdict_skips_none():
    metrics = _good_metrics()
    metrics["motion_peak_div"] = None
    v = aggregate_shot_verdict(3, metrics, GateThresholds())
    assert "motion_peak_div" not in v.gates_passed
    assert "motion_peak_div" not in v.gates_failed


def test_shot_verdict_to_dict():
    v = aggregate_shot_verdict(0, _good_metrics(), GateThresholds())
    d = v.to_dict()
    for key in ("shot_id", "composite", "verdict", "gates_passed", "gates_failed"):
        assert key in d


def test_build_phase4_report():
    metrics = {1: _good_metrics(), 2: _bad_metrics()}
    report = build_phase4_report(metrics)
    assert report["summary"]["n_shots"] == 2
    assert report["summary"]["n_accept"] == 1
    assert report["summary"]["n_reject"] == 1
    assert report["summary"]["mean_composite"] == 0.5
    assert 1 in report["verdicts"]
    assert 2 in report["verdicts"]
