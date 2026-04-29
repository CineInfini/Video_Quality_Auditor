"""Tests for inter_shot_loss, prompt_engineering, shot_registry, calibrate."""
from __future__ import annotations

import csv
import numpy as np
import pytest

from cineinfini.core.inter_shot_loss import (
    InterShotCoherenceLoss, InterShotLossResult,
)
from cineinfini.core.prompt_engineering import (
    ShotPrompt, build_prompt, build_all_prompts,
)
from cineinfini.core.shot_registry import ShotMetadata
from cineinfini.core.calibrate import (
    CalibrationResult, calibrate_from_csv,
    grid_search_thresholds, logistic_regression_weights,
)


# ---------------------------------------------------------------------------
# InterShotCoherenceLoss
# ---------------------------------------------------------------------------
class TestInterShotLoss:
    def test_identical_embeddings_zero_distance(self):
        loss = InterShotCoherenceLoss()
        emb = [1.0, 0.0, 0.0]
        r = loss.evaluate_pair(0, 1, semantic_a=emb, semantic_b=emb)
        assert r.semantic_distance == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_embeddings_unit_distance(self):
        loss = InterShotCoherenceLoss()
        a, b = [1.0, 0.0], [0.0, 1.0]
        r = loss.evaluate_pair(0, 1, semantic_a=a, semantic_b=b)
        assert r.semantic_distance == pytest.approx(1.0, abs=1e-6)

    def test_ssim_distance(self):
        loss = InterShotCoherenceLoss()
        r = loss.evaluate_pair(0, 1, visual_ssim=0.9)
        assert r.visual_distance == pytest.approx(0.1, abs=1e-6)

    def test_all_none_yields_zero(self):
        loss = InterShotCoherenceLoss()
        r = loss.evaluate_pair(0, 1)
        assert r.composite_loss == 0.0

    def test_evaluate_sequence(self):
        loss = InterShotCoherenceLoss()
        results = loss.evaluate_sequence(
            [1, 2, 3],
            visual_ssims={2: 0.9, 3: 0.7},
            semantic_embs={1: [1, 0], 2: [1, 0], 3: [0, 1]},
        )
        assert len(results) == 2
        assert results[0].a == 1 and results[0].b == 2
        assert results[1].a == 2 and results[1].b == 3

    def test_aggregate(self):
        loss = InterShotCoherenceLoss()
        results = [
            InterShotLossResult(a=1, b=2, composite_loss=0.1),
            InterShotLossResult(a=2, b=3, composite_loss=0.3),
        ]
        agg = loss.aggregate(results)
        assert agg["n_pairs"] == 2
        assert agg["mean_loss"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# ShotPrompt / build_prompt
# ---------------------------------------------------------------------------
class TestPromptEngineering:
    def test_build_prompt_default(self):
        p = build_prompt(3, video_name="my_video")
        assert isinstance(p, ShotPrompt)
        assert p.shot_id == 3
        assert "shot 3" in p.text
        assert "my_video" in p.text

    def test_build_prompt_alignment(self):
        p = build_prompt(1, "vid", template="alignment", description="a cat")
        assert "a cat" in p.text

    def test_build_prompt_raw_template(self):
        p = build_prompt(5, "x", template="custom: {shot_id}!")
        assert p.text == "custom: 5!"

    def test_build_prompt_safe_format_handles_missing_keys(self):
        p = build_prompt(1, "vid", template="ok {missing} stop")
        assert "ok " in p.text and "stop" in p.text

    def test_build_all_prompts(self):
        prompts = build_all_prompts([1, 2, 3], "vid")
        assert len(prompts) == 3
        assert {p.shot_id for p in prompts} == {1, 2, 3}

    def test_to_dict(self):
        p = build_prompt(7, "v")
        d = p.to_dict()
        assert d["shot_id"] == 7
        assert d["template"] == "default"


# ---------------------------------------------------------------------------
# ShotMetadata
# ---------------------------------------------------------------------------
class TestShotMetadata:
    def test_construction(self):
        s = ShotMetadata(shot_id=1, start_frame=0, end_frame=23)
        assert s.n_frames == 24

    def test_from_tuple(self):
        s = ShotMetadata.from_tuple(2, (10, 33, 1.0))
        assert s.shot_id == 2
        assert s.start_frame == 10
        assert s.duration_s == 1.0

    def test_from_tuple_no_duration(self):
        s = ShotMetadata.from_tuple(3, (0, 23), fps=24.0)
        assert s.duration_s == pytest.approx(1.0)

    def test_to_dict_and_back(self):
        s = ShotMetadata(shot_id=1, start_frame=0, end_frame=23, label="hero")
        d = s.to_dict()
        s2 = ShotMetadata.from_dict(d)
        assert s2.shot_id == s.shot_id
        assert s2.label == "hero"

    def test_time_seconds(self):
        s = ShotMetadata(shot_id=1, start_frame=24, end_frame=48, fps=24.0)
        assert s.start_time_s == pytest.approx(1.0)
        assert s.end_time_s == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------
class TestCalibrate:
    def test_grid_search_basic(self):
        rng = np.random.default_rng(0)
        n = 50
        x_good = rng.normal(0.2, 0.05, n)
        x_bad = rng.normal(0.9, 0.05, n)
        X = np.concatenate([x_good, x_bad]).reshape(-1, 1)
        y = np.array([0] * n + [1] * n)
        # higher value means "bad" (label=1) so direction = higher_is_better
        res = grid_search_thresholds(X, y, ["flicker"], n_grid=20,
                                     higher_is_better={"flicker": True})
        assert isinstance(res, CalibrationResult)
        assert "flicker" in res.thresholds
        assert 0.2 < res.thresholds["flicker"] < 1.0
        assert res.score > 0.5

    def test_logistic_regression(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (40, 3))
        y = (X[:, 0] > 0).astype(int)
        res = logistic_regression_weights(X, y, ["a", "b", "c"])
        assert "a" in res.weights
        assert res.score >= 0.5

    def test_calibrate_from_csv(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["motion", "label"])
            for i in range(20):
                w.writerow([0.1 if i < 10 else 0.9, 0 if i < 10 else 1])
        res = calibrate_from_csv(csv_path, method="grid")
        assert "motion" in res.thresholds

    def test_calibrate_missing_label_raises(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["a", "b"])
            w.writerow([1, 2])
        with pytest.raises(ValueError, match="label"):
            calibrate_from_csv(csv_path)
