import pytest
import numpy as np
from cineinfini.core.coherence import compute_inter_shot_coherence, compute_narrative_coherence


class TestInterShotCoherence:
    def test_identical_shots(self, sample_shot_frames):
        shot_frames = {1: sample_shot_frames[1], 2: sample_shot_frames[1]}
        results = compute_inter_shot_coherence(shot_frames, None, None, "cpu")
        assert len(results) == 1
        r = results[0]
        assert r["structure"] > 0.99
        assert r["style"] > 0.99
        assert r["semantic"] is None
        assert r["clip_available"] is False

    def test_different_shots(self, sample_shot_frames):
        results = compute_inter_shot_coherence(sample_shot_frames, None, None, "cpu")
        r = results[0]
        assert r["structure"] < 0.99
        assert 0.0 <= r["total"] <= 1.0


class TestNarrativeCoherence:
    def test_no_dinov2(self):
        result = compute_narrative_coherence({}, None, None, "cpu")
        assert result is None
