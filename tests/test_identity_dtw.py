"""
Tests for the DTW-based identity coherence module.

These tests do NOT require any face detection model — they exercise the
DTW math directly on synthetic embedding sequences. The key assertion is
that on a "glitch-and-revert" trajectory, DTW reveals the temporal
anomaly that the mean-based metric averages out.
"""
import numpy as np
import pytest

from cineinfini.core.identity_dtw import (
    dtw_distance,
    dtw_available,
    identity_within_shot_dtw,
    identity_drift_compare,
    IdentityDtwResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _slerp(a, b, t):
    omega = np.arccos(np.clip(np.dot(a, b), -1, 1))
    if omega < 1e-6:
        return a
    return (
        np.sin((1 - t) * omega) / np.sin(omega) * a
        + np.sin(t * omega) / np.sin(omega) * b
    )


# Three near-orthogonal "identity" vectors in 8-D
V1 = _normalize(np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
V2 = _normalize(np.array([0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0]))
V3 = _normalize(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0]))


def _build_seq(scenario: str, T: int = 16, noise: float = 0.02, seed: int = 42):
    """Construct a synthetic embedding sequence."""
    rng = np.random.default_rng(seed)
    embs = []
    for i in range(T):
        if scenario == "stable":
            e = V1
        elif scenario == "drift":
            e = _slerp(V1, V2, i / (T - 1))
        elif scenario == "glitch_revert":
            # mostly V1, brief excursion to V3 in the middle, then back to V1
            e = V3 if 6 <= i <= 9 else V1
        else:
            raise ValueError(scenario)
        e = _normalize(e + rng.normal(0, noise, size=e.shape))
        embs.append(e)
    return np.array(embs)


def _mean_drift(embs: np.ndarray) -> float:
    """Replicates `identity_within_shot` mean-to-first-frame logic."""
    ref = embs[0]
    return float(np.mean([1.0 - float(np.dot(ref, e)) for e in embs[1:]]))


# ---------------------------------------------------------------------------
# Tests on dtw_distance directly (no detector/embedder needed)
# ---------------------------------------------------------------------------

class TestDtwDistance:

    def test_identical_sequences_zero_distance(self):
        seq = _build_seq("stable")
        d, pl, backend = dtw_distance(seq, seq)
        assert d == pytest.approx(0.0, abs=1e-6)
        assert pl == len(seq)
        assert backend == "manual"

    def test_orthogonal_vectors_unit_distance(self):
        """Cosine distance between orthogonal unit vectors = 1."""
        a = np.array([[1.0, 0.0, 0.0]] * 5)
        b = np.array([[0.0, 1.0, 0.0]] * 5)
        d, pl, _ = dtw_distance(a, b)
        # 5 alignments × cost 1.0 each = 5.0; normalized = 1.0
        assert d == pytest.approx(5.0, abs=1e-6)
        assert d / pl == pytest.approx(1.0, abs=1e-6)

    def test_empty_sequences_raise(self):
        with pytest.raises(ValueError):
            dtw_distance(np.zeros((0, 8)), np.zeros((5, 8)))

    def test_different_lengths_handled(self):
        """DTW should handle T_a != T_b gracefully."""
        a = _build_seq("stable", T=10)
        b = _build_seq("stable", T=15)
        d, pl, _ = dtw_distance(a, b)
        assert d >= 0
        assert pl >= max(len(a), len(b))


# ---------------------------------------------------------------------------
# Core scientific claim: DTW reveals what mean averages out
# ---------------------------------------------------------------------------

class TestDtwRevealsAnomalies:
    """These tests are the JUSTIFICATION for adding DTW. If they fail,
    DTW provides no extra signal beyond the mean and should not be added."""

    def test_stable_shot_both_metrics_low(self):
        """Both metrics agree: stable shot has low drift."""
        embs = _build_seq("stable")
        mean = _mean_drift(embs)
        half = len(embs) // 2
        dtw, pl, _ = dtw_distance(embs[:half], embs[half:])
        dtw_norm = dtw / pl
        assert mean < 0.05
        assert dtw_norm < 0.05

    def test_glitch_revert_dtw_higher_than_mean(self):
        """KEY TEST: a brief glitch-and-revert pattern is detected MORE
        strongly by DTW than by the mean.

        Setup: 16 frames, 12 of which match V1, 4 in the middle morph to V3,
        then revert to V1. The mean-to-first-frame averages over all 15
        non-reference frames, diluting the 4 anomalous ones. DTW between
        the two halves explicitly exposes the temporal mismatch.
        """
        embs = _build_seq("glitch_revert")
        mean = _mean_drift(embs)
        half = len(embs) // 2
        dtw, pl, _ = dtw_distance(embs[:half], embs[half:])
        dtw_norm = dtw / pl
        # Both should be > 0
        assert mean > 0.05
        assert dtw_norm > 0.05
        # DTW should be strictly higher (this is the value-add claim)
        assert dtw_norm > mean, (
            f"Expected DTW ({dtw_norm:.3f}) > mean ({mean:.3f}) on "
            f"glitch_revert; if this fails, DTW provides no extra signal."
        )

    def test_continuous_drift_both_detect(self):
        """Both metrics should flag continuous drift, even if magnitudes differ."""
        embs = _build_seq("drift")
        mean = _mean_drift(embs)
        half = len(embs) // 2
        dtw, pl, _ = dtw_distance(embs[:half], embs[half:])
        dtw_norm = dtw / pl
        # Both should signal nonzero drift
        assert mean > 0.1
        assert dtw_norm > 0.05


# ---------------------------------------------------------------------------
# High-level API with mock detector/embedder
# ---------------------------------------------------------------------------

class _MockDetector:
    """Always returns one face box."""
    def detect(self, frame):
        return [(10, 10, 50, 50)]


class _ScenarioEmbedder:
    """Embedder that returns a pre-computed sequence based on call count.
    This lets us simulate glitch-and-revert at the embedding level."""
    def __init__(self, scenario, T=16, seed=42):
        self.embeddings = _build_seq(scenario, T=T, seed=seed)
        self.idx = 0

    def embed(self, crop):
        if self.idx >= len(self.embeddings):
            return None
        v = self.embeddings[self.idx]
        self.idx += 1
        return v


class TestHighLevelApi:

    def test_identity_within_shot_dtw_returns_result(self):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(16)]
        detector = _MockDetector()
        embedder = _ScenarioEmbedder("stable", T=16)
        result = identity_within_shot_dtw(frames, detector, embedder, max_samples=16)
        assert isinstance(result, IdentityDtwResult)
        assert result.distance is not None
        assert result.normalized is not None
        assert 0.0 <= result.normalized <= 0.05  # stable scenario
        assert result.n_a + result.n_b <= 16
        assert result.backend == "manual"

    def test_identity_within_shot_dtw_too_few_embeddings(self):
        """If fewer than 4 valid embeddings, return None."""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(2)]
        detector = _MockDetector()
        embedder = _ScenarioEmbedder("stable", T=2)
        result = identity_within_shot_dtw(frames, detector, embedder, max_samples=16)
        assert result.distance is None
        assert result.normalized is None

    def test_identity_drift_compare_returns_both(self):
        """The compare helper returns both mean and dtw_self values."""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(16)]
        detector = _MockDetector()
        embedder = _ScenarioEmbedder("glitch_revert", T=16)
        comp = identity_drift_compare(
            frames, detector, embedder, n_samples=5, max_samples_dtw=16
        )
        assert "mean" in comp
        assert "dtw_self" in comp
        assert "n_embeddings" in comp
        assert comp["n_embeddings"] == 16
        assert comp["mean"] is not None
        assert comp["dtw_self"] is not None


# ---------------------------------------------------------------------------
# Sanity check on the optional dependency declaration
# ---------------------------------------------------------------------------

def test_dtw_available_returns_bool():
    """`dtw_available()` should return True or False, never crash."""
    val = dtw_available()
    assert isinstance(val, bool)
